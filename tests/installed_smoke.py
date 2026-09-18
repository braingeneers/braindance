"""Distribution smoke test: run with python -I outside the source checkout.

Uses only the installed package and the standard library. Pass --gpu to also
exercise both bundled pretrained detectors on CUDA (no private data required).
Pass --games to exercise all three games included in the default install.
"""
import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request


def main(gpu=False, games=False):
    import braindance
    import braindance.core.simulation
    import braindance.core.replay
    import braindance.core.maxwell_env
    import braindance.core.phases_v3
    import braindance.examples.streaming_workshop.main

    package = Path(braindance.__file__).resolve()
    assert package.is_relative_to(Path(sys.prefix).resolve()), package
    assert braindance.__version__ == importlib.metadata.version('braindance')
    for resource in (
        'core/maxwell/stim_buffers.npy',
        'utils/rt_linear_art_removal.py',
        'utils/data_manager/__init__.py',
        'core/spikedetector/detection_models/mea/init_dict.json',
        'core/spikedetector/detection_models/mea/state_dict.pt',
        'core/spikedetector/detection_models/neuropixels/init_dict.json',
        'core/spikedetector/detection_models/neuropixels/state_dict.pt',
    ):
        assert (package.parent / resource).is_file(), f'Missing packaged resource: {resource}'
    if not gpu:
        for name in ('torch', 'torch_tensorrt', 'maxlab',
                     'stable_baselines3', 'spikeinterface'):
            assert importlib.util.find_spec(name) is None, f'CPU environment contains {name}'
        from spikelab import SpikeData
        from braindance.utils.data_manager import load_catalog
        assert SpikeData and load_catalog
    print(f'Installed package: {package}', flush=True)
    subprocess.run([sys.executable, '-I', '-m', 'pip', 'check'], check=True)

    if games:
        import numpy as np
        from braindance.examples.streaming_workshop.environments import WorkshopGame

        # Missing optional dependencies must fail, rather than silently skip.
        for name in ('cartpole', 'foodland', 'ant'):
            game = WorkshopGame(name)
            try:
                observation = game.reset()
                assert np.isfinite(observation).all()
                for _ in range(10):
                    observation, reward, done = game.step(np.full(len(game.action_names), .25))
                    assert np.isfinite(observation).all() and np.isfinite(reward)
                    scene = game.scene(observation)
                    assert scene['kind'] == name
                    if name == 'ant':
                        assert scene['geometry']
                    json.dumps(scene, allow_nan=False)
                    if done:
                        game.reset()
                print(f'{name}: reset, actions and browser scene passed')
            finally:
                game.close()

    with tempfile.TemporaryDirectory(prefix='braindance-smoke-') as directory:
        work = Path(directory)
        env = {**os.environ, 'PYTHONNOUSERSITE': '1', 'PYTHONIOENCODING': 'utf-8',
               'MPLBACKEND': 'Agg'}
        command = [sys.executable, '-I', '-X', 'utf8', '-m', 'braindance.examples.streaming_workshop.main']
        for mode in ('events', 'threshold'):
            result = subprocess.run(
                command + ['--headless', '--record-seconds', '0.1', '--causal-repeats', '1',
                           '--environment-seconds', '0.2', '--speed', 'max',
                           '--detection', mode, '--output-dir', str(work / mode)],
                cwd=work, env=env, capture_output=True, text=True, encoding='utf-8', timeout=90)
            print(result.stdout)
            assert result.returncode == 0, result.stderr
            assert '"status": "completed"' in result.stdout, result.stdout

        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        base = f'http://127.0.0.1:{port}'
        with (work / 'server.log').open('w', encoding='utf-8') as log:
            server = subprocess.Popen(command + ['--no-browser', '--port', str(port),
                                      '--output-dir', str(work / 'browser')],
                                      cwd=work, env=env, stdout=log, stderr=log)
            try:
                for _ in range(100):
                    try:
                        with urllib.request.urlopen(base + '/api/state', timeout=1) as response:
                            assert json.load(response)['execution']['source'] == 'Simulation'
                        break
                    except OSError:
                        if server.poll() is not None:
                            raise RuntimeError((work / 'server.log').read_text(encoding='utf-8'))
                        time.sleep(0.1)
                else:
                    raise RuntimeError('Workshop server did not start')
                for asset in ('/', '/watcher.js', '/ui.js', '/builder.js', '/code_workspace.js'):
                    with urllib.request.urlopen(base + asset, timeout=5) as response:
                        body = response.read()
                        assert response.status == 200 and len(body) > 100, asset
                        assert '__TOKEN__' not in body.decode('utf-8'), asset
                        assert ('text/html' if asset == '/' else 'text/javascript') in response.headers['Content-Type']
                with urllib.request.urlopen(base + '/api/phase-catalog', timeout=5) as response:
                    assert json.load(response), 'Empty phase catalog'
                print('Workshop HTTP, phase catalog, HTML and all four JS assets: passed')
            finally:
                server.terminate()
                server.wait(timeout=15)

    if gpu:
        from types import SimpleNamespace
        import numpy as np
        import torch
        from braindance.core.spikedetector.model import ModelSpikeSorter
        from braindance.core.spikesorter.rt_sort import RTSort

        assert torch.cuda.is_available(), 'CUDA unavailable'
        print(f'GPU: {torch.cuda.get_device_name(0)}; torch={torch.__version__}; CUDA={torch.version.cuda}')
        for kind in ('mea', 'neuropixels'):
            model = getattr(ModelSpikeSorter, f'load_{kind}')().eval()
            inputs = torch.randn(4, 1, model.sample_size, device='cuda', dtype=torch.float16)
            with torch.inference_mode():
                outputs = model(inputs)
                assert outputs.is_cuda and torch.isfinite(outputs).all()
                assert outputs.shape == (4, model.num_output_locs), outputs.shape
                compiled = model.compile(4)
                traced = compiled(inputs * model.input_scale)
                torch.testing.assert_close(traced.squeeze(1), outputs, rtol=1e-2, atol=1e-2)
                torch.cuda.synchronize()
            print(f'{kind}: pretrained CUDA forward and compiled inference passed {tuple(outputs.shape)}')
        # Synthetic sequence template: exercise the actual detector/matcher path
        # without claiming validation of biological sorting or sequence discovery.
        model = ModelSpikeSorter.load_mea().eval()
        sequence = SimpleNamespace(
            root_elec=0, spike_train=np.array([]), comp_elecs=[0, 1],
            inner_loose_elecs=[0, 1], loose_elecs=[0, 1], min_loose_detections=2,
            all_latencies=np.array([0., 0.]), all_amp_medians=np.array([5., 5.]),
            all_elec_probs=np.array([1., 1.]), root_to_amp_median_std={0: 1.})
        with tempfile.TemporaryDirectory(prefix='braindance-rtsort-') as directory:
            params = dict(
                samp_freq=20, elec_locs=np.array([[0., 0.], [10., 0.]]),
                model_inter_path=Path(directory), stringent_thresh=0.275,
                loose_thresh=0.1, inference_scaling_numerator=None,
                n_before=10, n_after=10, pre_median_frames=200, inner_radius=50,
                min_elecs_for_array_noise=3, min_inner_loose_detections=2,
                max_latency_diff_spikes=3.5, clip_latency_diff_factor=2,
                max_amp_median_diff_spikes=0.65, clip_amp_median_diff_factor=2,
                max_root_amp_median_std_spikes=2.5, repeated_detection_overlap_time=0.2)
            sorter = RTSort([sequence], model, params, device='cuda')
            sorter.pre_medians = torch.ones((1, 1, 2), device='cuda', dtype=torch.float16)
            with torch.inference_mode():
                sorter.sort_chunk(torch.randn((2, 200), device='cuda', dtype=torch.float16))
                torch.cuda.synchronize()
                chunk = torch.zeros((2, 200), device='cuda', dtype=torch.float16)
                chunk[:, 100] = -5
                logits = torch.full((2, 120), -10., device='cuda', dtype=torch.float16)
                logits[:, 60] = 10
                assigned = sorter.sort_chunk(chunk, torch_window=logits)
                assert assigned == [(0, 5.0)], assigned
            print('RTSort CUDA detector/matcher and controlled unit assignment: passed')
    print('Installed distribution smoke: passed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--games', action='store_true', help='Require and exercise all three workshop games')
    main(**vars(parser.parse_args()))
