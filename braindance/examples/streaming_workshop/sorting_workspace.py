"""Uploaded recording batches and background sorting for the click-only workspace."""
from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import threading
import uuid

from .batch_sorters import run_batch_sorting, sorter_choices
from .sorter_installation import installation_plans, install_sorter, validate_installation


class SortingWorkspace:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir) / 'sorting'
        self.files = {}
        self.jobs = {}
        self.installation = None
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='workshop-sorting')

    def close(self):
        self.executor.shutdown(wait=False, cancel_futures=True)

    def add_upload(self, path, name):
        import h5py
        from .analysis_workspace import _h5_info
        path = Path(path)
        if not h5py.is_hdf5(path):
            raise ValueError('Choose a raw Maxwell HDF5 or NWB recording, not a renamed data file.')
        if path.suffix == '.h5':
            try:
                info = _h5_info(path)
                with h5py.File(path, 'r') as handle:
                    raw = handle.get(info['raw_key']) if info['raw_key'] else None
                    valid = raw is not None and raw.ndim == 2 and all(raw.shape)
            except (IndexError, TypeError):
                valid = False
            if not valid:
                raise ValueError('This HDF5 file contains no supported Maxwell raw voltage data.')
        else:
            with h5py.File(path, 'r') as handle:
                def electrical_series(name, item):
                    kind = item.attrs.get('neurodata_type', '')
                    if kind in ('ElectricalSeries', b'ElectricalSeries') and isinstance(item, h5py.Group):
                        data = item.get('data')
                        if isinstance(data, h5py.Dataset) and data.ndim in (1, 2) and all(data.shape):
                            return True
                if not handle.visititems(electrical_series):
                    raise ValueError('This NWB file contains no raw ElectricalSeries data.')
        row = dict(id=uuid.uuid4().hex, name=name, path=str(path.resolve()), size=path.stat().st_size)
        with self.lock:
            self.files[row['id']] = row
        return copy.deepcopy(row)

    def control(self, command):
        action = command.get('action', 'status')
        if action == 'install':
            with self.lock:
                if self.installation:
                    raise ValueError('Restart the workshop before another installation attempt')
                if any(job['status'] in ('queued', 'running') for job in self.jobs.values()):
                    raise ValueError('Wait for the active sorting batch before installing')
                plan = command.get('plan')
                if plan not in {p['id'] for p in installation_plans()}:
                    raise ValueError('Choose a supported installation plan')
                validate_installation(plan)
                self.installation = dict(plan=plan, status='running', restart_required=True,
                    message='Installing packages…', log_path=str(self.output_dir / 'installs' / (uuid.uuid4().hex + '.log')))
                self.executor.submit(self._install, plan, self.installation['log_path'])
                return copy.deepcopy(self.installation)
        if action == 'status':
            with self.lock:
                result = dict(files=copy.deepcopy(list(self.files.values())),
                              jobs=copy.deepcopy(list(self.jobs.values())), installation=copy.deepcopy(self.installation))
            result['install_plans'] = installation_plans()
            result['sorters'] = [] if result['installation'] else sorter_choices()
            if result['installation']:
                try:
                    with Path(result['installation']['log_path']).open('rb') as log:
                        log.seek(0, 2)
                        log.seek(max(0, log.tell() - 16000))
                        result['installation']['log'] = log.read().decode('utf-8', errors='replace')
                except FileNotFoundError:
                    result['installation']['log'] = ''
            return result
        if action != 'run':
            raise ValueError('Unknown sorting action')
        with self.lock:
            if self.installation:
                raise ValueError('Restart the workshop after installing sorter dependencies')
            if any(job['status'] in ('queued', 'running') for job in self.jobs.values()):
                raise ValueError('A sorting batch is already running')
            ids = command.get('target_ids')
            if not isinstance(ids, list) or not ids or any(not isinstance(i, str) or i not in self.files for i in ids):
                raise ValueError('Select at least one uploaded recording to sort')
            baseline_id = command.get('baseline_id')
            if not isinstance(baseline_id, str) or baseline_id not in self.files:
                raise ValueError('Choose an uploaded baseline recording')
            selected = next((s for s in sorter_choices() if s['id'] == command.get('sorter')), None)
            if selected is None or not selected['available']:
                raise ValueError(selected['reason'] if selected else 'Choose an available sorter')
            targets = [copy.deepcopy(self.files[i]) for i in dict.fromkeys(ids)]
            baseline = copy.deepcopy(self.files[baseline_id])
            params = command.get('params') or {}
            if not isinstance(params, dict):
                raise ValueError('Sorter parameters must be an object')
            job = dict(id=uuid.uuid4().hex, status='queued', progress=0, message='Queued',
                       sorter=selected['id'], baseline=baseline, targets=targets,
                       params=copy.deepcopy(params), result=None, error=None)
            self.jobs[job['id']] = job
            self.executor.submit(self._worker, job['id'])
            return copy.deepcopy(job)

    def _install(self, plan, log_path):
        try:
            install_sorter(plan, log_path)
            status, message = 'completed', 'Installation complete. Restart the workshop to load the installed packages.'
        except Exception as exc:
            status, message = 'failed', f'{exc} Restart the workshop before retrying; pip may have changed some packages.'
        with self.lock:
            self.installation.update(status=status, message=message)

    def _worker(self, job_id):
        def progress(value, message):
            with self.lock:
                self.jobs[job_id].update(status='running', progress=value, message=message)
        with self.lock:
            job = copy.deepcopy(self.jobs[job_id])
        try:
            progress(0, 'Preparing sorting batch')
            result = run_batch_sorting(job['baseline']['path'], [f['path'] for f in job['targets']],
                                       self.output_dir, sorter=job['sorter'], params=job['params'], progress=progress)
            # Keep human filenames in the manifest alongside collision-safe storage paths.
            names = {f['path']: f['name'] for f in job['targets']}
            for recording in result['recordings']:
                recording['name'] = names.get(recording['recording_path'], Path(recording['recording_path']).name)
            manifest = {**job, 'status': 'completed', 'progress': 1, 'message': 'Batch complete', 'result': result}
            Path(result['output_dir']).mkdir(parents=True, exist_ok=True)
            (Path(result['output_dir']) / 'batch.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
            with self.lock:
                self.jobs[job_id].update(status='completed', progress=1, message='Batch complete', result=result)
        except Exception as exc:
            with self.lock:
                self.jobs[job_id].update(status='failed', message='Sorting failed', error=str(exc))

    def artifact(self, job_id, index):
        with self.lock:
            job = self.jobs.get(job_id)
            if not job or job['status'] != 'completed':
                raise ValueError('No completed sorting result for this job')
            recordings = job['result']['recordings']
            if not 0 <= index < len(recordings):
                raise ValueError('Unknown result file')
            return Path(recordings[index]['spikes_path'])
