"""End-to-end check for the real BusyBee workshop fixture."""

import json
import subprocess
import sys
from pathlib import Path


FIXTURE = Path(__file__).parents[1] / "braindance/examples/streaming_workshop/real_experiment_busy_bee.json"


def test_real_busy_bee_fixture_runs_and_exports(tmp_path):
    from braindance.examples.streaming_workshop.code_export import export_bundle
    from braindance.examples.streaming_workshop.native_runner import verify_native

    config = json.loads(FIXTURE.read_text(encoding="utf-8"))
    report = verify_native(config)
    assert report["ok"], report["errors"]
    expected = [f"cycle_01_{str(frequency).replace('.', 'p')}hz_{kind}"
                for frequency in (0.5, 1, 2, 4, 8)
                for kind in ("record", "stimulate")]
    assert [row["id"] for row in report["phases"]] == expected
    assert [row["params"].get("stim_freq") for row in report["phases"] if row["id"].endswith("stimulate")] == [0.5, 1, 2, 4, 8]
    assert all(row["params"].get("phase_length") == 200 for row in report["phases"] if row["id"].endswith("stimulate"))

    bundle = export_bundle(tmp_path, config, "# Native V3 phases do not use streaming hooks.\n")
    export_dir = Path(bundle["directory"])
    assert (export_dir / "builder_snapshot.json").exists()

    result = subprocess.run(
        [sys.executable, str(export_dir / "experiment.py"), "--output-dir", str(tmp_path / "runs")],
        cwd=export_dir,
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"status": "completed"' in result.stdout
