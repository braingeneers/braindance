"""Nested dataset discovery keeps experiment and recording boundaries intact."""
import json

import numpy as np
import pytest

from braindance.examples.streaming_workshop.analysis_workspace import AnalysisWorkspace
from braindance.examples.streaming_workshop.catalog_workspace import CatalogWorkspace


def make_experiment(root, name="experiment"):
    experiment = root / name
    phases = []
    for index in range(1, 4):
        recording = experiment / f"{index:03d}_rec"
        recording.mkdir(parents=True)
        np.savez(recording / f"{index:03d}.npz", times_ms=[1., 5.], unit_ids=[0, 0], duration_ms=10.)
        phases.append(dict(phase_idx=index, phase_name=f"record {index}",
                           phase_class="RecordPhaseV3", recording_dir=recording.name))
    (experiment / "experiment_log.json").write_text(json.dumps(dict(experiment_name=name, phase_log=phases)))
    return experiment


def test_nested_tutorial_discovery_and_loading_preserve_three_recordings(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAINDANCE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("BRAINDANCE_CATALOG_PATH", str(tmp_path / "missing.csv"))
    dataset = tmp_path / "tutorials/dataset"
    experiment = make_experiment(dataset / "project/chip")
    catalog = CatalogWorkspace(tmp_path / "outputs")
    entries = catalog.control({})["entries"]
    assert [(entry["path"], entry["source"]) for entry in entries] == [(str(experiment), "Tutorial data")]
    workspace = AnalysisWorkspace(tmp_path / "outputs")
    try:
        for selected in [dataset, dataset / "project", experiment.parent, experiment]:
            state = workspace.control(dict(action="load", path=str(selected)))["workspace"]
            assert state["path"] == str(experiment)
            assert [phase["name"] for phase in state["phases"]] == ["record 1", "record 2", "record 3"]
            assert len(state["files"]) == 3
            assert all(len(phase["files"]) == 1 for phase in state["phases"])
            assert state["logs"] == [str(experiment / "experiment_log.json")]
    finally:
        workspace.close()


def test_parent_catalog_refresh_expands_experiments_without_merging(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAINDANCE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("BRAINDANCE_CATALOG_PATH", str(tmp_path / "missing.csv"))
    parent = tmp_path / "external/project/chip"
    first = make_experiment(parent, "first")
    catalog = CatalogWorkspace(tmp_path / "outputs")
    assert [entry["path"] for entry in catalog.control(dict(action="add", path=str(parent)))["entries"]] == [str(first)]
    second = make_experiment(parent, "second")
    entries = CatalogWorkspace(tmp_path / "outputs").control({})["entries"]
    assert {entry["path"] for entry in entries} == {str(first), str(second)}
    workspace = AnalysisWorkspace(tmp_path / "outputs")
    try:
        previous = workspace.control(dict(action="load", path=str(first)))["workspace"]
        with pytest.raises(ValueError, match="2 experiments.*Catalog"):
            workspace.control(dict(action="load", path=str(parent)))
        assert workspace.control({})["workspace"] == previous
    finally:
        workspace.close()


def test_nested_experiment_files_do_not_leak_into_outer_experiment(tmp_path):
    outer = make_experiment(tmp_path, "outer")
    make_experiment(outer / "saved", "inner")
    workspace = AnalysisWorkspace(tmp_path / "outputs")
    try:
        state = workspace.control(dict(action="load", path=str(outer)))["workspace"]
        assert len(state["files"]) == 3
        assert all("/saved/" not in row["path"] for row in state["files"])
    finally:
        workspace.close()


def test_configured_tutorial_catalog_uses_bundle_relative_paths(tmp_path, monkeypatch):
    bundle = tmp_path / 'tutorials/dataset'
    bundle.mkdir(parents=True)
    raw = bundle / 'project/chip/experiment/recording/002.raw.h5'
    raw.parent.mkdir(parents=True)
    raw.touch()
    (bundle / '.complete.json').write_text('{}')
    catalog_path = bundle / 'catalog.csv'
    catalog_path.write_text('proj,chip,experiment,raw_data_path\nproject,chip,experiment/recording,project/chip/experiment/recording/002.raw.h5\n')
    monkeypatch.setenv('BRAINDANCE_DATA_DIR', str(tmp_path))
    monkeypatch.setenv('BRAINDANCE_CATALOG_PATH', str(catalog_path))
    entries = CatalogWorkspace(tmp_path / 'outputs').control({})['entries']
    assert len(entries) == 1
    assert entries[0]['available']
    assert entries[0]['path'] == str(raw)
