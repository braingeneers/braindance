import json

from braindance import config


def test_set_output_dir_works_without_a_configured_data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config.Path, "home", classmethod(lambda cls: tmp_path))

    output_dir = tmp_path / "plots"
    config.set_output_dir(output_dir)

    saved = json.loads((tmp_path / ".braindance" / "config.json").read_text())
    assert saved == {"output_dir": str(output_dir)}


def test_get_data_dir_cli_flag_does_not_require_a_value(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(config.Path, "home", classmethod(lambda cls: tmp_path))

    config.main(["--get_data_dir"])

    assert f"Current data directory:  {tmp_path / 'braindance_data'}" in capsys.readouterr().out


def test_global_settings_roundtrip_and_defaults(tmp_path, monkeypatch):
    monkeypatch.setattr(config.Path, 'home', classmethod(lambda cls: tmp_path))
    monkeypatch.setenv('HOME', str(tmp_path))
    for key in ('DATA_DIR', 'CATALOG_PATH', 'OUTPUT_DIR', 'AUTO_EXTRACT_SPIKE_INFO'):
        monkeypatch.delenv('BRAINDANCE_' + key, raising=False)
    folder = tmp_path / '.braindance'
    folder.mkdir()
    (folder / 'config.json').write_text('{"unrelated": "preserve"}')
    result = config.save_global_settings({'data_dir': '~/recordings', 'auto_extract_spike_info': True})
    assert result['settings']['catalog_path']['effective'] == str(tmp_path / 'recordings/catalog.csv')
    assert config.get_auto_extract_spike_info() is True
    assert json.loads((folder / 'config.json').read_text())['unrelated'] == 'preserve'
    monkeypatch.setenv('BRAINDANCE_DATA_DIR', '/override')
    setting = config.get_global_settings()['settings']['data_dir']
    assert setting == {'saved': str(tmp_path / 'recordings'), 'effective': '/override',
                       'environment': 'BRAINDANCE_DATA_DIR'}
    config.save_global_settings({'data_dir': None, 'auto_extract_spike_info': None})
    assert json.loads((folder / 'config.json').read_text()) == {'unrelated': 'preserve'}


def test_global_settings_reject_invalid_without_partial_write(tmp_path, monkeypatch):
    import pytest
    monkeypatch.setattr(config.Path, 'home', classmethod(lambda cls: tmp_path))
    config.save_global_settings({'data_dir': str(tmp_path / 'old')})
    path = tmp_path / '.braindance/config.json'
    original = path.read_bytes()
    for invalid in ({'data_dir': str(tmp_path / 'new'), 'auto_extract_spike_info': 'false'},
                    {'output_dir': 'relative'}, {'unknown': True}, {'data_dir': ''},
                    {'catalog_path': str(tmp_path)}, {'output_dir': str(path)}, []):
        with pytest.raises(ValueError):
            config.save_global_settings(invalid)
        assert path.read_bytes() == original
    path.write_text('{broken')
    with pytest.raises(ValueError):
        config.save_global_settings({'data_dir': str(tmp_path)})
    assert path.read_text() == '{broken'
