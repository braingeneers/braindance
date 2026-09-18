"""Dependency installation remains opt-in, fixed, and isolated from sorting jobs."""
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from braindance.examples.streaming_workshop import sorter_installation as installer
from braindance.examples.streaming_workshop import sorting_workspace as workspace


@pytest.fixture(autouse=True)
def supported_platform(monkeypatch):
    monkeypatch.setattr(installer.sys, 'platform', 'darwin')
    monkeypatch.setattr(installer.platform, 'machine', lambda: 'arm64')


@pytest.mark.parametrize('plan', ['rt-sort', 'spikeinterface'])
def test_intel_macos_rejected_before_pip_or_restart(tmp_path, monkeypatch, plan):
    monkeypatch.setattr(installer.platform, 'machine', lambda: 'x86_64')
    def unexpected_pip(*args, **kwargs):
        pytest.fail('pip must not run on unsupported macOS Python')
    monkeypatch.setattr(installer.subprocess, 'run', unexpected_pip)
    log = tmp_path / 'install.log'
    with pytest.raises(ValueError, match='Intel.*macOS') as error:
        installer.install_sorter(plan, log)
    assert 'arm64' in str(error.value)
    assert 'pip was not started' in str(error.value)
    assert not log.exists()
    service = workspace.SortingWorkspace(tmp_path)
    try:
        with pytest.raises(ValueError, match='Intel.*macOS'):
            service.control(dict(action='install', plan=plan))
        assert service.installation is None
        with pytest.raises(ValueError, match='Select at least one'):
            service.control(dict(action='run'))
    finally:
        service.close()


@pytest.mark.parametrize('system,machine', [('darwin', 'arm64'), ('linux', 'x86_64'), ('win32', 'AMD64')])
def test_supported_platforms_pass_preflight(monkeypatch, system, machine):
    monkeypatch.setattr(installer.sys, 'platform', system)
    monkeypatch.setattr(installer.platform, 'machine', lambda: machine)
    for plan in installer.PLANS:
        installer.validate_installation(plan)


@pytest.mark.parametrize('returncode', [0, 1])
def test_fixed_pip_command_uses_workshop_python(tmp_path, monkeypatch, returncode):
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        kwargs['stdout'].write('pip output\n')
        return SimpleNamespace(returncode=returncode)
    monkeypatch.setattr(installer.subprocess, 'run', run)
    log = tmp_path / 'install.log'
    if returncode:
        with pytest.raises(RuntimeError, match='pip exited'):
            installer.install_sorter('spikeinterface', log)
    else:
        installer.install_sorter('spikeinterface', log)
    command, kwargs = calls[0]
    assert command[:4] == [sys.executable, '-m', 'pip', 'install']
    assert '--no-input' in command
    assert 'hdbscan>=0.8.33' in command
    assert 'pynwb>=2.6' in command
    assert 'shell' not in kwargs
    assert kwargs['timeout'] == 1800
    assert 'pip output' in log.read_text()
    with pytest.raises(ValueError, match='Unknown'):
        installer.install_sorter('--index-url=https://other.invalid', log)
    assert len(calls) == 1


@pytest.mark.parametrize('failure', [False, True])
def test_installation_blocks_sorting_and_requires_restart(tmp_path, monkeypatch, failure):
    entered, release = threading.Event(), threading.Event()
    def install(plan, log_path):
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text('download progress')
        entered.set()
        assert release.wait(5)
        if failure:
            raise RuntimeError('network unavailable')
    monkeypatch.setattr(workspace, 'install_sorter', install)
    service = workspace.SortingWorkspace(tmp_path)
    try:
        with pytest.raises(ValueError, match='supported installation'):
            service.control(dict(action='install', plan='arbitrary-package'))
        service.jobs['active'] = dict(status='running')
        with pytest.raises(ValueError, match='active sorting batch'):
            service.control(dict(action='install', plan='rt-sort'))
        service.jobs.clear()
        started = service.control(dict(action='install', plan='rt-sort'))
        assert started['status'] == 'running'
        assert entered.wait(5)
        status = service.control({})
        assert status['installation']['log'] == 'download progress'
        assert status['sorters'] == []
        with pytest.raises(ValueError, match='Restart'):
            service.control(dict(action='run'))
        with pytest.raises(ValueError, match='Restart'):
            service.control(dict(action='install', plan='rt-sort'))
        release.set()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            status = service.control({})['installation']
            if status['status'] != 'running':
                break
            time.sleep(.01)
        assert status['status'] == ('failed' if failure else 'completed')
        assert status['restart_required']
        assert 'Restart' in status['message']
    finally:
        release.set()
        service.close()
