"""Catalog persistence, discovery and selection through the workshop UI."""
import json
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from braindance.examples.streaming_workshop.catalog_workspace import CatalogWorkspace
from test_workshop_browser import workshop_server  # noqa: F401
from test_workshop_analysis_browser import analysis_recording  # noqa: F401


def test_catalog_reload_discovery_and_persistence(tmp_path, monkeypatch):
    csv = tmp_path / 'catalog.csv'
    monkeypatch.setenv('BRAINDANCE_CATALOG_PATH', str(csv))
    monkeypatch.setenv('BRAINDANCE_DATA_DIR', str(tmp_path / 'data'))
    runs = tmp_path / 'runs'
    run = runs / 'saved-run'
    run.mkdir(parents=True)
    (run / 'experiment.json').write_text('{}')
    workspace = CatalogWorkspace(runs)
    initial = workspace.control({})
    assert initial['warnings']
    assert initial['entries'][0]['path'] == str(run)
    recording = tmp_path / 'local.raw.h5'
    recording.touch()
    csv.write_text(f'proj,chip,experiment,full_path\nproject,12,local,{recording}\nproject,12,remote,s3://bucket/remote\n')
    refreshed = workspace.control({})
    local = next(e for e in refreshed['entries'] if e['name'] == 'local')
    remote = next(e for e in refreshed['entries'] if e['name'] == 'remote')
    assert local['available'] and local['path'] == str(recording)
    assert not remote['available'] and remote['reason']
    other = tmp_path / 'other'
    other.mkdir()
    workspace.control({'action': 'add', 'path': str(other)})
    workspace.control({'action': 'add', 'path': str(other)})
    saved = CatalogWorkspace(runs).control({})
    assert len([e for e in saved['entries'] if e['path'] == str(other)]) == 1
    assert json.loads(workspace.index_path.read_text()) == [str(other)]
    with pytest.raises(ValueError, match='does not exist'):
        workspace.control({'action': 'add', 'path': str(tmp_path / 'missing')})


def test_catalog_api_requires_token(workshop_server):
    url, token = workshop_server
    def call(auth):
        request = Request(url + '/api/catalog', data=b'{"action":"refresh"}',
                          headers={'Content-Type': 'application/json', 'X-Workshop-Token': auth})
        with urlopen(request, timeout=30) as response:
            return json.load(response)
    with pytest.raises(HTTPError) as denied:
        call('wrong')
    assert denied.value.code == 403
    assert 'entries' in call(token)


def test_current_experiment_opens_in_analysis_and_catalog(workshop_server):
    url, token = workshop_server
    def call(endpoint, command):
        request = Request(url + endpoint, data=json.dumps(command).encode(),
                          headers={'Content-Type': 'application/json', 'X-Workshop-Token': token})
        with urlopen(request, timeout=30) as response:
            return json.load(response)
    call('/api/control', {'kind': 'start', 'config': {
        'source': None, 'live_config': None, 'speed': 0, 'loop': False,
        'write_output': True, 'baseline_hz': [0.] * 8,
        'phases': [{'id': 'record', 'type': 'recording', 'params': {'record_seconds': .04}}],
    }})
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        with urlopen(url + '/api/state', timeout=5) as response:
            state = json.load(response)
        if state['status'] in ('completed', 'error'):
            break
        time.sleep(.05)
    assert state['status'] == 'completed', state
    selected = call('/api/analysis', {'action': 'load', 'source': 'current'})
    assert selected['workspace']['path'] == state['output']
    assert any(entry['path'] == state['output'] for entry in call('/api/catalog', {})['entries'])


def test_catalog_browser_selection_refresh_and_current(workshop_server, analysis_recording, tmp_path):
    playwright = pytest.importorskip('playwright.sync_api')
    with playwright.sync_playwright() as driver:
        if not Path(driver.chromium.executable_path).exists():
            pytest.skip('Install Chromium for Playwright')
        browser = driver.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            errors = []
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.goto(workshop_server[0] + '/#catalog')
            expect = playwright.expect
            expect(page.locator('#view-catalog')).to_be_visible()
            expect(page.locator('#catalogMessage')).to_contain_text('Refreshed', timeout=30000)
            expect(page.locator('#catalogCurrent')).to_be_disabled()
            page.locator('#catalogAddPath').fill(str(analysis_recording))
            page.locator('#catalogAdd').click()
            expect(page.locator('#catalogRows')).to_contain_text('baseline.raw.h5')
            page.locator('#catalogSearch').fill('baseline.raw.h5')
            page.locator('#catalogRows button').first.click()
            expect(page.locator('#view-analysis')).to_be_visible()
            expect(page.locator('#analysisFile option')).to_have_count(1)
            page.locator('[data-view=catalog]').click()
            expect(page.locator('#catalogSelected')).to_have_text(str(analysis_recording))
            # Failed loads preserve the selected analysis experiment.
            missing = tmp_path / 'removed'
            missing.mkdir()
            page.locator('#catalogAddPath').fill(str(missing))
            page.locator('#catalogAdd').click()
            expect(page.locator('#catalogMessage')).to_contain_text('Refreshed')
            page.locator('#catalogSearch').fill('removed')
            expect(page.locator('#catalogRows button').first).to_be_enabled()
            missing.rmdir()
            page.locator('#catalogRows button').first.click()
            expect(page.locator('#analysisWorkspaceMessage')).to_contain_text('Path does not exist')
            page.locator('[data-view=catalog]').click()
            expect(page.locator('#catalogSelected')).to_have_text(str(analysis_recording))
            # Refresh discovers a newly saved run, and live status follows server state.
            run = tmp_path / 'runs' / 'new-run'
            run.mkdir(parents=True)
            (run / 'experiment.json').write_text('{}')
            page.locator('#catalogSearch').fill('new-run')
            page.locator('#catalogRefresh').click()
            expect(page.locator('#catalogRows')).to_contain_text('new-run')
            page.route('**/api/state', lambda route: route.fulfill(json={'status': 'running', 'phase': 'baseline', 'output': str(run), 'execution': {'source': 'Simulation'}}))
            expect(page.locator('#catalogCurrentStatus')).to_contain_text('running · baseline · Simulation')
            expect(page.locator('#catalogCurrent')).to_be_enabled()
            page.set_viewport_size({'width': 390, 'height': 844})
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
            page.reload()
            expect(page.locator('#catalogMessage')).to_contain_text('Refreshed', timeout=30000)
            expect(page.locator('#catalogSelected')).to_have_text(str(analysis_recording))
            page.locator('#catalogSearch').fill('baseline.raw.h5')
            expect(page.locator('#catalogRows')).to_contain_text('baseline.raw.h5')
            assert not errors
        finally:
            browser.close()
