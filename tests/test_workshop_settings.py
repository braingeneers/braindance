"""Persistence, protected API writes and settings navigation in the HTML workshop."""
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from test_workshop_browser import workshop_server
from braindance.examples.streaming_workshop import global_settings


@pytest.fixture
def settings_server(tmp_path, monkeypatch, request):
    import os
    monkeypatch.setenv('PLAYWRIGHT_BROWSERS_PATH', os.environ.get(
        'PLAYWRIGHT_BROWSERS_PATH', str(Path.home() / '.cache/ms-playwright')))
    monkeypatch.setenv('HOME', str(tmp_path))
    for key in ('DATA_DIR', 'CATALOG_PATH', 'OUTPUT_DIR', 'AUTO_EXTRACT_SPIKE_INFO'):
        monkeypatch.delenv('BRAINDANCE_' + key, raising=False)
    return request.getfixturevalue('workshop_server')


def test_feature_status_reports_missing_dependencies(monkeypatch):
    monkeypatch.setattr(global_settings, 'get_global_settings', lambda: {})
    monkeypatch.setattr(global_settings.importlib.util, 'find_spec', lambda name: None if name == 'mujoco' else object())
    monkeypatch.setattr(global_settings, 'sorting_capabilities', lambda: {'available': False, 'reason': 'Model missing'})
    features = {feature['id']: feature for feature in global_settings.settings_payload()['features']}
    assert features['cartpole']['available']
    assert not features['ant']['available']
    assert 'mujoco' in features['ant']['detail']
    assert not features['rtsort']['available']
    assert features['rtsort']['detail'] == 'Model missing'


def test_settings_api_persists_and_requires_token(settings_server, tmp_path):
    url, token = settings_server
    data = {'settings': {'data_dir': str(tmp_path / 'recordings'), 'auto_extract_spike_info': True}}
    with pytest.raises(HTTPError) as exc:
        urlopen(Request(url + '/api/settings', data=json.dumps(data).encode()), timeout=10)
    assert exc.value.code == 403
    assert not (tmp_path / '.braindance/config.json').exists()
    request = Request(url + '/api/settings', data=json.dumps(data).encode(),
                      headers={'X-Workshop-Token': token, 'Content-Type': 'application/json'})
    with urlopen(request, timeout=10) as response:
        result = json.load(response)
    assert result['settings']['data_dir']['effective'] == str(tmp_path / 'recordings')
    assert json.loads((tmp_path / '.braindance/config.json').read_text()) == data['settings']
    with urlopen(url + '/api/settings', timeout=10) as response:
        assert json.load(response) == result
    assert any(feature['id'] == 'rtsort' for feature in result['features'])


def test_settings_browser_save_reload_and_status(settings_server, tmp_path):
    playwright = pytest.importorskip('playwright.sync_api')
    url, _ = settings_server
    with playwright.sync_playwright() as driver:
        if not Path(driver.chromium.executable_path).exists():
            pytest.skip('Chromium is not installed')
        browser = driver.chromium.launch(headless=True)
        try:
            page = browser.new_page(viewport={'width': 1280, 'height': 900})
            errors = []
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.goto(url)
            page.locator('button[data-view=settings]').click()
            expect = playwright.expect
            expect(page.locator('#view-settings')).to_be_visible()
            expect(page.locator('#saveGlobalSettings')).to_be_enabled()
            button = page.locator('button[data-view=settings]').bounding_box()
            assert button['y'] > 750
            page.locator('#global-data_dir').fill(str(tmp_path / 'new-data'))
            page.locator('#global-auto_extract_spike_info').select_option('true')
            page.locator('#saveGlobalSettings').click()
            expect(page.locator('#globalSettingsStatus')).to_contain_text('Settings saved')
            page.reload()
            expect(page.locator('#view-settings')).to_be_visible()
            expect(page.locator('#global-data_dir')).to_have_value(str(tmp_path / 'new-data'))
            expect(page.locator('#global-auto_extract_spike_info')).to_have_value('true')
            expect(page.locator('.feature-badge')).to_have_count(9)
            for feature in page.request.get(url + '/api/settings').json()['features']:
                badge = page.locator(f'[data-feature="{feature["id"]}"] .feature-badge')
                expect(badge).to_have_attribute('data-available', str(feature['available']).lower())
            page.locator('#global-data_dir').fill('relative/path')
            page.locator('#saveGlobalSettings').click()
            expect(page.locator('#globalSettingsStatus')).to_contain_text('absolute path')
            page.locator('#global-data_dir').fill('')
            page.locator('#saveGlobalSettings').click()
            expect(page.locator('#globalSettingsStatus')).to_contain_text('Settings saved')
            assert 'data_dir' not in json.loads((tmp_path / '.braindance/config.json').read_text())
            assert not errors
        finally:
            browser.close()
