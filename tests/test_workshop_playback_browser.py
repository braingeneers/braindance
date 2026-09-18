"""Saved experiment selection and playback through the workshop HTTP/UI boundary."""
import json
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from test_workshop_browser import workshop_server  # noqa: F401


def call(server, endpoint, command, token=None):
    request = Request(server[0] + endpoint, data=json.dumps(command).encode(),
                      headers={'Content-Type': 'application/json',
                               'X-Workshop-Token': server[1] if token is None else token})
    with urlopen(request, timeout=30) as response:
        return json.load(response)


@pytest.fixture
def cartpole_log(tmp_path):
    directory = tmp_path / 'cartpole_saved'
    directory.mkdir()
    (directory / 'experiment_log.json').write_text('{}')
    (directory / 'cartpole_game_log.csv').write_text(
        'time,pole_angle,reward,action,spike_rates,state\n'
        '0.02,0.1,1,0,"[2, 3]","[0.5, 0, 0.1, 0]"\n'
        '0.04,0.2,2,1,"[3, 4]","[0.6, 0, 0.2, 0]"\n')
    return directory


def test_playback_api_saved_cartpole(workshop_server, cartpole_log):
    with pytest.raises(HTTPError) as denied:
        call(workshop_server, '/api/playback', {'path': str(cartpole_log)}, token='wrong')
    assert denied.value.code == 403
    call(workshop_server, '/api/playback', {'path': str(cartpole_log)})
    call(workshop_server, '/api/control', {'kind': 'playback', 'path': str(cartpole_log), 'speed': 0})
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        with urlopen(workshop_server[0] + '/api/state', timeout=5) as response:
            state = json.load(response)
        if state['status'] in ('completed', 'error'):
            break
        time.sleep(.02)
    assert state['status'] == 'completed', state
    assert state['execution']['engine'] == 'Playback'
    assert state['scene']['kind'] == 'cartpole'
    assert state['scene']['observation'][2] == .2


def test_catalog_loads_and_plays_saved_game(workshop_server, cartpole_log):
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
            page.locator('#catalogAddPath').fill(str(cartpole_log))
            page.locator('#catalogAdd').click()
            page.locator('#catalogSearch').fill('cartpole_saved')
            page.locator('#catalogRows').get_by_role('button', name='Load playback', exact=True).click()
            playwright.expect(page.locator('#view-experiment')).to_be_visible()
            playwright.expect(page.locator('#playbackPlay')).to_be_enabled()
            page.locator('#playbackPlay').click()
            playwright.expect(page.locator('#view-monitor')).to_be_visible()
            playwright.expect(page.locator('#gameTitle')).to_have_text('CartPole')
            playwright.expect(page.locator('#status')).to_contain_text('completed')
            assert not errors
        finally:
            browser.close()
