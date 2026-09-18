from pathlib import Path

import numpy as np
import pytest

from test_streaming_workshop import workshop_config  # noqa: F401
from test_workshop_browser import workshop_server  # noqa: F401


def test_preview_preserves_acquired_samples_and_phase_type(tmp_path, workshop_config, monkeypatch):
    from braindance.core.simulation import NeuralSimulationSource
    from braindance.examples.streaming_workshop.session import WorkshopSession

    acquired = []
    original = NeuralSimulationSource.read

    def read(self, *args, **kwargs):
        batch = original(self, *args, **kwargs)
        if batch is not None:
            acquired.append(batch['raw_float32'].copy())
        return batch

    monkeypatch.setattr(NeuralSimulationSource, 'read', read)
    workshop_config['phases'] = [dict(id='custom-recording', type='recording', params={'record_seconds': .04})]
    session = WorkshopSession(tmp_path, workshop_config)
    session.run()
    assert session.status == 'completed', session.error
    assert session.snapshot['phase_kind'] == 'recording'
    np.testing.assert_allclose(np.asarray(session.snapshot['raw']).T,
                               acquired[-1][:, :16] * 1000, atol=.00051)


def test_game_visibility_follows_phase_without_moving_panels(workshop_server):
    playwright = pytest.importorskip('playwright.sync_api')
    url, _ = workshop_server
    current = dict(status='running', phase='custom-id', phase_kind='recording',
                   scene=dict(kind='cartpole', observation=[0, 0, 0, 0]))
    with playwright.sync_playwright() as driver:
        if not Path(driver.chromium.executable_path).exists():
            pytest.skip('Chromium is not installed')
        browser = driver.chromium.launch(headless=True)
        page = browser.new_page(viewport={'width': 1500, 'height': 1000})
        page.route('**/api/state', lambda route: route.fulfill(json=current))
        page.goto(url)
        page.locator('[data-view=monitor]').click()
        expect = playwright.expect
        expect(page.locator('#environmentPlaceholder')).to_be_visible()
        expect(page.locator('#scene')).to_be_hidden()
        initial = page.locator('#gamePanel').bounding_box()
        current['phase_kind'] = 'environment'
        expect(page.locator('#scene')).to_be_visible()
        expect(page.locator('#environmentPlaceholder')).to_be_hidden()
        assert page.locator('#gamePanel').bounding_box() == initial
        current['phase_kind'] = 'recording'
        expect(page.locator('#scene')).to_be_hidden()
        expect(page.locator('#encodingPanel')).to_have_class('panel mapping-idle')
        page.screenshot(path='/tmp/workshop-live-inactive.png', full_page=True)
        current.pop('phase_kind')
        current.update(playback=True, status='completed', reward=12.)
        expect(page.locator('#scene')).to_be_visible()
        expect(page.locator('#rewardLabel')).to_have_text('Total reward')
        expect(page.locator('#reward')).to_have_text('12.0')
        current['episode_reward'] = 3.
        expect(page.locator('#rewardLabel')).to_have_text('Episode reward')
        expect(page.locator('#reward')).to_have_text('3.0')
        browser.close()
