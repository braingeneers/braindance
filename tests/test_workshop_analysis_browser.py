"""Analysis HTTP and browser workflows against deterministic Maxwell-shaped files."""
import json
import re
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest

from test_workshop_browser import workshop_server  # noqa: F401


@pytest.fixture
def analysis_recording(tmp_path):
    import h5py
    path = tmp_path / 'baseline.raw.h5'
    with h5py.File(path, 'w') as handle:
        routed = handle.create_group('data_store/data0000/groups/routed')
        routed.create_dataset('raw', data=np.vstack([np.arange(2000), np.arange(2000) + 100]).astype('uint16'))
        routed.create_dataset('frame_nos', data=np.arange(2000, dtype='uint64'))
        events = np.array([(100, 0, -30.), (200, 1, -40.), (500, 0, -35.), (900, 1, -45.)],
                          dtype=[('frame', 'uint64'), ('channel', 'int32'), ('amplitude', 'float32')])
        handle.create_dataset('data_store/data0000/spikes', data=events)
    return path


def call_analysis(server, command, token=None):
    url, real_token = server
    request = Request(url + '/api/analysis', data=json.dumps(command).encode(),
                      headers={'Content-Type': 'application/json',
                               'X-Workshop-Token': real_token if token is None else token})
    with urlopen(request, timeout=10) as response:
        return json.load(response)


def wait_job(server, job):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        status = call_analysis(server, {'action': 'status', 'job_id': job['id']})
        if status['status'] not in ('queued', 'running'):
            return status
        time.sleep(.05)
    pytest.fail(f'Analysis job did not finish: {status}')


def test_analysis_api_auth_capabilities_and_raw_values(workshop_server, analysis_recording):
    with pytest.raises(HTTPError) as denied:
        call_analysis(workshop_server, {'action': 'load', 'path': str(analysis_recording)}, token='wrong')
    assert denied.value.code == 403
    assert call_analysis(workshop_server, {'action': 'status'})['workspace']['files'] == []
    loaded = call_analysis(workshop_server, {'action': 'load', 'path': str(analysis_recording)})
    recording = loaded['workspace']['files'][0]
    assert recording['capabilities']['raw']
    assert recording['capabilities']['spikes']
    params = {'channel': 1, 'start_ms': 5, 'duration_ms': 1}
    result = wait_job(workshop_server, call_analysis(workshop_server,
                      {'action': 'run', 'kind': 'raw', 'file_id': recording['id'], 'params': params}))
    assert result['status'] == 'completed', result
    assert result['kind'] == 'raw'
    assert result['params'] == params
    assert result['result']['plots'][0]['y'] == list(range(200, 220))
    missing_log = wait_job(workshop_server, call_analysis(workshop_server,
                          {'action': 'run', 'kind': 'overlap', 'file_id': recording['id']}))
    assert missing_log['status'] == 'failed'
    assert missing_log['error']
    failed = wait_job(workshop_server, call_analysis(workshop_server,
                      {'action': 'run', 'kind': 'raw', 'file_id': recording['id'], 'params': {'channel': 999}}))
    assert failed['status'] == 'failed'
    assert failed['error']
    assert call_analysis(workshop_server, {'action': 'status'})['workspace']['files'][0]['id'] == recording['id']


def test_analysis_browser_raw_spikes_gating_and_mobile(workshop_server, analysis_recording):
    playwright = pytest.importorskip('playwright.sync_api')
    with playwright.sync_playwright() as driver:
        if not Path(driver.chromium.executable_path).exists():
            pytest.skip('Install Chromium for Playwright')
        browser = driver.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            errors = []
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.goto(workshop_server[0])
            expect = playwright.expect
            page.locator('[data-view=analysis]').click()
            expect(page.locator('#view-analysis')).to_be_visible()
            expect(page.locator('#analysisRun')).to_be_disabled()
            page.locator('#analysisPath').fill(str(analysis_recording))
            page.locator('#analysisLoad').click()
            expect(page.locator('#analysisFile option')).to_have_count(1)
            expect(page.locator('#analysisRun')).to_be_enabled()
            page.locator('#analysisRun').click()
            expect(page.locator('.analysis-job h3').first).to_have_text(re.compile(r'^Raw data .*completed$'), timeout=15000)
            expect(page.locator('.analysis-job canvas').first).to_be_visible()
            page.get_by_role('tab', name='Spikes', exact=True).click()
            expect(page.locator('#analysisRun')).to_be_enabled()
            page.locator('#analysisRun').click()
            expect(page.locator('.analysis-job h3').first).to_have_text(re.compile(r'^Spikes .*completed$'), timeout=15000)
            expect(page.locator('.analysis-job').first).to_contain_text('4 spikes')
            expect(page.locator('.analysis-job').first.get_by_role('button', name='Download result JSON')).to_be_visible()
            page.get_by_role('tab', name='Stimulus overlap', exact=True).click()
            expect(page.locator('[name=stim_log]')).to_be_visible()
            page.get_by_role('tab', name='Spike sorting', exact=True).click()
            expect(page.locator('[name=baseline_id] option')).to_have_count(1)
            expect(page.locator('[name=device]')).to_be_visible()
            page.set_viewport_size({'width': 390, 'height': 844})
            page.get_by_role('tab', name='Raw data', exact=True).click()
            expect(page.locator('#analysisRun')).to_be_visible()
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth + 1')
            # A raw-only recording must not advertise spike analysis.
            import h5py
            raw_only = analysis_recording.with_name('raw_only.h5')
            with h5py.File(raw_only, 'w') as handle:
                handle.create_dataset('sig', data=np.zeros((2, 100), dtype='uint16'))
            page.locator('#analysisPath').fill(str(raw_only))
            page.locator('#analysisLoad').click()
            expect(page.locator('#analysisFileInfo')).to_contain_text('raw_only.h5')
            page.get_by_role('tab', name='Spikes', exact=True).click()
            expect(page.locator('#analysisRun')).to_be_disabled()
            assert not errors
        finally:
            browser.close()


def test_analysis_folder_browser(workshop_server, analysis_recording):
    playwright = pytest.importorskip('playwright.sync_api')
    with playwright.sync_playwright() as driver:
        if not Path(driver.chromium.executable_path).exists():
            pytest.skip('Install Chromium for Playwright')
        browser = driver.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            page.goto(workshop_server[0] + '/#analysis')
            expect = playwright.expect
            default = call_analysis(workshop_server, {'action': 'browse'})['default_path']
            expect(page.locator('#analysisPath')).to_have_value(default)
            nested = analysis_recording.parent / 'nested'
            nested.mkdir()
            page.locator('#analysisPath').fill(str(analysis_recording.parent))
            page.locator('#analysisBrowse').click()
            expect(page.locator('#analysisChoices')).to_contain_text('baseline.raw.h5')
            page.locator('#analysisChoices').select_option(str(nested))
            page.locator('#analysisOpenFolder').click()
            expect(page.locator('#analysisBrowseStatus')).to_contain_text('0 options')
            page.locator('#analysisParent').click()
            expect(page.locator('#analysisChoices')).to_contain_text('baseline.raw.h5')
            page.locator('#analysisChoices').select_option(str(analysis_recording))
            expect(page.locator('#analysisOpenFolder')).to_be_disabled()
            page.locator('#analysisLoad').click()
            expect(page.locator('#analysisFile option')).to_have_count(1)
            page.locator('#analysisDataFolder').click()
            expect(page.locator('#analysisPath')).to_have_value(default)
        finally:
            browser.close()
