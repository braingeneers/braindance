"""Source navigation opens latest files and leaves the editor alone."""
from pathlib import Path
import subprocess

import pytest

from braindance.examples.streaming_workshop import source_links
from test_workshop_browser import workshop_server  # noqa: F401


def test_links_use_remote_head_instead_of_unpushed_commits(tmp_path, monkeypatch):
    def git(*args):
        return subprocess.check_output(['git', '-C', str(tmp_path), *args], text=True).strip()

    git('init', '-q')
    git('config', 'user.email', 'test@example.com')
    git('config', 'user.name', 'Test')
    git('remote', 'add', 'origin', 'git@github.com:example/BrainDance.git')
    path = tmp_path / 'example.py'
    path.write_text('\n\nraise RuntimeError("Do not import")\nclass Phase:\n    def run(self):\n        pass\n')
    git('add', 'example.py')
    git('commit', '-qm', 'Fixture')
    path.write_text('\n' * 20 + path.read_text() + '\nclass LocalOnly:\n    pass\n')
    monkeypatch.setattr(source_links, '__file__', str(tmp_path / 'source_links.py'))
    source_links._repository.cache_clear()
    try:
        assert source_links.source_url('example', 'Phase.run') == (
            'https://github.com/example/BrainDance/blob/HEAD/example.py')
        assert source_links.source_url('example', 'LocalOnly') is None
        assert source_links.source_url('missing', 'Phase') is None
        git('remote', 'set-url', 'origin', 'https://other.example/repo.git')
        source_links._repository.cache_clear()
        assert source_links.source_url('example', 'Phase') is None
    finally:
        source_links._repository.cache_clear()
        source_links._definitions.cache_clear()


def test_phase_and_analysis_source_navigation(workshop_server):
    playwright = pytest.importorskip('playwright.sync_api')
    url, _ = workshop_server
    with playwright.sync_playwright() as driver:
        if not Path(driver.chromium.executable_path).exists():
            pytest.skip('Chromium is not installed')
        browser = driver.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            errors = []
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.goto(url)
            page.wait_for_function("typeof builderLoaded !== 'undefined' && builderLoaded")
            page.evaluate("phaseSpecs=[{id:'record',type:'native:phases3.RecordPhaseV3',params:{}}];renderPhases();showView('sequence');renderPhaseFlow();")
            link = page.locator('.flow-node-actions a.source-link').first
            playwright.expect(link).to_be_visible()
            assert '/blob/HEAD/braindance/core/phases_v3/phases3.py' in link.get_attribute('href')
            assert link.get_attribute('rel') == 'noopener noreferrer'
            # Block the remote request: test navigation without requiring GitHub access.
            page.context.route('https://github.com/**', lambda route: route.fulfill(body='source'))
            before = page.evaluate('JSON.stringify(phaseSpecs)')
            with page.expect_popup() as popup:
                link.click()
            popup.value.close()
            assert page.evaluate('JSON.stringify(phaseSpecs)') == before
            playwright.expect(page.locator('#flowInspector')).to_be_hidden()
            page.evaluate("showView('analysis')")
            for title, filename in [('Raw data', 'analysis_workspace.py'),
                                    ('Connectivity (STTC)', 'analysis_workspace.py'),
                                    ('Stimulus overlap', 'analysis_evoked.py'),
                                    ('Spike sorting', 'analysis_sorting.py')]:
                page.get_by_role('tab', name=title, exact=True).click()
                source = page.locator('#analysisToolHelp a.source-link')
                playwright.expect(source).to_be_visible()
                assert source.get_attribute('href').endswith(f'/{filename}')
                assert '/blob/HEAD/' in source.get_attribute('href')
            assert not errors
        finally:
            browser.close()
