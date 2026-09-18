"""Local API and optional browser workflow regression tests.

Browser checks require ``pip install playwright`` and ``playwright install chromium``.
"""
import json
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
from urllib.request import Request, urlopen

import pytest


@pytest.fixture
def workshop_server(tmp_path):
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    url = f'http://127.0.0.1:{port}'
    with (tmp_path / 'server.log').open('w+') as log:
        process = subprocess.Popen([
            sys.executable, '-m', 'braindance.examples.streaming_workshop.main',
            '--no-browser', '--port', str(port), '--output-dir', str(tmp_path / 'runs'),
            '--channels', '8', '--num-neurons', '8', '--no-write-output',
            '--source', str(tmp_path / 'missing.raw.h5'),
        ], cwd=Path(__file__).resolve().parents[1], stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                try:
                    with urlopen(url, timeout=1) as response:
                        html = response.read().decode()
                    break
                except OSError:
                    if process.poll() is not None:
                        log.seek(0)
                        pytest.fail(log.read())
                    time.sleep(.05)
            else:
                pytest.fail('Workshop server did not start')
            token = re.search(r"WORKSHOP_TOKEN='([^']+)'", html).group(1)
            yield url, token
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def test_export_style_switches_preview(workshop_server):
    playwright = pytest.importorskip('playwright.sync_api')
    url, _ = workshop_server
    with playwright.sync_playwright() as driver:
        if not Path(driver.chromium.executable_path).exists():
            pytest.skip('Install browser with python -m playwright install chromium')
        browser = driver.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            page.goto(url)
            page.get_by_role('button', name='View / export Python', exact=True).click()
            playwright.expect(page.locator('#exportStyle')).to_have_value('phases')
            playwright.expect(page.locator('#generatedCode')).to_contain_text('exp.add_phase(', timeout=15000)
            page.locator('#exportStyle').select_option('config')
            playwright.expect(page.locator('#generatedCode')).to_contain_text('PHASES =', timeout=15000)
            playwright.expect(page.locator('#generatedCode')).not_to_contain_text('exp.add_phase(')
            with page.expect_response('**/api/code-export') as exported:
                page.locator('#exportCode').click()
            assert exported.value.request.post_data_json['style'] == 'config'
            assert exported.value.ok, exported.value.text()
            exported_code = Path(exported.value.json()['files']['experiment.py']).read_text()
            assert 'PHASES =' in exported_code
            page.locator('#exportStyle').select_option('phases')
            playwright.expect(page.locator('#generatedCode')).to_contain_text('exp.add_phase(', timeout=15000)
            playwright.expect(page.locator('#generatedCode')).not_to_contain_text('PHASES =')
            with page.expect_response('**/api/code-export') as exported:
                page.locator('#exportCode').click()
            assert exported.value.request.post_data_json['style'] == 'phases'
            assert exported.value.ok, exported.value.text()
            exported_code = Path(exported.value.json()['files']['experiment.py']).read_text()
            assert 'exp.add_phase(' in exported_code and 'PHASES =' not in exported_code
            page.screenshot(path='/tmp/braindance-phase-calls-export.png', full_page=True)
        finally:
            browser.close()


def test_start_can_switch_startup_replay_to_simulation(workshop_server):
    url, token = workshop_server
    config = dict(source=None, live_config=None, speed=0, loop=False,
                  baseline_hz=[0.] * 8,
                  phases=[dict(id='record', type='recording', params={'record_seconds': .04})])
    command = Request(url + '/api/control', data=json.dumps(dict(kind='start', config=config)).encode(),
                      headers={'Content-Type': 'application/json', 'X-Workshop-Token': token})
    with urlopen(command, timeout=10) as response:
        assert json.load(response)['ok']
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        with urlopen(url + '/api/state', timeout=2) as response:
            state = json.load(response)
        if state['status'] in {'completed', 'error'}:
            break
        time.sleep(.02)
    assert state['status'] == 'completed', state
    assert state['execution']['source'] == 'Simulation'
    assert state['execution']['timing'] == 'Unpaced'


def test_experiment_source_navigation_and_launch(workshop_server):
    playwright = pytest.importorskip('playwright.sync_api')
    url, _ = workshop_server
    with playwright.sync_playwright() as driver:
        if not Path(driver.chromium.executable_path).exists():
            pytest.skip('Install browser with python -m playwright install chromium')
        browser = driver.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            errors = []
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.goto(url)
            expect = playwright.expect
            expect(page.locator('#view-experiment')).to_be_visible()
            expect(page.locator('#dataSource')).to_have_value('replay')
            assert not any(re.match(r'^\d+\.', label) for label in page.locator('.app-nav button').all_text_contents())
            page.locator('#dataSource').select_option('simulation')
            page.locator('#view-experiment .simulator-tab').click()
            expect(page.locator('#view-simulation')).to_be_visible()
            # Choose a neuron on the canvas, then make several outgoing edges.
            point = page.evaluate("""() => {
                const r=$('simCulture').getBoundingClientRect();
                return simProjection(r.width,r.height,simulatorElectrodes()).point(neuronPositions[2]);
            }""")
            page.locator('#simCulture').click(position={'x': point[0], 'y': point[1]})
            expect(page.locator('#simSelected')).to_have_value('2')
            page.get_by_role('button', name='Target neuron 3', exact=True).click()
            page.get_by_role('button', name='Target neuron 4', exact=True).click()
            page.locator('#simEdge').fill('0.7')
            page.locator('#simConnectMany').click()
            assert page.evaluate('[matrixValues[3][2],matrixValues[4][2],simulatorSelection]') == [.7, .7, 2]
            page.locator('#simDisconnectMany').click()
            assert page.evaluate('[matrixValues[3][2],matrixValues[4][2]]') == [0, 0]
            page.locator('[data-view=experiment]').first.click()
            page.locator('#dataSource').select_option('replay')
            expect(page.locator('#sourcePath')).to_be_visible()
            expect(page.locator('#detection')).to_be_visible()
            expect(page.locator('#view-experiment .simulator-tab')).to_be_hidden()
            page.locator('#dataSource').select_option('live')
            expect(page.locator('#liveConfigPath')).to_be_visible()
            page.locator('#dataSource').select_option('simulation')
            page.locator('#acquisitionSpeed').select_option('0')
            page.wait_for_function('typeof builderLoaded !== "undefined" && builderLoaded')
            page.evaluate("""() => {
                phaseSpecs = [{id:'record',type:'recording',params:{record_seconds:.04}}];
                renderPhases(); scheduleValidation();
            }""")
            page.locator('#launchExperiment').click()
            expect(page.locator('#view-monitor')).to_be_visible(timeout=15000)
            page.wait_for_function("state.status === 'completed' || state.status === 'error'", timeout=15000)
            assert page.evaluate('state.status') == 'completed', page.evaluate('state.error')
            assert page.evaluate('state.execution.timing') == 'Unpaced'
            # A paced recording remains responsive to pause, step and stop.
            page.locator('[data-view=experiment]').first.click()
            page.locator('#acquisitionSpeed').select_option('1')
            page.evaluate("""() => {
                phaseSpecs = [{id:'record',type:'recording',params:{record_seconds:10}}];
                renderPhases(); scheduleValidation();
            }""")
            page.locator('#launchExperiment').click()
            page.wait_for_function("state.status === 'running'")
            page.locator('#pause').click()
            page.wait_for_function('state.paused === true')
            expect(page.locator('#launchExperiment')).to_be_disabled()
            before = page.evaluate('state.history.at(-1).t')
            page.locator('#step').click()
            page.wait_for_function('(before) => state.history.at(-1).t > before', arg=before)
            assert page.evaluate('state.history.at(-1).t') == pytest.approx(before + .02)
            page.locator('#stop').click()
            page.wait_for_function("state.status === 'stopped'")
            # The task playground operates independently of the stopped run.
            page.locator('[data-view=playground]').click()
            page.locator('#pgLoad').click()
            expect(page.locator('#pgStep')).to_be_enabled()
            page.locator('#pgStep').click()
            expect(page.locator('#pgStatus')).to_contain_text('0.02 s')
            assert not errors
        finally:
            browser.close()


def test_custom_analysis_template_and_navigation(workshop_server):
    playwright = pytest.importorskip('playwright.sync_api')
    url, _ = workshop_server
    with playwright.sync_playwright() as driver:
        if not Path(driver.chromium.executable_path).exists():
            pytest.skip('Install Chromium for Playwright')
        browser = driver.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            errors = []
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.goto(url)
            page.wait_for_function('typeof builderLoaded !== "undefined" && builderLoaded')
            page.locator('#dataSource').select_option('simulation')
            page.evaluate("""() => {
                phaseSpecs = [{id:'record',type:'recording',params:{record_seconds:.04}}];
                renderPhases(); scheduleValidation(); showView('sequence');
            }""")
            page.locator('#flowCustomAnalysis').click()
            playwright.expect(page.locator('#view-functions')).to_be_visible()
            playwright.expect(page.locator('#code')).to_be_focused()
            playwright.expect(page.locator('#code')).to_be_in_viewport()
            assert page.evaluate('phaseSpecs.at(-1).params.function_name') == 'custom_analysis'
            assert page.locator('#analysisPanel input').count() == 1
            assert page.locator('#analysisPanel input[type=checkbox]').count() == 0
            page.locator('#analysisParameterReference summary').click()
            assert 'channels : int' in page.locator('#analysisParameters').inner_text()
            assert '8' in page.locator('#analysisParameters').inner_text()
            code = page.locator('#code').input_value()
            assert '@analysis_phase(inputs=[], outputs=["custom_analysis_result"])' in code
            # The decorator is the only editable contract, including multiline declarations.
            code = code.replace('inputs=[]', 'inputs=[\n    "recording_baseline_hz",\n]')
            code = code.replace('result = {}  # Replace with your analysis.',
                'result = sum(exp.data.get("recording_baseline_hz")) / exp.params["channels"]')
            page.locator('#code').fill(code)
            page.wait_for_function('phaseSpecs.at(-1).params.inputs.includes("recording_baseline_hz")')
            assert page.evaluate('verifyExperiment(false)') is True
            page.evaluate("showView('sequence')")
            card = page.locator('.flow-piece[data-id=custom_analysis]')
            assert 'list[float]' in card.inner_text()
            card.locator('h3').click()
            playwright.expect(page.locator('#analysisPanel')).to_be_visible()
            assert page.locator('#code').input_value() == code
            playwright.expect(page.locator('#createAnalysis')).to_be_hidden()
            page.locator('#profileName').fill('simple_analysis_test')
            page.locator('#saveProfile').click()
            page.wait_for_function("$('code').value === savedCode")
            page.evaluate("$('acquisitionSpeed').value='0'; showView('experiment')")
            page.locator('#launchExperiment').click()
            page.wait_for_function("state.status === 'completed' || state.status === 'error'", timeout=20000)
            assert page.evaluate('state.status') == 'completed', page.evaluate('state.error')
            page.evaluate("showView('functions')")
            page.locator('#addCustomAnalysis').click()
            playwright.expect(page.locator('#code')).to_be_focused()
            assert 'def custom_analysis_2(exp):' in page.locator('#code').input_value()
            assert page.evaluate('phaseSpecs.at(-1).params.function_name') == 'custom_analysis_2'
            page.wait_for_function('!!analysisContractsCache.custom_analysis_2')
            code = page.locator('#code').input_value().replace('def custom_analysis_2(exp):', 'def testy(exp):')
            page.locator('#code').fill(code)
            # Saving immediately must not depend on the debounced validation.
            with page.expect_response('**/api/profile') as saved:
                page.evaluate("() => { clearTimeout(validationTimer); return saveProfile('renamed_analysis_test'); }")
            assert saved.value.ok, saved.value.text()
            assert saved.value.request.post_data_json['settings']['phases'][-1]['params']['function_name'] == 'testy'
            page.wait_for_function("phaseSpecs.at(-1).params.function_name === 'testy'")
            assert page.locator('#analysisName').input_value() == 'testy'
            playwright.expect(page.locator('.flow-piece[data-id=custom_analysis_2] h3')).to_have_text('testy')
            assert page.evaluate('verifyExperiment(false)') is True
            page.evaluate('openAnalysisPhase(null, 1)')
            assert page.evaluate('phaseSpecs[1].params.function_name') == 'custom_analysis_2'
            # Rebind explicitly when a loaded profile refers to an old name.
            page.evaluate("openAnalysisPhase('custom_analysis_2')")
            page.locator('#analysisName').fill('custom_analysis')
            page.locator('#analysisName').press('Tab')
            page.wait_for_function("phaseSpecs.find(p=>p.id==='custom_analysis_2').params.function_name === 'custom_analysis'")
            assert not errors
        finally:
            browser.close()


def test_run_offers_to_save_unsaved_functions(workshop_server):
    playwright = pytest.importorskip('playwright.sync_api')
    url, _ = workshop_server
    with playwright.sync_playwright() as driver:
        browser = driver.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            starts = []
            page.on('request', lambda request: starts.append(request.post_data_json)
                    if request.url.endswith('/api/control') and request.method == 'POST' else None)
            page.goto(url)
            page.wait_for_function('typeof builderLoaded !== "undefined" && builderLoaded')
            page.locator('#dataSource').select_option('simulation')
            page.evaluate("""() => {
                phaseSpecs=[{id:'record',type:'recording',params:{record_seconds:.04}}];
                renderPhases();$('acquisitionSpeed').value='0';showView('sequence');
            }""")
            page.locator('#flowCustomAnalysis').click()
            code = page.locator('#code').input_value()
            page.evaluate("showView('experiment')")
            page.locator('#launchExperiment').click()
            dialog = page.get_by_role('dialog', name='Save your functions before running')
            playwright.expect(dialog).to_be_visible()
            playwright.expect(dialog.get_by_label('Profile name')).to_be_focused()
            dialog.get_by_role('button', name='Cancel', exact=True).click()
            assert page.locator('#code').input_value() == code
            assert not starts
            page.locator('#launchExperiment').click()
            dialog.get_by_label('Profile name').fill('save_and_run_test')
            # A rejected save must keep the prompt open and never start the run.
            page.evaluate("$('code').value += '\\ninvalid python !'")
            dialog.get_by_role('button', name='Save and run', exact=True).click()
            playwright.expect(page.locator('#saveRunError')).not_to_be_empty()
            playwright.expect(dialog).to_be_visible()
            assert not starts
            page.evaluate('(code) => {$("code").value=code}', code)
            dialog.get_by_role('button', name='Save and run', exact=True).click()
            playwright.expect(dialog).not_to_be_visible()
            page.wait_for_function("state.status === 'completed' || state.status === 'error'", timeout=20000)
            assert page.evaluate('state.status') == 'completed', page.evaluate('state.error')
            assert page.evaluate('selectedProfile') == 'save_and_run_test'
            assert page.evaluate("$('code').value === savedCode")
            assert len(starts) == 1
            assert starts[0]['profile'] == 'save_and_run_test'
        finally:
            browser.close()


def test_source_file_uploads(workshop_server):
    from urllib.error import HTTPError

    url, token = workshop_server
    paths = []
    for kind, name, body in [('replay', '../../recording.h5', b'first recording'),
                             ('replay', '../../recording.h5', b'replacement'),
                             ('config', 'routing.cfg', b'maxwell configuration')]:
        command = Request(url + '/api/upload/' + kind, data=body,
                          headers={'X-Workshop-Token': token, 'X-File-Name': name})
        with urlopen(command) as response:
            path = Path(json.load(response)['path'])
        assert path.parent.name == 'uploads'
        assert path.read_bytes() == body
        paths.append(path)
    assert paths[0] != paths[1]
    assert paths[0].read_bytes() == b'first recording'
    for name, body, auth, status in [('bad.txt', b'x', token, 400),
                                     ('empty.h5', b'', token, 400),
                                     ('valid.h5', b'x', 'wrong', 403)]:
        command = Request(url + '/api/upload/replay', data=body,
                          headers={'X-Workshop-Token': auth, 'X-File-Name': name})
        with pytest.raises(HTTPError) as exc:
            urlopen(command)
        assert exc.value.code == status


def test_replay_drop_replace_and_live_config_picker(workshop_server, tmp_path):
    import numpy as np

    playwright = pytest.importorskip('playwright.sync_api')
    url, _ = workshop_server
    recording = tmp_path / 'replay.npy'
    np.save(recording, np.zeros((8, 4000), dtype=np.float32))
    config = tmp_path / 'routing.cfg'
    config.write_text('selected live configuration')
    with playwright.sync_playwright() as driver:
        if not Path(driver.chromium.executable_path).exists():
            pytest.skip('Install Chromium for Playwright')
        browser = driver.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            page.goto(url)
            page.wait_for_function('typeof builderLoaded !== "undefined" && builderLoaded')
            expect = playwright.expect
            assert page.locator('#dataSource option').count() == 3
            page.locator('#replayFile').set_input_files(str(recording))
            expect(page.locator('#replayFileStatus')).to_have_text('Selected: replay.npy')
            first = page.locator('#sourcePath').input_value()
            page.evaluate('''bytes => {
                const data = new DataTransfer();
                data.items.add(new File([new Uint8Array(bytes)], 'replacement.npy'));
                $('replayDrop').dispatchEvent(new DragEvent('drop', {bubbles:true, dataTransfer:data}));
            }''', list(recording.read_bytes()))
            expect(page.locator('#replayFileStatus')).to_have_text('Selected: replacement.npy')
            second = page.locator('#sourcePath').input_value()
            assert first != second
            assert Path(first).read_bytes() == Path(second).read_bytes() == recording.read_bytes()
            assert page.evaluate('collect().live_config') is None
            page.locator('#dataSource').select_option('live')
            page.locator('#liveConfigFile').set_input_files(str(config))
            expect(page.locator('#liveConfigFileStatus')).to_have_text('Selected: routing.cfg')
            selected = page.evaluate('collect()')
            assert selected['source'] is None
            assert Path(selected['live_config']).read_text() == config.read_text()
            page.locator('#dataSource').select_option('simulation')
            expect(page.locator('#liveConfigFile')).to_be_hidden()
            expect(page.locator('#replayFile')).to_be_hidden()
            assert page.evaluate('[collect().source, collect().live_config]') == [None, None]
            page.locator('#dataSource').select_option('replay')
            page.locator('#acquisitionSpeed').select_option('0')
            page.evaluate('''() => {
                phaseSpecs=[{id:'record',type:'recording',params:{record_seconds:.04}}];
                renderPhases();scheduleValidation();
            }''')
            assert page.evaluate('verifyExperiment(true, false)'), page.locator('#experimentValidation').inner_text()
            page.locator('#launchExperiment').click()
            page.wait_for_function("state.status === 'completed' || state.status === 'error'", timeout=15000)
            assert page.evaluate('state.status') == 'completed', page.evaluate('state.error')
        finally:
            browser.close()


def test_parameter_forms_and_advanced_json(workshop_server):
    playwright = pytest.importorskip('playwright.sync_api')
    url, _ = workshop_server
    with playwright.sync_playwright() as driver:
        if not Path(driver.chromium.executable_path).exists():
            pytest.skip('Install Chromium for Playwright')
        browser = driver.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            errors = []
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.goto(url)
            page.wait_for_function('typeof builderLoaded !== "undefined" && builderLoaded')
            page.locator('.native-inputs > summary').click()
            fields = page.locator('#experimentParameterFields')
            fields.get_by_label('New parameter name').fill('gain')
            fields.get_by_label('New parameter type').select_option('float')
            fields.get_by_role('button', name='Add parameter', exact=True).click()
            fields.get_by_label('gain', exact=True).fill('2.5')
            assert page.evaluate('builderConfig().native_settings.gain') == 2.5
            playwright.expect(page.locator('#nativeSettings')).to_be_hidden()
            fields.locator('summary').click()
            page.locator('#nativeSettings').fill('{"gain": 4, "enabled": true}')
            assert fields.get_by_label('gain', exact=True).input_value() == '4'
            assert fields.get_by_label('enabled', exact=True).input_value() == 'true'
            page.evaluate("""() => {
                phaseSpecs=[{id:'record',type:'native:phases3.RecordPhaseV3',params:{duration:60}}];
                renderPhases();showView('sequence');
            }""")
            page.get_by_role('button', name='Parameters for record', exact=True).click()
            inspector = page.locator('#flowInspector')
            playwright.expect(inspector.locator('input').first).to_be_focused()
            playwright.expect(inspector.locator('h2')).to_be_in_viewport()
            assert page.get_by_role('button', name='Parameters for record', exact=True).get_attribute('aria-controls') == 'flowInspector'
            assert inspector.locator('textarea:visible').count() == 0
            inspector.get_by_label('duration', exact=True).fill('0.04')
            inspector.get_by_role('button', name='Apply settings', exact=True).click()
            assert page.evaluate('phaseSpecs[0].params.duration') == .04
            assert inspector.get_by_label('duration', exact=True).evaluate('(input) => input.checkValidity()')
            assert not errors
        finally:
            browser.close()


def test_sorting_tab_upload_select_and_submit(workshop_server, tmp_path):
    import h5py
    playwright = pytest.importorskip('playwright.sync_api')
    url, _ = workshop_server
    paths = []
    for name in ('baseline.hdf5', 'target one.h5', 'target two.h5'):
        path = tmp_path / name
        with h5py.File(path, 'w') as handle:
            handle.create_dataset('sig', shape=(4, 100), dtype='int16')
        paths.append(path)
    with playwright.sync_playwright() as driver:
        if not Path(driver.chromium.executable_path).exists():
            pytest.skip('Install Chromium for Playwright')
        browser = driver.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            page_errors = []
            page.on('pageerror', lambda error: page_errors.append(str(error)))
            submitted = []

            def intercept(route):
                command = route.request.post_data_json
                if command.get('action') != 'run':
                    response = route.fetch()
                    data = response.json()
                    data['sorters'] = [dict(id='rt-sort', label='RT-Sort', available=True, reason='')]
                    route.fulfill(response=response, json=data)
                else:
                    submitted.append(command)
                    route.fulfill(json=dict(id='test', sorter='rt-sort', targets=[{}, {}], status='failed', error='Test engine failure', result=None))

            page.route('**/api/sorting', intercept)
            page.goto(url)
            page.locator('[data-view="sorting"]').click()
            page.locator('#sortingUpload').set_input_files([str(path) for path in paths])
            expect = playwright.expect
            expect(page.locator('#sortingUploadStatus')).to_contain_text('Uploaded 3 recording(s)', timeout=20000)
            expect(page.locator('#sortingBaseline option')).to_have_count(3)
            page.locator('#sortingBaseline').select_option(label='baseline.hdf5')
            page.locator('#sortingFiles input').nth(0).uncheck()
            page.locator('#sortingRTOptions summary').click()
            page.locator('#sortingDevice').select_option('cpu')
            page.locator('#sortingRun').click()
            expect(page.locator('#sortingJobs')).to_contain_text('Test engine failure')
            assert len(submitted) == 1
            assert len(submitted[0]['target_ids']) == 2
            assert submitted[0]['baseline_id'] not in submitted[0]['target_ids']
            assert submitted[0]['params']['device'] == 'cpu'
            assert not page_errors
        finally:
            browser.close()


def test_sorter_install_button_displays_logs_and_restart(workshop_server):
    playwright = pytest.importorskip('playwright.sync_api')
    url, _ = workshop_server
    with playwright.sync_playwright() as driver:
        if not Path(driver.chromium.executable_path).exists():
            pytest.skip('Install Chromium for Playwright')
        browser = driver.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            commands = []
            installation = None
            def intercept(route):
                nonlocal installation
                command = route.request.post_data_json
                if command['action'] == 'install':
                    commands.append(command)
                    installation = dict(status='completed', restart_required=True,
                        message='Installation complete. Restart the workshop.', log='Successfully installed example')
                    route.fulfill(json=installation)
                else:
                    route.fulfill(json=dict(files=[], jobs=[], sorters=[], installation=installation,
                        install_plans=[dict(id='spikeinterface', label='SpikeInterface Python sorters', description='Install Python dependencies')]))
            page.route('**/api/sorting', intercept)
            page.goto(url + '/#sorting')
            page.get_by_text('Install sorter dependencies', exact=True).click()
            page.locator('#sortingInstall').click()
            expect = playwright.expect
            expect(page.locator('#sortingInstallStatus')).to_contain_text('Restart the workshop')
            expect(page.locator('#sortingInstallLog')).to_contain_text('Successfully installed')
            expect(page.locator('#sortingInstall')).to_be_disabled()
            expect(page.locator('#sortingRun')).to_be_disabled()
            assert commands == [dict(action='install', plan='spikeinterface')]
        finally:
            browser.close()


@pytest.mark.parametrize('kind,title', [('cartpole', 'CartPole'), ('foodland', 'Foodland'), ('ant', 'Ant')])
def test_native_game_scenes_render_in_monitor(workshop_server, kind, title):
    playwright = pytest.importorskip('playwright.sync_api')
    from braindance.examples.streaming_workshop.environments import WorkshopGame
    from braindance.examples.streaming_workshop.native_visualization import game_scene
    game = WorkshopGame(kind)
    try:
        observation = game.reset()
        payload = dict(status='running', native=True, phase='game', phase_kind='environment',
                       scene=game_scene(kind, game.env, observation), reward=10., episode_reward=1., episodes=0)
    finally:
        game.close()
    with playwright.sync_playwright() as driver:
        if not Path(driver.chromium.executable_path).exists():
            pytest.skip('Install Chromium for Playwright')
        browser = driver.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            errors = []
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.route('**/api/state', lambda route: route.fulfill(json=payload))
            page.goto(workshop_server[0])
            page.evaluate("showView('monitor')")
            playwright.expect(page.locator('#gameTitle')).to_have_text(title)
            playwright.expect(page.locator('#reward')).to_have_text('1.0')
            playwright.expect(page.locator('#environmentPlaceholder')).to_be_hidden()
            # Check drawn pixels, not just a visible blank canvas.
            assert page.evaluate("""() => {
                showView('monitor'); draw(); const c=$('scene'), pixels=c.getContext('2d').getImageData(0,0,c.width,c.height).data;
                let colored=0;for(let i=0;i<pixels.length;i+=4)if(pixels[i+3]&&pixels[i]<180)colored++;
                return colored>30;
            }""")
            payload.clear()
            payload.update(status='running', native=True, phase='record')
            playwright.expect(page.locator('#gameTitle')).to_have_text('Environment')
            playwright.expect(page.locator('#episodeStatus')).to_have_text('')
            playwright.expect(page.locator('#reward')).to_have_text('—')
            assert page.evaluate("""() => {
                showView('monitor'); draw(); const c=$('scene'), pixels=c.getContext('2d').getImageData(0,0,c.width,c.height).data;
                return pixels.every(value => value === 0);
            }""")
            assert not errors
        finally:
            browser.close()
