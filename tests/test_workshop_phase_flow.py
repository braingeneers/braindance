"""Browser integration for dependency-aware phase planning."""
import pytest
from test_workshop_browser import workshop_server


def test_phase_flow_dependencies_loops_and_roundtrip(workshop_server):
    playwright = pytest.importorskip('playwright.sync_api')
    url, token = workshop_server
    with playwright.sync_playwright() as driver:
        browser = driver.chromium.launch(headless=True)
        page = browser.new_page(viewport={'width': 1500, 'height': 1000})
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(url)
        page.wait_for_function('typeof builderLoaded !== "undefined" && builderLoaded')
        page.locator('#dataSource').select_option('simulation')
        page.get_by_role('button', name='New sequence', exact=True).click()
        page.locator('[data-experiment-tab=sequence]').first.click()
        probe = page.locator('#flowOptions button').filter(has_text='Response probes')
        assert probe.get_attribute('aria-disabled') == 'true'
        probe.click(force=True)
        assert page.locator('#flowOptions details[open]').inner_text().find('Recording') >= 0
        assert page.locator('.flow-piece').count() == 0
        page.locator('#flowOptions button').filter(has_text='＋ Recording').click()
        assert probe.get_attribute('aria-disabled') == 'false'
        probe.click()
        page.locator('#flowOptions button').filter(has_text='＋ CartPole').click()
        page.get_by_role('button', name='Remove recording', exact=True).click()
        assert page.locator('.flow-piece').count() == 3
        assert 'without its inputs' in page.locator('#flowStatus').inner_text()
        page.locator('.flow-piece input[type=checkbox]').nth(1).check()
        page.locator('.flow-piece input[type=checkbox]').nth(2).check()
        page.locator('#flowCount').fill('3')
        page.locator('#flowLoop').click()
        assert page.locator('.flow-loop').count() == 1
        assert '7 steps' in page.locator('#flowTotal').inner_text()
        assert 'Projected' in page.locator('#flowContext').inner_text()
        config = page.evaluate('builderConfig()')
        assert len(config['phases']) == 7
        assert len({p['id'] for p in config['phases']}) == 7
        page.evaluate('(values) => loadBuilder(values)', config)
        assert page.locator('.flow-loop').count() == 1
        report = page.evaluate('async () => request("/api/verify-experiment", {config: {...settings,...collect()}, preflight:false})')
        assert report['ok'], report
        assert len(report['phases']) == 7
        page.screenshot(path='/tmp/workshop-phase-flow.png', full_page=True)
        page.set_viewport_size({'width': 390, 'height': 844})
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        # Execute a short repeated recording section through the real control API.
        run_config = page.evaluate('''() => {
            phaseSpecs = [{id:'a',type:'recording',params:{record_seconds:.04},loop:{id:'repeat',count:3}},
                          {id:'b',type:'recording',params:{record_seconds:.04},loop:{id:'repeat',count:3}}];
            return {...collect(),speed:0};
        }''')
        response = page.request.post(url + '/api/control', data={'kind':'start', 'config':run_config},
                                     headers={'X-Workshop-Token':token})
        assert response.ok, response.text()
        page.wait_for_function('state.status === "completed" || state.status === "error"')
        state = page.evaluate('state')
        assert state['status'] == 'completed', state
        assert len(state['phases']) == 6
        assert not errors
        browser.close()


def test_phase_board_dragging_shapes_and_editing(workshop_server):
    playwright = pytest.importorskip('playwright.sync_api')
    url, _ = workshop_server
    with playwright.sync_playwright() as driver:
        browser = driver.chromium.launch(headless=True)
        page = browser.new_page(viewport={'width': 1700, 'height': 1100})
        page.set_default_timeout(5000)
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(url)
        page.wait_for_function('typeof builderLoaded !== "undefined" && builderLoaded')
        assert page.request.get(url + '/phase_flow.css').ok
        assert page.request.get(url + '/workshop_theme.css').ok
        page.evaluate('loadBuilder({phases:[]}); showView("sequence")')
        page.locator('#flowOptions .flow-add').filter(has_text='＋ Recording').drag_to(page.locator('.flow-gap'))
        assert page.locator('.flow-piece').count() == 1
        page.locator('#flowOptions .flow-add').filter(has_text='Response probes').drag_to(page.locator('.flow-gap').last)
        assert page.evaluate('phaseSpecs.map(p=>p.type)') == ['recording', 'causal']
        page.locator('.flow-piece').first.drag_to(page.locator('.flow-gap').last)
        assert page.evaluate('phaseSpecs.map(p=>p.type)') == ['recording', 'causal']
        assert 'without its inputs' in page.locator('#flowStatus').inner_text()
        page.locator('#flowOptions .flow-add').filter(has_text='＋ Recording').click()
        page.locator('.flow-piece').last.drag_to(page.locator('.flow-gap[data-position="1"]'))
        assert page.evaluate('phaseSpecs.map(p=>p.type)') == ['recording', 'recording', 'causal']
        page.locator('#flowUndo').click()
        assert page.evaluate('phaseSpecs.map(p=>p.type)') == ['recording', 'causal', 'recording']
        page.locator('#flowRedo').click()
        assert page.evaluate('phaseSpecs.map(p=>p.type)') == ['recording', 'recording', 'causal']
        page.locator('.flow-piece').first.click()
        page.locator('#flowInspector input[type=number]').fill('1.5')
        page.get_by_role('button', name='Apply settings', exact=True).click()
        assert page.evaluate('phaseSpecs[0].params.record_seconds') == 1.5
        page.locator('.flow-piece input[type=checkbox]').nth(1).check()
        page.locator('.flow-piece input[type=checkbox]').nth(2).check()
        page.locator('#flowLoop').click()
        page.locator('.flow-gap').last.click()
        page.locator('#flowOptions .flow-add').filter(has_text='＋ Recording').click()
        page.locator('.flow-loop-title').drag_to(page.locator('.flow-gap').last)
        assert page.evaluate('phaseSpecs.map(p=>p.type)') == ['recording', 'recording', 'recording', 'causal']
        assert page.evaluate('phaseSpecs.slice(2).every(p=>p.loop)')
        assert page.evaluate('phaseSpecs.slice(0,2).every(p=>!p.loop)')
        # Connected pieces fit on a wide screen; narrow the board to test panning.
        page.set_viewport_size({'width': 1100, 'height': 1100})
        page.locator('#flowViewport').evaluate('(el)=>el.scrollLeft=100')
        board = page.locator('#flowViewport').bounding_box()
        before_pan = page.locator('#flowViewport').evaluate('(el)=>el.scrollLeft')
        page.mouse.move(board['x'] + 200, board['y'] + 20)
        page.mouse.down()
        page.mouse.move(board['x'] + 140, board['y'] + 20, steps=5)
        page.mouse.up()
        assert page.locator('#flowViewport').evaluate('(el)=>el.scrollLeft') > before_pan
        page.set_viewport_size({'width': 1700, 'height': 1100})
        page.locator('#flowExpand').click()
        assert 'flow-expanded' in page.locator('#view-sequence').get_attribute('class')
        page.keyboard.press('Escape')
        assert 'flow-expanded' not in page.locator('#view-sequence').get_attribute('class')
        page.locator('#flowZoomOut').click()
        assert page.locator('#flowZoomReset').inner_text() == '90%'
        page.locator('#flowZoomReset').click()
        # Real catalog categories and their required recording data determine connectors.
        page.evaluate('''() => {
            const types=['native:phases3.RecordPhaseV3','native:phases_analysis_3.RTSortPhaseV3'];
            loadBuilder({phases:types.map((type,i)=>({id:'phase_'+i,type,params:{...catalog[type].params}}))});
        }''')
        assert page.locator('.flow-piece[data-kind=experiment]').count() == 1
        assert page.locator('.flow-piece[data-kind=analysis][data-input=experiment]').count() == 1
        experiment_color = page.locator('.flow-piece[data-kind=experiment] .flow-node-head').evaluate('(el)=>getComputedStyle(el).backgroundColor')
        analysis_color = page.locator('.flow-piece[data-kind=analysis] .flow-node-head').evaluate('(el)=>getComputedStyle(el).backgroundColor')
        assert experiment_color != analysis_color
        page.locator('.flow-piece[data-kind=analysis]').drag_to(page.locator('.flow-piece[data-kind=experiment]'))
        assert page.evaluate('phaseSpecs[0].type') == 'native:phases3.RecordPhaseV3'
        assert 'recording_file' in page.locator('#flowStatus').inner_text()
        page.locator('#flowCategory').select_option('analysis')
        page.screenshot(path='/tmp/workshop-board-light.png', full_page=True)
        page.locator('#themeToggle').click()
        page.screenshot(path='/tmp/workshop-board-dark.png', full_page=True)
        assert page.locator('.flow-piece[data-kind=experiment] .flow-node-head').evaluate('(el)=>getComputedStyle(el).backgroundColor') != experiment_color
        page.evaluate('showView("experiment")')
        page.screenshot(path='/tmp/workshop-setup-dark.png', full_page=True)
        page.evaluate('showView("sequence")')
        page.set_viewport_size({'width': 390, 'height': 844})
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        page.screenshot(path='/tmp/workshop-board-mobile.png', full_page=True)
        assert not errors
        browser.close()


def test_drag_onto_node_reorders_and_loop_selection_is_visible(workshop_server):
    playwright = pytest.importorskip('playwright.sync_api')
    url, _ = workshop_server
    with playwright.sync_playwright() as driver:
        browser = driver.chromium.launch(headless=True)
        page = browser.new_page(viewport={'width': 1600, 'height': 1000})
        page.set_default_timeout(5000)
        page.goto(url)
        page.wait_for_function('typeof builderLoaded !== "undefined" && builderLoaded')
        page.evaluate('''() => {
            loadBuilder({phases:['first','second','third'].map(id=>({id,type:'recording',params:{record_seconds:.04}}))});
            showView('sequence');
        }''')
        # Users drop on the node they want to move past, not only a tiny + target.
        page.locator('.flow-piece[data-id=third]').drag_to(page.locator('.flow-piece[data-id=first]'))
        assert page.evaluate('phaseSpecs.map(p=>p.id)') == ['third', 'first', 'second']
        page.locator('#flowLoop').click()
        assert 'first phase' in page.locator('#flowSelectionHelp').inner_text()
        page.locator('.flow-piece[data-id=third] h3').click()
        page.locator('.flow-piece[data-id=second] h3').click()
        assert page.locator('.flow-piece[data-selected=true]').count() == 3
        assert 'Create loop' in page.locator('#flowLoop').inner_text()
        page.screenshot(path='/tmp/workshop-loop-selecting.png', full_page=True)
        page.locator('#flowLoop').click()
        assert page.locator('.flow-loop').count() == 1
        assert page.evaluate('phaseSpecs.every(p=>p.loop.count===2)')
        page.screenshot(path='/tmp/workshop-loop-selection.png', full_page=True)
        browser.close()


def test_real_busy_bee_builds_and_runs_from_board(workshop_server):
    import json
    from pathlib import Path

    playwright = pytest.importorskip('playwright.sync_api')
    config = json.loads((Path(__file__).parents[1] / 'braindance/examples/streaming_workshop/real_experiment_busy_bee.json').read_text())
    url, _ = workshop_server
    with playwright.sync_playwright() as driver:
        browser = driver.chromium.launch(headless=True)
        page = browser.new_page(viewport={'width': 1600, 'height': 1000})
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(url)
        page.wait_for_function('typeof builderLoaded !== "undefined" && builderLoaded')
        page.evaluate('(config)=>{hydrate(config);showView("sequence");}', config)
        assert page.locator('.flow-piece').count() == 10
        assert page.evaluate('builderConfig().phases') == config['phases']
        report = page.evaluate('async()=>request("/api/verify-experiment",{config:collect(),preflight:false})')
        assert report['ok'], report
        assert len(report['phases']) == 10
        # Use the actual Run experiment UI and its preflight, not a mock runner.
        page.locator('#flowSetup').click()
        page.locator('#launchExperiment').click()
        page.wait_for_function('state.status === "completed" || state.status === "error"', timeout=20000)
        state = page.evaluate('state')
        assert state['status'] == 'completed', state
        assert len(state['phases']) == 10
        assert not errors
        browser.close()


def test_board_play_button_runs_dummy_maxwell_recording_and_stimulation(workshop_server):
    import csv
    from pathlib import Path
    import h5py
    import numpy as np

    playwright = pytest.importorskip('playwright.sync_api')
    url, _ = workshop_server
    with playwright.sync_playwright() as driver:
        browser = driver.chromium.launch(headless=True)
        page = browser.new_page(viewport={'width': 1500, 'height': 1000})
        errors, starts = [], []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.on('request', lambda request: starts.append(request.post_data_json)
                if request.url.endswith('/api/control') and request.method == 'POST' else None)
        page.goto(url)
        page.wait_for_function('typeof builderLoaded !== "undefined" && builderLoaded')
        page.evaluate('loadBuilder({phases:[]});showView("sequence")')
        assert page.get_by_role('button', name='Run experiment from phase board', exact=True).is_disabled()
        page.evaluate('''() => hydrate({
            experiment_name:'dummy_maxwell_play', source:null, speed:0,
            channels:8, num_neurons:8, native_settings:{stim_electrodes:[0],verbose:false},
            phases:[{id:'record',type:'native:phases3.RecordPhaseV3',params:{duration:.04}},
                    {id:'stim',type:'native:phases3.FrequencyStimPhaseV3',
                     params:{stim_command:[[0],250,100],stim_freq:100,duration:.04}}]
        })''')
        page.locator('#flowSetup').click()
        page.locator('#dataSource').select_option('replay')
        page.locator('#sourcePath').fill('sine')
        assert page.locator('#sourcePathField').is_visible()
        assert not page.locator('#liveConfigField').is_visible()
        page.locator('[data-experiment-tab=sequence]').first.click()
        page.get_by_role('button', name='Run experiment from phase board', exact=True).click()
        page.wait_for_function('state.status === "completed" || state.status === "error"', timeout=20000)
        state = page.evaluate('state')
        assert state['status'] == 'completed', state
        assert state['execution']['source'] == 'Dummy Maxwell (sine)'
        assert state['execution']['engine'] == 'Native V3'
        commands = [command for command in starts if command['kind'] == 'start']
        assert len(commands) == 1
        assert commands[0]['config']['source'] == 'sine'
        assert commands[0]['config']['live_config'] is None
        assert state['phases'] == ['record', 'stim']
        output = Path(state['output'])
        recordings = list(output.rglob('*.raw.h5'))
        assert len(recordings) == 2
        # Backend evidence: the dummy Maxwell source's deterministic 1024-channel
        # sine signal, not NeuralSimulationSource's network activity.
        for recording in recordings:
            with h5py.File(recording) as file:
                routed = file['wells/well000/rec0000/groups/routed']
                raw = routed['raw']
                assert raw.shape[0] == 1024 and raw.shape[1] >= 800
                frames = routed['frame_nos'][:200]
                expected = (512 + 100*np.sin(2*np.pi*frames/20000)).astype(np.uint16)
                np.testing.assert_array_equal(raw[0,:200], expected)
        pulses = []
        for log in output.rglob('*_log.csv'):
            with log.open() as handle:
                pulses.extend(csv.DictReader(handle))
        assert len(pulses) == 4, pulses
        assert [int(row['replay_frame']) for row in pulses] == [0,200,400,600]
        playwright.expect(page.locator('#runBadge')).to_have_text('completed')
        page.wait_for_function("[...document.querySelectorAll('#phaseProgress small')].length === 2 && [...document.querySelectorAll('#phaseProgress small')].every(el=>el.textContent==='completed')")
        page.screenshot(path='/tmp/workshop-dummy-maxwell-play.png', full_page=True)
        assert not errors
        browser.close()


def test_phase_board_pinch_zoom_trackpad_and_touch(workshop_server):
    playwright = pytest.importorskip('playwright.sync_api')
    url, _ = workshop_server
    with playwright.sync_playwright() as driver:
        browser = driver.chromium.launch(headless=True)
        page = browser.new_page(viewport={'width': 1500, 'height': 1000}, has_touch=True)
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(url)
        page.wait_for_function('typeof builderLoaded !== "undefined" && builderLoaded')
        page.evaluate('''() => {
            loadBuilder({phases:['a','b','c','d','e'].map(id=>({id,type:'recording',params:{record_seconds:.04}}))});
            showView('sequence');
        }''')
        board = page.locator('#flowViewport')
        bounds = board.bounding_box()
        x, y = bounds['x'] + 250, bounds['y'] + 70
        board.evaluate('(el)=>el.scrollLeft=300')
        before = board.evaluate('(el)=>(el.scrollLeft+250)/Number(document.querySelector("#flowSequence").style.zoom||1)')
        page.mouse.move(x, y)
        page.keyboard.down('Control')
        page.mouse.wheel(0, -20)
        page.keyboard.up('Control')
        playwright.expect(page.locator('#flowZoomReset')).to_have_text('122%')
        after = board.evaluate('(el)=>(el.scrollLeft+250)/Number(document.querySelector("#flowSequence").style.zoom)')
        assert abs(before-after) < 2
        page.keyboard.down('Control')
        page.mouse.wheel(0, 40)
        page.keyboard.up('Control')
        playwright.expect(page.locator('#flowZoomReset')).to_have_text('82%')
        page.locator('#flowZoomReset').click()
        board.evaluate('(el)=>el.scrollLeft=300')
        page.mouse.move(x,y)
        page.mouse.wheel(100, 0)
        page.wait_for_function('document.querySelector("#flowViewport").scrollLeft>300')
        assert page.locator('#flowZoomReset').inner_text() == '100%'
        # Actual browser touch input, including centroid anchoring and no reordering.
        cdp = page.context.new_cdp_session(page)
        def points(half_distance):
            return [{'x':x-half_distance,'y':y,'id':1}, {'x':x+half_distance,'y':y,'id':2}]
        before = board.evaluate('(el)=>el.scrollLeft+250')
        cdp.send('Input.dispatchTouchEvent', {'type':'touchStart','touchPoints':points(50)})
        cdp.send('Input.dispatchTouchEvent', {'type':'touchMove','touchPoints':points(65)})
        playwright.expect(page.locator('#flowZoomReset')).to_have_text('130%')
        after = board.evaluate('(el)=>(el.scrollLeft+250)/Number(document.querySelector("#flowSequence").style.zoom)')
        assert abs(before-after) < 2
        cdp.send('Input.dispatchTouchEvent', {'type':'touchMove','touchPoints':points(40)})
        playwright.expect(page.locator('#flowZoomReset')).to_have_text('80%')
        cdp.send('Input.dispatchTouchEvent', {'type':'touchEnd','touchPoints':[]})
        assert page.evaluate('phaseSpecs.map(p=>p.id)') == ['a','b','c','d','e']
        assert page.evaluate('visualViewport.scale') == 1
        assert not page.locator('#flowInspector').is_visible()
        assert 'flow-dragging' not in page.locator('#view-sequence').get_attribute('class')
        assert not errors
        browser.close()
