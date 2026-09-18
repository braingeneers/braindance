"""Distinct game choices retain their identity through validation and saved plans."""
import json

import pytest

from braindance.examples.streaming_workshop.experiment_spec import phase_catalog, verify_spec
from braindance.examples.streaming_workshop.session import build_phase
from test_workshop_browser import workshop_server


@pytest.mark.parametrize('game', ['cartpole', 'foodland', 'ant'])
@pytest.mark.parametrize('skip', [False, True])
def test_game_type_selects_game_independently_of_global_settings(game, skip):
    definition = phase_catalog()[game]
    assert 'environment' not in definition['params']
    report = verify_spec(dict(channels=8, baseline_hz=[0.] * 8,
        environment='foodland' if game == 'cartpole' else 'cartpole',
        phases=[dict(id='play', type=game, params={})]), skip=skip)
    assert report['ok'], report
    row = report['phases'][0]
    assert row['type'] == game
    assert row['params']['environment'] == game
    assert row['params']['sensory_index'] == (2 if game == 'cartpole' else 0)
    phase = build_phase(game, identifier=row['id'], params=row['params'])
    assert phase.phase_params['environment'] == game
    assert phase.kind == 'environment'


def test_native_episode_reward_resets_without_losing_total(tmp_path):
    import gymnasium as gym
    from braindance.examples.streaming_workshop.native_visualization import ObservedGame

    progress = tmp_path / 'progress.json'
    game = ObservedGame(gym.make('CartPole-v1', max_episode_steps=2), 'cartpole', progress, 'play')
    try:
        game.reset(seed=7)
        game.step(0)
        game.step(1)
        state = json.loads(progress.read_text())
        assert state['reward'] == state['episode_reward'] == 2.
        assert state['episodes'] == 1
        game.reset()
        state = json.loads(progress.read_text())
        assert state['reward'] == 2.
        assert state['episode_reward'] == 0.
        game.step(0)
        assert game.episode_reward == 1.
        assert game.reward == 3.
    finally:
        game.close()


def test_mixed_game_plan_roundtrip_and_legacy_compatibility():
    phases = [dict(id='record', type='recording', params={})]
    phases += [dict(id=game, type=game, params={'environment_seconds': .04})
               for game in ['foodland', 'cartpole', 'ant']]
    config = json.loads(json.dumps(dict(channels=8, phases=phases)))
    report = verify_spec(config)
    assert report['ok'], report
    assert [row['params']['environment'] for row in report['phases'][1:]] == ['foodland', 'cartpole', 'ant']
    config['phases'][1] = dict(id='old_foodland', type='environment',
                              params={'environment': 'foodland', 'environment_seconds': .04})
    report = verify_spec(config)
    assert report['ok'], report
    assert report['phases'][1]['params']['environment'] == 'foodland'


def test_game_choices_and_legacy_plan_migration_in_browser(workshop_server):
    playwright = pytest.importorskip('playwright.sync_api')
    url, _ = workshop_server
    with playwright.sync_playwright() as driver:
        browser = driver.chromium.launch(headless=True)
        try:
            page = browser.new_page(viewport={'width': 1600, 'height': 1000})
            page.goto(url)
            page.wait_for_function('typeof builderLoaded !== "undefined" && builderLoaded')
            page.evaluate('''() => {
                loadBuilder({phases:[{id:'record',type:'recording',params:{}}]});
                showView('sequence');
            }''')
            page.locator('.flow-gap').last.click()
            for game, label in [('foodland', 'FoodLand'), ('cartpole', 'CartPole'), ('ant', 'Ant')]:
                assert page.locator(f'#phaseType option[value="{game}"]').count() == 1
                page.locator('#flowOptions .flow-add').filter(has_text='＋ ' + label).click()
            assert page.locator('#phaseType option[value="environment"]').count() == 0
            assert 'Mapped environment' not in page.locator('#flowOptions').inner_text()
            assert page.evaluate('phaseSpecs.map(p=>p.type)') == ['recording', 'foodland', 'cartpole', 'ant']
            page.locator('.flow-piece').last.click()
            assert page.locator('#flowInspector select').count() == 0
            page.screenshot(path='/tmp/workshop-distinct-game-phases.png', full_page=True)
            config = page.evaluate('builderConfig()')
            assert all('environment' not in phase['params'] for phase in config['phases'][1:])
            page.evaluate('(config)=>loadBuilder(config)', config)
            assert page.evaluate('builderConfig()') == config
            page.evaluate('''() => loadBuilder({environment:'cartpole', phases:[
                {id:'record',type:'recording',params:{}},
                {id:'old_ant',type:'environment',params:{environment:'ant',environment_seconds:2}}
            ]})''')
            migrated = page.evaluate('builderConfig()')
            assert migrated['phases'][1]['type'] == 'ant'
            assert migrated['phases'][1]['params']['environment_seconds'] == 2
            assert 'environment' not in migrated['phases'][1]['params']
        finally:
            browser.close()
