// Manual games are independent from neural acquisition and experiment phases.
(() => {
  const host = views.playground;
  host.insertAdjacentHTML('beforeend', `
    <p class="help">Try the real environment inputs before connecting neural activity. This playground does not start a recording.</p>
    <section class="panel"><div class="fields">
      <div><label for="pgEnvironment">Environment</label><select id="pgEnvironment"><option value="cartpole">CartPole</option><option value="foodland">FoodLand</option><option value="ant">Ant (MuJoCo)</option></select></div>
      <div><label for="pgSeed">Random seed</label><input id="pgSeed" type="number" min="0" max="4294967295" step="1" value="7"></div>
    </div><div class="toolbar"><button id="pgLoad" class="primary">Load / reset</button><button id="pgRun" disabled>Play</button><button id="pgStep" disabled>Single step (20 ms)</button><button id="pgZero">Zero inputs</button></div>
    <div id="pgError" class="error" role="alert"></div><div id="pgStatus" role="status">Choose an environment, then load it.</div></section>
    <div class="grid"><section class="panel wide"><h2>Environment</h2><canvas id="pgScene" style="height:300px" tabindex="0" aria-label="Environment keyboard controls"></canvas><p id="pgHelp" class="help">Hold a direction key or an on-screen button to play. Release to stop applying that input.</p><div id="pgJoints" class="pg-joints"></div><div id="pgActions" class="pg-controls"></div><p id="pgInput" class="help" role="status">Inputs released</p><p id="pgApplied" class="help">Applied inputs: —</p></section>
    <section class="panel"><h2>Observation monitor</h2><label for="pgObservation">State variable</label><select id="pgObservation"></select><canvas id="pgPlot"></canvas><div id="pgValue" class="help"></div><p class="help">Last 10 simulated seconds; vertical scale follows the selected state variable. Pausing stops simulation time.</p></section></div>`);
  let state = null, playing = false, busy = false, samples = [], buttons = [], joint = 0;
  const keys = new Set(), pointers = new Map();
  function pause() { playing = false; keys.clear(); pointers.clear(); $('pgRun').textContent = 'Play'; draw(); }
  function zero() { keys.clear(); pointers.clear(); feedback(); }
  function held(direction) {
    const aliases = {left:['arrowleft','a'], right:['arrowright','d'], forward:['arrowup','w']};
    return aliases[direction].some(key => keys.has(key)) || [...pointers.values()].includes(direction);
  }
  function action() {
    const values = state.action_names.map(() => 0);
    // FoodLand advances heading by action * 0.5 rad each 20 ms: 0.08 gives 2 rad/s.
    const turn = Number(held('right')) - Number(held('left'));
    values[state.environment === 'ant' ? joint : 0] = turn * (state.environment === 'foodland' ? .08 : 1);
    if (state.environment === 'foodland' && held('forward')) values[1] = 1;
    return values;
  }
  function feedback() {
    if (!state) return;
    for (const button of buttons) {
      button.setAttribute('aria-pressed', String(held(button.dataset.direction)));
      button.disabled = state.done;
    }
    for (const button of $('pgJoints').children) button.setAttribute('aria-pressed', String(Number(button.dataset.joint) === joint));
    const values = action();
    $('pgInput').textContent = values.some(v => v !== 0)
      ? 'Current input: '+values.map((v,i) => `${state.action_names[i]}: ${v.toFixed(2)}`).join(' · ')
      : 'Inputs released · zero force / turn / torque';
  }
  function playOnInput() {
    if (!state || state.done) return;
    playing = true; $('pgRun').textContent = 'Pause'; draw();
  }
  function draw() {
    if (!state || host.hidden) return;
    feedback();
    scene(state.scene, 'pgScene');
    $('pgApplied').textContent = 'Last applied inputs: '+(state.action ? state.action.map((v,i) => `${i+1}: ${v.toFixed(2)}`).join(' · ') : 'none yet');
    const index = Number($('pgObservation').value || 0), values = samples.map(s => s.observation[index]);
    plot('pgPlot', values, []);
    $('pgValue').textContent = `${state.observation_names[index]}: ${Number(state.observation[index]).toFixed(4)} · ${samples.length ? samples[0].time.toFixed(2) : '0.00'}–${state.time.toFixed(2)} s · range ${values.length ? Math.min(...values).toFixed(3) : '—'} to ${values.length ? Math.max(...values).toFixed(3) : '—'}`;
    $('pgStatus').textContent = `${playing ? 'Playing' : 'Paused'} · ${state.time.toFixed(2)} s · reward ${state.reward.toFixed(2)}${state.done ? ' · '+state.end_reason+' — reset to play again' : ''}`;
    $('pgRun').disabled = $('pgStep').disabled = state.done;
  }
  async function send(command) {
    if (busy) return;
    busy = true; $('pgError').textContent = '';
    try {
      state = await request('/api/playground', command);
      if (command.kind === 'reset') {
        samples = [{time: 0, observation: state.observation}];
        $('pgObservation').replaceChildren(...state.observation_names.map((name, i) => new Option(name, i)));
        $('pgActions').replaceChildren(); $('pgJoints').replaceChildren(); joint = 0;
        keys.clear(); pointers.clear();
        if (state.environment === 'ant') state.action_names.forEach((name,i) => {
          const button = document.createElement('button'); button.type = 'button';
          button.textContent = `${i+1}. ${name}`; button.dataset.joint = i;
          button.onclick = () => { joint = i; feedback(); };
          $('pgJoints').append(button);
        });
        const directions = [['left','← / A',state.environment === 'ant' ? 'Negative torque' : 'Left'],
          ['right','→ / D',state.environment === 'ant' ? 'Positive torque' : 'Right']];
        if (state.environment === 'foodland') directions.splice(1,0,['forward','↑ / W','Forward']);
        buttons = directions.map(([direction,key,label]) => {
          const button = document.createElement('button'); button.type = 'button';
          button.dataset.direction = direction; button.setAttribute('aria-label', `${label}: hold ${key}`);
          const cap = document.createElement('kbd'), caption = document.createElement('span');
          cap.textContent = key; caption.textContent = label; button.append(cap,caption);
          button.onpointerdown = event => {
            if (event.button !== 0 || state.done) return;
            event.preventDefault(); pointers.set(event.pointerId,direction);
            button.setPointerCapture(event.pointerId); playOnInput();
          };
          for (const name of ['pointerup','pointercancel','lostpointercapture']) button.addEventListener(name,event => {
            pointers.delete(event.pointerId); feedback();
          });
          $('pgActions').append(button); return button;
        });
        const help = {cartpole:'Hold ←/A or →/D to push the cart.', foodland:'Hold ←/A or →/D to turn gently (about 115°/s), ↑/W to move forward. Combine them to steer.', ant:'Select a joint with 1–8 or the joint buttons, then hold ←/A or →/D for negative/positive torque. These are actual MuJoCo torques, not a walking policy.'};
        $('pgHelp').textContent = help[state.environment]+' Hold keys or buttons to start playing; release for zero input. Space pauses/plays; 0 clears inputs.';
        $('pgScene').focus();
      } else samples.push(...state.samples);
      samples = samples.filter(s => s.time >= state.time - 10);
      if (state.done) pause();
      draw();
    } catch (err) { pause(); $('pgError').textContent = err.message; }
    finally { busy = false; }
  }
  function toggle() { if (!state || state.done) return; if (playing) pause(); else playOnInput(); }
  $('pgLoad').onclick = () => { pause(); send({kind:'reset',environment:$('pgEnvironment').value,seed:Number($('pgSeed').value)}); };
  $('pgRun').onclick = toggle;
  $('pgStep').onclick = () => { pause(); if (state && !state.done) send({kind:'step',action:action(),steps:1}); };
  $('pgZero').onclick = zero;
  $('pgObservation').onchange = draw;
  host.addEventListener('keydown', event => {
    if (host.hidden || !state || state.done || event.target.matches('input,select,textarea') || event.target.isContentEditable) return;
    const key = event.key.toLowerCase();
    const movement = ['arrowleft','arrowright','a','d', ...(state.environment === 'foodland' ? ['arrowup','w'] : [])];
    if (![...movement,' ','0', ...(state.environment === 'ant' ? ['1','2','3','4','5','6','7','8'] : [])].includes(key)) return;
    event.preventDefault();
    if (key === ' ') { if (!event.repeat) toggle(); }
    else if (key === '0') zero();
    else if (/^[1-8]$/.test(key)) { joint = Number(key)-1; feedback(); }
    else if (!event.repeat) { keys.add(key); playOnInput(); }
  });
  window.addEventListener('keyup', event => { keys.delete(event.key.toLowerCase()); feedback(); });
  host.addEventListener('focusout', event => { if (!host.contains(event.relatedTarget)) zero(); });
  window.addEventListener('blur', pause);
  document.addEventListener('visibilitychange', () => { if (document.hidden) pause(); });
  window.addEventListener('resize', draw);
  new MutationObserver(() => { if (host.hidden) pause(); else draw(); }).observe(host,{attributes:true,attributeFilter:['hidden']});
  setInterval(() => { if (playing && !busy && !host.hidden && !document.hidden) send({kind:'step',action:action(),steps:2}); }, 40);
})();
