const $ = id => document.getElementById(id);
let state = {}, settings = {}, matrixValues = [], selectedProfile = '', localError = '';
let namesKey = '', dragging = false, painted = false, matrixDirty = false;
let sourceUploadPending = false;
let savedCode = '', previousStatus = '', lastServerMatrix = '';

const fields = {
  environment: 'environment', detection: 'detection', channels: 'channels', seed: 'seed',
  sensory: 'sensory_index', egain: 'encoder_gain', dgain: 'decoder_gain', record: 'record_seconds',
  repeats: 'causal_repeats', duration: 'environment_seconds', episode: 'episode_seconds'
};
const listFields = {left: 'left_channels', right: 'right_channels', stim: 'stim_electrodes', baseline: 'baseline_hz'};
const defaults = {environment:'cartpole', detection:'events', channels:400, seed:7, sensory_index:2,
  encoder_gain:100, decoder_gain:.025, record_seconds:3, causal_repeats:4,
  environment_seconds:120, episode_seconds:10, left_channels:[0,1,2,3], right_channels:[4,5,6,7],
  stim_electrodes:[0,1], baseline_hz:[]};

function error(message) { localError = message || ''; $('error').textContent = localError || state.error || ''; }
async function request(path, body) {
  const response = await fetch(path, body === undefined ? {} : {
    method:'POST', headers:{'Content-Type':'application/json','X-Workshop-Token':WORKSHOP_TOKEN},
    body:JSON.stringify(body)
  });
  const result = await response.json();
  if (!response.ok) throw Error(result.error || 'Request failed');
  return result;
}
async function control(kind, extra = {}) {
  try { await request('/api/control', {kind,...extra}); error(''); return true; }
  catch (e) { error(e.message); return false; }
}
function running() { return ['starting','running'].includes(state.status); }
function numbers(id) {
  const values = $(id).value.split(',').filter(v => v.trim()).map(Number);
  if (values.some(v => !Number.isFinite(v))) throw Error(`Invalid numbers in ${id}`);
  return values;
}
function collect() {
  if(sourceUploadPending)throw Error('Wait for the selected file to finish uploading.');
  const mode=$('dataSource').value;
  const result = {source:mode==='dummy'?'sine':mode==='replay'?$('sourcePath').value.trim():null,
    live_config:mode==='live'?$('liveConfigPath').value.trim():null,
    speed:mode==='live'?1:Number($('acquisitionSpeed').value)};
  if(mode==='replay'&&!result.source)throw Error('Choose or drop a recording file, or enter its server path.');
  if(mode==='live'&&!result.live_config)throw Error('Choose a Maxwell configuration file, or enter its server path.');
  for (const [id,key] of Object.entries(fields)) {
    result[key] = ['environment','detection'].includes(id) ? $(id).value : Number($(id).value);
    if (typeof result[key] === 'number' && !Number.isFinite(result[key])) throw Error(`Invalid ${key}`);
  }
  for (const [id,key] of Object.entries(listFields)) result[key] = numbers(id);
  if (mode==='simulation') result.adjacency = matrixValues;
  if (typeof builderConfig === 'function') Object.assign(result, builderConfig());
  if (typeof simulatorConfig === 'function') Object.assign(result, simulatorConfig());
  return result;
}
function ring(n) {
  return Array.from({length:n}, (_,target) => Array.from({length:n}, (_,source) => target === (source+1)%n ? .35 : 0));
}
function hydrate(values) {
  settings = {...defaults,...values};
  if (values.sensory_index === undefined) settings.sensory_index = settings.environment === 'cartpole' ? 2 : 0;
  for (const [id,key] of Object.entries(fields)) $(id).value = settings[key];
  for (const [id,key] of Object.entries(listFields)) $(id).value = (settings[key] || []).join(',');
  matrixValues = settings.adjacency ? settings.adjacency.map(row=>row.slice()) : ring(settings.num_neurons || settings.channels);
  $('selfConnections').checked = matrixValues.some((row,i)=>row[i] !== 0);
  matrixDirty = false;
  const speed=Number(settings.speed??1);
  if(![0,1].includes(speed)&&!Array.from($('acquisitionSpeed').options).some(option=>Number(option.value)===speed))$('acquisitionSpeed').add(new Option(`${speed}×`,speed));
  $('acquisitionSpeed').value=String(speed);
  if(settings.source==='sine'&&!Array.from($('dataSource').options).some(option=>option.value==='dummy'))$('dataSource').add(new Option('Dummy Maxwell (sine)', 'dummy'));
  $('dataSource').value=settings.live_config?'live':settings.source==='sine'?'dummy':settings.source?'replay':'simulation';
  $('sourcePath').value=settings.source||'';$('liveConfigPath').value=settings.live_config||'';
  $('channels').disabled = !!(settings.source || settings.live_config);
  $('connectivityPanel').hidden = !!(settings.source || settings.live_config);
  $('sourceLabel').textContent = `Source: ${settings.live_config ? 'live Maxwell — '+settings.live_config : settings.source ? 'recorded file — '+settings.source : 'neural simulator'}. Bin width: 20 ms.`;
  if(typeof updateSourceUI==='function')updateSourceUI();
  renderMatrix(); explainSetup();
  if (typeof loadBuilder === 'function') loadBuilder(values);
  if (typeof loadSimulator === 'function') loadSimulator(settings);
}
function explainSetup() {
  const repeats = Math.max(1,Number($('repeats').value));
  $('probeHelp').textContent = `Probes: ${(repeats*1.2).toFixed(1)} seconds total. Each trial: 100 ms before a pulse, 100 ms response, 100 ms recovery. ${repeats} repeats × 2 inputs × stimulated/sham trials. Sham trials send no pulse.`;
  const env = $('environment').value;
  $('sensoryHelp').textContent = env === 'cartpole' ? 'CartPole: 0 position, 1 velocity, 2 pole angle, 3 angular velocity.' : env === 'foodland' ? 'Foodland: 0 food signal, 1 hazard signal, 2 x, 3 y, 4 direction.' : 'Ant: 0 torso height; remaining raw observations include orientation, joint positions, velocities, and contact forces. See the observation dropdown during a run.';
}
const saveRunDialog=document.createElement('dialog');saveRunDialog.id='saveRunDialog';
saveRunDialog.setAttribute('aria-labelledby','saveRunTitle');
saveRunDialog.innerHTML='<form><h2 id="saveRunTitle">Save your functions before running</h2><p>Your functions have unsaved changes, including any new custom analysis. The experiment runs the saved version.</p><label for="saveRunName">Profile name</label><input id="saveRunName" required maxlength="64" pattern="[A-Za-z0-9][A-Za-z0-9_-]{0,63}" placeholder="e.g. my_experiment"><p class="help">A profile saves your Python functions and setup. Saving an existing name keeps its previous revisions.</p><p id="saveRunError" role="alert"></p><div class="toolbar"><button type="button" id="cancelSaveRun">Cancel</button><button type="submit" id="confirmSaveRun">Save and run</button></div></form>';
document.body.append(saveRunDialog);
let saveRunSkip=false;
$('cancelSaveRun').onclick=()=>saveRunDialog.close();
saveRunDialog.addEventListener('cancel',event=>{if($('confirmSaveRun').disabled)event.preventDefault();});
saveRunDialog.querySelector('form').onsubmit=async event=>{
  event.preventDefault();$('confirmSaveRun').disabled=true;$('cancelSaveRun').disabled=true;$('saveRunError').textContent='';
  try{
    await saveProfile($('saveRunName').value.trim());
    saveRunDialog.close();await start(saveRunSkip);
  }catch(exc){$('saveRunError').textContent=exc.message;}
  finally{$('confirmSaveRun').disabled=false;$('cancelSaveRun').disabled=false;}
};
async function start(skip) {
  if (running()) return error('Stop the current run before starting another.');
  if ($('code').value !== savedCode) {
    saveRunSkip=skip;$('saveRunName').value=$('profileName').value||selectedProfile;
    $('saveRunError').textContent='';error('');saveRunDialog.showModal();$('saveRunName').focus();return;
  }
  try {
    if (typeof verifyExperiment === 'function' && !await verifyExperiment(true, skip)) return;
    const config=collect();
    if (typeof builderConfig === 'function' && skip) config.phases=config.phases.filter(p=>['environment','cartpole','foodland','ant'].includes(p.type));
    if(await control('start', {skip:typeof builderConfig === 'function'?false:skip, config, profile:selectedProfile}))showView('monitor');
  } catch (e) { error(e.message); }
}
$('run').onclick = ()=>start(false);
$('skip').onclick = ()=>start(true);
for (const kind of ['pause','step','stop']) $(kind).onclick = ()=>control(kind);
$('environment').onchange = ()=>{ $('sensory').value = $('environment').value === 'cartpole' ? 2 : 0; explainSetup(); };
$('repeats').oninput = explainSetup;
$('channels').onchange = ()=>{
  const n = Number($('channels').value);
  if (!Number.isInteger(n) || n<2 || n>1024) return error('Choose 2–1024 simulated channels.');
  if (typeof resizeSimulatorGrid === 'function') resizeSimulatorGrid(n);
  $('left').value = Array.from({length:Math.floor(n/2)},(_,i)=>i).join(',');
  $('right').value = Array.from({length:n-Math.floor(n/2)},(_,i)=>i+Math.floor(n/2)).join(',');
};

async function profileList() {
  const data = await request('/api/profiles');
  $('profileSelect').replaceChildren(new Option('Unsaved / default functions',''), ...data.names.map(name=>new Option(name,name)));
  $('profileSelect').value = selectedProfile;
  return data;
}
async function loadProfile() {
  if (running()) return error('Stop the run before loading a different setup.');
  const name = $('profileSelect').value;
  if (!name) return;
  try {
    const data = await request('/api/profile?name='+encodeURIComponent(name));
    selectedProfile = name; $('profileName').value = name;
    $('code').value = savedCode = data.code;
    $('profilePath').textContent = data.path;
    $('profileMessage').textContent = 'Loaded. Setup values will apply on the next run.';
    hydrate(data.settings); error('');
  } catch(e) { error(e.message); }
}
$('loadProfile').onclick = loadProfile;
$('template').onclick = async()=>{
  if (running()) return error('Stop the run before creating a new profile.');
  try {
    const data = await profileList();
    $('code').value = data.template; savedCode = ''; selectedProfile = '';
    $('profileSelect').value=''; $('profileName').value=''; $('profilePath').textContent='Not saved yet';
    $('profileMessage').textContent='Edit the functions and choose a name, then save.'; error('');
  } catch(e) { error(e.message); }
};
async function saveProfile(name){
    if(typeof syncAnalysisContracts==='function'){
      const errors=await syncAnalysisContracts();
      if(errors.length)throw Error(errors.join('\n'));
    }
    const data = await request('/api/profile', {name, code:$('code').value,
      settings:{...settings,...collect()}});
    selectedProfile = data.name; $('profileName').value=data.name; $('code').value = savedCode = data.code;
    $('profilePath').textContent = data.path;
    $('profileMessage').textContent='Saved. Older revisions are kept in the profiles/revisions folder.';
    await profileList(); error('');
}
$('saveProfile').onclick = async()=>{
  try {await saveProfile($('profileName').value);} catch(e) { error(e.message); }
};
$('reload').onclick = async()=>{
  if (!running()) return error('Start a run before reloading its functions.');
  if ($('code').value !== savedCode) return error('Save the profile before reloading.');
  if (await control('reload', {profile:selectedProfile})) $('profileMessage').textContent='Reload requested. The run will pause. Check for errors, then resume.';
};
$('code').oninput = ()=>{ $('profileMessage').textContent = $('code').value===savedCode ? 'Saved code.' : 'Unsaved changes.'; };

function renderMatrix() {
  const element=$('adjMatrix'), n=matrixValues.length;
  // Window the editable matrix: neuron count must not create 65,536 DOM buttons.
  let paging=$('matrixPaging');
  if(!paging){paging=document.createElement('div');paging.id='matrixPaging';paging.className='selectors';paging.innerHTML='<div><label for="matrixRows">Target neurons</label><select id="matrixRows"></select></div><div><label for="matrixCols">Source neurons</label><select id="matrixCols"></select></div>';element.before(paging);$('matrixRows').onchange=renderMatrix;$('matrixCols').onchange=renderMatrix;}
  paging.hidden=n<=32;
  for(const id of ['matrixRows','matrixCols']){const old=Number($(id).value)||0;$(id).replaceChildren(...Array.from({length:Math.ceil(n/32)},(_,i)=>new Option(`${i*32}–${Math.min(n-1,i*32+31)}`,i*32)));$(id).value=old<n?old:0;}
  const rowStart=Number($('matrixRows').value),colStart=Number($('matrixCols').value);
  element.style.gridTemplateColumns=`repeat(${Math.min(32,n-colStart)}, minmax(32px,1fr))`;
  element.replaceChildren(...matrixValues.slice(rowStart,rowStart+32).flatMap((row,ri)=>row.slice(colStart,colStart+32).map((value,cj)=>{
    const i=ri+rowStart,j=cj+colStart,cell=document.createElement('button'); cell.className='cell';
    cell.dataset.row=i; cell.dataset.col=j;
    cell.disabled = i===j && !$('selfConnections').checked;
    cell.setAttribute('aria-label',`Source ${j} to target ${i}, weight ${value.toFixed(2)}`);
    cell.title=`Source ${j} → target ${i}`; styleCell(cell,value);
    return cell;
  })));
}
function styleCell(cell,value) {
  cell.textContent=value===0?'0':value.toFixed(2);
  cell.style.background=value===0?'var(--field)':value<0?`rgba(180,92,40,${.12+Math.abs(value)*.28})`:`rgba(43,101,163,${.12+Math.abs(value)*.28})`;
}
function paint(cell, clear) {
  if (!cell || !cell.dataset.row || cell.disabled) return;
  const row=Number(cell.dataset.row), col=Number(cell.dataset.col);
  const value=clear?0:Number($('weightNumber').value);
  if (!Number.isFinite(value)||value < -2||value > 2) return error('Weight must be between -2 and 2.');
  matrixValues[row][col]=value; styleCell(cell,value);
  cell.setAttribute('aria-label',`Source ${col} to target ${row}, weight ${value.toFixed(2)}`);
  $('connectionStatus').textContent=`Source ${col} → target ${row}: ${value.toFixed(2)}`;
  painted=matrixDirty=true;
}
async function applyMatrix() {
  if (typeof scheduleValidation === 'function') scheduleValidation();
  if (running() && state.adjacency) {
    if (await control('adjacency',{value:matrixValues})) $('connectionStatus').textContent='Connections sent to the simulator.';
  } else $('connectionStatus').textContent='Connections ready for the next run. Save a profile to keep them.';
}
$('adjMatrix').onpointerdown=e=>{const cell=e.target.closest('button');if(!cell || cell.disabled)return;e.preventDefault();dragging=true;painted=false;paint(cell,e.shiftKey);};
document.addEventListener('pointermove',e=>{if(dragging){const cell=document.elementFromPoint(e.clientX,e.clientY)?.closest('#adjMatrix button');paint(cell,e.shiftKey);}});
document.addEventListener('pointerup',()=>{if(dragging){dragging=false;if(painted)applyMatrix();}});
$('adjMatrix').onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();paint(e.target,e.shiftKey);applyMatrix();}};
$('weight').oninput=()=>{$('weightNumber').value=$('weight').value;};
$('weightNumber').oninput=()=>{$('weight').value=$('weightNumber').value;};
$('selfConnections').onchange=renderMatrix;
$('ring').onclick=()=>{matrixValues=ring(matrixValues.length);matrixDirty=true;renderMatrix();applyMatrix();};
$('clearMatrix').onclick=()=>{matrixValues=matrixValues.map(row=>row.map(()=>0));matrixDirty=true;renderMatrix();applyMatrix();};

function canvas(id) {
  const element=$(id), rect=element.getBoundingClientRect(), d=devicePixelRatio||1;
  if (element.width!==Math.round(rect.width*d)||element.height!==Math.round(rect.height*d)) {element.width=rect.width*d;element.height=rect.height*d;}
  const ctx=element.getContext('2d');ctx.setTransform(d,0,0,d,0,0);ctx.clearRect(0,0,rect.width,rect.height);
  return [ctx,rect.width,rect.height];
}
function line(c, values, w, h, color) {
  if(!values.length)return;
  let lo=Math.min(...values), hi=Math.max(...values);if(hi-lo<1e-6){lo-=1;hi+=1;}
  c.strokeStyle=color;c.lineWidth=1.4;c.beginPath();
  values.forEach((v,i)=>{const x=8+i/Math.max(1,values.length-1)*(w-16), y=h-12-(v-lo)/(hi-lo)*(h-24);i?c.lineTo(x,y):c.moveTo(x,y);});c.stroke();
}
function plot(id,a,b) {
  const[c,w,h]=canvas(id);c.strokeStyle='#e4e7ea';
  for(let i=1;i<4;i++){c.beginPath();c.moveTo(0,h*i/4);c.lineTo(w,h*i/4);c.stroke();}
  const colors=getComputedStyle(document.documentElement);
  line(c,a,w,h,colors.getPropertyValue('--blue'));line(c,b,w,h,colors.getPropertyValue('--orange'));
}
function options(id,names) {const old=$(id).value||'0';$(id).replaceChildren(...names.map((name,i)=>new Option(name,i)));if(Number(old)<names.length)$(id).value=old;}
function scene(s, target='scene') {
  const[c,w,h]=canvas(target); if(!s)return;
  if(s.kind==='cartpole') {
    const o=s.observation,x=w/2+o[0]*w/6,y=h*.7;
    c.strokeStyle='#999';c.beginPath();c.moveTo(10,y+30);c.lineTo(w-10,y+30);c.stroke();
    c.fillStyle='#718ba7';c.fillRect(x-22,y,44,22);
    for(const dx of [-14,14]){c.beginPath();c.arc(x+dx,y+25,5,0,7);c.fill();}
    c.strokeStyle='#354f6b';c.lineWidth=6;c.lineCap='round';c.beginPath();c.moveTo(x,y);c.lineTo(x+Math.sin(o[2])*h*.5,y-Math.cos(o[2])*h*.5);c.stroke();
  } else if(s.kind==='foodland') {
    const sx=w/800,sy=h/600;
    for(const[points,color] of [[s.food,'#41854b'],[s.hazards,'#b35b3e']])for(const xy of points||[]){c.fillStyle=color;c.beginPath();c.arc(xy[0]*sx,xy[1]*sy,5,0,7);c.fill();}
    c.save();c.translate(s.agent[0]*sx,s.agent[1]*sy);c.rotate(s.direction);c.fillStyle='#354f6b';c.beginPath();c.moveTo(11,0);c.lineTo(-6,-6);c.lineTo(-6,6);c.fill();c.restore();
  } else if(s.kind==='ant' && s.geometry) {
    const root=s.root, scale=Math.min(w/3,h/2.5);
    const project=p=>[w/2+((p[0]-root[0])*.8-(p[1]-root[1])*.6)*scale,h*.72+((p[0]-root[0])*.3+(p[1]-root[1])*.4-p[2])*scale];
    c.strokeStyle='#e2e5e8';c.lineWidth=1;
    for(let j=-3;j<=3;j++)for(const axis of [0,1]){const a=[root[0]-2,root[1]-2,0],b=[root[0]+2,root[1]+2,0];a[axis]=b[axis]=Math.floor(root[axis]*2)/2+j*.5;const p=project(a),q=project(b);c.beginPath();c.moveTo(...p);c.lineTo(...q);c.stroke();}
    for(const g of s.geometry.toSorted((a,b)=>(a.a[1]+a.a[0])-(b.a[1]+b.a[0]))) {
      const a=project(g.a),b=project(g.b);c.strokeStyle='#526c88';c.fillStyle='#526c88';c.lineWidth=Math.max(3,g.radius*2*scale);c.lineCap='round';
      c.beginPath();c.moveTo(...a);c.lineTo(...b);c.stroke();if(Math.hypot(a[0]-b[0],a[1]-b[1])<1){c.beginPath();c.arc(...a,Math.max(3,g.radius*scale),0,7);c.fill();}
    }
    c.fillStyle='#626970';c.font='11px system-ui';c.fillText(`x ${root[0].toFixed(2)} m   y ${root[1].toFixed(2)} m`,8,h-8);
  }
}
function draw() {
  $('error').textContent=localError||state.error||'';
  $('status').textContent=(state.status||'ready')+(state.paused?' · paused':'');
  $('pause').textContent=state.paused?'Resume':'Pause';
  for(const id of ['run','skip','loadProfile','template'])$(id).disabled=running();
  for(const id of ['pause','step','stop','reload'])$(id).disabled=!running();
  for(const id of ['pause','step','reload'])if(state.native)$(id).disabled=true;
  document.querySelectorAll('.phase').forEach(e=>e.classList.toggle('active',e.dataset.phase===state.phase));
  $('trial').textContent=state.trial||'';
  const perf=state.performance;
  $('timing').textContent=perf?`Raw sampling: ${(perf.sampling_hz/1000).toFixed(1)} kHz · acquired: ${(perf.acquired_hz/1000).toFixed(1)} ksample/s · ${Number(perf.realtime_factor).toFixed(2)}× real time. Processing capacity: ${(perf.processing_capacity_hz/1000).toFixed(1)} ksample/s. Processing p95: ${Number(perf.p95_ms).toFixed(2)} ms / 20 ms budget · late bins: ${perf.deadline_misses}. Worker: ${perf.worker}.`:`20 ms bins. Processing p95: ${(state.p95_ms||0).toFixed(2)} ms.`;
  $('activeFile').textContent='Running functions: '+(state.functions_file||'not started');
  $('output').textContent='Output: '+(state.output||'not started');
  const game=state.scene?.kind;
  const phaseKind=state.phase_kind??state.phases?.find(p=>p.id===state.phase)?.type;
  const environmentActive=phaseKind==='environment'||(state.playback&&phaseKind==null&&!!state.scene);
  $('gamePanel').classList.toggle('environment-idle',!environmentActive);
  $('environmentPlaceholder').hidden=environmentActive;
  $('environmentPlaceholder').textContent=state.playback?'No environment visualization in this phase':'Waiting for an environment phase';
  for(const id of ['encodingPanel','decodingPanel'])$(id).classList.toggle('mapping-idle',!environmentActive);
  $('gameTitle').textContent=state.playback&&!state.scene?'No recorded game visualization':game==='ant'?'Ant':game==='foodland'?'Foodland':game==='cartpole'?'CartPole':'Environment';
  if(!environmentActive)$('gameTitle').textContent='Environment';
  if(!environmentActive)scene(null);
  else if(!$('view-monitor')?.hidden)scene(state.scene);
  $('episodeStatus').textContent=environmentActive?`Completed episodes: ${state.episodes||0}. Current: ${(state.episode_seconds||0).toFixed(1)} / ${state.episode_limit||Number($('episode').value)} s`:'';
  $('resetReason').textContent=environmentActive?(state.playback_notice||(state.last_episode_end?'Last reset: '+state.last_episode_end:'')):'';
  const legacyPlaybackReward=state.playback&&state.episode_reward==null;
  $('rewardLabel').textContent=legacyPlaybackReward?'Total reward':'Episode reward';
  $('reward').textContent=environmentActive&&game?Number((legacyPlaybackReward?state.reward:state.episode_reward)||0).toFixed(1):'—';
  if(state.status==='running'&&previousStatus!=='running') lastServerMatrix='';
  previousStatus=state.status;
  if(state.adjacency&&!dragging&&!matrixDirty&&JSON.stringify(state.adjacency)!==lastServerMatrix){matrixValues=state.adjacency.map(row=>row.slice());lastServerMatrix=JSON.stringify(matrixValues);renderMatrix();}
  if (typeof drawSimulator === 'function') drawSimulator();
  if($('view-monitor')?.hidden)return;
  const history=state.history||[],last=history.at(-1);if(!last)return;
  const key=JSON.stringify([state.observation_names,state.action_names,last.counts.length]);
  if(key!==namesKey){options('obsSelect',state.observation_names);options('actSelect',state.action_names);options('chSelect',last.counts.map((_,i)=>`Channel / unit ${i}`));$('obsSelect').value=Math.min(Number($('sensory').value),state.observation_names.length-1);namesKey=key;}
  const oi=Number($('obsSelect').value),si=Number($('stimSelect').value),ci=Number($('chSelect').value),ai=Number($('actSelect').value);
  $('obsValue').textContent=Number(last.observation[oi]||0).toFixed(3);
  $('stimValue').textContent=Number(last.rates[si]||0).toFixed(1)+' Hz';
  $('countValue').textContent=last.counts[ci]+' spikes';
  $('actionValue').textContent=Number(last.action[ai]||0).toFixed(3);
  plot('encodePlot',history.map(v=>v.observation[oi]||0),history.map(v=>v.rates[si]||0));
  plot('decodePlot',history.map(v=>v.counts[ci]||0),history.map(v=>v.action[ai]||0));
  $('encDetail').textContent=`Before clipping: ${(state.encoder?.before_clip_hz||[]).map(v=>Number(v).toFixed(1)).join(', ')} Hz. Pulses sent to inputs: ${(last.delivered||[]).join(', ')||'none'}.`;
  const before=state.decoder?.before_clip||0;
  $('decDetail').textContent=`Before clipping: ${Number(Array.isArray(before)?before[ai]:before).toFixed(3)}. Smoothed rate: ${Number(state.decoder?.rates_hz?.[ci]||0).toFixed(1)} Hz.`;
  const[c,w,h]=canvas('raw');(state.raw||[]).forEach((row,i)=>{const y=(i+.5)*h/state.raw.length;c.strokeStyle=i===ci?'#2b65a3':'#8c9baa';c.lineWidth=1;c.beginPath();row.forEach((v,j)=>{const x=j/(row.length-1)*w,yy=y-v*.13;j?c.lineTo(x,yy):c.moveTo(x,yy);});c.stroke();c.fillStyle='#626970';c.font='10px system-ui';c.fillText('ch '+i,3,y-4);});
  const[r,rw,rh]=canvas('raster');history.forEach((bin,i)=>bin.counts.forEach((count,ch)=>{if(count){r.fillStyle=`rgba(43,101,163,${Math.min(1,.2+count*.2)})`;r.fillRect(i/history.length*rw,ch/last.counts.length*rh,2,Math.max(2,rh/last.counts.length-3));}}));
  const matrix=$('causalMatrix');if(state.causal){matrix.style.gridTemplateColumns=`repeat(${state.causal[0].length},1fr)`;matrix.replaceChildren(...state.causal.flat().map(value=>{const cell=document.createElement('div');cell.className='cell';styleCell(cell,value);cell.textContent=value.toFixed(1);return cell;}));}
}
let displayFrames=0, displayWindow=performance.now();
async function poll() {
  try { state=await request('/api/state'); }
  catch(e) { error('Cannot reach the workshop server: '+e.message);setTimeout(poll,500);return; }
  try { if(!document.hidden){draw();displayFrames++;const now=performance.now();if(now-displayWindow>=1000){$('displayRate').textContent=`Display ${(displayFrames*1000/(now-displayWindow)).toFixed(0)} updates/s`;displayFrames=0;displayWindow=now;}} } catch(e) { error('Display error: '+e.message); }
  setTimeout(poll,document.hidden?500:33);
}
(async()=>{
  try {
    const data=await profileList();hydrate(data.settings);selectedProfile=data.selected;
    $('profileSelect').value=selectedProfile;$('profileName').value=selectedProfile;
    const loaded=selectedProfile?await request('/api/profile?name='+encodeURIComponent(selectedProfile)):null;
    $('code').value=savedCode=loaded?loaded.code:data.template;
    $('profilePath').textContent=loaded?loaded.path:data.path;
    $('profileMessage').textContent='Edit here or in the Python file. Save under a profile name to keep your changes.';
  } catch(e) {error(e.message);}
  poll();
})();
