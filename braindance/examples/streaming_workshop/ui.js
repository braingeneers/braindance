// Page organization and editor UI. Experiment behavior stays in watcher.js.
const main = document.querySelector('main');
const runBar = $('run').parentElement;
runBar.classList.add('run-controls');
main.prepend(runBar);
const monitorGrid = main.querySelector('.grid');
const progress = $('phaseProgress');
// This text describes this workshop's start/skip behavior, not global navigation.
const workshopHelp = [...main.children].find(el => el.matches('p.help'));
const views = {};
for (const [id,title] of [['playground','Task playground'],['monitor','Live recording'],['experiment','Experiment setup'],['sequence','Phase builder'],['functions','Python functions'],['simulation','Neural simulator'],['catalog','Catalog'],['analysis','Analysis'],['sorting','Spike sorting'],['settings','Settings']]) {
  const section = document.createElement('section');
  section.id = 'view-'+id; section.className = 'view';
  const heading = document.createElement('h1'); heading.className='view-title'; heading.textContent=title;
  section.append(heading); main.append(section); views[id]=section;
}
views.monitor.append(progress,monitorGrid);
main.insertBefore(runBar,main.firstChild);

const restart=document.createElement('button');restart.id='restartPhase';restart.textContent='Restart phase';restart.title='Start a new attempt at the next 20 ms boundary. Previous frames, delivered stimuli and attempt metadata remain saved; acquisition time does not rewind.';runBar.insertBefore(restart,$('stop'));
restart.onclick=()=>control('restart_phase');
const nativeLive=document.createElement('section');nativeLive.className='panel';nativeLive.id='nativeLive';nativeLive.hidden=true;
nativeLive.innerHTML='<h2>Native V3 execution</h2><p class="help">Native phases use their own acquisition, plots and controllers. Output files and console log are retained. Stop ends the process; launch again for a new run. Cooperative pause/restart is available only for streaming phases.</p><pre id="nativeLog" style="white-space:pre-wrap;max-height:420px;overflow:auto"></pre>';
views.monitor.append(nativeLive);
views.experiment.append($('setup'));
if(workshopHelp) views.experiment.append(workshopHelp);
views.functions.append($('functionsPanel'));
const simulationSettings=document.createElement('div');
simulationSettings.className='panel simulation-settings fields';
for(const id of ['channels','seed']) simulationSettings.append($(id).parentElement);
const simulationHelp=document.createElement('p'); simulationHelp.className='help';
simulationHelp.textContent='Build a neural culture, then record its activity. Setup changes apply on the next run; connection weights can change while it runs.';
views.simulation.append(simulationHelp,simulationSettings,$('connectivityPanel'));
for(const id of ['setup','functionsPanel','connectivityPanel']) $(id).open=true;
// Remove numbering tied to the old single-page layout.
document.querySelectorAll('legend').forEach(el=>el.textContent=el.textContent.replace(/^\d+\. /,''));
// Simulator is a source-specific tab within the experiment workspace.
for(const view of [views.experiment,views.sequence,views.simulation]){
  const tabs=document.createElement('div');tabs.className='toolbar experiment-tabs';
  const setupTab=document.createElement('button');setupTab.textContent='Parameters & phases';setupTab.onclick=()=>showView('experiment');
  const simulatorTab=document.createElement('button');simulatorTab.textContent='Simulator';simulatorTab.className='simulator-tab';simulatorTab.onclick=()=>showView('simulation');
  setupTab.dataset.experimentTab='experiment';simulatorTab.dataset.experimentTab='simulation';
  const sequenceTab=document.createElement('button');sequenceTab.textContent='Phase builder';sequenceTab.dataset.experimentTab='sequence';sequenceTab.onclick=()=>showView('sequence');
  tabs.append(setupTab,sequenceTab,simulatorTab);view.querySelector('h1').after(tabs);
}
// Saved experiment playback shares the live monitor and transport controls.
$('sourcePathField').insertAdjacentHTML('afterbegin', `
  <div class="toolbar"><button id="replayCatalog" type="button">Choose from catalog</button></div>
  <label for="playbackPath">Saved experiment, game log, or recording path</label>
  <input id="playbackPath" placeholder="/path/to/experiment">
  <div class="toolbar"><button id="playbackLoad" type="button">Load playback</button><button id="playbackPlay" type="button" disabled>Play saved recording / experiment</button></div>
  <p id="playbackStatus" role="status" aria-live="polite"></p>
  <p class="help">Playback shows saved activity and supported game states in Live recording. Speed, Pause, Step and Stop control playback. The recording input below can also supply neural data to a new experiment.</p>`);
let playbackPath='';
window.workshopPlayback={load:async path=>{
  if(running()){error('Stop the current run before loading playback.');return false;}
  $('dataSource').value='replay';updateSourceUI();
  $('playbackPath').value=path;playbackPath='';$('playbackPlay').disabled=true;
  $('playbackStatus').textContent='Inspecting saved experiment…';
  try{
    const info=await request('/api/playback',{path});
    playbackPath=path;$('playbackPlay').disabled=false;
    $('playbackStatus').textContent=info.message||info.description||'Ready to play saved data.';
    error('');return true;
  }catch(e){$('playbackStatus').textContent=e.message;error(e.message);return false;}
}};
$('replayCatalog').onclick=()=>showView('catalog');
$('playbackPath').oninput=()=>{playbackPath='';$('playbackPlay').disabled=true;};
$('playbackLoad').onclick=()=>window.workshopPlayback.load($('playbackPath').value.trim());
$('playbackPlay').onclick=async()=>{
  if(!playbackPath||running())return;
  if(await control('playback',{path:playbackPath,speed:Number($('acquisitionSpeed').value)}))showView('monitor');
};
function updateSourceUI(){
  const simulated=$('dataSource').value==='simulation';
  $('pacingField').hidden=$('dataSource').value==='live';
  $('sourcePathField').hidden=$('dataSource').value!=='replay';
  $('liveConfigField').hidden=$('dataSource').value!=='live';
  document.querySelectorAll('.simulator-tab').forEach(button=>button.hidden=!simulated);
  $('channels').disabled=!simulated;
  $('connectivityPanel').hidden=!simulated;
  $('sourceLabel').textContent=simulated?'Simulated neural activity · raw samples at 20 kHz · control bins of 20 ms. Open Simulator to edit neurons and connections.':$('dataSource').value==='dummy'?'Dummy Maxwell · deterministic sine data on 1024 channels · hardware-free acquisition and stimulation.':$('dataSource').value==='replay'?'Replay the selected recording. No Maxwell configuration is needed.':'Choose a configuration to acquire from the Maxwell system.';
  if(!simulated && !views.simulation.hidden)showView('experiment');
}
$('dataSource').onchange=()=>{
  settings.source=$('dataSource').value==='dummy'?'sine':$('dataSource').value==='replay'?$('sourcePath').value:null;
  settings.live_config=$('dataSource').value==='live'?$('liveConfigPath').value:null;
  updateSourceUI();
  if(typeof scheduleValidation==='function')scheduleValidation();
};
for(const id of ['sourcePath','liveConfigPath','acquisitionSpeed'])$(id).oninput=()=>{if(typeof scheduleValidation==='function')scheduleValidation();};
async function uploadSourceFile(file, kind){
  if(!file)return;
  if(running())return error('Stop the current run before replacing its input file.');
  if(sourceUploadPending)return error('Wait for the current upload to finish.');
  const replay=kind==='replay', status=$(replay?'replayFileStatus':'liveConfigFileStatus');
  sourceUploadPending=true;
  status.textContent=`Uploading ${file.name}…`;
  updateNavigation();
  try{
    const response=await fetch('/api/upload/'+kind,{method:'POST',headers:{'Content-Type':'application/octet-stream','X-Workshop-Token':WORKSHOP_TOKEN,'X-File-Name':encodeURIComponent(file.name)},body:file});
    const result=await response.json();
    if(!response.ok)throw Error(result.error||'File upload failed.');
    $(replay?'sourcePath':'liveConfigPath').value=result.path;
    status.textContent=`Selected: ${file.name}`;
    error('');
    if(replay)await window.workshopPlayback.load(result.path);
  }catch(exc){status.textContent=`Upload failed: ${exc.message} Previous selection retained.`;error(exc.message);}
  finally{
    sourceUploadPending=false;
    $(replay?'replayFile':'liveConfigFile').value='';
    updateNavigation();
    if(typeof scheduleValidation==='function')scheduleValidation();
  }
}
$('replayFile').onchange=event=>uploadSourceFile(event.target.files[0],'replay');
$('liveConfigFile').onchange=event=>uploadSourceFile(event.target.files[0],'config');
for(const name of ['dragenter','dragover'])$('replayDrop').addEventListener(name,event=>{event.preventDefault();if(!running()&&!sourceUploadPending)$('replayDrop').classList.add('drag-over');});
for(const name of ['dragleave','drop'])$('replayDrop').addEventListener(name,event=>{event.preventDefault();$('replayDrop').classList.remove('drag-over');});
$('replayDrop').addEventListener('drop',event=>{
  if(event.dataTransfer.files.length!==1)return error('Drop one replay recording at a time.');
  uploadSourceFile(event.dataTransfer.files[0],'replay');
});
function showView(id) {
  if(!views[id])id='experiment';
  if(id==='simulation' && $('dataSource').value!=='simulation')id='experiment';
  for(const [key,section] of Object.entries(views)) section.hidden=key!==id;
  document.querySelectorAll('[data-view]').forEach(button=>{
    if(button.dataset.view===id || (['simulation','sequence'].includes(id) && button.dataset.view==='experiment'))button.setAttribute('aria-current','page');else button.removeAttribute('aria-current');
  });
  document.querySelectorAll('[data-experiment-tab]').forEach(button=>{if(button.dataset.experimentTab===id)button.setAttribute('aria-current','page');else button.removeAttribute('aria-current');});
  runBar.hidden=!['experiment','monitor'].includes(id);
  document.body.dataset.view=id;
  history.replaceState(null,'','#'+id);
  window.scrollTo(0,0);
  if(innerWidth<750) {document.body.classList.add('nav-closed');$('navToggle').setAttribute('aria-expanded','false');}
  if(id==='functions')highlightPython();
}
document.querySelectorAll('[data-view]').forEach(button=>button.onclick=()=>showView(button.dataset.view));
$('navToggle').onclick=()=>{const closed=document.body.classList.toggle('nav-closed');$('navToggle').setAttribute('aria-expanded',String(!closed));};
function setTheme(theme) {
  document.documentElement.dataset.theme=theme;
  $('themeToggle').textContent=theme==='dark'?'Light mode':'Dark mode';
  $('themeToggle').setAttribute('aria-label',`Switch to ${theme==='dark'?'light':'dark'} mode`);
  try{localStorage.setItem('braindance-theme',theme);}catch{}
}
let initialTheme;try{initialTheme=localStorage.getItem('braindance-theme');}catch{}
setTheme(initialTheme|| (matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light'));
$('themeToggle').onclick=()=>setTheme(document.documentElement.dataset.theme==='dark'?'light':'dark');

// Highlight text as DOM text nodes: participant code is never interpreted as HTML.
let highlighted=null, verifiedCode=null;
function highlightPython() {
  const code=$('code').value;
  if(code===highlighted)return;
  highlighted=code;
  if(verifiedCode!==null && code!==verifiedCode){$('verifyResult').textContent='Code changed. Verify again to check this version.';$('verifyResult').removeAttribute('data-ok');}
  paintPython($('highlight'),code); syncEditor();
}
function paintPython(element,code) {
  const tokens=/(#[^\n]*|"""[\s\S]*?(?:"""|$)|'''[\s\S]*?(?:'''|$)|"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*'|\b(?:False|None|True|and|as|assert|async|await|break|class|continue|def|del|elif|else|except|finally|for|from|global|if|import|in|is|lambda|nonlocal|not|or|pass|raise|return|try|while|with|yield)\b|\b\d+(?:\.\d*)?(?:[eE][+-]?\d+)?\b)/g;
  const fragment=document.createDocumentFragment();let end=0;
  for(const match of code.matchAll(tokens)){
    fragment.append(document.createTextNode(code.slice(end,match.index)));
    const span=document.createElement('span'),value=match[0];
    span.className='tok-'+(value.startsWith('#')?'comment':/^['"]/.test(value)?'string':/^\d/.test(value)?'number':'keyword');
    span.textContent=value;fragment.append(span);end=match.index+value.length;
  }
  fragment.append(document.createTextNode(code.slice(end)+'\n'));
  element.replaceChildren(fragment);
}
function syncEditor(){ $('highlight').scrollTop=$('code').scrollTop;$('highlight').scrollLeft=$('code').scrollLeft; }
function cursorPosition(){const before=$('code').value.slice(0,$('code').selectionStart);$('cursorPosition').textContent=`Line ${before.split('\n').length}, column ${before.length-before.lastIndexOf('\n')}`;}
$('code').addEventListener('input',()=>{highlightPython();cursorPosition();});
$('code').addEventListener('scroll',syncEditor);
$('code').addEventListener('click',cursorPosition);
$('code').addEventListener('keyup',cursorPosition);
$('code').addEventListener('keydown',event=>{
  if(event.key==='Escape'){$('verify').focus();return;}
  if(event.key==='Tab'){
    event.preventDefault();const editor=$('code');editor.setRangeText('    ',editor.selectionStart,editor.selectionEnd,'end');editor.dispatchEvent(new Event('input'));
  }
  if(event.key==='Enter'){
    event.preventDefault();const editor=$('code'),line=editor.value.slice(0,editor.selectionStart).split('\n').at(-1);
    const indent=line.match(/^\s*/)[0]+(line.trimEnd().endsWith(':')?'    ':'');
    editor.setRangeText('\n'+indent,editor.selectionStart,editor.selectionEnd,'end');editor.dispatchEvent(new Event('input'));
  }
});
$('verify').onclick=async()=>{
  const code=$('code').value;$('verify').disabled=true;
  try {
    const result=await request('/api/verify',{code});
    verifiedCode=code;
    $('verifyResult').dataset.ok=String(result.ok);
    $('verifyResult').textContent=(result.line?`Line ${result.line}${result.column?', column '+result.column:''}: `:'')+result.message;
    if($('code').value!==code){$('verifyResult').textContent='Code changed while checking. Verify again.';return;}
    if(!result.ok&&result.line){const editor=$('code'),offset=code.split('\n').slice(0,result.line-1).reduce((n,line)=>n+line.length+1,0);editor.focus();editor.setSelectionRange(offset,offset+code.split('\n')[result.line-1].length);editor.scrollTop=Math.max(0,(result.line-3)*20);syncEditor();cursorPosition();}
  }catch(exc){$('verifyResult').textContent=exc.message;$('verifyResult').dataset.ok='false';}
  finally{$('verify').disabled=false;}
};
let phaseKey='';
function updateNavigation(){
  highlightPython();
  $('dataSource').disabled=running()||sourceUploadPending;$('acquisitionSpeed').disabled=running();
  for(const id of ['sourcePath','liveConfigPath','replayFile','liveConfigFile','playbackPath','playbackLoad','replayCatalog'])$(id).disabled=running()||sourceUploadPending;
  $('restartPhase').disabled=!running()||!state.restart_supported;
  $('playbackPlay').disabled=!playbackPath||running()||sourceUploadPending;
  if(state.playback)$('reload').disabled=true;
  for(const id of ['pause','step','reload'])if(state.native)$(id).disabled=true;
  nativeLive.hidden=!state.native;monitorGrid.hidden=!!state.native;
  $('nativeLog').textContent=state.log||state.output||'Waiting for native phase output…';
  const phases=state.phases||[],key=JSON.stringify([phases,state.phase,state.status]);
  if(key===phaseKey)return;phaseKey=key;
  const current=phases.indexOf(state.phase);
  progress.replaceChildren(...phases.map((phase,index)=>{
    const item=document.createElement('li');item.textContent=phase;
    const status=document.createElement('small');
    status.textContent=state.status==='completed'||index<current?'completed':index===current?state.status==='error'?'failed':state.status==='stopped'?'stopped':'current':'pending';
    item.append(status);if(index===current)item.setAttribute('aria-current','step');return item;
  }));
  if(!phases.length){const hint=document.createElement('li');hint.textContent='No active run. Phases appear when the experiment starts.';progress.append(hint);}
}
setInterval(updateNavigation,250);
showView(location.hash.slice(1)||'experiment');
// Close the drawer when resizing into the narrow layout so it does not cover the workspace.
matchMedia('(max-width: 750px)').addEventListener('change', event=>{
  if(event.matches){document.body.classList.add('nav-closed');$('navToggle').setAttribute('aria-expanded','false');}
});
