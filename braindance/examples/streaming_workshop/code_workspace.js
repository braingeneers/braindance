// Builder-to-code previews are one-way. Export always creates a fresh directory.
(() => {
const workspace=document.createElement('details');workspace.className='panel code-workspace';
workspace.innerHTML='<summary>Code previews and export</summary><p class="help">Use forms or Python. These previews follow the builder. Export creates editable files in a new folder; future exports never overwrite your edits.</p><div class="controls"><button id="participantTab">Participant functions</button><button data-code-file="environment_setup.py">Environment setup</button><button data-code-file="experiment.py">Experiment / phase order</button><label for="exportStyle">Python style <select id="exportStyle"><option value="phases">Phase objects (.add_phase)</option><option value="config">Config (PHASES)</option></select></label><button id="exportCode" class="primary">Convert setup to code</button></div><div id="exportResult" role="status" aria-live="polite" class="help"></div><div id="generatedPanel" hidden><p class="help" id="generatedLabel"></p><pre id="generatedCode" tabindex="0" aria-label="Generated Python preview"></pre><p class="help">Preview only. Export to edit this file by hand. Exported Python needs the BrainDance repository and selected phase dependencies.</p></div><details><summary>V3 phases and execution (?)</summary><p class="help">All phases declare inputs and outputs using the V3 contract. Interactive phases use your encode/decode/train functions; scientific phases retain their own controllers, timing and files. Browser observation and transport belong to the runners. Scientific CartPole, FoodLand and Ant phases can also publish live game geometry to the monitor.</p><p class="help">Interactive and scientific runners currently use separate acquisition lifecycles, so keep them in separate sequences. Participant functions are connected to interactive phases; scientific phases use their constructor settings.</p></details>';
$('view-functions').insertBefore(workspace,$('functionsPanel'));
const functionNav=document.createElement('div');functionNav.className='controls';functionNav.id='functionNav';
for(const name of ['encode','decode','train']){
  const button=document.createElement('button');button.textContent=name;button.onclick=()=>jumpToFunction(name);functionNav.append(button);
}
const addTrain=document.createElement('button');addTrain.textContent='Add train hook';addTrain.id='addTrain';
functionNav.append(addTrain);$('functionsPanel').insertBefore(functionNav,$('functionsPanel').querySelector('.profile-fields'));
const style=document.createElement('style');style.textContent='.view>details.code-workspace>summary{display:list-item;cursor:pointer}button.primary:hover{background:var(--blue)}[data-code-file][aria-pressed=true],#participantTab[aria-pressed=true]{border-color:var(--blue);color:var(--blue)}#generatedCode{font:13px/1.6 Consolas,monospace;white-space:pre;overflow:auto;max-height:520px;padding:16px;background:var(--field);border:1px solid var(--border);border-radius:4px;color:var(--ink)}#exportResult{white-space:pre-wrap;padding:10px 0;overflow-wrap:anywhere}.execution-status{display:flex;gap:8px;align-items:center;margin-left:auto;font-size:12px;color:var(--muted);flex-wrap:wrap}.execution-status span{border:1px solid var(--border);border-radius:4px;padding:4px 7px}.execution-status span:first-child:before{content:"●";color:var(--blue);margin-right:6px}.execution-status [data-error=true]{color:var(--orange)}.app-header #themeToggle{margin-left:8px}@media(max-width:750px){.execution-status{font-size:10px;gap:3px}.app-header .help{display:none}}';document.head.append(style);
const execution=document.createElement('div');execution.className='execution-status';execution.setAttribute('role','status');execution.title='Source and timing of the server session. Pacing is a wall-clock target, not a hard real-time guarantee. Native phases control their own step sizes.';
execution.innerHTML='<span id="sourceBadge">Connecting…</span><span id="timingBadge"></span><span id="runBadge"></span>';
document.querySelector('.app-header').insertBefore(execution,$('themeToggle'));
let currentCodeFile=null, bundlePreview=null, previewKey='', previewVersion=0, previewTimer;
function showCodeFile(name){
  currentCodeFile=name;document.querySelectorAll('[data-code-file]').forEach(button=>button.setAttribute('aria-pressed',String(button.dataset.codeFile===name)));$('participantTab').setAttribute('aria-pressed',String(!name));$('functionsPanel').hidden=!!name;$('generatedPanel').hidden=!name;
  if(name){workspace.open=true;$('generatedLabel').textContent=name;renderPreview();schedulePreview();}
}
document.querySelectorAll('[data-code-file]').forEach(button=>button.onclick=()=>showCodeFile(button.dataset.codeFile));
$('participantTab').onclick=()=>showCodeFile(null);
function jumpToFunction(name){
  showCodeFile(null);const editor=$('code'),match=new RegExp('^def '+name+'\\s*\\(', 'm').exec(editor.value);
  if(!match){$('profileMessage').textContent=name==='train'?'No train hook in this profile. Add one to customize learning in streaming phases.':'No '+name+' function found.';return;}
  editor.focus();editor.setSelectionRange(match.index,match.index);editor.scrollTop=Math.max(0,(editor.value.slice(0,match.index).split('\n').length-2)*20);syncEditor();cursorPosition();
}
addTrain.onclick=()=>{
  if(/^def train\s*\(/m.test($('code').value)){jumpToFunction('train');return;}
  $('code').value += "\n\ndef train(transition, dt_s, params, state):\n    \"\"\"Optional learning hook, called after game.step and before encode.\n\n    Args:\n        transition (dict): A copy of the current step, with these keys:\n            observation: List of observation values before the game step.\n            next_observation: List after the step; see encode for layouts.\n            action: List of applied action values; see decode for ordering.\n            spike_counts: List of counts per channel/unit in this bin.\n            reward: Float reward returned by the game step.\n            done: Bool indicating that the episode ended on this step.\n        dt_s (float): Bin duration in seconds (0.02 in the workshop).\n        params (dict): Copy of the shared configuration documented in encode\n            and decode. Changes here do not update the runtime configuration.\n        state (dict): Mutable training memory. state['decode'] and\n            state['encode'] reference the dictionaries passed to those hooks;\n            update their contents to share learned values. Other keys can\n            hold training-only memory. State resets on episode reset, phase\n            restart, or function reload, so learning here is episode-local.\n\n    Returns:\n        dict: JSON-compatible diagnostics (an empty dict disables reporting).\n\n    Runs in the order decode -> game.step -> train -> encode, including the\n    terminal step before state resets. Loading/reloading functions also calls\n    the hooks with dummy inputs and fresh state to validate their outputs.\n    Native V3 phases use their own trainers and do not call this hook.\n    \"\"\"\n    return {}\n";
  $('code').dispatchEvent(new Event('input'));jumpToFunction('train');
};
const analysisPanel=document.createElement('section');analysisPanel.className='panel';analysisPanel.id='analysisPanel';analysisPanel.hidden=true;
analysisPanel.innerHTML='<h2>Custom analysis</h2><p class="help">Edit the new function template below, then save your profile. The phase runs once at its position in the sequence.</p><label>Function name<input id="analysisName" value="custom_analysis" maxlength="64"></label><button id="createAnalysis" class="primary">Create analysis</button><p id="analysisMessage" role="status" aria-live="polite"></p><details id="analysisDataReference"><summary>Available data · names, types and shapes</summary><p class="help">Read earlier outputs with exp.data.get("name"). Declare the names you use in inputs and the results you return in outputs, in the decorator below. The board follows your code automatically. n_channels means detected channels or sorted units.</p><div id="analysisInputs" class="analysis-reference"></div></details><details id="analysisParameterReference"><summary>Experiment parameters · names, values and types</summary><p class="help">All experiment parameters are available through exp.params["name"]. Change their values in Experiment setup. Constructor settings on other phases belong to those phases.</p><div id="analysisParameters" class="analysis-reference"></div></details>';
$('view-functions').insertBefore(analysisPanel,$('functionsPanel'));
style.textContent+='.analysis-reference{display:grid;gap:6px;max-height:250px;overflow:auto;margin:12px 0}.analysis-reference div{display:flex;flex-wrap:wrap;gap:10px;padding:6px;border-bottom:1px solid var(--border)}.analysis-reference code{overflow-wrap:anywhere;white-space:normal}.analysis-reference small{color:var(--muted)}#analysisPanel>label{display:block;max-width:400px;margin-bottom:12px}#analysisPanel details{margin-top:10px}#analysisMessage{white-space:pre-wrap}';
let analysisId=null,analysisPosition=0;
$('analysisName').onchange=async()=>{
  if(!analysisId)return;
  const phase=phaseSpecs.find(p=>p.id===analysisId),name=$('analysisName').value.trim();
  const report=await request('/api/analysis-contracts',{code:$('code').value});
  if(!phase||report.errors?.length||!report.contracts?.[name]){
    $('analysisMessage').textContent='Enter the name of an existing @analysis_phase function in your code.';return;
  }
  phase.params={function_name:name,...report.contracts[name]};
  $('analysisMessage').textContent='Phase now uses '+name+'. Save profile when ready.';
  renderPhases();scheduleValidation();jumpToFunction(name);
};
window.openAnalysisPhase=(id=null,position=phaseSpecs.length)=>{
  if(!catalog)return;
  analysisId=id;analysisPosition=id?phaseSpecs.findIndex(p=>p.id===id):position;
  const existing=id?phaseSpecs[analysisPosition]:null;
  showView('functions');showCodeFile(null);analysisPanel.hidden=false;
  let name=existing?.params.function_name||'custom_analysis',suffix=1;
  if(!existing)while(new RegExp('^def '+name+'\\s*\\(', 'm').test($('code').value))name='custom_analysis_'+(++suffix);
  $('analysisName').value=name;$('analysisName').disabled=false;
  $('analysisMessage').textContent=existing?'Edit inputs and outputs in the decorator. Function name selects an existing decorated function. Save profile when ready.':'';
  $('createAnalysis').hidden=!!existing;
  refreshAnalysisReference();
  if(!existing){createAnalysis();return;}
  jumpToFunction(name);
  $('code').scrollIntoView({block:'center'});
};
function refreshAnalysisReference(){
  if(analysisPanel.hidden)return;
  try{
    if(analysisId){analysisPosition=phaseSpecs.findIndex(p=>p.id===analysisId);if(analysisPosition<0){analysisPanel.hidden=true;return;}}
    const native=phaseSpecs.some(p=>!!catalog[p.type]?.native)||phaseSpecs.every(p=>p.type==='custom_analysis'),available={};
    if(native)Object.assign(available,JSON.parse($('initialData').value),JSON.parse($('nativeSettings').value));
    else{if($('baseline').value.trim()||settings.calibration)available.recording_baseline_hz=[];}
    for(const phase of phaseSpecs.slice(0,analysisPosition))for(const key of phaseDefinition(phase).outputs||[])available[key]=null;
    $('analysisInputs').replaceChildren(...Object.keys(available).map(key=>{const row=document.createElement('div'),code=document.createElement('code');code.textContent=contractLabel(key);row.append(code);return row;}));
    if(!Object.keys(available).length)$('analysisInputs').textContent='No earlier outputs at this position. You can still use experiment parameters.';
    const params={...settings,...collect(),...JSON.parse($('nativeSettings').value)};
    $('analysisParameters').replaceChildren(...Object.keys(params).filter(k=>!['phases','phase_plan','code','native_settings','initial_data','functions_file','headless'].includes(k)).sort().map(key=>{
      const row=document.createElement('div'),name=document.createElement('code'),value=document.createElement('small'),v=params[key];
      const type=Array.isArray(v)?'list'+(v.length?' · length '+v.length:''):v===null?'None':typeof v==='object'?'dict':typeof v==='number'?(Number.isInteger(v)?'int':'float'):typeof v==='boolean'?'bool':'str';
      name.textContent=key+' : '+type;const full=JSON.stringify(v);value.textContent=full?.length>150?full.slice(0,150)+'…':full;value.title=full;row.append(name,value);return row;
    }));
  }catch(exc){$('analysisMessage').textContent=exc.message;}
}
const customButton=document.createElement('button');customButton.id='addCustomAnalysis';customButton.textContent='＋ Custom analysis';customButton.onclick=()=>openAnalysisPhase();functionNav.append(customButton);
const boardCustom=customButton.cloneNode(true);boardCustom.id='flowCustomAnalysis';boardCustom.onclick=()=>openAnalysisPhase();$('flowOptions').before(boardCustom);
function createAnalysis(){
  try{
    const name=$('analysisName').value.trim();
    const reserved='False None True and as assert async await break class continue def del elif else except finally for from global if import in is lambda nonlocal not or pass raise return try while with yield encode decode train analysis_phase'.split(' ');
    if(!/^[A-Za-z_][A-Za-z0-9_]{0,63}$/.test(name)||reserved.includes(name))throw Error('Choose a Python function name (letters, numbers and underscores).');
    if(new RegExp('^def '+name+'\\s*\\(', 'm').test($('code').value))throw Error('That function already exists. Choose a new name.');
    let id=name,suffix=1;while(phaseSpecs.some(p=>p.id===id))id=name.slice(0,55)+'_'+(++suffix);
    const next=structuredClone(phaseSpecs),output=name+'_result';
    next.splice(analysisPosition,0,{id,type:'custom_analysis',params:{function_name:name,inputs:[],outputs:[output]}});
    const code=$('code').value+'\n\nfrom braindance.examples.streaming_workshop.custom_analysis import analysis_phase\n\n@analysis_phase(inputs=[], outputs=['+JSON.stringify(output)+'])\ndef '+name+'(exp):\n    """Analyze earlier outputs; return named results for later phases."""\n    # All experiment parameters are already available: exp.params["name"]\n    # Read an earlier output with exp.data.get("name").\n    # Add that name to inputs above. See Available data for types and shapes.\n    result = {}  # Replace with your analysis.\n    return {'+JSON.stringify(output)+': result}\n';
    if(!flowCommit(next))throw Error('The phase could not be added. Check the sequence.');
    $('code').value=code;$('code').dispatchEvent(new Event('input'));openAnalysisPhase(id);jumpToFunction(name);
  }catch(exc){$('analysisMessage').textContent=exc.message;}
}
$('createAnalysis').onclick=createAnalysis;

function renderPreview(){
  if(!currentCodeFile)return;
  const code=bundlePreview?.files?.[currentCodeFile]||'Generating Python preview…';
  // All content is text; participant Python is never interpreted as markup.
  paintPython($('generatedCode'),code);
}
function exportRequest(){return {config:{...settings,...collect()},code:$('code').value,style:$('exportStyle').value};}
function schedulePreview(){clearTimeout(previewTimer);previewTimer=setTimeout(refreshPreview,300);}
async function refreshPreview(){
  const version=++previewVersion;
  try{
    const payload=exportRequest(),key=JSON.stringify(payload);
    if(key===previewKey&&bundlePreview){renderPreview();return;}
    const result=await request('/api/code-preview',payload);if(version!==previewVersion)return;
    bundlePreview=result;previewKey=key;renderPreview();
    if(result.errors.length)$('exportResult').textContent='Code needs attention: '+result.errors.join('\n');
  }catch(exc){$('exportResult').textContent=exc.message;}
}
$('exportStyle').onchange=()=>{bundlePreview=null;renderPreview();refreshPreview();};
$('exportCode').onclick=async()=>{
  $('exportCode').disabled=true;
  try{
    const result=await request('/api/code-export',exportRequest());
    $('exportResult').textContent='Exported to '+result.directory+'\nRun experiment.py --verify, then experiment.py. Files are independent of the builder.';
  }catch(exc){$('exportResult').textContent='Export failed: '+exc.message;}
  finally{$('exportCode').disabled=false;}
};
const builderCode=document.createElement('button');builderCode.textContent='View / export Python';builderCode.onclick=()=>{showView('functions');showCodeFile('experiment.py');};general.append(builderCode);
let observedConfig='';
setInterval(()=>{
  const metadata=state.execution;
  if(metadata){
    $('sourceBadge').textContent=metadata.source;
    $('timingBadge').textContent=metadata.timing+' · '+(metadata.bin_ms?metadata.bin_ms+' ms bins':'native phase timing');
    $('runBadge').textContent=(state.paused?'Paused':state.status||'ready')+(!state.native&&state.p95_ms>20?' · p95 over 20 ms':'');
    $('runBadge').dataset.error=String(state.status==='error'||(!state.native&&state.p95_ms>20));
  }
  addTrain.hidden=/^def train\s*\(/m.test($('code').value);
  if(!$('view-functions').hidden){try{const key=JSON.stringify(exportRequest());if(key!==observedConfig){observedConfig=key;refreshAnalysisReference();schedulePreview();}}catch{}}
},400);
})();
