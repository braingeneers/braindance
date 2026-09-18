// The phase catalog comes from the same definitions used by the Python runner.
let catalog=null, phaseSpecs=[], builderLoaded=false, validationTimer, validationVersion=0;
const experimentView=$('view-experiment');
const general=document.createElement('section');general.className='panel';
general.innerHTML='<h2>Experiment identity</h2><div class="fields"><div><label for="experimentName">Name</label><input id="experimentName" value="streaming_workshop" maxlength="64"></div><div><label for="experimentNotes">Notes</label><input id="experimentNotes" placeholder="Purpose or session notes"></div></div><p class="help">Source and acquisition are shared. Game, encoding electrodes and decoding channels belong to each phase.</p>';
experimentView.insertBefore(general,$('setup'));
const builder=document.createElement('section');builder.className='panel';
builder.innerHTML='<h2>Phases</h2><p class="help">Phases run from top to bottom. Inputs must come from explicit settings or an earlier phase. Reorder or repeat phases as needed.</p><div class="controls"><label for="phaseType">Add phase</label><select id="phaseType" style="width:auto"></select><button id="addPhase">Add</button><button id="zeroBaseline">Use zero baseline</button><button id="verifyExperiment">Verify experiment</button></div><p class="help">Verify experiment executes your current Python functions on sample inputs and steps the local game. It does not start acquisition or deliver stimulation.</p><div id="phaseBuilder"></div><div id="experimentValidation" role="status" aria-live="polite"></div>';
experimentView.append(builder);
const sequenceReset=document.createElement('button');sequenceReset.textContent='New sequence';let previousSequence=null;
const sequenceRestore=document.createElement('button');sequenceRestore.textContent='Restore previous sequence';sequenceRestore.disabled=true;
sequenceReset.onclick=()=>{previousSequence=structuredClone(phaseSpecs);phaseSpecs=[];sequenceRestore.disabled=false;renderPhases();scheduleValidation();};
sequenceRestore.onclick=()=>{if(previousSequence){phaseSpecs=previousSequence;previousSequence=null;sequenceRestore.disabled=true;renderPhases();scheduleValidation();}};
builder.querySelector('.controls').append(sequenceReset,sequenceRestore);
const verificationHelp=document.createElement('details');verificationHelp.innerHTML='<summary>What does verification check? (?)</summary><p class="help">Binned V3: dependencies, local source/game setup and sample mapping outputs. Native V3: constructor parameters, imports and declared inputs; scientific execution still happens at launch. Both use the V3 phase contract. Keep separate sequences because binned mappings and native controllers use different acquisition lifecycles.</p>';
builder.insertBefore(verificationHelp,$('phaseBuilder'));
const nativeInputs=document.createElement('details');nativeInputs.className='panel native-inputs';
nativeInputs.innerHTML='<summary>Experiment parameters and initial data</summary><p class="help">Add experiment parameters here; analysis functions can read them through exp.params. Earlier phases can supply initial data for you.</p><h3>Experiment parameters</h3><div id="experimentParameterFields"></div><h3>Initial data</h3><div id="initialDataFields"></div>';
const nativeSettingsArea=document.createElement('textarea');nativeSettingsArea.id='nativeSettings';
const initialDataArea=document.createElement('textarea');initialDataArea.id='initialData';
const experimentParameterEditor=parameterEditor(nativeInputs.querySelector('#experimentParameterFields'),{}, {},()=>scheduleValidation(),nativeSettingsArea);
const initialDataEditor=parameterEditor(nativeInputs.querySelector('#initialDataFields'),{}, {},()=>scheduleValidation(),initialDataArea);
experimentView.insertBefore(nativeInputs,builder);
for(const id of ['environment','left','right','stim'])$(id).parentElement.hidden=true;
$('environment').closest('fieldset').querySelector('legend').textContent='Acquisition source';
$('left').closest('fieldset').querySelector('legend').textContent='Initial data';
const poolHelp=$('left').closest('fieldset').querySelector('.help');if(poolHelp)poolHelp.textContent='A recording phase can supply baseline rates. Channels and stimulation mappings are configured within each phase.';
for(const id of ['record','sensory'])$(id).closest('fieldset').hidden=true;
const baselineLabel=document.querySelector('label[for=baseline]');baselineLabel.textContent='Explicit baseline Hz (one value per channel)';
$('baseline').placeholder='Leave blank when an earlier recording phase outputs it';
$('run').textContent='Run experiment';
const oldHelp=[...experimentView.querySelectorAll('p.help')].find(el=>el.textContent.startsWith('Run all:'));
if(oldHelp)oldHelp.remove();
const style=document.createElement('style');
style.textContent='.phase-card{border:1px solid var(--border);border-radius:4px;padding:14px;margin-top:14px}.phase-card h3{font-size:14px;margin:0}.phase-card .fields{margin-top:12px}.phase-contract{font:12px/1.6 Consolas,monospace;overflow-wrap:anywhere;margin-top:10px}.phase-contract .missing{color:var(--orange)}#experimentValidation{margin-top:14px;white-space:pre-wrap;line-height:1.6}#experimentValidation[data-ok=false]{color:var(--orange)}.phase-actions{display:flex;gap:5px}.phase-actions button{padding:4px 8px}';
style.textContent+=':root{--success:#246947;--success-bg:#eaf5ee;--warning-bg:#fff3e4}:root[data-theme=dark]{--success:#9cdbb5;--success-bg:#243d31;--warning-bg:#423525}.phase-card{border-left:3px solid var(--blue);padding:18px}.phase-card h3{color:var(--blue)}#experimentValidation{padding:12px;border-radius:4px;background:var(--hover)}#experimentValidation[data-ok=true]{color:var(--success);background:var(--success-bg)}#experimentValidation[data-ok=false]{background:var(--warning-bg)}.phase-description{margin:10px 0;color:var(--muted);font-size:12px}.native-code{width:100%;min-height:160px;font:12px/1.5 Consolas,monospace;margin:8px 0}.phase-contract{padding:10px;background:var(--page);border-radius:3px}.contract-toggle{margin-top:12px}';
document.head.append(style);
style.textContent+='.view>details.native-inputs>summary{display:list-item}.fields small{display:block;color:var(--muted);font-size:11px}.native-inputs details,.phase-card details{margin:10px 0}';
const launch=document.createElement('button');launch.id='launchExperiment';launch.className='primary';launch.textContent='Run experiment';
let launchPending=false;
launch.onclick=async()=>{launchPending=true;launch.disabled=true;try{await start(false);}finally{launchPending=false;launch.disabled=running();}};
const launchBar=document.createElement('div');launchBar.className='toolbar launch-area';launchBar.append(launch);builder.append(launchBar);
setInterval(()=>{launch.disabled=launchPending||running()||!builderLoaded;},250);
function builderConfig(){return {experiment_name:$('experimentName').value,experiment_notes:$('experimentNotes').value,native_settings:JSON.parse($('nativeSettings').value),initial_data:JSON.parse($('initialData').value),phase_plan:structuredClone(phaseSpecs),phases:expandPhasePlan(phaseSpecs).map(p=>({id:p.id,type:p.type,params:{...p.params},...(p.settings?{settings:p.settings}:{})}))};}
function loadBuilder(values){
  if(!catalog)return;
  $('experimentName').value=values.experiment_name||'streaming_workshop';$('experimentNotes').value=values.experiment_notes||'';
  experimentParameterEditor.load(values.native_settings||{});initialDataEditor.load(values.initial_data||{});analysisCodeCache=null;analysisContractsCache={};analysisContractErrors=[];
  phaseSpecs=(values.phase_plan||values.phases)?(values.phase_plan||values.phases).map(p=>({...p,params:{...p.params}})):['recording','causal','environment'].map(type=>({id:type,type,params:Object.fromEntries(Object.entries(catalog[type].params).map(([key,value])=>[key,values[key]??(key==='sensory_index'&&values.environment!=='cartpole'?0:value)]))}));
  phaseSpecs=phaseSpecs.map(phase=>{
    if(phase.type!=='environment')return phase;
    const game=phase.params.environment??values.environment??'cartpole';
    if(!['cartpole','foodland','ant'].includes(game))return phase;
    const params={...phase.params};delete params.environment;
    return {...phase,type:game,params};
  });
  builderLoaded=true;renderPhases();scheduleValidation();
}
// Keep one object behind the form and its optional JSON view.
function parameterEditor(parent, values={}, metadata={}, onChange=()=>{}, backing=null){
  let current=structuredClone(values);const section=document.createElement('div'),fields=document.createElement('div');fields.className='fields';section.append(fields);
  const advanced=document.createElement('details'),summary=document.createElement('summary');summary.textContent='Advanced · JSON';advanced.append(summary);
  const area=backing||document.createElement('textarea');area.classList.add('native-code');area.setAttribute('aria-label',area.id||'Advanced parameters JSON');advanced.append(area);section.append(advanced);
  const add=document.createElement('div');add.className='controls';
  const keyInput=document.createElement('input');keyInput.placeholder='Parameter name';keyInput.setAttribute('aria-label','New parameter name');
  const typeInput=document.createElement('select');typeInput.setAttribute('aria-label','New parameter type');for(const [value,label] of [['str','Text'],['float','Number'],['bool','Yes / no'],['list','Numbers / channels']])typeInput.add(new Option(label,value));
  const addButton=document.createElement('button');addButton.type='button';addButton.textContent='Add parameter';add.append(keyInput,typeInput,addButton);section.insertBefore(add,advanced);parent.append(section);
  function publish(){area.value=JSON.stringify(current,null,2);area.setCustomValidity('');onChange(structuredClone(current));}
  function draw(){
    fields.replaceChildren();
    for(const key of new Set([...Object.keys(metadata),...Object.keys(current)])){
      const meta=metadata[key]||{},present=Object.hasOwn(current,key),v=present?current[key]:meta.default;
      if(meta.supported===false)continue;
      const annotation=meta.annotation||'',complex=v!==null&&typeof v==='object'&&(!Array.isArray(v)||v.some(x=>x!==null&&typeof x==='object'));
      const row=document.createElement('div'),label=document.createElement('label');label.textContent=key.replaceAll('_',' ')+(meta.required?' *':'');row.append(label);
      if(complex){const help=document.createElement('small');help.textContent='Structured value · edit in Advanced';row.append(help);fields.append(row);continue;}
      const list=Array.isArray(v)||(v==null&&/\b(List|list|Sequence|Tuple|tuple)\b/.test(annotation)),bool=typeof v==='boolean'||(v==null&&/\bbool\b/.test(annotation)),number=typeof v==='number'||(!list&&/\b(float|int)\b/.test(annotation));
      const input=document.createElement(bool?'select':'input');input.setAttribute('aria-label',key);label.append(input);
      if(bool){input.add(new Option('Use default',''));input.add(new Option('Yes','true'));input.add(new Option('No','false'));input.value=v==null?'':String(v);}
      else{input.type=number?'number':'text';if(number)input.step='any';input.value=v==null?'':list?v.join(', '):String(v);input.placeholder=list?'e.g. 0, 1, 2':meta.required?'Required':meta.default_python||'Use default';}
      const help=document.createElement('small');help.textContent=list?'Comma-separated values'+((v||[]).every?.(x=>typeof x==='number')?' (numbers / channel IDs)':''):meta.note||annotation;row.append(help);
      input.oninput=()=>{
        try{
          if(input.value===''){if(present&&v===null)current[key]=null;else delete current[key];}
          else if(bool)current[key]=input.value==='true';
          else if(list){const items=input.value.split(',').map(x=>x.trim()).filter(Boolean),strings=(Array.isArray(v)&&v.some(x=>typeof x==='string'))||/\bstr\b/.test(annotation);current[key]=strings?items:items.map(x=>{const n=Number(x);if(!Number.isFinite(n))throw Error('Use comma-separated numbers.');return n;});}
          else if(number){const n=Number(input.value);if(!Number.isFinite(n))throw Error('Enter a finite number.');current[key]=n;}
          else current[key]=input.value;
          input.setCustomValidity('');publish();
        }catch(exc){input.setCustomValidity(exc.message);}
      };
      fields.append(row);
    }
  }
  keyInput.oninput=()=>keyInput.setCustomValidity('');
  addButton.onclick=()=>{const key=keyInput.value.trim();if(!key)return;keyInput.setCustomValidity('');if(Object.hasOwn(current,key)){keyInput.setCustomValidity('That parameter already exists.');keyInput.reportValidity();return;}current[key]=typeInput.value==='bool'?false:typeInput.value==='float'?0:typeInput.value==='list'?[]:'';keyInput.value='';publish();draw();};
  area.oninput=()=>{try{const value=JSON.parse(area.value);if(!value||Array.isArray(value)||typeof value!=='object')throw Error('Use a JSON object.');current=value;area.setCustomValidity('');draw();onChange(structuredClone(current));}catch(exc){area.setCustomValidity(exc.message);}};
  area.value=JSON.stringify(current,null,2);draw();
  return {read(){if(section.querySelector(':invalid'))throw Error('Check the highlighted parameter fields.');return structuredClone(current);},load(value){current=structuredClone(value);area.value=JSON.stringify(current,null,2);area.setCustomValidity('');draw();}};
}
let analysisCodeCache=null,analysisContractErrors=[],analysisContractsCache={};
async function syncAnalysisContracts(){
  const code=$('code').value;
  if(!phaseSpecs.some(p=>p.type==='custom_analysis'))return [];
  const previousErrors=analysisContractErrors.length||phaseSpecs.some(p=>p.type==='custom_analysis'&&!analysisContractsCache[p.params.function_name]);
  let changed=false;
  if(code!==analysisCodeCache){
    const result=await request('/api/analysis-contracts',{code});
    if(code!==$('code').value)return syncAnalysisContracts();
    if(!(result.errors||[]).length){
      const contracts=result.contracts||{},selected=new Set(phaseSpecs.filter(p=>p.type==='custom_analysis').map(p=>p.params.function_name));
      const removed=[...selected].filter(name=>analysisContractsCache[name]&&!contracts[name]);
      const added=Object.keys(contracts).filter(name=>!analysisContractsCache[name]&&!selected.has(name));
      // Follow a single unambiguous code rename, retaining phase IDs and loops.
      if(removed.length===1&&added.length===1){
        for(const phase of phaseSpecs)if(phase.type==='custom_analysis'&&phase.params.function_name===removed[0])phase.params.function_name=added[0];
        if($('analysisName')?.value===removed[0])$('analysisName').value=added[0];
        changed=true;
      }
    }
    analysisCodeCache=code;analysisContractErrors=result.errors||[];
    if(!analysisContractErrors.length)analysisContractsCache=result.contracts||{};
  }
  if(!analysisContractErrors.length){
    for(const phase of phaseSpecs)if(phase.type==='custom_analysis'){
      const contract=analysisContractsCache[phase.params.function_name]||{inputs:[],outputs:[]};
      const params={function_name:phase.params.function_name,...contract};
      if(JSON.stringify(params)!==JSON.stringify(phase.params)){phase.params=params;changed=true;}
    }
    if(changed&&typeof renderPhaseFlow==='function')renderPhaseFlow();
  }
  const errors=[...analysisContractErrors];
  if(!errors.length)for(const phase of phaseSpecs)if(phase.type==='custom_analysis'&&!analysisContractsCache[phase.params.function_name])errors.push(phase.id+': missing @analysis_phase function '+phase.params.function_name+'. Restore its name or select the intended function in Functions.');
  if($('analysisMessage')){
    if(errors.length)$('analysisMessage').textContent=errors.join('\n');
    else if(previousErrors)$('analysisMessage').textContent='The board follows your decorator. Save profile when ready.';
  }
  return errors;
}

function phaseDefinition(phase, definitions=catalog){
  const definition=definitions[phase.type]||{};
  return phase.type==='custom_analysis'?{...definition,label:phase.params?.function_name||definition.label,inputs:phase.params?.inputs||phase.params?.requires||[],outputs:phase.params?.outputs||phase.params?.provides||[]}:definition;
}
function contractLabel(key, definition={}){
  const info=definition.data_contracts?.[key]||Object.values(catalog||{}).map(d=>d.data_contracts?.[key]).find(Boolean);
  return key+' : '+(typeof info==='string'?info:info?.type||'type not declared')+(info?.shape?' · '+info.shape:'');
}
function sourceLink(url){
  const link=document.createElement('a');
  link.className='source-link';link.textContent='View source ↗';link.href=url;
  link.target='_blank';link.rel='noopener noreferrer';link.title='View the Python definition on GitHub (committed version)';
  return link;
}
function renderPhases(){
  const nodes=phaseSpecs.map((phase,index)=>{
    const card=document.createElement('div');card.className='phase-card';
    const header=document.createElement('div');header.className='row';
    const title=document.createElement('h3');title.textContent=catalog[phase.type]?.label||phase.type;header.append(title);
    const actions=document.createElement('div');actions.className='phase-actions';
    for(const [label,change,disabled] of [['Move up',-1,index===0],['Move down',1,index===phaseSpecs.length-1],['Remove',0,false]]){
      const button=document.createElement('button');button.textContent=label;button.disabled=disabled;
      button.onclick=()=>{const next=structuredClone(phaseSpecs);if(change){if(next[index].loop||next[index+change].loop){showView('sequence');return;}[next[index],next[index+change]]=[next[index+change],next[index]];}else next.splice(index,1);flowCommit(next);};actions.append(button);
    }
    header.append(actions);card.append(header);
    const fields=document.createElement('div');fields.className='fields';
    const identity=document.createElement('div'),label=document.createElement('label'),input=document.createElement('input');
    label.textContent='Phase ID';input.value=phase.id;input.setAttribute('aria-label',`Phase ${index+1} ID`);
    input.oninput=()=>{phase.id=input.value;scheduleValidation();};identity.append(label,input);fields.append(identity);
    const definition=phaseDefinition(phase);card.dataset.kind=definition.category||'experiment';
    if(definition.source_url)actions.prepend(sourceLink(definition.source_url));
    const help=document.createElement('p');help.className='phase-description';
    help.textContent=definition.native?'V3 phase with its own scientific controller and constructor settings.':`${definition.description||''} Inputs: ${definition.required_stim_electrodes||0} stimulation electrodes.`;
    card.append(help);
    if(definition.mapping_contract)help.textContent+=' Mapping: '+JSON.stringify(definition.mapping_contract);
    if(definition.caveats?.length)help.textContent+=' Review: '+definition.caveats.join(' ');
    if(phase.type==='custom_analysis'){
      const edit=document.createElement('button');edit.textContent='Edit analysis in Functions';edit.onclick=()=>openAnalysisPhase(phase.id);card.append(edit);
    }else if(definition.native){
      for(const [key,caption] of [['params','Phase parameters'],['settings','Phase inputs / stimulation mapping']]){
        const heading=document.createElement('h4');heading.textContent=caption;card.append(heading);
        parameterEditor(card,phase[key]||{},key==='params'?definition.parameters:{},value=>{phase[key]=value;scheduleValidation();});
      }
    }
    for(const [key,fallback] of Object.entries(definition.native||phase.type==='custom_analysis'?{}:definition.params||{})){
      const field=document.createElement('div'),caption=document.createElement('label'),value=document.createElement(key==='environment'?'select':'input');
      if(key==='environment')value.append(...['cartpole','foodland','ant'].map(name=>new Option(name,name)));
      caption.textContent=key.replaceAll('_',' ');const current=phase.params[key]??settings[key]??fallback;
      value.type=Array.isArray(fallback)||typeof fallback==='string'?'text':'number';value.step=key.endsWith('_seconds')?'.02':key==='sensory_index'||key==='causal_repeats'?'1':'any';
      value.value=Array.isArray(current)?current.join(','):current;phase.params[key]=current;
      value.setAttribute('aria-label',`Phase ${index+1} ${key}`);
      value.oninput=()=>{phase.params[key]=Array.isArray(fallback)?value.value.split(',').filter(v=>v.trim()).map(Number):typeof fallback==='string'?value.value:value.value===''?null:Number(value.value);scheduleValidation();};field.append(caption,value);fields.append(field);
    }
    card.append(fields);const contract=document.createElement('div');contract.className='phase-contract';contract.dataset.contract=index;card.append(contract);return card;
  });
  $('phaseBuilder').replaceChildren(...nodes);
  if(typeof renderPhaseFlow==='function')renderPhaseFlow();
}
function displayValidation(report,preflight=false){
  $('experimentValidation').dataset.ok=String(report.ok);
  $('experimentValidation').textContent=(report.ok?(preflight?'✓ Verification passed. '+report.preflight:'✓ Structure valid. Run Verify experiment for dependency/runtime preflight.'):'! '+(report.errors||[]).join('\n'))+(report.caveats?.length?'\nReview: '+report.caveats.join('\n'):'');
  document.querySelectorAll('[data-contract]').forEach((el,index)=>{
    const row=report.phases?.find(row=>row.id===phaseSpecs[index]?.id);el.replaceChildren();if(!row)return;
    for(const [key,origin] of Object.entries(row.inputs||row.requirements||{})){
      const line=document.createElement('div');line.textContent=`Inputs: ${contractLabel(key,phaseDefinition(phaseSpecs[index]))} ← ${origin||'MISSING'}`;if(!origin)line.className='missing';el.append(line);
    }
    const output=document.createElement('div');output.textContent='Outputs: '+(row.outputs||row.provides||[]).map(key=>contractLabel(key,phaseDefinition(phaseSpecs[index]))).join(', ');el.append(output);
  });
}
function scheduleValidation(){
  validationVersion++;clearTimeout(validationTimer);
  if(typeof renderPhaseFlow==='function')renderPhaseFlow();
  $('experimentValidation').textContent='Changes need verification…';
  $('experimentValidation').removeAttribute('data-ok');
  validationTimer=setTimeout(()=>verifyExperiment(false),350);
}
async function verifyExperiment(preflight=false,skip=false){
  if(!builderLoaded){error('Phase definitions are still loading.');return false;}
  clearTimeout(validationTimer);const version=++validationVersion;
  try{
    if(document.querySelector('#view-experiment :invalid'))throw Error('Check the highlighted parameter fields before verifying or launching.');
    const analysisErrors=await syncAnalysisContracts();if(analysisErrors.length)throw Error(analysisErrors.join('\n'));
    const config={...settings,...collect()};
    if(skip)config.phases=expandPhasePlan(phaseSpecs).filter(p=>['environment','cartpole','foodland','ant'].includes(p.type));
    if(preflight)$('experimentValidation').textContent='Checking dependencies, local environment and Python outputs…';
    const report=await request('/api/verify-experiment',{config,code:$('code').value,preflight});
    if(version!==validationVersion)return false;
    displayValidation(report,preflight);
    if(preflight&&!report.ok){showView('experiment');error('Experiment verification failed. See phase requirements and errors.');}
    else if(preflight)error('');
    return report.ok;
  }catch(exc){if(version===validationVersion){$('experimentValidation').textContent=exc.message;$('experimentValidation').dataset.ok='false';}return false;}
}
$('addPhase').onclick=()=>{
  const type=$('phaseType').value;if(type==='custom_analysis'){openAnalysisPhase();return;}let suffix=1,id=type.replace(/[^A-Za-z0-9_-]/g,'_');
  while(phaseSpecs.some(p=>p.id===id))id=type.replace(/[^A-Za-z0-9_-]/g,'_')+'_'+(++suffix);
  const next=structuredClone(phaseSpecs);next.push({id,type,params:{...catalog[type].params}});
  if(!['environment','cartpole','foodland','ant'].includes(type))delete next.at(-1).params.sensory_index;
  else if(type==='environment')next.at(-1).params.sensory_index=Number($('sensory').value);
  flowCommit(next);
};
$('zeroBaseline').onclick=()=>{$('baseline').value=Array(Number($('channels').value)).fill(0).join(',');scheduleValidation();};
$('verifyExperiment').onclick=()=>verifyExperiment(true);
experimentView.addEventListener('input',scheduleValidation);
$('view-simulation').addEventListener('input',scheduleValidation);
$('code').addEventListener('input',scheduleValidation);
(async()=>{try{
  catalog=await request('/api/phase-catalog');
  const groups=[['Interactive V3 phases',false],['Scientific V3 phases',true]].map(([label,native])=>{
    const group=document.createElement('optgroup');group.label=label;
    group.append(...Object.entries(catalog).filter(([,entry])=>!entry.hidden&&!!entry.native===native).map(([key,entry])=>new Option(entry.label,key)));return group;
  });
  $('phaseType').replaceChildren(...groups);loadBuilder(settings);
}catch(exc){error(exc.message);}})();
