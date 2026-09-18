// A loop is an editor grouping; every consumer receives the same expanded sequence.
function expandPhasePlan(plan) {
  const result=[], used=new Set(plan.map(p=>p.id));
  for(let i=0;i<plan.length;){
    let end=i+1;const loop=plan[i].loop;
    if(loop)while(end<plan.length&&plan[end].loop?.id===loop.id)end++;
    const count=loop?loop.count:1;
    if(!Number.isInteger(count)||count<1||count>32)throw Error('Loop repetitions must be between 1 and 32.');
    for(let repeat=0;repeat<count;repeat++)for(const phase of plan.slice(i,end)){
      let id=phase.id;
      if(repeat){let suffix=2;id=phase.id.slice(0,52)+'_r'+(repeat+1);while(used.has(id))id=phase.id.slice(0,46)+'_r'+(repeat+1)+'_'+suffix++;used.add(id);}
      result.push({...phase,id});
      if(result.length>32)throw Error('This sequence expands beyond 32 phases. Reduce the loop count or section size.');
    }
    i=end;
  }
  return result;
}
function phaseDataKey(key) {
  return ({workshop_baseline_hz:'recording_baseline_hz',workshop_causal_hz:'response_probe_hz',workshop_causal_trials:'response_probe_trials',workshop_episodes:'environment_episodes',workshop_reward:'environment_reward'})[key]||key;
}
function phaseFlowTrace(plan, definitions, initial) {
  const available=Object.fromEntries(Object.entries(initial).map(([key,value])=>[phaseDataKey(key),value])), rows=[];let mode=null;
  for(const phase of plan){
    const base=definitions[phase.type]||{},definition=phase.type==='custom_analysis'?{...base,inputs:phase.params?.inputs||phase.params?.requires||[],outputs:phase.params?.outputs||phase.params?.provides||[]}:base, native=phase.type==='custom_analysis'?(mode??plan.some(p=>!!definitions[p.type]?.native)):!!definition.native;
    const local={...available};
    if(native)for(const [key,value] of Object.entries(phase.settings||{}))if(value!==null)local[phaseDataKey(key)]={origin:phase.id+' settings',value};
    const missing=(definition.inputs||definition.requires||[]).filter(key=>!Object.hasOwn(local,phaseDataKey(key)));
    const mixed=mode!==null&&mode!==native;if(phase.type!=='custom_analysis')mode??=native;
    rows.push({before:{...available},missing,mixed});
    if(!missing.length&&!mixed)for(const key of definition.outputs||definition.provides||[])available[phaseDataKey(key)]={origin:phase.id,projected:true};
  }
  return {rows,available,ok:rows.every(row=>!row.missing.length&&!row.mixed)};
}
function phaseFlowLineage(key, definitions, available, native, seen=new Set()) {
  key=phaseDataKey(key);
  if(Object.hasOwn(available,key))return key+' (available)';
  if(seen.has(key))return key+' (cyclic dependency; supply an initial input)';
  const next=new Set([...seen,key]);
  const producers=Object.values(definitions).filter(d=>!d.hidden&&!!d.native===native&&(d.outputs||d.provides||[]).map(phaseDataKey).includes(key));
  if(!producers.length)return key+' ← supply in experiment inputs';
  return key+' ← '+producers.map(d=>d.label+((d.inputs||d.requires||[]).length?' ['+(d.inputs||d.requires||[]).map(k=>phaseFlowLineage(k,definitions,available,native,next)).join('; ')+']':'')).join(' OR ');
}
// Targets are gaps in the original plan, including its final gap.
function movePhasePlan(plan, source, target, section=false) {
  if(!Number.isInteger(source)||source<0||source>=plan.length||!Number.isInteger(target)||target<0||target>plan.length)throw Error('Choose a connection on the board.');
  const next=structuredClone(plan);let start=source,end=source+1;
  if(section&&plan[source].loop){
    while(start>0&&plan[start-1].loop?.id===plan[source].loop.id)start--;
    while(end<plan.length&&plan[end].loop?.id===plan[source].loop.id)end++;
  }
  if(target>=start&&target<=end)return next;
  const items=next.splice(start,end-start),destination=target>end?target-items.length:target;
  const enclosing=next[destination-1]?.loop&&next[destination-1].loop.id===next[destination]?.loop?.id?next[destination].loop:null;
  if(section&&enclosing)throw Error('Place a loop between sections; nested loops are not supported.');
  if(!section)for(const item of items){delete item.loop;if(enclosing)item.loop={...enclosing};}
  next.splice(destination,0,...items);return next;
}
if(typeof module!=='undefined')module.exports={expandPhasePlan,phaseFlowTrace,phaseFlowLineage,movePhasePlan};
if(typeof document!=='undefined'){
let flowPosition=0,flowSelected=new Set(),flowMessage='',flowFocus=null,flowDrag=null,flowZoom=1,flowSelecting=false,flowSelectionAnchor=null;
const flowUndo=[],flowRedo=[];
let flowTouchPinching=false,flowPinchStart=null,flowGestureZoom=null,flowSuppressClickUntil=0;
const flowView=$('view-sequence');
flowView.insertAdjacentHTML('beforeend',`
<div class="flow-heading"><p>Build a sequence. Snap in phases, connect their data, and repeat sections.</p><div class="flow-key"><span data-kind="experiment">● Experiment</span><span data-kind="analysis">◆ Analysis</span><span data-kind="loop">↻ Loop</span></div></div>
<div class="flow-tools"><div><button id="flowUndo" aria-label="Undo change" title="Undo">↶</button><button id="flowRedo" aria-label="Redo change" title="Redo">↷</button><button id="flowLoop">↻ Select loop section</button><label>× <input id="flowCount" aria-label="New loop repetitions" type="number" min="1" max="32" value="2"></label><button id="flowClear">Clear selection</button></div><div><button id="flowSetup">Setup</button><button id="flowVerify">✓ Verify experiment</button><button id="flowExpand" aria-pressed="false">Expand board</button></div></div>
<div id="flowSelectionHelp" role="status" aria-live="polite"></div>
<div id="flowStatus" role="status" aria-live="polite"></div>
<div class="flow-layout"><aside class="flow-library"><div class="flow-library-head"><h2>Phases <small id="flowAvailable"></small></h2><input id="flowSearch" aria-label="Find a phase" type="search" placeholder="Find a phase…"><select id="flowCategory" aria-label="Phase category"><option value="all">All phases</option><option value="experiment">Experiments</option><option value="analysis">Analysis</option></select><p id="flowInsertion"></p></div><div id="flowOptions"></div></aside>
<div class="flow-workspace"><section class="flow-board"><div class="flow-board-bar"><span><strong>Sequence</strong> <small id="flowTotal"></small></span><div><button id="flowZoomOut" aria-label="Zoom out">−</button><button id="flowZoomReset" title="Reset zoom">100%</button><button id="flowZoomIn" aria-label="Zoom in">+</button></div></div><div id="flowViewport" tabindex="0" aria-label="Phase board. Drag the empty board to pan; use connections to add or move phases."><div id="flowSequence"></div></div><div class="flow-board-foot"><span id="flowDragHint">Drag to reorder · Click a node to edit · Select a section to loop</span><span>Pinch to zoom · Drag the board to pan</span></div></section>
<section id="flowInspector" class="flow-inspector" hidden></section>
<details class="flow-context" open><summary>Data at this connection <small id="flowContextCount"></small></summary><p class="help">Input values are available now. Projected outputs become available when their phase runs.</p><div id="flowContext"></div></details></div></div>`);
const sheet=document.createElement('link');sheet.rel='stylesheet';sheet.href='/phase_flow.css';document.head.append(sheet);
function flowInputs(native){
  const data={};
  if(native){
    for(const [key,value] of Object.entries(JSON.parse($('initialData').value)))data[key]={origin:'Initial data',value};
    for(const [key,value] of Object.entries(JSON.parse($('nativeSettings').value)))if(value!==null)data[key]={origin:'Experiment settings',value};
  }else{
    if($('baseline').value.trim())data.recording_baseline_hz={origin:'Explicit baseline',value:$('baseline').value};
    else if(settings.calibration)data.recording_baseline_hz={origin:'Saved calibration (verify before running)'};
  }
  return data;
}
function flowProblem(next){
  try{
    const trace=phaseFlowTrace(expandPhasePlan(next),catalog,flowInputs(next.some(p=>!!catalog[p.type]?.native)||next.every(p=>p.type==='custom_analysis')));
    const bad=trace.rows.findIndex(row=>row.missing.length||row.mixed);
    if(bad>=0){const row=trace.rows[bad],phase=expandPhasePlan(next)[bad];return row.mixed?'Use separate sequences for streaming and native phases.':`${phase.id} would be without its inputs: ${row.missing.join(', ')}. Keep its producer earlier in the sequence.`;}
    return '';
  }catch(exc){return exc.message;}
}
function flowCommit(next,remember=true){
  const problem=flowProblem(next);
  if(problem){flowMessage=problem;renderPhaseFlow();showView('sequence');return false;}
  if(remember&&JSON.stringify(next)!==JSON.stringify(phaseSpecs)){flowUndo.push(structuredClone(phaseSpecs));if(flowUndo.length>50)flowUndo.shift();flowRedo.length=0;}
  phaseSpecs=next;flowMessage='';flowSelected.clear();flowSelecting=false;flowSelectionAnchor=null;renderPhases();scheduleValidation();return true;
}
function flowNewPhase(type){
  const base=type.replace(/[^A-Za-z0-9_-]/g,'_').slice(0,55);let id=base,suffix=1;
  while(phaseSpecs.some(p=>p.id===id))id=base+'_'+(++suffix);
  return {id,type,params:structuredClone(catalog[type].params||{})};
}
function flowPropose(payload,target){
  if(payload.type){
    const next=structuredClone(phaseSpecs),phase=flowNewPhase(payload.type);
    if(target>0&&next[target-1]?.loop?.id===next[target]?.loop?.id&&next[target]?.loop)phase.loop={...next[target].loop};
    next.splice(target,0,phase);return next;
  }
  return movePhasePlan(phaseSpecs,payload.index,target,payload.section);
}
function flowDrop(payload,target){
  try{const next=flowPropose(payload,target);if(JSON.stringify(next)===JSON.stringify(phaseSpecs)){flowMessage='Already in this position. Drop on another node to move it.';renderPhaseFlow();flowEndDrag();return;}if(flowCommit(next)){flowPosition=Math.min(target+Number(!!payload.type),next.length);flowMessage=payload.type?'Phase connected.':'Sequence updated.';renderPhaseFlow();}}
  catch(exc){flowMessage=exc.message;renderPhaseFlow();}
  flowEndDrag();
}
function flowEndDrag(){$('flowDragHint').textContent='Drag to reorder · Click a node to edit · Select a section to loop';document.querySelectorAll('[data-drop-side]').forEach(el=>{delete el.dataset.dropSide;delete el.dataset.dropBlocked;});flowDrag=null;flowView.classList.remove('flow-dragging');document.querySelectorAll('.flow-gap').forEach(el=>{delete el.dataset.drop;delete el.dataset.over;el.removeAttribute('title');});document.querySelectorAll('.flow-drag-source').forEach(el=>el.classList.remove('flow-drag-source'));}
function flowStartDrag(payload,node,event){
  flowDrag=payload;flowView.classList.add('flow-dragging');node.classList.add('flow-drag-source');
  if(event.dataTransfer){event.dataTransfer.effectAllowed=payload.type?'copy':'move';event.dataTransfer.setData('text/plain',JSON.stringify(payload));}
  document.querySelectorAll('.flow-gap').forEach(el=>{try{const reason=flowProblem(flowPropose(payload,Number(el.dataset.position)));el.dataset.drop=reason?'blocked':'ready';el.title=reason||'Drop to connect';}catch(exc){el.dataset.drop='blocked';el.title=exc.message;}});
}
function flowDraggable(node,payload,handle=node){
  node.draggable=true;
  node.ondragstart=event=>{if(event.target.closest('input,select,textarea')){event.preventDefault();return;}event.stopPropagation();flowStartDrag(payload,node,event);};
  node.ondragend=flowEndDrag;
  // Touch uses the same drop targets without relying on browser HTML drag support.
  handle.addEventListener('pointerdown',event=>{
    if(event.pointerType!=='touch')return;
    event.preventDefault();event.stopPropagation();flowStartDrag(payload,node,event);handle.setPointerCapture(event.pointerId);
    const move=e=>flowPreviewTarget(e.clientX,e.clientY);
    const end=e=>{handle.removeEventListener('pointermove',move);handle.removeEventListener('pointerup',end);handle.removeEventListener('pointercancel',cancel);const target=flowTarget(e.clientX,e.clientY);if(flowDrag&&!flowTouchPinching&&target!==null)flowDrop(payload,target);else flowEndDrag();};
    const cancel=()=>{handle.removeEventListener('pointermove',move);handle.removeEventListener('pointerup',end);handle.removeEventListener('pointercancel',cancel);flowEndDrag();};
    handle.addEventListener('pointermove',move);handle.addEventListener('pointerup',end);handle.addEventListener('pointercancel',cancel);
  });
}
$('flowSetup').onclick=()=>showView('experiment');
$('flowVerify').onclick=async()=>{await verifyExperiment(true);flowMessage=$('experimentValidation').textContent;renderPhaseFlow();};
$('flowSearch').oninput=$('flowCategory').onchange=()=>renderPhaseFlow();
$('flowClear').onclick=()=>{flowSelected.clear();flowSelecting=false;flowSelectionAnchor=null;renderPhaseFlow();};
for(const [id,from,to] of [['flowUndo',flowUndo,flowRedo],['flowRedo',flowRedo,flowUndo]])$(id).onclick=()=>{if(!from.length)return;const next=from.at(-1),previous=structuredClone(phaseSpecs);if(flowCommit(next,false)){from.pop();to.push(previous);renderPhaseFlow();}};
$('flowLoop').onclick=()=>{
  if(!flowSelected.size){flowSelecting=true;flowSelectionAnchor=null;flowMessage='';flowSelectionUI();return;}
  const indices=phaseSpecs.flatMap((p,i)=>flowSelected.has(p.id)?[i]:[]),count=Number($('flowCount').value);
  if(!indices.length||indices.some((v,i)=>v!==indices[0]+i)||indices.some(i=>phaseSpecs[i].loop)){flowMessage='Select consecutive nodes outside existing loops. Unwrap a loop before regrouping it.';renderPhaseFlow();return;}
  const next=structuredClone(phaseSpecs),id='loop_'+Date.now();for(const i of indices)next[i].loop={id,count};flowCommit(next);
};
$('flowExpand').onclick=()=>{const expanded=flowView.classList.toggle('flow-expanded');$('flowExpand').textContent=expanded?'Exit expanded board':'Expand board';$('flowExpand').setAttribute('aria-pressed',String(expanded));};
function flowSetZoom(value,clientX=null,clientY=null){
  const board=$('flowViewport'),bounds=board.getBoundingClientRect();
  const x=clientX===null?board.clientWidth/2:clientX-bounds.left,y=clientY===null?board.clientHeight/2:clientY-bounds.top;
  const contentX=(board.scrollLeft+x)/flowZoom,contentY=(board.scrollTop+y)/flowZoom;
  flowZoom=Math.max(.55,Math.min(1.4,value));$('flowSequence').style.zoom=flowZoom;$('flowZoomReset').textContent=Math.round(flowZoom*100)+'%';
  board.scrollLeft=contentX*flowZoom-x;board.scrollTop=contentY*flowZoom-y;
}
$('flowZoomIn').onclick=()=>flowSetZoom(flowZoom+.1);$('flowZoomOut').onclick=()=>flowSetZoom(flowZoom-.1);$('flowZoomReset').onclick=()=>flowSetZoom(1);
flowView.addEventListener('keydown',e=>{if(e.key==='Escape'){flowSelecting=false;flowSelectionAnchor=null;flowSelected.clear();flowSelectionUI();flowEndDrag();flowView.classList.remove('flow-expanded');$('flowExpand').textContent='Expand board';$('flowExpand').setAttribute('aria-pressed','false');}});
const viewport=$('flowViewport');
viewport.addEventListener('pointerdown',e=>{
  if(flowTouchPinching||e.button!==0||e.target.closest('button,input,select,.flow-piece,.flow-loop-title'))return;
  const x=e.clientX,y=e.clientY,left=viewport.scrollLeft,top=viewport.scrollTop;viewport.setPointerCapture(e.pointerId);viewport.classList.add('panning');
  const move=event=>{if(flowTouchPinching)return;viewport.scrollLeft=left+x-event.clientX;viewport.scrollTop=top+y-event.clientY;};
  const end=()=>{viewport.removeEventListener('pointermove',move);viewport.removeEventListener('pointerup',end);viewport.removeEventListener('pointercancel',end);viewport.classList.remove('panning');};
  viewport.addEventListener('pointermove',move);viewport.addEventListener('pointerup',end);viewport.addEventListener('pointercancel',end);
});
// Trackpad pinches arrive as Ctrl+wheel; ordinary wheel scrolling remains native.
viewport.addEventListener('wheel',event=>{
  if(!event.ctrlKey)return;
  event.preventDefault();if(flowGestureZoom!==null)return;
  const delta=event.deltaY*(event.deltaMode===1?16:event.deltaMode===2?viewport.clientHeight:1);
  flowSetZoom(flowZoom*Math.exp(-delta*.01),event.clientX,event.clientY);
},{passive:false});
// Two-finger touchscreen gestures cancel any one-finger node drag or pan.
const pinchGeometry=touches=>({distance:Math.hypot(touches[0].clientX-touches[1].clientX,touches[0].clientY-touches[1].clientY),x:(touches[0].clientX+touches[1].clientX)/2,y:(touches[0].clientY+touches[1].clientY)/2});
viewport.addEventListener('touchstart',event=>{
  if(event.touches.length!==2)return;
  event.preventDefault();flowTouchPinching=true;flowEndDrag();viewport.classList.remove('panning');
  const point=pinchGeometry(event.touches),bounds=viewport.getBoundingClientRect();
  flowPinchStart={...point,zoom:flowZoom,contentX:(viewport.scrollLeft+point.x-bounds.left)/flowZoom,contentY:(viewport.scrollTop+point.y-bounds.top)/flowZoom};
},{passive:false,capture:true});
viewport.addEventListener('touchmove',event=>{
  if(!flowTouchPinching)return;event.preventDefault();
  if(event.touches.length!==2||!flowPinchStart||flowPinchStart.distance<1)return;
  const point=pinchGeometry(event.touches),bounds=viewport.getBoundingClientRect();
  flowSetZoom(flowPinchStart.zoom*point.distance/flowPinchStart.distance,point.x,point.y);
  viewport.scrollLeft=flowPinchStart.contentX*flowZoom-(point.x-bounds.left);
  viewport.scrollTop=flowPinchStart.contentY*flowZoom-(point.y-bounds.top);
},{passive:false,capture:true});
for(const name of ['touchend','touchcancel'])viewport.addEventListener(name,event=>{
  if(!flowTouchPinching)return;event.preventDefault();flowSuppressClickUntil=Date.now()+350;
  if(event.touches.length<2)flowPinchStart=null;
  if(!event.touches.length){flowTouchPinching=false;flowEndDrag();}
},{passive:false,capture:true});
viewport.addEventListener('click',event=>{if(flowTouchPinching||Date.now()<flowSuppressClickUntil){event.preventDefault();event.stopImmediatePropagation();}},true);
// Safari exposes trackpad pinch as gesture events instead of Ctrl+wheel.
viewport.addEventListener('gesturestart',event=>{event.preventDefault();flowGestureZoom=flowZoom;});
viewport.addEventListener('gesturechange',event=>{event.preventDefault();if(flowGestureZoom!==null&&Number.isFinite(event.scale))flowSetZoom(flowGestureZoom*event.scale,Number.isFinite(event.clientX)?event.clientX:null,Number.isFinite(event.clientY)?event.clientY:null);});
viewport.addEventListener('gestureend',event=>{event.preventDefault();flowGestureZoom=null;});
// Resolve generous board-wide drop areas, including either half of a node.
function flowTarget(x,y){
  const bounds=viewport.getBoundingClientRect();if(x<bounds.left||x>bounds.right||y<bounds.top||y>bounds.bottom)return null;
  const hit=document.elementFromPoint(x,y),gap=hit?.closest('.flow-gap');
  if(gap)return Number(gap.dataset.position);
  const card=hit?.closest('.flow-piece');
  if(card){const index=phaseSpecs.findIndex(p=>p.id===card.dataset.id),rect=card.getBoundingClientRect(),middle=rect.left+rect.width/2;
    const after=Math.abs(x-middle)<12&&flowDrag?.index!==undefined?flowDrag.index<index:x>middle;
    return index+Number(after);
  }
  const gaps=[...viewport.querySelectorAll('.flow-gap')];
  if(!gaps.length)return null;
  return Number(gaps.reduce((best,el)=>Math.abs(x-(el.getBoundingClientRect().left+el.getBoundingClientRect().width/2))<Math.abs(x-(best.getBoundingClientRect().left+best.getBoundingClientRect().width/2))?el:best).dataset.position);
}
function flowPreviewTarget(x,y){
  document.querySelectorAll('[data-drop-side]').forEach(el=>{delete el.dataset.dropSide;delete el.dataset.dropBlocked;});
  document.querySelectorAll('.flow-gap[data-over]').forEach(el=>delete el.dataset.over);
  const target=flowTarget(x,y);if(target===null||!flowDrag)return;
  let problem='';try{problem=flowProblem(flowPropose(flowDrag,target));}catch(exc){problem=exc.message;}
  const gap=viewport.querySelector(`.flow-gap[data-position="${target}"]`);if(gap)gap.dataset.over='true';
  const cards=[...viewport.querySelectorAll('.flow-piece')],card=cards[target]||cards.at(-1);
  if(card){card.dataset.dropSide=target<cards.length?'before':'after';card.dataset.dropBlocked=String(!!problem);}
  $('flowDragHint').textContent=problem||('Release to place '+(target===phaseSpecs.length?'at the end.':'before '+phaseSpecs[target].id+'.'));
}
viewport.addEventListener('dragover',e=>{if(!flowDrag)return;e.preventDefault();e.dataTransfer.dropEffect=flowDrag.type?'copy':'move';flowPreviewTarget(e.clientX,e.clientY);const box=viewport.getBoundingClientRect();if(e.clientX>box.right-65)viewport.scrollLeft+=18;else if(e.clientX<box.left+65)viewport.scrollLeft-=18;});
viewport.addEventListener('drop',e=>{if(!flowDrag)return;e.preventDefault();const target=flowTarget(e.clientX,e.clientY);if(target!==null)flowDrop(flowDrag,target);else flowEndDrag();});
function flowSelectionUI(){
  const count=flowSelected.size;if(flowSelecting)$('flowStatus').textContent=flowMessage;flowView.classList.toggle('flow-selecting',flowSelecting);
  $('flowLoop').textContent=count?`↻ Create loop (${count} ${count===1?'phase':'phases'})`:'↻ Select loop section';
  $('flowLoop').setAttribute('aria-pressed',String(flowSelecting));
  $('flowClear').textContent=flowSelecting?'Cancel selection':'Clear selection';
  $('flowSelectionHelp').textContent=flowSelecting?(count>1?`${count} phases selected. Choose Create loop, or click a different last phase to adjust the range.`:count?'Click the last phase to extend the section, or create a loop with this phase.':'Click the first phase in the section you want to repeat.'):(count?`${count} phases selected for a loop. Choose Create loop above.`:'');
  viewport.querySelectorAll('.flow-piece').forEach(card=>{const selected=flowSelected.has(card.dataset.id);card.dataset.selected=String(selected);const input=card.querySelector('input[type=checkbox]');input.checked=selected;const text=card.querySelector('.flow-select-label span');if(text)text.textContent=input.disabled?'Inside loop':selected?'Selected for loop':'Include in loop';});
}
function flowSelectRange(index){
  if(phaseSpecs[index].loop){flowMessage='Unwrap this existing loop before selecting a new section.';$('flowStatus').textContent=flowMessage;return;}
  if(flowSelectionAnchor===null)flowSelectionAnchor=index;
  const start=Math.min(flowSelectionAnchor,index),end=Math.max(flowSelectionAnchor,index);
  if(phaseSpecs.slice(start,end+1).some(p=>p.loop)){$('flowStatus').textContent='A selection cannot cross an existing loop. Unwrap it first.';return;}
  flowSelected=new Set(phaseSpecs.slice(start,end+1).map(p=>p.id));flowMessage='';flowSelectionUI();
}

function flowButton(text,label,action){const button=document.createElement('button');button.textContent=text;button.title=label;button.setAttribute('aria-label',label);button.onclick=action;return button;}
function flowLabel(definition){if(definition.environment)return definition.label;return (definition.label||'Phase').replace(/PhaseV3$|Phase$|V3$/g,'').replace(/([a-z])([A-Z])/g,'$1 $2').replace(/RTSort/g,'RT Sort');}
function flowKind(definition){return definition.category==='analysis'?'analysis':'experiment';}
function flowInspector(){
  const inspector=$('flowInspector'),index=phaseSpecs.findIndex(p=>p.id===flowFocus);inspector.replaceChildren();inspector.hidden=index<0;if(index<0)return;
  const phase=phaseSpecs[index];if(phase.type==='custom_analysis'){inspector.append(flowButton('Edit analysis in Functions','Edit analysis in Functions',()=>openAnalysisPhase(phase.id)));return;}const definition=catalog[phase.type],header=document.createElement('div');header.className='flow-inspector-head';
  const heading=document.createElement('h2');heading.textContent=flowLabel(definition)+' · Parameters';header.append(heading,flowButton('×','Close phase settings',()=>{flowFocus=null;renderPhaseFlow();}));inspector.append(header);
  const form=document.createElement('form');form.className='flow-form';
  const idLabel=document.createElement('label'),idInput=document.createElement('input');idLabel.textContent='Phase ID';idInput.value=phase.id;idInput.required=true;idInput.pattern='[A-Za-z0-9_-]{1,64}';idLabel.append(idInput);form.append(idLabel);
  const editors={};
  if(definition.native){
    for(const key of ['params','settings']){const group=document.createElement('div'),heading=document.createElement('h3');heading.textContent=key==='params'?'Phase parameters':'Phase inputs';group.append(heading);form.append(group);editors[key]=parameterEditor(group,phase[key]||{},key==='params'?definition.parameters:{});}
  }else for(const [key,fallback] of Object.entries(definition.params||{})){
    const label=document.createElement('label'),input=document.createElement(key==='environment'?'select':'input');label.textContent=key.replaceAll('_',' ');
    if(key==='environment')input.append(...['cartpole','foodland','ant'].map(name=>new Option(name,name)));
    else if(typeof fallback==='number'){input.type='number';input.step='any';input.required=true;}
    const value=phase.params[key]??fallback;input.value=Array.isArray(value)?value.join(', '):value;label.append(input);form.append(label);editors[key]=input;
  }
  const submit=document.createElement('button');submit.type='submit';submit.textContent='Apply settings';form.append(submit);
  form.onsubmit=e=>{e.preventDefault();try{
    const next=structuredClone(phaseSpecs),updated=next[index];updated.id=idInput.value;
    if(next.some((p,i)=>i!==index&&p.id===updated.id))throw Error('Choose a unique phase ID.');
    if(definition.native){for(const key of ['params','settings']){updated[key]=editors[key].read();}}
    else for(const [key,fallback] of Object.entries(definition.params||{})){const value=editors[key].value;updated.params[key]=Array.isArray(fallback)?value.split(',').filter(v=>v.trim()).map(Number):typeof fallback==='number'?Number(value):value;}
    if(flowCommit(next)){flowFocus=updated.id;flowMessage='Settings applied.';renderPhaseFlow();}
  }catch(exc){$('flowStatus').textContent=exc.message;}};
  inspector.append(form);
  if(definition.caveats?.length){const note=document.createElement('p');note.className='help';note.textContent=definition.caveats.join(' ');inspector.append(note);}
}
function renderPhaseFlow(){
  if(!catalog)return;
  flowPosition=Math.min(flowPosition,phaseSpecs.length);const native=phaseSpecs.some(p=>!!catalog[p.type]?.native)||phaseSpecs.every(p=>p.type==='custom_analysis');
  let initial={},trace;
  try{initial=flowInputs(native);trace=phaseFlowTrace(phaseSpecs,catalog,initial);$('flowTotal').textContent=expandPhasePlan(phaseSpecs).length+' steps';}
  catch(exc){flowMessage=exc.message;trace=phaseFlowTrace(phaseSpecs,catalog,initial);}
  $('flowStatus').textContent=flowMessage;$('flowStatus').dataset.success=String(['Phase connected.','Sequence updated.','Settings applied.'].includes(flowMessage)||flowMessage.startsWith('✓'));$('flowUndo').disabled=!flowUndo.length;$('flowRedo').disabled=!flowRedo.length;
  const available=flowPosition<phaseSpecs.length?trace.rows[flowPosition].before:trace.available;
  $('flowInsertion').textContent=flowPosition===phaseSpecs.length?'Adding at the end of the sequence.':'Adding before '+phaseSpecs[flowPosition].id+'.';
  const options=[];let ready=0;
  for(const [type,definition] of Object.entries(catalog)){
    if(definition.hidden)continue;
    const kind=flowKind(definition);
    if(!(flowLabel(definition)+' '+definition.label).toLowerCase().includes($('flowSearch').value.toLowerCase())||!['all',kind].includes($('flowCategory').value))continue;
    let candidateInputs=available,inputError='';try{if(!phaseSpecs.length)candidateInputs=flowInputs(!!definition.native);}catch{inputError='Fix the JSON in Experiment setup inputs first.';}
    const missing=(definition.inputs||definition.requires||[]).filter(key=>!Object.hasOwn(candidateInputs,phaseDataKey(key))),mixed=type!=='custom_analysis'&&phaseSpecs.some(p=>p.type!=='custom_analysis')&&!!definition.native!==native;
    const reason=inputError||(mixed?'Use a separate sequence for native V3 phases and streaming adapters.':missing.map(key=>phaseFlowLineage(key,catalog,candidateInputs,!!definition.native)).join('\n'));
    if(!reason)ready++;
    const item=document.createElement('div');item.className='flow-option';item.dataset.kind=kind;item.dataset.available=String(!reason);
    const button=document.createElement('button');button.className='flow-add';button.textContent=(reason?'◌ ':'＋ ')+flowLabel(definition);button.setAttribute('aria-disabled',String(!!reason));button.title=reason||'Click to add, or drag to a connection';
    const caption=document.createElement('small');caption.textContent=(kind==='analysis'?'◆ Analysis':'● Experiment')+' · '+(mixed?'Other runner':reason?'Needs inputs':'Ready');
    const details=document.createElement('details'),summary=document.createElement('summary'),description=document.createElement('div');summary.textContent=reason?'Why unavailable?':'Data connections';description.textContent=reason||'Inputs: '+(definition.inputs.map(key=>contractLabel(key,definition)).join(', ')||'none')+' → Outputs: '+(definition.outputs.map(key=>contractLabel(key,definition)).join(', ')||'none');details.append(summary,description);
    button.onclick=()=>{if(type==='custom_analysis'){openAnalysisPhase(null,flowPosition);return;}if(reason){details.open=!details.open;return;}flowDrop({type},flowPosition);};
    // Availability is evaluated again at the drop destination.
    if(!reason&&type!=='custom_analysis')flowDraggable(button,{type});
    item.append(button,caption,details);if(definition.source_url)item.append(sourceLink(definition.source_url));options.push(item);
  }
  $('flowAvailable').textContent=ready+' ready';$('flowOptions').replaceChildren(...options);
  const sequence=$('flowSequence');sequence.replaceChildren();let container=sequence;
  const start=flowButton('▶','Run experiment from phase board',()=>{if(!launchPending&&!running()&&phaseSpecs.length)launch.click();});start.className='flow-start';start.disabled=launchPending||running()||!builderLoaded||!phaseSpecs.length;sequence.append(start);
  function gap(index,parent){
    const button=flowButton('','Insert at position '+(index+1),()=>{flowPosition=index;flowMessage='';renderPhaseFlow();$('flowSearch').focus({preventScroll:true});});button.className='flow-gap';button.dataset.position=index;button.setAttribute('aria-pressed',String(flowPosition===index));
    button.ondragover=e=>{if(!flowDrag)return;e.preventDefault();e.dataTransfer.dropEffect=flowDrag.type?'copy':'move';button.dataset.over='true';};button.ondragleave=()=>delete button.dataset.over;
    button.ondrop=e=>{e.preventDefault();e.stopPropagation();if(flowDrag)flowDrop(flowDrag,index);};parent.append(button);
  }
  phaseSpecs.forEach((phase,index)=>{
    const definition=phaseDefinition(phase),row=trace.rows[index],kind=flowKind(definition);
    if(!phase.loop||phase.loop.id!==phaseSpecs[index-1]?.loop?.id){
      container=sequence;gap(index,container);
      if(phase.loop){
        const group=document.createElement('section');group.className='flow-loop';sequence.append(group);
        const heading=document.createElement('div');heading.className='flow-loop-title';const handle=document.createElement('span');handle.className='flow-handle';handle.textContent='⠿';handle.title='Drag loop section';heading.append(handle,'↻ Repeat');
        const count=document.createElement('input');count.type='number';count.min=1;count.max=32;count.value=phase.loop.count;count.setAttribute('aria-label','Section repetitions');count.onchange=()=>{const next=structuredClone(phaseSpecs);for(const p of next)if(p.loop?.id===phase.loop.id)p.loop.count=Number(count.value);flowCommit(next);};
        heading.append(count,'times',flowButton('Unwrap','Unwrap loop',()=>{const next=structuredClone(phaseSpecs);for(const p of next)if(p.loop?.id===phase.loop.id)delete p.loop;flowCommit(next);}));
        const loopStart=index,loopEnd=phaseSpecs.findLastIndex(p=>p.loop?.id===phase.loop.id)+1;
        heading.append(flowButton('←','Move loop left',()=>flowDrop({index,section:true},Math.max(0,loopStart-1))),flowButton('→','Move loop right',()=>flowDrop({index,section:true},Math.min(phaseSpecs.length,loopEnd+1))));
        flowDraggable(heading,{index,section:true},handle);group.append(heading);container=document.createElement('div');container.className='flow-loop-body';group.append(container);
      }
    }else gap(index,container);
    const card=document.createElement('article');card.className='flow-piece';card.dataset.id=phase.id;card.dataset.kind=kind;card.dataset.invalid=String(!!row.missing.length||row.mixed);card.dataset.focused=String(flowFocus===phase.id);card.tabIndex=0;card.setAttribute('aria-label',(definition.label||phase.type)+', '+kind+' phase. Enter to edit.');
    const producer=(definition.inputs||definition.requires||[]).map(key=>row.before[phaseDataKey(key)]?.origin).map(id=>phaseSpecs.find(p=>p.id===id)).filter(Boolean).at(-1);
    card.dataset.input=producer?flowKind(catalog[producer.type]):'initial';
    const header=document.createElement('div');header.className='flow-node-head';const handle=document.createElement('span');handle.className='flow-handle';handle.textContent='⠿';handle.title='Drag to move';
    const selection=document.createElement('input');selection.type='checkbox';selection.setAttribute('aria-label','Select '+phase.id+' for loop');selection.checked=flowSelected.has(phase.id);selection.disabled=!!phase.loop;selection.onchange=()=>{if(selection.checked)flowSelected.add(phase.id);else flowSelected.delete(phase.id);flowSelectionUI();};card.dataset.selected=String(selection.checked);
    const typeLabel=document.createElement('span');typeLabel.textContent=(kind==='analysis'?'◆ Analysis':'● Experiment')+' · '+(index+1);
    header.append(handle,typeLabel,flowButton('×','Remove '+phase.id,()=>{const next=structuredClone(phaseSpecs);next.splice(index,1);flowCommit(next);}));card.append(header);
    const selectionLabel=document.createElement('label');selectionLabel.className='flow-select-label';const selectionText=document.createElement('span');selectionText.textContent=phase.loop?'Inside loop':'Include in loop';selectionLabel.append(selection,selectionText);card.append(selectionLabel);
    const title=document.createElement('h3');title.textContent=flowLabel(definition);card.append(title);
    const identity=document.createElement('small');identity.className='flow-node-id';identity.textContent=phase.id;card.append(identity);
    for(const [prefix,keys] of [['Inputs',definition.inputs||definition.requires||[]],['Outputs',definition.outputs||definition.provides||[]]]){
      const line=document.createElement('div');line.className='flow-contract';const label=document.createElement('small');label.textContent=prefix;line.append(label);
      if(!keys.length){const empty=document.createElement('span');empty.textContent='—';line.append(empty);}
      for(const key of keys){const chip=document.createElement('span');chip.className='flow-chip';chip.textContent=contractLabel(key,definition);chip.title=prefix==='Inputs'?key+' ← '+(row.before[phaseDataKey(key)]?.origin||(phase.settings?.[key]!==undefined?'Phase settings':'Missing input')):'Produced by '+phase.id;line.append(chip);}card.append(line);
    }
    if(row.missing.length||row.mixed){const warning=document.createElement('p');warning.className='flow-warning';warning.textContent=row.mixed?'Incompatible execution mode':'Missing: '+row.missing.join(', ');card.append(warning);}
    const actions=document.createElement('div');actions.className='flow-node-actions';
    const edit=()=>{if(phase.type==='custom_analysis'){openAnalysisPhase(phase.id);return;}flowFocus=phase.id;flowPosition=index+1;renderPhaseFlow();const inspector=$('flowInspector');inspector.scrollIntoView({block:'start'});inspector.querySelector('input,select,textarea')?.focus({preventScroll:true});};
    const parameters=flowButton(phase.type==='custom_analysis'?'Edit function':'Edit parameters','Parameters for '+phase.id,edit);
    parameters.setAttribute('aria-controls',phase.type==='custom_analysis'?'functionsPanel':'flowInspector');parameters.setAttribute('aria-expanded',String(flowFocus===phase.id));
    actions.append(parameters);if(definition.source_url)actions.append(sourceLink(definition.source_url));card.append(actions);
    const order=document.createElement('div');order.className='flow-node-order';
    const left=flowButton('← Earlier','Move '+phase.id+' earlier in sequence',()=>flowDrop({index},index-1)),right=flowButton('Later →','Move '+phase.id+' later in sequence',()=>flowDrop({index},index+2));left.disabled=index===0;right.disabled=index===phaseSpecs.length-1;
    order.append(left,right);card.append(order);
    card.onclick=e=>{if(e.target.closest('button,input,label,a'))return;if(flowSelecting)flowSelectRange(index);else edit();};card.onkeydown=e=>{if(e.target===card&&e.key==='Enter'){e.preventDefault();if(flowSelecting)flowSelectRange(index);else edit();}};
    flowDraggable(card,{index},handle);container.append(card);
  });gap(phaseSpecs.length,sequence);
  if(!phaseSpecs.length){const hint=document.createElement('div');hint.className='flow-empty';hint.innerHTML='<strong>Your experiment starts here</strong><p>Drag a phase onto the board,<br>or choose one from the library.</p>';sequence.append(hint);}
  $('flowContextCount').textContent=Object.keys(available).length+' values';
  $('flowContext').replaceChildren(...Object.entries(available).map(([key,entry])=>{const item=document.createElement('div');item.className='flow-data';const name=document.createElement('strong'),origin=document.createElement('small');name.textContent=contractLabel(key);origin.textContent=(entry.projected?'Projected · ':'Input · ')+entry.origin;item.append(name,origin);if(entry.value!==undefined){const value=document.createElement('small');value.textContent=JSON.stringify(entry.value);item.append(value);}return item;}));
  flowInspector();flowSelectionUI();
}
setInterval(()=>{const play=flowView.querySelector('.flow-start');if(play)play.disabled=launchPending||running()||!builderLoaded||!phaseSpecs.length;},250);
window.renderPhaseFlow=renderPhaseFlow;window.flowCommit=flowCommit;
renderPhaseFlow();
}
