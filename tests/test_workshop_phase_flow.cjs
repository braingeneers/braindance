const assert=require('node:assert/strict');
const {expandPhasePlan,phaseFlowTrace,phaseFlowLineage}=require('../braindance/examples/streaming_workshop/phase_flow.js');
const catalog={record:{label:'Record',requires:['source'],provides:['baseline']},probe:{label:'Probe',requires:['baseline'],provides:['response']},analyse:{label:'Analyse',requires:['response'],provides:['result']},native:{label:'Native',native:true,requires:['file'],provides:['units']}};
const phase=(id,type,loop)=>({id,type,params:{},...(loop?{loop}:{})});
const initial={source:{origin:'settings'}};
assert.equal(phaseFlowTrace([phase('p','probe')],catalog,initial).ok,false);
assert.equal(phaseFlowTrace([phase('r','record'),phase('p','probe')],catalog,initial).ok,true);
assert.equal(phaseFlowTrace([phase('p','probe'),phase('r','record')],catalog,initial).available.response,undefined);
assert.equal(phaseFlowTrace([phase('r','record'),{...phase('n','native'),settings:{file:'x'}}],catalog,initial).ok,false);
assert.equal(phaseFlowTrace([{...phase('n','native'),settings:{file:'x'}}],catalog,{}).ok,true);
const loop={id:'section',count:3};
const plan=[phase('r','record'),phase('p','probe',loop),phase('a','analyse',loop),phase('p_r2','analyse')];
const expanded=expandPhasePlan(plan);
assert.deepEqual(expanded.map(p=>p.type),['record','probe','analyse','probe','analyse','probe','analyse','analyse']);
assert.equal(new Set(expanded.map(p=>p.id)).size,8);
assert.equal(phaseFlowTrace(expanded,catalog,initial).ok,true);
assert.deepEqual(plan[1].loop,loop);
assert.throws(()=>expandPhasePlan([phase('r','record',{id:'loop',count:0})]));
assert.throws(()=>expandPhasePlan([phase('r','record',{id:'loop',count:32}),phase('r2','record')]));
const lineage=phaseFlowLineage('response',catalog,initial,false);
assert.match(lineage,/Probe.*baseline.*Record.*source/);
assert.match(phaseFlowLineage('missing',catalog,{},false),/supply/);
assert.match(phaseFlowLineage('cycle',{cycle:{label:'Cycle',requires:['cycle'],provides:['cycle']}},{},false),/cyclic/);
console.log('Phase flow: dependency order, modes, local settings, loop expansion, IDs and lineage passed');
const {movePhasePlan}=require('../braindance/examples/streaming_workshop/phase_flow.js');
const movePlan=[phase('r','record'),phase('p','probe',loop),phase('a','analyse',loop),phase('b','record')];
assert.deepEqual(movePhasePlan(movePlan,3,1).map(p=>p.id),['r','b','p','a']);
assert.deepEqual(movePhasePlan(movePlan,1,4,true).map(p=>p.id),['r','b','p','a']);
assert.equal(movePhasePlan(movePlan,3,2)[2].loop.id,'section');
assert.equal(movePhasePlan(movePlan,1,4).at(-1).loop,undefined);
assert.deepEqual(movePhasePlan(movePlan,2,2),movePlan);
assert.throws(()=>movePhasePlan(movePlan,0,9));
assert.equal(phaseFlowTrace(movePhasePlan(movePlan,0,4),catalog,initial).ok,false);
console.log('Phase moves: order, whole loops, loop entry/exit, bounds and dependency rejection passed');

// Custom decorated analysis carries per-instance contracts and can join either runner.
const customCatalog = {
  record: {requires: [], provides: ['rates']},
  native: {native: true, requires: [], provides: ['rates']},
  custom_analysis: {category: 'analysis', requires: [], provides: ['placeholder']},
};
const customPhase = {id: 'analysis', type: 'custom_analysis', params: {requires: ['rates'], provides: ['result']}};
for (const type of ['record', 'native']) {
  const trace = phaseFlowTrace([{id: 'source', type}, customPhase], customCatalog, {});
  assert.equal(trace.ok, true);
  assert.equal(trace.available.result.origin, 'analysis');
  assert.equal(trace.available.placeholder, undefined);
  assert.equal(phaseFlowTrace([customPhase, {id:'source', type}], customCatalog, {}).ok, false);
}

// Canonical V3 workshop contracts need no artificial workshop configuration input.
const canonicalCatalog = {
  recording: {inputs: [], outputs: ['recording_baseline_hz']},
  environment: {inputs: ['recording_baseline_hz'], outputs: ['environment_reward']},
  custom_analysis: {inputs: [], outputs: []},
};
assert.equal(phaseFlowTrace([phase('r','recording'),phase('e','environment')],canonicalCatalog,{}).ok,true);
assert.equal(phaseFlowTrace([phase('e','environment')],canonicalCatalog,{}).ok,false);
assert.equal(phaseFlowTrace([phase('e','environment')],canonicalCatalog,{recording_baseline_hz:{origin:'Explicit baseline'}}).ok,true);
const legacyAnalysis={id:'legacy',type:'custom_analysis',params:{requires:['workshop_baseline_hz'],provides:['mean_rate']}};
assert.equal(phaseFlowTrace([phase('r','recording'),legacyAnalysis],canonicalCatalog,{}).ok,true);
assert.match(phaseFlowLineage('workshop_baseline_hz',{recording:{label:'Recording',inputs:[],outputs:['recording_baseline_hz']}},{},false),/Recording/);
