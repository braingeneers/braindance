// Manual analysis is independent of the acquisition controls and phase builder.
(() => {
  const root = views.analysis;
  root.insertAdjacentHTML('beforeend', `
    <p class="help">Load an experiment, select a recorded phase, then inspect or analyze its files. Sorting creates neuron spike trains for further analysis.</p>
    <section class="panel analysis-loader"><h2>Experiment & recordings</h2>
      <div class="toolbar"><button id="analysisCurrent">Use current experiment</button><label class="analysis-path">Saved experiment directory, experiment JSON, or recording file<input id="analysisPath" placeholder="/path/to/experiment or recording.raw.h5"></label><button id="analysisBrowse">📁 Browse folder</button><button id="analysisLoad">Load</button></div>
      <div class="toolbar analysis-browser"><button id="analysisDataFolder">Data folder</button><button id="analysisParent">Up one folder</button><label class="analysis-path">Folders, experiments & recordings<select id="analysisChoices" aria-label="Folders, experiments and recordings"></select></label><button id="analysisOpenFolder">Open folder</button></div>
      <p id="analysisBrowseStatus" class="help" role="status"></p>
      <p id="analysisWorkspaceMessage" role="status" aria-live="polite">Choose the current experiment or enter a local path.</p>
    </section>
    <div class="analysis-layout"><aside class="panel"><h2>Recorded phases</h2><div id="analysisPhases"><p class="help">No experiment loaded.</p></div></aside>
    <section class="panel analysis-workbench"><div class="fields"><label>Recording<select id="analysisFile"></select></label></div><p id="analysisFileInfo" class="help"></p><details id="analysisOverview"><summary>Experiment files & phase details</summary><pre id="analysisOverviewText"></pre></details>
      <div id="analysisTabs" class="toolbar" role="tablist" aria-label="Analysis tools"></div>
      <h2 id="analysisToolTitle"></h2><p id="analysisToolHelp" class="help"></p>
      <form id="analysisForm"><div id="analysisParams" class="fields"></div><p id="analysisCapability" role="status"></p><button id="analysisRun" type="submit">Run analysis</button></form>
    </section></div>
    <section class="panel"><h2>Jobs & results</h2><p class="help">Each run retains its parameters and results below. You can continue using the workshop while a job runs.</p><div id="analysisJobs"><p class="help">No analysis jobs yet.</p></div></section>`);
  const tools = {
    raw: ['Raw data', 'Inspect a bounded trace from one acquisition channel. Values use the recording’s units.', [['channel','Channel',0],['start_ms','Start (ms)',0],['duration_ms','Duration (ms)',1000]]],
    spikes: ['Spikes', 'View detected channel events or sorted neuron spikes. The result identifies its spike source.', [['start_ms','Start (ms)',0],['duration_ms','Duration (ms)',10000],['max_units','Maximum units',64]]],
    sttc: ['Connectivity (STTC)', 'Pairwise spike-time tiling coefficients describe functional association, not causal connections.', [['start_ms','Start (ms)',0],['duration_ms','Duration (ms)',10000],['max_units','Maximum units',64],['delta_ms','Coincidence window (ms)',20]]],
    latency: ['Neuron latencies', 'Select a unit to inspect relative spike timing to other units. Peaks do not establish synaptic causality.', [['unit_id','Reference unit ID',0],['start_ms','Start (ms)',0],['duration_ms','Duration (ms)',10000],['window_ms','Lag window (ms)',100],['bin_ms','Lag bin (ms)',2],['max_units','Maximum units',64]]],
    overlap: ['Stimulus overlap', 'Compare repeated stimulation responses using the causal-connectivity workflow: raw and cleaned overlap, evoked spikes, and early/late response matrices. Inspect timing, artifact blanking, and unit source in the results.', [['channel','Channel',0],['start_ms','Start (ms)',0],['duration_ms','Duration (ms)',10000],['pre_ms','Before stimulus (ms)',20],['post_ms','After stimulus (ms)',200],['blank_ms','Artifact blanking (ms)',3],['first_order_ms','Early response window (ms)',10],['threshold_sigma','Raw detection threshold (noise SD)',5],['bin_ms','Response histogram bin (ms)',2],['max_events','Maximum stimuli',50],['stim_log','Stimulus log path (optional)','']]],
    sorting: ['Spike sorting', 'Initialize RT-Sort from a baseline recording, then apply it to the selected recording. Baseline and target must share channel routing. Sorting processes whole target files, including when a phase is selected. Results become available to spike analyses.', []],
  };
  let workspace={files:[],phases:[]}, selectedPhase=null, tool='raw', busy=false, poll=null;
  const drafts={};
  let sourceUrls={};
  const el=id=>document.getElementById(id);
  const file=()=>workspace.files.find(f=>f.id===el('analysisFile').value);
  const fileLabel=f=>`${workspace.phases.find(p=>p.id===f.phase_id)?.name||'Recording'} · ${f.name}`;
  const message=(text)=>{el('analysisWorkspaceMessage').textContent=text;};
  async function api(command){return request('/api/analysis',command);}
  function saveDraft(){const form=el('analysisParams');drafts[tool]=Object.fromEntries([...form.querySelectorAll('[name]')].map(input=>[input.name,input.value]));}
  function field(name,label,value,type='number'){
    const wrapper=document.createElement('label');wrapper.textContent=label;
    const input=document.createElement('input');input.name=name;input.type=type;input.value=drafts[tool]?.[name]??value;
    if(type==='number'){input.step='any';input.required=true;if(name!=='unit_id')input.min='0';}
    wrapper.append(input);return wrapper;
  }
  function renderTool(){
    el('analysisTabs').replaceChildren(...Object.entries(tools).map(([key,[title]])=>{
      const button=document.createElement('button');button.textContent=title;button.type='button';button.role='tab';button.setAttribute('aria-selected',String(key===tool));button.onclick=()=>{saveDraft();tool=key;renderTool();};return button;
    }));
    el('analysisToolTitle').textContent=tools[tool][0];el('analysisToolHelp').textContent=tools[tool][1];
    if(sourceUrls[tool])el('analysisToolHelp').append(' ',sourceLink(sourceUrls[tool]));
    el('analysisParams').replaceChildren(...tools[tool][2].map(([name,label,value])=>field(name,label,value,typeof value==='number'?'number':'text')));
    if(tool==='sorting'){
      const label=document.createElement('label');label.textContent='Baseline recording';const select=document.createElement('select');select.name='baseline_id';
      for(const f of workspace.files.filter(f=>f.capabilities?.raw)){const option=new Option(fileLabel(f),f.id);select.add(option);}
      select.value=drafts.sorting?.baseline_id||select.value;select.onchange=()=>{const baseline=workspace.files.find(f=>f.id===select.value);if(baseline?.bounds){const fs=baseline.sample_rate||20000;const start=Math.max(0,baseline.bounds[0]+1-(baseline.info?.first_frame||0));el('analysisParams').querySelector('[name=baseline_start_ms]').value=start*1000/fs;el('analysisParams').querySelector('[name=baseline_duration_ms]').value=(baseline.bounds[1]-baseline.bounds[0])*1000/fs;}};label.append(select);el('analysisParams').append(label);
      el('analysisParams').append(field('baseline_path','Or baseline file path (overrides selection)','','text'));
      const targetsLabel=document.createElement('label');targetsLabel.textContent='Additional target recordings (optional; Ctrl/Cmd to select several)';
      const targets=document.createElement('select');targets.name='target_ids';targets.multiple=true;targets.size=4;
      for(const f of workspace.files.filter(f=>f.capabilities?.raw)){const option=new Option(fileLabel(f)+(f.bounds?' · whole file':''),f.id);targets.add(option);}targetsLabel.append(targets);el('analysisParams').append(targetsLabel);
      for(const [name,label,value] of [['device','Device (cuda or cpu)','cuda'],['baseline_start_ms','Baseline start (ms)',0],['baseline_duration_ms','Baseline duration (ms)',60000],['num_processes','Worker processes',1],['stringent_thresh','Stringent detection threshold',.275],['loose_thresh','Loose detection threshold',.1],['min_activity_hz','Minimum activity (Hz)',.05],['min_seq_spikes_n','Minimum sequence spikes',10],['min_seq_spikes_hz','Minimum sequence rate (Hz)',.05]])el('analysisParams').append(field(name,label,value,typeof value==='number'?'number':'text'));
      if(!drafts.sorting?.baseline_duration_ms)select.onchange();
    }
    const f=file(),cap=f?.capabilities?.[tool];
    const enabled=typeof cap==='object'?cap.available:cap;
    el('analysisCapability').textContent=!f?'Select a recording to begin.':enabled?(tool==='sorting'?'Ready to configure sorting.':tool==='overlap'&&!f.stim_log?'Enter a stimulation log CSV path below before running.':'Ready to analyze this recording.'):(f.reasons?.[tool]||cap?.reason||'This recording does not provide the data needed for this tool.');
    el('analysisRun').disabled=!enabled||busy;el('analysisRun').textContent=tool==='sorting'?'Initialize & sort':'Run analysis';
  }
  function renderFiles(preferred){
    const files=workspace.files.filter(f=>!selectedPhase||f.phase_id===selectedPhase||workspace.phases.find(p=>p.id===selectedPhase)?.files?.includes(f.id));
    el('analysisFile').replaceChildren(...files.map(f=>new Option(fileLabel(f),f.id)));
    if(files.some(f=>f.id===preferred))el('analysisFile').value=preferred;
    const f=file();el('analysisFileInfo').textContent=f?[f.path,f.spike_source,f.description].filter(Boolean).join(' · '):'No recording file is available for this phase.';
    renderTool();
  }
  function renderWorkspace(data){
    const prior=el('analysisFile').value;workspace=data.workspace||data;
    workspace.files ||= [];workspace.phases ||= [];
    if(!workspace.phases.some(p=>p.id===selectedPhase))selectedPhase=null;
    el('analysisPhases').replaceChildren(...[{id:null,name:'All recordings'},...workspace.phases].map(p=>{
      const button=document.createElement('button');button.className='analysis-phase';button.textContent=p.name||p.id;if(p.status){const status=document.createElement('small');status.textContent=p.status;status.style.display='block';button.append(status);}button.setAttribute('aria-pressed',String(p.id===selectedPhase));button.onclick=()=>{saveDraft();selectedPhase=p.id;renderWorkspace(workspace);};return button;
    }));
    const phase=workspace.phases.find(p=>p.id===selectedPhase);
    el('analysisOverviewText').textContent=[workspace.name||'',phase?`${phase.name} · ${phase.status||''}${phase.duration_s!=null?' · '+phase.duration_s+' seconds':''}`:'All recorded phases',...(phase?.logs||workspace.logs||[]),...(workspace.warnings||[])].join('\n');
    renderFiles(prior);
  }
  let directory=null, browseVersion=0;
  async function browse(path){
    const version=++browseVersion;
    el('analysisBrowseStatus').textContent='Reading folder…';
    try{
      const data=await api({action:'browse',path});
      if(version!==browseVersion)return;
      directory=data;
      el('analysisPath').value=data.path;
      el('analysisChoices').replaceChildren(new Option('Select a folder, experiment or recording…',''),...data.entries.map(entry=>new Option(`${entry.directory?'📁':'▤'} ${entry.name} · ${entry.kind}`,entry.path)));
      el('analysisOpenFolder').disabled=true;
      el('analysisParent').disabled=data.parent===data.path;
      el('analysisBrowseStatus').textContent=data.warning||`${data.path} · ${data.entries.length} options. Open a folder to explore it, or select an experiment/recording and Load.`;
    }catch(error){if(version===browseVersion)el('analysisBrowseStatus').textContent=error.message;}
  }
  el('analysisBrowse').onclick=()=>browse(el('analysisPath').value);
  el('analysisDataFolder').onclick=()=>browse();
  el('analysisParent').onclick=()=>browse(directory?.parent);
  el('analysisOpenFolder').onclick=()=>browse(el('analysisChoices').value);
  el('analysisChoices').onchange=()=>{
    const entry=directory?.entries.find(entry=>entry.path===el('analysisChoices').value);
    el('analysisPath').value=entry?.path||directory?.path||'';
    el('analysisOpenFolder').disabled=!entry?.directory;
  };
  async function load(source, path){
    if(busy){message('Wait for the current analysis request to finish before selecting another experiment.');return false;}
    if(path!==undefined)el('analysisPath').value=path;
    busy=true;renderTool();message('Loading experiment metadata…');
    try{const data=await api({action:'load',source,path:el('analysisPath').value});selectedPhase=null;renderWorkspace(data);message(`${workspace.path||'Experiment'} · ${workspace.phases.length} recorded phases · ${workspace.files.length} files. ${(workspace.warnings||[]).join(' ')}`);window.dispatchEvent(new CustomEvent('workshop-analysis-selected',{detail:{path:workspace.path,name:workspace.name}}));return true;}
    catch(e){message(e.message);return false;}finally{busy=false;renderTool();}
  }
  window.workshopAnalysis={load};
  el('analysisCurrent').onclick=()=>load('current');el('analysisLoad').onclick=()=>load('path');
  el('analysisPath').onkeydown=e=>{if(e.key==='Enter'){e.preventDefault();load('path');}};
  el('analysisFile').onchange=()=>{saveDraft();const f=file();el('analysisFileInfo').textContent=f?.path||'';renderTool();};
  const jobNodes=new Map();
  function renderJob(job){
    let node=jobNodes.get(job.id);
    if(!node){if(!jobNodes.size)el('analysisJobs').replaceChildren();node=document.createElement('article');node.className='analysis-job';el('analysisJobs').prepend(node);jobNodes.set(job.id,node);}
    const signature=JSON.stringify(job);if(node.dataset.signature===signature)return;node.dataset.signature=signature;node.replaceChildren();
    const heading=document.createElement('h3');heading.textContent=`${tools[job.kind]?.[0]||'Analysis'} · ${job.file_name||''} · ${job.status}`;node.append(heading);
    const status=document.createElement('p');status.textContent=job.error||job.message||'';status.setAttribute('role','status');node.append(status);
    if(['queued','running'].includes(job.status)){const progress=document.createElement('progress');progress.max=1;if(Number.isFinite(job.progress)&&job.progress>0)progress.value=job.progress>1?job.progress/100:job.progress;progress.setAttribute('aria-label','Analysis progress');node.append(progress);}
    if(job.params){const details=document.createElement('details');const summary=document.createElement('summary');summary.textContent='Parameters';const pre=document.createElement('pre');pre.textContent=JSON.stringify(job.params,null,2);details.append(summary,pre);node.append(details);}
    if(job.result){
      const summary=document.createElement('p');summary.textContent=Array.isArray(job.result.summary)?job.result.summary.join(' · '):typeof job.result.summary==='object'?JSON.stringify(job.result.summary):job.result.summary||'';node.append(summary);
      for(const plot of job.result.plots||[])node.append(drawPlot(plot));
      const download=document.createElement('button');download.textContent='Download result JSON';download.onclick=()=>{const url=URL.createObjectURL(new Blob([JSON.stringify(job.result,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download=`analysis-${job.id}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};node.append(download);
    }
  }
  function drawPlot(plot){
    const figure=document.createElement('figure'),caption=document.createElement('figcaption');caption.textContent=plot.title||'Analysis result';figure.append(caption);
    const canvas=document.createElement('canvas');canvas.width=1000;canvas.height=360;canvas.setAttribute('role','img');canvas.setAttribute('aria-label',`${plot.title}. ${plot.x_label||'x'} versus ${plot.y_label||'y'}. Download result JSON for values.`);figure.append(canvas);
    const ctx=canvas.getContext('2d'),left=75,top=20,width=900,height=280;
    ctx.fillStyle='#fafcfb';ctx.fillRect(0,0,1000,360);ctx.font='13px sans-serif';ctx.fillStyle='#23372e';ctx.fillText(plot.x_label||'',430,348);ctx.save();ctx.translate(17,220);ctx.rotate(-Math.PI/2);ctx.fillText(plot.y_label||'',0,0);ctx.restore();
    const palette=['#256d51','#397bb3','#ae6044','#8862aa'];
    if(plot.type==='heatmap'){
      const z=plot.z||[],cols=Math.max(1,...z.map(r=>r.length)),rows=z.length;
      const isSTTC=/STTC/i.test(plot.title||''),scale=isSTTC?1:Math.max(1,...z.flat().filter(Number.isFinite));
      z.forEach((row,i)=>row.forEach((v,j)=>{ctx.fillStyle=!Number.isFinite(v)?'#ddd':`hsl(${v<0?25:153} 45% ${96-Math.min(1,Math.abs(v)/scale)*62}%)`;ctx.fillRect(left+j*width/cols,top+i*height/rows,width/cols+.5,height/rows+.5);}));
      ctx.fillStyle='#23372e';ctx.fillText(isSTTC?'−1 orange · 0 white · +1 green · unavailable gray':`${/probability/i.test(plot.title||'')?'Probability':'Value'}: 0 white → ${scale} green`,left,top+height+23);
      if(plot.x?.length)ctx.fillText(`${plot.x[0]} … ${plot.x.at(-1)}`,left+width-180,top+height+23);
      if(plot.y?.length){ctx.fillText(String(plot.y[0]),25,top+12);ctx.fillText(String(plot.y.at(-1)),25,top+height);}
    }else{
      const series=plot.series?.length?plot.series:[{x:plot.x||[],y:plot.y||[]}];let xmin=Infinity,xmax=-Infinity,ymin=Infinity,ymax=-Infinity;
      for(const s of series)for(let i=0;i<(s.y||[]).length;i++){const x=Number(s.x?.[i]??i),y=s.y[i];if(y===null||!Number.isFinite(x)||!Number.isFinite(y))continue;xmin=Math.min(xmin,x);xmax=Math.max(xmax,x);ymin=Math.min(ymin,y);ymax=Math.max(ymax,y);}
      if(!Number.isFinite(xmin)){ctx.fillText('No events in this window.',left,100);return figure;}
      if(plot.type==='bar')ymin=Math.min(0,ymin);if(xmax===xmin)xmax=xmin+1;if(ymax===ymin)ymax=ymin+1;
      const px=x=>left+(x-xmin)/(xmax-xmin)*width,py=y=>top+height-(y-ymin)/(ymax-ymin)*height;
      ctx.strokeStyle='#b8c6bf';ctx.strokeRect(left,top,width,height);ctx.fillText(xmin.toFixed(1),left,325);ctx.fillText(xmax.toFixed(1),left+width-40,325);ctx.fillText(ymax.toFixed(2),20,top+10);ctx.fillText(ymin.toFixed(2),20,top+height);
      series.forEach((s,index)=>{ctx.strokeStyle=ctx.fillStyle=palette[index%palette.length];ctx.globalAlpha=series.length>12?.35:.85;ctx.beginPath();let started=false;for(let i=0;i<(s.y||[]).length;i++){const x=Number(s.x?.[i]??i),y=s.y[i];if(y===null||!Number.isFinite(x)||!Number.isFinite(y)){started=false;continue;}if(plot.type==='scatter')ctx.fillRect(px(x)-1,py(y)-2,2,4);else if(plot.type==='bar')ctx.fillRect(px(x),py(y),Math.max(1,width/s.y.length-1),py(0)-py(y));else{if(started)ctx.lineTo(px(x),py(y));else ctx.moveTo(px(x),py(y));started=true;}}ctx.stroke();});ctx.globalAlpha=1;
    }
    return figure;
  }
  async function refresh(){
    try{const data=await api({action:'status'});if(JSON.stringify(data.source_urls||{})!==JSON.stringify(sourceUrls)){sourceUrls=data.source_urls||{};saveDraft();renderTool();}for(const job of data.jobs||[])renderJob(job);if(data.workspace&&JSON.stringify(data.workspace)!==JSON.stringify(workspace)){saveDraft();renderWorkspace(data.workspace);}if(!(data.jobs||[]).some(j=>['queued','running'].includes(j.status))){clearInterval(poll);poll=null;}}
    catch(e){message(e.message);}
  }
  el('analysisForm').onsubmit=async e=>{
    e.preventDefault();saveDraft();const params={};for(const input of el('analysisParams').querySelectorAll('[name]'))if(input.value!=='')params[input.name]=input.multiple?[...input.selectedOptions].map(o=>o.value):input.type==='number'?Number(input.value):input.value;
    if(params.target_ids&&!params.target_ids.includes(file()?.id))params.target_ids.unshift(file()?.id);
    busy=true;renderTool();try{const result=await api({action:'run',kind:tool,file_id:file()?.id,params});renderJob(result.job||result);if(!poll)poll=setInterval(refresh,1000);await refresh();}catch(exc){message(exc.message);}finally{busy=false;renderTool();}
  };
  renderTool();refresh();browse();
})();
