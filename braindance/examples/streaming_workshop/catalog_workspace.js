// Catalog metadata and local experiment selection share the Analysis workspace.
(()=>{
  const root=views.catalog;
  root.insertAdjacentHTML('beforeend',`
    <p class="help">Browse recordings and saved experiments, then select one for analysis or playback.</p>
    <div class="catalog-summary">
      <section class="panel"><h2>Current experiment</h2><p id="catalogCurrentStatus" role="status"></p><p id="catalogCurrentPath" class="help"></p><div class="toolbar"><button id="catalogCurrent">Analyze current experiment</button><button id="catalogMonitor">Live recording</button></div></section>
      <section class="panel"><h2>Selected for analysis</h2><p id="catalogSelected">No experiment selected.</p><button id="catalogAnalysis">Open Analysis</button></section>
    </div>
    <section class="panel">
      <div class="toolbar"><h2>Experiments & recordings</h2><button id="catalogRefresh">Refresh catalog</button></div>
      <p class="help">Refresh reloads the configured CSV and discovers saved workshop runs and tutorial experiments. Add a parent folder to discover the experiments inside it; added paths are saved in the workshop index.</p>
      <form id="catalogAddForm" class="toolbar"><label class="analysis-path">Experiment or parent directory, experiment JSON, or recording<input id="catalogAddPath" placeholder="/path/to/experiment" required></label><button id="catalogAdd" type="submit">Add experiment</button></form>
      <div class="toolbar"><label class="analysis-path">Search catalog<input id="catalogSearch" type="search" placeholder="Experiment, project, chip, type or path"></label><label>Source<select id="catalogSource"><option value="">All sources</option><option>Configured catalog</option><option>Workshop run</option><option>Tutorial data</option><option>Added locally</option></select></label></div>
      <p id="catalogMessage" role="status" aria-live="polite"></p><p id="catalogWarnings" class="help"></p>
      <div class="catalog-table"><table><thead><tr><th>Experiment</th><th>Project / chip</th><th>Type / source</th><th>Actions</th></tr></thead><tbody id="catalogRows"></tbody></table></div>
      <div class="toolbar"><button id="catalogPrevious">Previous</button><span id="catalogCount"></span><button id="catalogNext">Next</button></div>
      <p id="catalogLocation" class="help"></p>
    </section>`);
  const el=id=>document.getElementById(id);
  let entries=[], selectedPath='', page=0, busy=false, loaded=false;
  function render(){
    const query=el('catalogSearch').value.toLowerCase(), source=el('catalogSource').value;
    const filtered=entries.filter(entry=>(!source||entry.source===source)&&[entry.name,entry.project,entry.chip,entry.kind,entry.path].join(' ').toLowerCase().includes(query));
    page=Math.min(page,Math.max(0,Math.ceil(filtered.length/100)-1));
    const rows=filtered.slice(page*100,(page+1)*100).map(entry=>{
      const row=document.createElement('tr');
      const name=document.createElement('td'), title=document.createElement('strong'), path=document.createElement('small');
      title.textContent=entry.name;path.textContent=entry.path||'No local path';name.append(title,path);
      const project=document.createElement('td');project.textContent=[entry.project,entry.chip].filter(Boolean).join(' / ')||'—';
      const source=document.createElement('td');source.textContent=[entry.kind,entry.source].filter(Boolean).join(' · ');
      const action=document.createElement('td'), button=document.createElement('button');
      button.textContent=selectedPath===entry.path?'Selected · open Analysis':'Select for analysis';button.disabled=!entry.available||busy;
      button.onclick=()=>select('path',entry.path);action.append(button);
      if(entry.available){
        const replay=document.createElement('button');replay.textContent='Load playback';replay.disabled=busy||running();
        replay.onclick=async()=>{
          if(await window.workshopPlayback.load(entry.path))showView('experiment');
        };action.append(replay);
      }
      if(!entry.available){const reason=document.createElement('small');reason.textContent=entry.reason;action.append(reason);}
      row.append(name,project,source,action);return row;
    });
    if(!rows.length){const row=document.createElement('tr'), cell=document.createElement('td');cell.colSpan=4;cell.textContent=entries.length?'No matching experiments.':'No experiments yet. Run an experiment or add a local path above.';row.append(cell);rows.push(row);}
    el('catalogRows').replaceChildren(...rows);
    el('catalogCount').textContent=`${filtered.length} entries${filtered.length?` · page ${page+1} of ${Math.ceil(filtered.length/100)}`:''}`;
    el('catalogPrevious').disabled=page===0;el('catalogNext').disabled=(page+1)*100>=filtered.length;
  }
  function current(){
    const snapshot=state||{};
    el('catalogCurrentStatus').textContent=[snapshot.status||'Idle',snapshot.phase,snapshot.execution?.source].filter(Boolean).join(' · ');
    el('catalogCurrentPath').textContent=snapshot.output||'No saved experiment output yet.';
    el('catalogCurrent').disabled=!snapshot.output||busy;
  }
  async function select(source,path){
    busy=true;render();current();el('catalogMessage').textContent='Loading experiment for analysis…';
    try{
      const ok=await window.workshopAnalysis.load(source,path);
      el('catalogMessage').textContent=ok?'Experiment selected for analysis.':'Could not select experiment. See the Analysis message for details.';
      showView('analysis');
    }finally{busy=false;render();current();}
  }
  async function refresh(command={action:'refresh'}){
    if(busy)return;
    busy=true;el('catalogRefresh').disabled=true;el('catalogAdd').disabled=true;render();current();
    el('catalogMessage').textContent=command.action==='add'?'Adding experiment…':'Refreshing catalog…';
    try{
      const data=await request('/api/catalog',command);entries=data.entries;loaded=true;
      el('catalogWarnings').textContent=data.warnings.join(' ');
      el('catalogLocation').textContent=`Catalog: ${data.catalog_path||'not configured'} · Local index: ${data.index_path}`;
      el('catalogMessage').textContent=`Refreshed ${new Date(data.refreshed_at).toLocaleTimeString()}. ${entries.length} entries.`;
      if(command.action==='add')el('catalogAddPath').value='';
    }catch(error){el('catalogMessage').textContent=error.message;}
    finally{busy=false;el('catalogRefresh').disabled=false;el('catalogAdd').disabled=false;render();current();}
  }
  window.addEventListener('workshop-analysis-selected',event=>{
    selectedPath=event.detail.path;el('catalogSelected').textContent=selectedPath||'No experiment selected.';render();
  });
  el('catalogRefresh').onclick=()=>refresh();
  el('catalogAddForm').onsubmit=event=>{event.preventDefault();refresh({action:'add',path:el('catalogAddPath').value});};
  el('catalogSearch').oninput=el('catalogSource').onchange=()=>{page=0;render();};
  el('catalogPrevious').onclick=()=>{page--;render();};el('catalogNext').onclick=()=>{page++;render();};
  el('catalogCurrent').onclick=()=>select('current');
  el('catalogMonitor').onclick=()=>showView('monitor');el('catalogAnalysis').onclick=()=>showView('analysis');
  new MutationObserver(()=>{if(!root.hidden){current();if(!loaded)refresh();}}).observe(root,{attributes:true,attributeFilter:['hidden']});
  // Reuse the workshop state poll; no second acquisition request loop.
  setInterval(()=>{if(!root.hidden&&!document.hidden)current();},1000);
  request('/api/analysis',{action:'status'}).then(data=>{if(!selectedPath&&data.workspace.path){selectedPath=data.workspace.path;el('catalogSelected').textContent=selectedPath;render();}}).catch(()=>{});
  render();current();if(!root.hidden)refresh();
})();
