// Click-only batch sorting, independent of experiment setup and acquisition.
(() => {
  views.sorting.insertAdjacentHTML('beforeend', `
    <p class="help">Upload raw recordings, choose a baseline and sorter, then sort your selected files together in one batch.</p>
    <section class="panel"><h2>1. Upload recordings</h2>
      <label>Raw Maxwell H5/HDF5 or NWB files (one or more)<input id="sortingUpload" type="file" accept=".h5,.hdf5,.nwb" multiple></label>
      <p class="help">Include your baseline here. Files are copied to the workshop server; each file may be up to 64 GiB.</p>
      <p id="sortingUploadStatus" role="status" aria-live="polite"></p>
      <div id="sortingFiles"></div>
    </section>
    <section class="panel"><h2>2. Choose baseline & sorter</h2>
      <form id="sortingForm"><div class="fields">
        <label>Baseline recording<select id="sortingBaseline" required></select></label>
        <label>Sorter<select id="sortingSorter" required></select></label>
      </div><p id="sortingSorterHelp" class="help"></p>
      <details><summary>Install sorter dependencies</summary>
        <p class="help">Installs packages into the Python environment running this workshop. Internet access and write access to that environment are required. Stop acquisition and analysis first. Restart the workshop afterward. GPU drivers, MATLAB and external sorter executables require separate setup.</p>
        <label>Install option<select id="sortingInstallPlan"></select></label>
        <p id="sortingInstallHelp" class="help"></p>
        <button id="sortingInstall" type="button" disabled>Install dependencies</button>
        <p id="sortingInstallStatus" role="status" aria-live="polite"></p>
        <pre id="sortingInstallLog" style="white-space:pre-wrap;max-height:240px;overflow:auto"></pre>
      </details>
      <details id="sortingRTOptions"><summary>RT-Sort options</summary><div class="fields">
        <label>Device<select id="sortingDevice"><option value="cuda">CUDA GPU</option><option value="cpu">CPU</option></select></label>
        <label>Baseline start (ms)<input id="sortingStart" type="number" value="0" min="0" step="any" required></label>
        <label>Baseline duration (ms)<input id="sortingDuration" type="number" value="60000" min="1" step="any" required></label>
      </div></details>
      <button id="sortingRun" type="submit" disabled>Sort selected recordings</button>
      <p id="sortingStatus" role="status" aria-live="polite"></p></form>
    </section>
    <section class="panel"><h2>3. Results</h2><div id="sortingJobs"><p class="help">No sorting jobs yet.</p></div></section>`);
  const el = id => document.getElementById(id);
  let files = [], sorters = [], jobs = [], loaded = false, uploading = false, submitting = false, refreshing = false, poll = null;
  let installation = null, installPlans = [];
  const selected = new Set(), jobNodes = new Map();
  const api = command => request('/api/sorting', command);
  function controls() {
    const sorter = sorters.find(s => s.id === el('sortingSorter').value);
    const rt = sorter?.id === 'rt-sort';
    el('sortingRTOptions').hidden = !rt;
    for (const id of ['sortingStart', 'sortingDuration', 'sortingDevice']) el(id).disabled = !rt;
    el('sortingSorterHelp').textContent = installation ? 'Restart the workshop after installation before sorting.' : !loaded ? 'Checking installed sorters…' : !sorter ? 'No sorter is available. Install the desired sorter on the workshop server, then refresh this page.' :
      !sorter.available ? sorter.reason : rt ? 'RT-Sort learns units from the baseline and applies them to every selected recording. All recordings must have matching electrode routing and sampling rate.' :
      'This SpikeInterface sorter fits each selected recording independently using its default parameters. The baseline selection is not used; unit IDs do not match across recordings.';
    const active = jobs.some(j => ['queued', 'running'].includes(j.status));
    el('sortingInstall').disabled = !installPlans.length || !!installation || active || submitting;
    el('sortingInstallPlan').disabled = !!installation || submitting;
    el('sortingRun').disabled = !!installation || uploading || submitting || active || !selected.size || !el('sortingBaseline').value || !sorter?.available;
    el('sortingUpload').disabled = uploading || submitting;
  }
  function renderFiles() {
    const previous = el('sortingBaseline').value;
    el('sortingBaseline').replaceChildren(...files.map(f => new Option(f.name, f.id)));
    if (files.some(f => f.id === previous)) el('sortingBaseline').value = previous;
    el('sortingFiles').replaceChildren(...files.map(file => {
      const label = document.createElement('label'); label.className = 'sorting-file';
      const input = document.createElement('input'); input.type = 'checkbox'; input.checked = selected.has(file.id);
      input.onchange = () => {if (input.checked) selected.add(file.id); else selected.delete(file.id); controls();};
      label.append(input, document.createTextNode(`${file.name} · ${(file.size / 1024**2).toFixed(1)} MiB`));
      return label;
    }));
    controls();
  }
  function renderJobs() {
    for (const job of jobs) {
      let node = jobNodes.get(job.id);
      if (!node) {
        if (!jobNodes.size) el('sortingJobs').replaceChildren();
        node = document.createElement('article'); node.className = 'analysis-job';
        el('sortingJobs').prepend(node); jobNodes.set(job.id, node);
      }
      const signature = JSON.stringify(job);
      if (node.dataset.signature === signature) continue;
      node.dataset.signature = signature; node.replaceChildren();
      const heading = document.createElement('h3'); heading.textContent = `${job.sorter} · ${job.targets.length} recording(s) · ${job.status}`;
      const status = document.createElement('p'); status.textContent = job.error || job.message; status.setAttribute('role', 'status');
      node.append(heading, status);
      if (['queued', 'running'].includes(job.status)) {
        const progress = document.createElement('progress'); progress.max = 1;
        if (job.progress > 0) progress.value = job.progress;
        progress.setAttribute('aria-label', 'Sorting batch progress'); node.append(progress);
      }
      if (job.result) {
        const location = document.createElement('p'); location.textContent = `Saved to ${job.result.output_dir}`; node.append(location);
        if (job.result.note) {const note = document.createElement('p'); note.textContent = job.result.note; node.append(note);}
        job.result.recordings.forEach((recording, index) => {
          const row = document.createElement('p'), download = document.createElement('a');
          download.textContent = `Download ${recording.name} spikes (${recording.unit_count} units, ${recording.spike_count} spikes)`;
          download.href = `/api/sorting/download?job=${encodeURIComponent(job.id)}&index=${index}&token=${encodeURIComponent(WORKSHOP_TOKEN)}`;
          download.download = `${recording.name}_spikes.npz`; row.append(download);
          const analyze = document.createElement('button'); analyze.type = 'button'; analyze.textContent = 'Open in Analysis';
          analyze.onclick = async () => {if (await window.workshopAnalysis.load('path', recording.spikes_path)) showView('analysis');};
          row.append(' ', analyze); node.append(row);
        });
        const manifest = document.createElement('button'); manifest.type = 'button'; manifest.textContent = 'Download batch manifest';
        manifest.onclick = () => {
          const url = URL.createObjectURL(new Blob([JSON.stringify(job, null, 2)], {type: 'application/json'}));
          const link = document.createElement('a'); link.href = url; link.download = `sorting-${job.id}.json`; link.click();
          setTimeout(() => URL.revokeObjectURL(url), 1000);
        }; node.append(manifest);
      }
    }
    controls();
  }
  async function refresh() {
    if (refreshing) return;
    refreshing = true;
    try {
      const data = await api({action: 'status'});
      const previous = el('sortingSorter').value;
      sorters = data.sorters; loaded = true;
      installation = data.installation || installation;
      installPlans = data.install_plans || [];
      const installChoice = el('sortingInstallPlan').value;
      el('sortingInstallPlan').replaceChildren(...installPlans.map(plan => new Option(plan.label, plan.id)));
      if (installPlans.some(plan => plan.id === installChoice)) el('sortingInstallPlan').value = installChoice;
      el('sortingInstallHelp').textContent = installPlans.find(plan => plan.id === el('sortingInstallPlan').value)?.description || '';
      if (installation) {
        el('sortingInstallStatus').textContent = installation.message;
        el('sortingInstallLog').textContent = installation.log || '';
      }
      el('sortingSorter').replaceChildren(...sorters.map(sorter => {
        const option = new Option(sorter.label + (sorter.available ? '' : ' (unavailable)'), sorter.id);
        option.disabled = !sorter.available; option.title = sorter.reason || ''; return option;
      }));
      el('sortingSorter').value = sorters.find(s => s.id === previous && s.available)?.id || sorters.find(s => s.available)?.id || sorters[0]?.id || '';
      for (const file of data.files) if (!files.some(f => f.id === file.id)) selected.add(file.id);
      // A status response may have started before an upload or submission completed.
      // The server never removes rows during this session, so retain newer local rows.
      files = [...new Map([...files, ...data.files].map(file => [file.id, file])).values()];
      jobs = [...new Map([...jobs, ...data.jobs].map(job => [job.id, job])).values()];
      renderFiles(); renderJobs();
      const active = installation?.status === 'running' || jobs.some(j => ['queued', 'running'].includes(j.status));
      if (active && !poll) poll = setInterval(refresh, 2000);
      if (!active && poll) {clearInterval(poll); poll = null;}
    } catch (error) {el('sortingStatus').textContent = error.message;}
    finally {refreshing = false;}
  }
  el('sortingUpload').onchange = async event => {
    const pending = [...event.target.files]; if (!pending.length) return;
    uploading = true; controls(); const failures = [];
    for (let index = 0; index < pending.length; index++) {
      const file = pending[index];
      el('sortingUploadStatus').textContent = `Uploading ${index + 1}/${pending.length}: ${file.name}…`;
      try {
        const response = await fetch('/api/upload/sorting', {method: 'POST', headers: {
          'Content-Type': 'application/octet-stream', 'X-Workshop-Token': WORKSHOP_TOKEN, 'X-File-Name': encodeURIComponent(file.name)
        }, body: file});
        const result = await response.json(); if (!response.ok) throw Error(result.error || 'Upload failed');
        if (!files.some(f => f.id === result.id)) files.push(result); selected.add(result.id); renderFiles();
      } catch (error) {failures.push(`${file.name}: ${error.message}`);}
    }
    el('sortingUploadStatus').textContent = failures.length ? failures.join(' · ') : `Uploaded ${pending.length} recording(s). Choose a baseline below; uncheck any files you do not want to sort.`;
    el('sortingUpload').value = ''; uploading = false; controls();
  };
  el('sortingInstallPlan').onchange = () => {
    el('sortingInstallHelp').textContent = installPlans.find(plan => plan.id === el('sortingInstallPlan').value)?.description || '';
  };
  el('sortingInstall').onclick = async () => {
    submitting = true; controls(); el('sortingInstallStatus').textContent = 'Starting installation…';
    try {
      installation = await api({action: 'install', plan: el('sortingInstallPlan').value});
      el('sortingInstallStatus').textContent = installation.message;
      if (!poll) poll = setInterval(refresh, 2000);
      await refresh();
    } catch (error) {el('sortingInstallStatus').textContent = error.message;}
    finally {submitting = false; controls();}
  };
  el('sortingSorter').onchange = controls; el('sortingBaseline').onchange = controls;
  el('sortingForm').onsubmit = async event => {
    event.preventDefault(); submitting = true; controls(); el('sortingStatus').textContent = 'Starting batch…';
    const sorter = el('sortingSorter').value;
    const params = sorter === 'rt-sort' ? {device: el('sortingDevice').value, baseline_start_ms: Number(el('sortingStart').value), baseline_duration_ms: Number(el('sortingDuration').value)} : {};
    try {
      const job = await api({action: 'run', sorter, baseline_id: el('sortingBaseline').value, target_ids: [...selected], params});
      jobs.push(job); renderJobs(); el('sortingStatus').textContent = 'Batch started. Progress and results appear below.';
      if (!poll) poll = setInterval(refresh, 2000); await refresh();
    } catch (error) {el('sortingStatus').textContent = error.message;}
    finally {submitting = false; controls();}
  };
  refresh();
})();
