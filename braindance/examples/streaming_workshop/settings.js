// Global defaults belong to the Python host, shared by every BrainDance session.
(() => {
  const panel = document.createElement('div');
  panel.className = 'global-settings';
  panel.innerHTML = `
    <p class="help">Global defaults for BrainDance on this computer. Restart BrainDance after saving to apply them everywhere. Explicit launch options and environment variables take precedence.</p>
    <form id="globalSettingsForm" class="panel">
      <h2>Data &amp; output</h2>
      <p class="help">Saved in <span id="globalConfigPath" class="file-path">~/.braindance/config.json</span>. Paths refer to the computer running BrainDance. Leave a path blank to use its default. Existing files are not moved.</p>
      <div id="globalSettingsFields" class="fields"></div>
      <div class="toolbar"><button id="saveGlobalSettings" class="primary" type="submit" disabled>Save settings</button><button id="reloadGlobalSettings" type="button">Reload saved settings</button></div>
      <p id="globalSettingsStatus" role="status" aria-live="polite"></p>
    </form>
    <section class="panel"><div class="row"><h2>Installed features</h2><button id="refreshGlobalFeatures" type="button">Refresh status</button></div>
      <p class="help">Green means dependencies were detected in the Python environment running BrainDance; red means dependencies or model files are missing. These checks do not test hardware, GPU support, package compatibility, or game ROMs.</p>
      <div id="globalFeatures" class="global-features" aria-live="polite"></div>
    </section>`;
  views.settings.append(panel);
  const fields = [
    ['data_dir', 'Data directory', 'Default: ~/braindance_data'],
    ['catalog_path', 'Catalog file', 'Default: catalog.csv inside the data directory'],
    ['output_dir', 'Output directory', 'Default: outputs inside the data directory'],
    ['auto_extract_spike_info', 'Automatic RT-Sort spike-info extraction', 'Extract spike_info automatically when loading RT-Sort results.'],
  ];
  for (const [key, title, help] of fields) {
    const field = document.createElement('div'); field.className = 'field full';
    const label = document.createElement('label'); label.htmlFor = 'global-' + key; label.textContent = title;
    const input = document.createElement(key === 'auto_extract_spike_info' ? 'select' : 'input');
    input.id = 'global-' + key;
    if (key === 'auto_extract_spike_info') {
      for (const [value, text] of [['', 'Default (disabled)'], ['true', 'Enabled'], ['false', 'Disabled']]) {
        const option = document.createElement('option'); option.value = value; option.textContent = text; input.append(option);
      }
    } else { input.type = 'text'; input.autocomplete = 'off'; input.spellcheck = false; }
    const hint = document.createElement('p'); hint.className = 'help'; hint.textContent = help;
    const effective = document.createElement('p'); effective.className = 'help settings-effective'; effective.id = 'effective-' + key;
    field.append(label, input, hint, effective); $('globalSettingsFields').append(field);
  }
  let busy = false;
  let loaded = false;
  function message(text, failed = false) {
    $('globalSettingsStatus').textContent = text;
    $('globalSettingsStatus').dataset.failed = String(failed);
  }
  function render(result, updateForm) {
    if (updateForm) {
      $('globalConfigPath').textContent = result.config_path;
      for (const [key] of fields) {
        const setting = result.settings[key], input = $('global-' + key);
        input.value = setting.saved == null ? '' : String(setting.saved);
        if (input.tagName === 'INPUT') input.placeholder = setting.effective;
        $('effective-' + key).textContent = `Currently effective: ${setting.effective}` + (setting.environment ? ` · Overridden by ${setting.environment}. Saved changes take effect when that override is removed.` : '');
      }
      loaded = true;
    }
    $('globalFeatures').replaceChildren(...result.features.map(feature => {
      const row = document.createElement('div'); row.className = 'global-feature'; row.dataset.feature = feature.id;
      const heading = document.createElement('strong'); heading.textContent = feature.label;
      const badge = document.createElement('span'); badge.className = 'feature-badge'; badge.dataset.available = String(feature.available);
      badge.textContent = feature.available ? '● Installed' : '● Missing';
      const detail = document.createElement('p'); detail.className = 'help'; detail.textContent = feature.detail;
      row.append(heading, badge, detail); return row;
    }));
  }
  async function request(save = false, updateForm = true) {
    if (busy) return;
    busy = true;
    for (const id of ['saveGlobalSettings', 'reloadGlobalSettings', 'refreshGlobalFeatures']) $(id).disabled = true;
    message(save ? 'Saving…' : 'Loading…');
    try {
      const options = {};
      if (save) {
        const settings = {};
        for (const [key] of fields) {
          const value = $('global-' + key).value.trim();
          settings[key] = value === '' ? null : key === 'auto_extract_spike_info' ? value === 'true' : value;
        }
        options.method = 'POST'; options.headers = {'Content-Type': 'application/json', 'X-Workshop-Token': WORKSHOP_TOKEN};
        options.body = JSON.stringify({settings});
      }
      const response = await fetch('/api/settings', options), result = await response.json();
      if (!response.ok) throw Error(result.error || 'Could not load settings');
      render(result, updateForm);
      message(save ? 'Settings saved. Restart BrainDance to apply the new defaults everywhere.' : updateForm ? 'Saved settings loaded.' : 'Feature status refreshed.');
    } catch (exc) { message(exc.message, true); }
    finally {
      busy = false;
      $('saveGlobalSettings').disabled = !loaded;
      $('reloadGlobalSettings').disabled = false;
      $('refreshGlobalFeatures').disabled = false;
    }
  }
  $('globalSettingsForm').onsubmit = event => { event.preventDefault(); request(true); };
  $('reloadGlobalSettings').onclick = () => request();
  $('refreshGlobalFeatures').onclick = () => request(false, false);
  request();
})();
