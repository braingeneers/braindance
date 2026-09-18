"""File-backed, bounded workshop analyses, independent of the live acquisition worker.

Spike times are milliseconds. Maxwell channel detections are explicitly not sorted
neurons. No pickle or participant Python is executed when opening an experiment.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import copy
import importlib.util
import json
import threading
import uuid

from .discovery import discover_experiments, recording_files

import numpy as np


def _number(params, key, default, low=0, high=1e9):
    value = float(params.get(key, default))
    if not np.isfinite(value) or not low <= value <= high:
        raise ValueError(f'{key} must be between {low} and {high}')
    return value


def _h5_info(path):
    import h5py
    with h5py.File(path, 'r') as handle:
        raw = next((key for key in ('sig', 'data_store/data0000/groups/routed/raw', 'recordings/rec0000/well000/groups/routed/raw') if key in handle), None)
        spikes = next((key for key in ('data_store/data0000/spikes', 'recordings/rec0000/well000/spikes', 'proc0/spikeTimes') if key in handle), None)
        frame_key = (raw.rsplit('/',1)[0] + '/frame_nos') if raw and raw != 'sig' else 'data_store/data0000/groups/routed/frame_nos'
        first = int(handle[frame_key][0]) if frame_key in handle and len(handle[frame_key]) else 0
        if raw == 'sig' and handle[raw].shape[0] >= 1028:
            first = int(handle[raw][-2, 0]) + (int(handle[raw][-1, 0]) << 16)
        return dict(raw_key=raw, spike_key=spikes, frames=int(handle[raw].shape[1]) if raw else None, first_frame=first)


def _raw(path, info, channel, start, stop):
    import h5py
    with h5py.File(path, 'r') as handle:
        data = handle[info['raw_key']]
        if channel < 0 or channel >= data.shape[0]:
            raise ValueError('Channel is outside this recording')
        return np.asarray(data[channel, start:stop], dtype=float)


class AnalysisWorkspace:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='workshop-analysis')
        self.lock = threading.RLock()
        self.workspace = dict(path='', name='', phases=[], files=[], warnings=[])
        self.jobs = {}

    def close(self):
        self.executor.shutdown(wait=False, cancel_futures=True)

    def control(self, command):
        action = command.get('action', 'status')
        if action == 'browse':
            from braindance.config import get_data_dir
            default = get_data_dir().expanduser().resolve()
            directory = Path(command.get('path') or default).expanduser().resolve()
            if directory.is_file():
                directory = directory.parent
            result = dict(path=str(directory), default_path=str(default),
                          parent=str(directory.parent), entries=[], warning='')
            if not directory.exists():
                result['warning'] = 'This folder does not exist. Choose another folder or edit the path.'
                return result
            entries = []
            for child in directory.iterdir():
                if child.name.startswith('.'):
                    continue
                if child.is_dir():
                    kind = 'experiment' if (child / 'experiment.json').is_file() or (child / 'experiment_log.json').is_file() else 'folder'
                elif child.suffix.lower() in ('.h5', '.hdf5', '.npz'):
                    kind = 'recording'
                elif child.name in ('experiment.json', 'experiment_log.json'):
                    kind = 'experiment file'
                else:
                    continue
                entries.append(dict(name=child.name, path=str(child), kind=kind,
                                    directory=child.is_dir()))
            result['entries'] = sorted(entries, key=lambda item: (not item['directory'], item['name'].casefold()))
            return result
        if action == 'load':
            with self.lock:
                if any(j['status'] in ('queued', 'running') for j in self.jobs.values()):
                    raise ValueError('Wait for the active analysis before loading another experiment')
            self._load(command.get('path', ''))
        elif action == 'run':
            with self.lock:
                if any(j['status'] in ('queued', 'running') for j in self.jobs.values()):
                    raise ValueError('An analysis is already running')
                row = next((f for f in self.workspace['files'] if f['id'] == command.get('file_id')), None)
                if row is None:
                    raise ValueError('Select a recording file')
                kind = command.get('kind')
                if not row['capabilities'].get(kind):
                    raise ValueError(row['reasons'].get(kind, 'Analysis is unavailable for this file'))
                job = dict(id=uuid.uuid4().hex, kind=kind, params=dict(command.get('params') or {}), file_name=row['name'], file_path=row['path'], status='queued', progress=0, message='Queued', result=None, error=None)
                self.jobs[job['id']] = job
                if len(self.jobs) > 50:
                    del self.jobs[next(iter(self.jobs))]
                self.executor.submit(self._worker, job['id'], copy.deepcopy(row), kind, dict(command.get('params') or {}))
                return copy.deepcopy(job)
        elif action != 'status':
            raise ValueError('Unknown analysis action')
        with self.lock:
            if command.get('job_id'):
                if command['job_id'] not in self.jobs:
                    raise ValueError('Unknown analysis job')
                return copy.deepcopy(self.jobs[command['job_id']])
            from .source_links import analysis_source_urls
            return dict(workspace=copy.deepcopy(self.workspace), jobs=copy.deepcopy(list(self.jobs.values())),
                        source_urls=analysis_source_urls())

    def _file(self, path, phase_id, bounds=None):
        path = Path(path).resolve()
        row = dict(id=uuid.uuid5(uuid.NAMESPACE_URL, str(path) + str(bounds)).hex, name=path.name,
                   path=str(path), phase_id=phase_id, kind=path.suffix.lstrip('.'), bounds=bounds,
                   capabilities={}, reasons={})
        raw = spikes = False
        try:
            if path.suffix in ('.h5', '.hdf5'):
                row['info'] = _h5_info(path)
                raw, spikes = bool(row['info']['raw_key']), bool(row['info']['spike_key'])
                row['unit_kind'] = 'channel detections (unsorted)'
            elif path.suffix == '.npz':
                with np.load(path, allow_pickle=False) as values:
                    spikes = {'times_ms', 'unit_ids', 'duration_ms'} <= set(values.files)
                    if 'source_path' in values:
                        source = Path(str(values['source_path'].item()))
                        if source.is_file():
                            row['source_path'] = str(source)
                            row['info'] = _h5_info(source)
                    if 'sampling_frequency_hz' in values:
                        row['sample_rate'] = float(values['sampling_frequency_hz'])
                    if spikes and 'info' not in row:
                        row['info'] = dict(first_frame=0, frames=int(float(values['duration_ms']) * row.get('sample_rate',20000) / 1000))
                row['unit_kind'] = 'sorted units'
        except (OSError, ValueError, KeyError) as exc:
            row['warning'] = str(exc)
        log_source = Path(row.get('source_path', path))
        stem = log_source.name.removesuffix('.raw.h5').removesuffix('.h5').removesuffix('.npz')
        logs = [log_source.with_name(stem + '_log.csv'), log_source.with_name(stem + '_stim_log.csv')]
        row['stim_log'] = str(next((p for p in logs if p.exists()), ''))
        if raw and not row['stim_log']:
            row['reasons']['overlap'] = 'Select a stimulation CSV log to run an aligned overlap'
        available = importlib.util.find_spec('spikelab') is not None
        row['capabilities'] = dict(raw=raw, spikes=spikes, sttc=spikes and available,
                                   latency=spikes and available, overlap=raw or spikes, sorting=raw and path.suffix == '.h5')
        for kind, enabled in row['capabilities'].items():
            if not enabled:
                row['reasons'][kind] = ('Install spikelab to enable this analysis' if kind in ('sttc','latency') and spikes else
                    'Requires raw data and a matching stimulation log' if kind == 'overlap' else 'Required data are not available in this file')
        try:
            from .analysis_sorting import sorting_capabilities
            sorting = sorting_capabilities()
            row['sorting'] = sorting
            if not sorting.get('available', False):
                row['capabilities']['sorting'] = False
                row['reasons']['sorting'] = sorting.get('reason', 'RT-Sort dependencies unavailable')
        except ImportError:
            row['capabilities']['sorting'] = False
            row['reasons']['sorting'] = 'RT-Sort integration unavailable'
        return row

    def _load(self, location):
        if not location:
            raise ValueError('Choose an experiment directory, experiment JSON, or recording file')
        selected = Path(location).expanduser().resolve()
        if not selected.exists():
            raise ValueError(f'Path does not exist: {selected}')
        if selected.is_dir():
            experiments = discover_experiments(selected)
            if len(experiments) > 1:
                raise ValueError(f'Found {len(experiments)} experiments. Add this folder in Catalog and select one experiment.')
            if experiments:
                selected = experiments[0]
        root = selected if selected.is_dir() else selected.parent
        from .analysis_sorting import sorting_capabilities
        state = dict(path=str(selected), name=root.name, phases=[], files=[], warnings=[], sorting=sorting_capabilities())
        from .native_catalog import native_catalog
        catalog = native_catalog()
        analysis_classes = {entry['class_name'] for entry in catalog.values() if entry['category'] == 'analysis'}
        config = {}
        if selected.suffix == '.json':
            config = json.loads(selected.read_text())
        elif (root / 'experiment.json').exists():
            config = json.loads((root / 'experiment.json').read_text())
        candidates = []
        for candidate in ([selected] if selected.is_file() and selected.suffix.lower() in ('.h5', '.hdf5', '.npz') else recording_files(root)):
            candidates.append(candidate)
            if len(candidates) > 1000:
                raise ValueError('More than 1000 files; select a specific experiment directory')
        assigned = set()
        log_path = root / 'experiment_log.json'
        if log_path.exists():
            log = json.loads(log_path.read_text())
            state['name'] = log.get('experiment_name', log.get('name', root.name))
            state['logs'] = [str(log_path)]
            for entry in log.get('phase_log', []):
                cls = entry.get('phase_class', entry.get('class', ''))
                if not entry.get('recording_dir') or (cls in analysis_classes or 'group' in cls.lower()):
                    continue
                directory = (root / entry['recording_dir']).resolve()
                phase = dict(id=str(entry.get('phase_idx', len(state['phases']))), name=entry.get('phase_name', directory.name), files=[], status=entry.get('status', entry.get('event', 'recorded')), duration_s=entry.get('duration'), logs=[str(p) for p in sorted(directory.glob('*.csv'))] + [str(p) for p in sorted(directory.glob('*metadata.json'))])
                for p in candidates:
                    if directory in p.parents and p not in assigned:
                        row = self._file(p, phase['id']); state['files'].append(row); phase['files'].append(row['id']); assigned.add(p)
                state['phases'].append(phase)
        for attempt in sorted(root.glob('attempts/*/*/attempt.json')):
            entry = json.loads(attempt.read_text())
            phase = dict(log_path=str(attempt), logs=[str(attempt)], id=str(attempt.relative_to(root)), name=f"{entry.get('phase', attempt.parent.parent.name)} · attempt {entry.get('attempt', 1)}", files=[], status=entry.get('status'))
            bounds = [entry.get('start_frame', -1), entry.get('end_frame', -1)]
            if bounds[0] >= -1 and bounds[1] > max(0, bounds[0]):
                for p in candidates:
                    if p.suffix in ('.h5','.hdf5'):
                        row = self._file(p, phase['id'], bounds); state['files'].append(row); phase['files'].append(row['id']); assigned.add(p)
            else:
                phase['reason'] = 'No completed recording frame interval available'
            state['phases'].append(phase)
        for spec in config.get('phases', config.get('phase_plan', [])):
            kind = str(spec.get('category', spec.get('type', ''))).lower()
            if catalog.get(spec.get('type'), {}).get('category') == 'analysis' or any(term in kind for term in ('analysis', 'sort', 'connectivity', 'footprint')):
                continue
            name = spec.get('id', spec.get('name', kind))
            if not any(phase['name'].split(' · ')[0] == name for phase in state['phases']):
                state['phases'].append(dict(id='configured:'+name, name=name, files=[], status='No recording matched'))
        for p in candidates:
            if p in assigned:
                continue
            phase_id = str(p.parent.relative_to(root))
            phase = next((v for v in state['phases'] if v['id'] == phase_id), None)
            if phase is None:
                phase = dict(id=phase_id, name=p.parent.name, files=[], status='files discovered')
                state['phases'].append(phase)
            row = self._file(p, phase_id); state['files'].append(row); phase['files'].append(row['id'])
        state.setdefault('logs', [])
        state['logs'].extend(str(p) for p in sorted(root.glob('*.csv')))
        if not state['files']:
            state['warnings'].append('No supported recording files found. Load the run output directory, or an HDF5 / sorted NPZ file.')
        with self.lock:
            self.workspace = state

    def _worker(self, job_id, row, kind, params):
        def progress(value, message):
            with self.lock:
                self.jobs[job_id].update(status='running', progress=float(value), message=message)
        try:
            progress(.02, 'Loading recording')
            result = self._analyze(row, kind, params, progress)
            with self.lock:
                self.jobs[job_id].update(status='completed', progress=1, message='Complete', result=result)
        except Exception as exc:
            with self.lock:
                self.jobs[job_id].update(status='failed', message='Analysis failed', error=str(exc))

    def _analyze(self, row, kind, params, progress):
        path = Path(row['path'])
        fs = _number(params, 'sample_rate', row.get('sample_rate',20000), 1, 1e6)
        start_ms = _number(params, 'start_ms', 0)
        duration = _number(params, 'duration_ms', 1000 if kind == 'raw' else 10000, .01, 3600000)
        info = row.get('info', {})
        offset, end_frame = 0, info.get('frames')
        if row.get('bounds'):
            offset = max(0, row['bounds'][0] + 1 - info.get('first_frame', 0))
            end_frame = min(end_frame or row['bounds'][1], row['bounds'][1] + 1 - info.get('first_frame',0))
        start = offset + int(start_ms * fs / 1000)
        stop = min(end_frame or 2**63, start + int(duration * fs / 1000))
        if kind in ('raw', 'overlap'):
            channel = int(_number(params, 'channel', 0, 0, 100000))
            if start >= stop:
                raise ValueError('Requested time window is outside this phase recording')
            if kind == 'raw':
                if stop-start > 2000000:
                    raise ValueError('Choose a raw preview of at most 2 million samples')
                values = _raw(path, info, channel, start, stop)
                step = max(1, int(np.ceil(len(values)/5000)))
                x = (np.arange(start, stop, step)-offset) * 1000/fs
                return dict(summary=[f'Channel {channel}; ADC values, sample rate {fs:g} Hz; display every {step} sample(s).'], plots=[dict(type='line', title='Raw voltage', x=x.tolist(), y=values[::step].tolist(), x_label='Phase time (ms)', y_label='ADC counts')])
            import pandas as pd
            from .analysis_evoked import analyze_evoked
            log_path = params.get('stim_log') or row['stim_log']
            if not log_path:
                raise ValueError('Provide the stimulation log CSV path')
            log = pd.read_csv(log_path)
            time_col = 'time_mod' if 'time_mod' in log else 'time'
            if 'replay_frame' in log:
                events = np.asarray(log['replay_frame'], dtype=float)-info.get('first_frame',0)
                time_col = 'replay_frame (recording frames)'
            elif time_col in log:
                events = np.asarray(log[time_col], dtype=float)*fs
            else:
                raise ValueError('Stimulation log needs replay_frame or time/time_mod in seconds')
            pre = _number(params,'pre_ms',20,0,1000); post = _number(params,'post_ms',200,.01,5000)
            padding = 3 if row['capabilities'].get('raw') else 0
            mask = np.isfinite(events) & (events-(pre+padding)*fs/1000>=start) & (events+(post+padding)*fs/1000+(2 if padding else 0)<=stop)
            indices = np.flatnonzero(mask)[:int(_number(params,'max_events',50,1,200))]
            if not len(indices):
                raise ValueError('No stimulation events with complete windows in this phase')
            patterns = log['stim_pattern'].astype(str) if 'stim_pattern' in log else log['stim_electrodes'].astype(str) if 'stim_electrodes' in log else pd.Series(['all stimuli']*len(log))
            arguments = {}
            if row['capabilities'].get('raw'):
                arguments['raw_reader'] = lambda ch, lo, hi: _raw(path,info,ch,lo,hi)
            elif path.suffix == '.npz':
                with np.load(path,allow_pickle=False) as values:
                    arguments['spike_times_ms'] = np.asarray(values['times_ms'],dtype=float)
                    arguments['unit_ids'] = np.asarray(values['unit_ids'],dtype=int)
            else:
                import h5py
                with h5py.File(path,'r') as handle:
                    data=np.asarray(handle[info['spike_key']])
                key = 'frameno' if 'frameno' in data.dtype.names else 'frame'
                arguments['spike_times_ms'] = (data[key].astype(float)-info.get('first_frame',0))*1000/fs
                arguments['unit_ids'] = data['channel'].astype(int)
            result = analyze_evoked(events[indices]*1000/fs,patterns.iloc[indices].tolist(),sampling_frequency=fs,params=params,progress=progress,**arguments)
            result['summary'].insert(0, f'{len(indices)} complete stimuli; aligned with {time_col}. Confirm log timing against artifacts for recorded wall-clock timestamps.')
            if not row['capabilities'].get('raw'):
                result['summary'].append(row.get('unit_kind','Spike events'))
            return result
        if kind == 'sorting':
            from .analysis_sorting import run_sorting
            baseline = params.get('baseline_path')
            files = {f['id']: f for f in self.workspace['files']}
            if not baseline and params.get('baseline_id') in files:
                baseline = files[params['baseline_id']]['path']
            if not baseline:
                raise ValueError('Select a baseline recording')
            targets = params.get('recording_paths') or [files[i]['path'] for i in params.get('target_ids', []) if i in files] or [str(path)]
            options = {k:v for k,v in params.items() if k not in ('baseline_path','baseline_id','recording_paths','target_ids')}
            result = run_sorting(baseline, targets, self.output_dir / 'analysis' / uuid.uuid4().hex, params=options, progress=progress)
            with self.lock:
                for output in result['recordings']:
                    sources = [f for f in self.workspace['files'] if f['path'] == output['recording_path']] or [row]
                    for source in sources:
                        added = self._file(output['spikes_path'], source['phase_id'], source.get('bounds'))
                        added['stim_log'] = source.get('stim_log','')
                        if source.get('bounds'):
                            added['info'] = source['info']
                        self.workspace['files'].append(added)
                        for phase in self.workspace['phases']:
                            if phase['id'] == source['phase_id']:
                                phase['files'].append(added['id'])
            result['summary'] = [f"Sorted {len(result['recordings'])} recording(s); portable spike files are now available for analysis.", 'Sorting uses whole target recordings. Phase boundaries are applied only to manual analyses.']
            result['plots'] = []
            return result
        if path.suffix == '.npz':
            with np.load(path,allow_pickle=False) as values:
                times=np.asarray(values['times_ms'],dtype=float); units=np.asarray(values['unit_ids'],dtype=int)
                full_duration=float(values['duration_ms'])
                all_units=np.asarray(values['all_unit_ids'],dtype=int) if 'all_unit_ids' in values else np.unique(units)
        else:
            # Read only event rows, avoiding voltage reads and retaining recorded channel IDs.
            import h5py
            with h5py.File(path,'r') as handle:
                data=np.asarray(handle[info['spike_key']])
            names=data.dtype.names or ()
            frame_name=next((n for n in ('frameno','frame') if n in names),None)
            if frame_name is None or 'channel' not in names:
                raise ValueError('Spike table needs structured frameno/frame and channel columns')
            times=(np.asarray(data[frame_name],dtype=float)-info.get('first_frame',0))*1000/fs
            units=np.asarray(data['channel'],dtype=int)
            full_duration=(info.get('frames') or 0)*1000/fs
            if not full_duration and len(times):
                full_duration=float(np.max(times))+1000/fs
            all_units=np.unique(units)
        if times.ndim != 1 or units.shape != times.shape or not np.all(np.isfinite(times)):
            raise ValueError('Invalid spike arrays')
        base=offset*1000/fs+start_ms
        end=min(full_duration,(end_frame*1000/fs if end_frame else full_duration),base+duration)
        if end<=base:
            raise ValueError('Requested spike window is outside this recording')
        mask=(times>=base)&(times<end); times=times[mask]-base; units=units[mask]
        ids=all_units; cap=int(_number(params,'max_units',64,1,256)); ids=ids[:cap]
        summary=[f'{row.get("unit_kind", "units")}; {base:g}–{end:g} ms; first {len(ids)} units (limit {cap}).']
        if path.suffix != '.npz' and not info.get('frames'):
            summary.append('No raw recording duration available: window ends at the last observed spike, not the acquisition end.')
        if not len(ids):
            raise ValueError('No spikes in the selected window')
        if kind=='spikes':
            mask=np.isin(units,ids); times=times[mask]; units=units[mask]
            count=len(times); step=max(1,int(np.ceil(count/20000)))
            summary.append(f'{count} spikes; display every {step} event(s).')
            return dict(summary=summary,unit_ids=ids.tolist(),plots=[dict(type='scatter',title='Spike raster',x=(times[::step]+start_ms).tolist(),y=units[::step].tolist(),x_label='Phase time (ms)',y_label='Unit / channel')])
        from spikelab import SpikeData
        sd=SpikeData([np.sort(times[units==unit]) for unit in ids],length=end-base)
        progress(.3,'Computing spike analysis')
        if kind=='sttc':
            delta=_number(params,'delta_ms',20,.001,10000)
            matrix=np.asarray(sd.spike_time_tilings(delt=delta).matrix)
            summary.append('STTC measures association, not causal connectivity.')
            plot=dict(type='heatmap',title=f'STTC (±{delta:g} ms)',x=ids.tolist(),y=ids.tolist(),z=np.where(np.isfinite(matrix),matrix,None).tolist(),x_label='Unit / channel',y_label='Unit / channel')
        elif kind=='latency':
            unit=int(params.get('unit_id',int(ids[0])))
            if unit not in ids:
                raise ValueError(f'Select a unit in this window: {ids.tolist()}')
            window=_number(params,'window_ms',100,.01,10000); width=_number(params,'bin_ms',2,.01,1000)
            if 2*window/width>2000:
                raise ValueError('Choose at most 2000 latency bins')
            if not len(sd.train[int(np.flatnonzero(ids==unit)[0])]):
                raise ValueError('Selected reference unit has no spikes in this window')
            values=sd.latencies_to_index(int(np.flatnonzero(ids==unit)[0]),window_ms=window)
            bins=np.linspace(-window,window,int(np.ceil(2*window/width))+1)
            other=[i for i,u in enumerate(ids) if u!=unit]
            counts=[np.histogram(values[i][np.isfinite(values[i])],bins=bins)[0].tolist() for i in other]
            summary.append('Signed nearest-spike latencies; positive means target follows reference. No causality or significance inferred.')
            plot=dict(type='heatmap',title=f'Nearest-spike latency from {unit}',x=((bins[:-1]+bins[1:])/2).tolist(),y=ids[other].tolist(),z=counts,x_label='Latency (ms)',y_label='Target unit / channel')
        else:
            raise ValueError('Unknown analysis kind')
        return dict(summary=summary,unit_ids=ids.tolist(),plots=[plot])
