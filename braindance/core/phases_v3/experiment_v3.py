"""
Streamlined Experiment Framework V3

A middle-ground approach between the simple PhaseManager and complex Experiment class.
"""
import json
import hashlib
import os
import pickle
import re
import tempfile
import time
import datetime
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from .phase_base_v3 import PhaseV3, AnalysisPhaseV3, PhaseGroup, PhaseValidator, ValidationError
from .data_context import DataContext, DataDependencyTracker


class Experiment:
    """
    Streamlined experiment manager with automatic data passing and smart params.
    
    Supports a hierarchical directory layout where each recording-producing phase
    gets its own auto-incremented subdirectory under the experiment folder::
    
        base_dir/project_id/chip_id/experiment_name/   (when project/chip given)
            ├── RT_sort.pkl
            ├── experiment_summary.json
            ├── 001_rec/
            │   ├── 001_metadata.json
            │   ├── 001.raw.h5, 001_log.csv
            │   └── results/          (DataContext data)
            └── 002_closed_loop/
                └── ...
    
    Example:
        exp = (Experiment("my_experiment", project_id="proj1", chip_id="chipA")
               .add_phase(RecordPhase(duration=300))
               .add_phase(RTSortPhase())
               .add_phase(ConnectivityPhase())
               .run())
    """
    
    def __init__(self, name: str, params: Union[str, dict] = None, save_dir: str = None,
                 project_id: str = None, chip_id: str = None, base_dir: str = None,
                 results_dir_name: str = "results",
                 auto_load_data: bool = True, overwrite_existing: bool = False):
        """
        Initialize experiment.
        
        Args:
            name: Experiment name (also used as experiment directory name)
            params: Path to JSON params or params dict
            save_dir: Explicit directory for saving results (overrides project/chip path)
            project_id: Project identifier; when combined with chip_id, builds the
                path ``base_dir / project_id / chip_id / name``
            chip_id: Chip identifier (requires project_id)
            base_dir: Root data directory. Defaults to ``braindance.config.get_data_dir()``
            results_dir_name: Name of the results subdirectory inside each recording
                folder (default ``"results"``)
            auto_load_data: If True, automatically load existing data
            overwrite_existing: If True, allow overwriting existing data (default: False)
        """
        self.name = name
        self.project_id = project_id
        self.chip_id = chip_id
        self.results_dir_name = results_dir_name
        self.chip_dir: Optional[Path] = None

        # --- Resolve save_dir ---
        if save_dir is not None:
            self.save_dir = Path(save_dir)
        elif project_id and chip_id:
            if base_dir is None:
                from braindance.config import get_data_dir
                base_dir = str(get_data_dir())
            root = Path(base_dir)
            self.chip_dir = root / project_id / chip_id
            self.save_dir = self.chip_dir / name
        else:
            if base_dir is None:
                from braindance.config import get_data_dir
                base_dir = str(get_data_dir())
            self.save_dir = Path(base_dir) / name

        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Write chip metadata stub when project/chip hierarchy is used
        if self.chip_dir is not None:
            self._save_chip_metadata()
        
        # Core components
        self.params = {}
        self.data = DataContext(overwrite_existing=overwrite_existing)
        self.phases: List[Union[PhaseV3, PhaseGroup]] = []
        self.results: List[Dict] = []
        self.mapping = None
        
        # State tracking
        self.current_env = None
        self.current_phase_idx = 0
        self.current_recording_dir: Optional[Path] = None
        self.current_recording_count: Optional[str] = None
        self.start_time = None
        self.verbose = True
        
        # Data tracking
        self.data_tracker = DataDependencyTracker()
        self._phase_output_keys = set()
        
        # Metadata
        self.metadata = {
            "created": datetime.datetime.now().isoformat(),
            "name": name,
            "version": "3.0",
            "project_id": project_id,
            "chip_id": chip_id,
            "completed_phases": []
        }
        
        # Load initial params if provided
        if params:
            self.load_params(params)
        
        # Auto-load existing data if requested and available
        self._auto_loaded_keys = set()
        if auto_load_data:
            self._auto_load_existing_data()
            self._auto_loaded_keys = set(self.data.keys())
    
    def _auto_load_existing_data(self):
        """
        Automatically load existing data from the most recent recording's
        results directory, or fall back to the legacy ``save_dir/data`` path.
        """
        # Try recording-level results directories (most recent first)
        recording_dirs = self._list_recording_dirs()
        for rec_dir in reversed(recording_dirs):
            results_dir = rec_dir / self.results_dir_name
            if results_dir.exists():
                try:
                    self._try_load_from_dir(results_dir)
                    self.current_recording_dir = rec_dir
                    return
                except Exception as e:
                    if self.verbose:
                        print(f"⚠️  Warning: Could not load data from {results_dir}: {e}")

        # Fallback: save_dir/results (or legacy save_dir/data)
        for fallback_name in (self.results_dir_name, 'data'):
            fallback_dir = self.save_dir / fallback_name
            if fallback_dir.exists():
                try:
                    self._try_load_from_dir(fallback_dir)
                    return
                except Exception as e:
                    if self.verbose:
                        print(f"⚠️  Warning: Could not load existing data from {fallback_dir}: {e}")

    def _try_load_from_dir(self, data_dir: Path):
        """Attempt to load DataContext data from a directory."""
        index_file = data_dir / 'data_index.json'
        if index_file.exists():
            with open(index_file, 'r') as f:
                index_data = json.load(f)
            saved_files = index_data.get('saved_files', {})

            if saved_files and self.verbose:
                print(f"📂 Found existing data in {data_dir}")
                print(f"   Available data: {list(saved_files.keys())}")

            self.load_data(data_dir)

            if self.verbose and saved_files:
                print(f"✅ Loaded {len(saved_files)} data items from previous experiment")

        elif any(data_dir.glob('*.pkl')) or any(data_dir.glob('*.npy')) or any(data_dir.glob('*.npz')):
            if self.verbose:
                print(f"📂 Found existing data files in {data_dir} (no index)")
            self.load_data(data_dir)
            if self.verbose:
                print(f"✅ Loaded existing data from previous experiment")

    # ========== Checkpointing ==========

    def _get_checkpoint(self) -> int:
        """Read ``experiment_log.json`` and return the phase index to resume from.

        Return the first phase without a current successful checkpoint. Refuse
        corrupt/legacy logs, changed phase identities, incomplete groups, or
        missing/changed persisted outputs rather than silently restarting.
        """
        log_path = self.save_dir / 'experiment_log.json'
        if not log_path.exists():
            return 0

        try:
            with open(log_path, 'r') as f:
                log = json.load(f)
        except Exception as e:
            raise ValueError("Cannot resume: experiment_log.json is unreadable; "
                             "preserve it and review the saved run before restarting.") from e
        if log.get('checkpoint_schema_version') != 1:
            raise ValueError("Cannot resume: legacy experiment log has no verified "
                             "output checkpoints. Review or migrate it explicitly.")

        latest = {}
        for entry in log.get('phase_log', []):
            index = entry.get('phase_idx')
            if not isinstance(index, int) or not 0 <= index < len(self.phases):
                raise ValueError("Cannot resume: logged phase indices do not match this pipeline.")
            if entry.get('checkpoint_boundary', True):
                phase = self.phases[index]
                if entry.get('phase_name') != phase.name or entry.get('phase_class') != type(phase).__name__:
                    raise ValueError(f"Cannot resume: phase {index} differs from the saved pipeline.")
                if isinstance(phase, PhaseGroup):
                    children = [{'name': child.name, 'class': type(child).__name__} for child in phase]
                    if entry.get('group_phases') != children:
                        raise ValueError(f"Cannot resume: group {index} differs from the saved pipeline.")
                latest[index] = entry
        resume_from = 0
        while latest.get(resume_from, {}).get('success'):
            resume_from += 1
        if any(index > resume_from and entry.get('success') for index, entry in latest.items()):
            raise ValueError("Cannot resume: successful checkpoints are not contiguous.")
        if resume_from < len(self.phases) and isinstance(self.phases[resume_from], PhaseGroup):
            if any(entry.get('phase_idx') == resume_from for entry in log.get('phase_log', [])):
                raise ValueError("Cannot resume an interrupted PhaseGroup automatically. "
                                 "Its partial shared data and recording require explicit review.")

        # A later completed phase may intentionally replace a key. Verify the
        # latest persisted version of each file/key, without changing the data
        # context's existing overwrite policy.
        outputs = {}
        for entry in log.get('phase_log', []):
            if entry.get('success') and entry['phase_idx'] < resume_from:
                persisted = entry.get('persisted_outputs')
                provided = entry.get('provided_keys')
                if (not isinstance(persisted, list) or not isinstance(provided, list)
                        or not all(isinstance(output, dict) and isinstance(output.get('key'), str)
                                   for output in persisted)
                        or len(persisted) != len(provided)
                        or {output['key'] for output in persisted} != set(provided)):
                    raise ValueError("Cannot resume: successful phase has incomplete output checkpoint metadata.")
                for output in persisted:
                    if not all(field in output for field in ('path', 'format', 'sha256')):
                        raise ValueError("Cannot resume: saved output descriptor is incomplete.")
                    identity = (output['path'], output['key'] if output['format'] == 'json' else None)
                    outputs[identity] = output
        for output in outputs.values():
            path = self.save_dir / output['path']
            try:
                digest = self._output_digest(path, output['format'], output['key'])
            except Exception as e:
                raise ValueError(f"Cannot resume: saved output {output['key']!r} is unavailable or unreadable.") from e
            if digest != output['sha256']:
                raise ValueError(f"Cannot resume: saved output {output['key']!r} has changed.")
        if self.verbose and resume_from > 0:
            print(f"📌 Checkpoint found: {resume_from} phases already completed, "
                  f"resuming from phase {resume_from}")
        return resume_from

    @staticmethod
    def _output_digest(path: Path, file_format: str, key: str) -> str:
        if file_format == 'json':
            with path.open('r', encoding='utf-8') as handle:
                value = json.load(handle)[key]
            return hashlib.sha256(json.dumps(value, sort_keys=True).encode('utf-8')).hexdigest()
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(chunk)
        return digest.hexdigest()

    def _persist_phase_outputs(self, result: Dict) -> List[Dict]:
        if not result:
            return []
        data_dir = (self.current_recording_dir or self.save_dir) / self.results_dir_name
        self.data.save(data_dir, keys=list(result), overwrite_files=True, strict=True)
        with (data_dir / 'data_index.json').open('r', encoding='utf-8') as handle:
            saved_files = json.load(handle)['saved_files']
        outputs = []
        for key in result:
            file_format = saved_files[key]
            filename = ('results.json' if file_format == 'json' else
                        f"{key}.npy" if file_format == 'numpy' else
                        f"{key}.npz" if file_format == 'numpy_dict' else f"{key}.pkl")
            path = data_dir / filename
            outputs.append({'key': key, 'format': file_format,
                            'path': str(path.relative_to(self.save_dir)),
                            'sha256': self._output_digest(path, file_format, key)})
        return outputs

    def _read_checkpoint_output(self, output: Dict) -> Any:
        """Read the recorded artifact directly; never consult data_index.json.

        Verification is repeated on the same open file used for deserialization,
        so disappearing/changed artifacts cannot become silently missing keys
        between checkpoint selection and restoration.
        """
        path = self.save_dir / output['path']
        key = output['key']
        file_format = output['format']
        try:
            if file_format == 'json':
                with path.open('r', encoding='utf-8') as handle:
                    value = json.load(handle)[key]
                digest = hashlib.sha256(json.dumps(value, sort_keys=True).encode('utf-8')).hexdigest()
                if digest != output['sha256']:
                    raise ValueError('serialized value changed')
                return value
            with path.open('rb') as handle:
                digest = hashlib.sha256()
                for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                    digest.update(chunk)
                if digest.hexdigest() != output['sha256']:
                    raise ValueError('serialized file changed')
                handle.seek(0)
                if file_format == 'numpy':
                    import numpy as np
                    return np.load(handle, allow_pickle=True)
                if file_format == 'numpy_dict':
                    import numpy as np
                    with np.load(handle, allow_pickle=True) as archive:
                        return {name: archive[name] for name in archive.files}
                if file_format in ('pandas', 'pickle', 'rt_sort'):
                    return pickle.load(handle)
                raise ValueError(f'unsupported checkpoint format: {file_format}')
        except Exception as e:
            raise ValueError(f"Cannot resume: verified output {key!r} could not be restored.") from e

    def _load_all_recording_data(self, completed_before: Optional[int] = None):
        """Load DataContext data from *every* recording's results/ folder.

        This is used when resuming so that the full accumulated state is
        restored regardless of which recording directory each key was saved in.
        """
        if completed_before is not None:
            # Constructor auto-loading can include a phase's files written just
            # before a crash. Restore only committed keys when resuming; retain
            # explicit parameters and new caller-supplied inputs. This does not
            # change normal DataContext overwrite behavior.
            restore_excluded = self._auto_loaded_keys | self._phase_output_keys
            explicit_inputs = {key: self.data.get(key) for key in self.data.keys()
                               if key not in restore_excluded}
            self.data = DataContext(overwrite_existing=self.data.get_overwrite_policy())
            self.data.update({key: value for key, value in self.params.items() if key != 'maxwell_env'})
            self.data.update(explicit_inputs)
            self.current_recording_dir = None
            log_path = self.save_dir / 'experiment_log.json'
            prior = {}
            if log_path.exists():
                with log_path.open('r', encoding='utf-8') as handle:
                    prior = json.load(handle)
            restore_outputs = {}
            for entry in prior.get('phase_log', []):
                if entry.get('success') and entry['phase_idx'] < completed_before:
                    for output in entry.get('persisted_outputs', []):
                        restore_outputs[output['key']] = output
                    if entry.get('recording_dir'):
                        self.current_recording_dir = self.save_dir / entry['recording_dir']
            for output in restore_outputs.values():
                value = self._read_checkpoint_output(output)
                self.data.force_set(output['key'], value)
            return

        recording_dirs = self._list_recording_dirs()
        loaded_any = False
        # Analysis-only phases also persist outputs at experiment level.
        fallback_dir = self.save_dir / self.results_dir_name
        if fallback_dir.exists():
            self.data.load(fallback_dir)
        for rec_dir in recording_dirs:
            results_dir = rec_dir / self.results_dir_name
            if results_dir.exists():
                try:
                    self.data.load(results_dir)
                    self.current_recording_dir = rec_dir
                    loaded_any = True
                except Exception as e:
                    if self.verbose:
                        print(f"⚠️  Warning: Could not load data from {results_dir}: {e}")

        if loaded_any and self.verbose:
            print(f"✅ Restored {len(self.data.keys())} data keys from "
                  f"{len(recording_dirs)} recording(s)")

    # ========== Recording Directory Management ==========

    def _list_recording_dirs(self) -> List[Path]:
        """Return existing recording subdirectories sorted by name."""
        if not self.save_dir.exists():
            return []
        pattern = re.compile(r'^\d{3,}_.+$')
        dirs = [
            d for d in sorted(self.save_dir.iterdir())
            if d.is_dir() and pattern.match(d.name)
        ]
        return dirs

    def _get_next_recording_id(self, tag: str = "rec") -> str:
        """Return the next auto-incremented recording directory name for *tag*.

        Scans ``save_dir`` for all existing ``NNN_*`` directories (across all
        tags) and returns ``{N+1:03d}_{tag}``, so counts are globally
        sequential regardless of tag.
        """
        max_n = 0
        any_rec_pattern = re.compile(r'^(\d+)_.+$')
        if self.save_dir.exists():
            for d in self.save_dir.iterdir():
                if d.is_dir():
                    m = any_rec_pattern.match(d.name)
                    if m:
                        max_n = max(max_n, int(m.group(1)))

        return f"{max_n + 1:03d}_{tag}"

    def _create_recording_dir(self, tag: str = "rec") -> Path:
        """Create the next recording subdirectory and set it as current.

        Returns the newly created directory path.
        """
        rec_id = self._get_next_recording_id(tag)
        rec_dir = self.save_dir / rec_id
        rec_dir.mkdir(parents=True, exist_ok=True)
        self.current_recording_dir = rec_dir
        self.current_recording_count = rec_id.split("_")[0]

        if self.verbose:
            print(f"   📁 Created recording directory: {rec_dir}")

        return rec_dir

    # ========== Metadata Helpers ==========

    def _save_chip_metadata(self):
        """Write a stub ``chip_metadata.json`` at the chip directory level.

        No-op if ``chip_dir`` is None or the file already exists.
        """
        if self.chip_dir is None:
            return

        self.chip_dir.mkdir(parents=True, exist_ok=True)
        meta_path = self.chip_dir / 'chip_metadata.json'

        if meta_path.exists():
            return

        metadata = {
            'chip_id': self.chip_id,
            'project_id': self.project_id,
            'created': datetime.datetime.now().isoformat()
        }

        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def _save_recording_metadata(self, phase: PhaseV3, recording_dir: Path,
                                  duration: float = None):
        """Write ``[count]_metadata.json`` inside a recording directory."""
        count = self.current_recording_count or "rec"
        meta_path = recording_dir / f'{count}_metadata.json'
        metadata = {
            'recording_id': recording_dir.name,
            'experiment': self.name,
            'phase_name': phase.name,
            'phase_class': phase.__class__.__name__,
            'created': datetime.datetime.now().isoformat(),
            'duration': duration,
            'project_id': self.project_id,
            'chip_id': self.chip_id
        }

        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    # ========== Configuration Methods ==========
    
    def load_params(self, params: Union[str, dict, Path]) -> 'Experiment':
        """Load params from file or dict. Returns self for chaining."""
        if isinstance(params, (str, Path)):
            with open(params, 'r') as f:
                self.params = json.load(f)
        else:
            self.params = params.copy()
        # Configuration values can satisfy phase requirements just like values
        # produced by earlier phases. Environment construction settings stay
        # private because they can contain paths and backend objects.
        self.data.update({
            key: value for key, value in self.params.items()
            if key != 'maxwell_env'
        })
        return self
    
    def set_param(self, key: str, value: Any) -> 'Experiment':
        """Set a param value. Returns self for chaining."""
        self.params[key] = value
        return self
    
    def get_param(self, key: str, default: Any = None) -> Any:
        """Get a param value."""
        return self.params.get(key, default)
    
    def load_mapping(self, mapping_file: Union[str, Path, None] = None) -> 'Experiment':
        """Load electrode mapping from file. Returns self for chaining."""
        from braindance.analysis.mapping import Mapping
        if mapping_file is None:
            # load from the recording_file
            self.mapping = Mapping.from_maxwell(self.data.recording_file)
        else:
            self.mapping = Mapping(filepath=mapping_file)

        self.data.mapping = self.mapping
        return self
    
    # ========== Phase Management ==========
    
    def add_phase(self, phase: Union[PhaseV3, PhaseGroup]) -> 'Experiment':
        """Add a phase or phase group. Returns self for chaining."""
        self.phases.append(phase)
        
        # Register with data tracker
        if isinstance(phase, PhaseGroup):
            for p in phase:
                self.data_tracker.add_phase(p.name, p.inputs, p.outputs)
        else:
            self.data_tracker.add_phase(phase.name, phase.inputs, phase.outputs)
        
        return self
    
    def add_phase_group(self, phases: List[PhaseV3], name: str = None) -> 'Experiment':
        """Add a group of phases that share an environment. Returns self for chaining."""
        group = PhaseGroup(phases, name)
        return self.add_phase(group)
    
    def add_phases(self, *phases: Union[PhaseV3, PhaseGroup]) -> 'Experiment':
        """Add multiple phases at once. Returns self for chaining."""
        for phase in phases:
            self.add_phase(phase)
        return self
    
    # ========== Execution Methods ==========
    
    def run(self, start_from: int = 0, stop_at: int = None, 
            validate: bool = True, resume: bool = False) -> bool:
        """
        Run experiment phases.
        
        Args:
            start_from: Phase index to start from (ignored when *resume* is True)
            stop_at: Phase index to stop at (exclusive)
            validate: Whether to validate phase dependencies
            resume: If True, read ``experiment_log.json`` to find the last
                successful phase and continue from there.  All previously-saved
                recording data is reloaded into the DataContext automatically.
            
        Returns:
            True if all phases completed successfully
        """
        if not self.phases:
            print("❌ No phases to run!")
            return False

        # --- Resume from checkpoint ---
        if resume:
            try:
                start_from = self._get_checkpoint()
                self._load_all_recording_data(completed_before=start_from)
            except ValueError as e:
                print(f"❌ {e}")
                return False
            if start_from > 0:
                # Preload prior phase_log entries so incremental log writes
                # during this run don't clobber the record of skipped phases.
                # Without this, _save_experiment_log() overwrites the log
                # with only the phases that ran *this* process, which makes
                # the next resume think nothing has been completed.
                log_path = self.save_dir / 'experiment_log.json'
                if log_path.exists():
                    try:
                        with open(log_path, 'r') as f:
                            prior = json.load(f)
                        prior_log = prior.get('phase_log', [])
                        if prior_log:
                            self.results = list(prior_log)
                            self.metadata['completed_phases'] = list(prior_log)
                    except Exception as e:
                        if self.verbose:
                            print(f"⚠️  Could not preload prior phase_log: {e}")
            if start_from >= len(self.phases):
                print("✅ All phases already completed in previous run.")
                return True
        
        # Validate dependencies
        if validate:
            try:
                PhaseValidator.validate_pipeline(self.phases, existing_data=self.data)
                if self.verbose:
                    existing_keys = list(self.data.keys()) if self.data.keys() else []
                    if existing_keys:
                        print(f"✅ Phase dependencies validated (with existing data: {existing_keys})")
                    else:
                        print("✅ Phase dependencies validated")
            except ValidationError as e:
                print(f"❌ Validation error: {e}")
                return False
        
        self.start_time = datetime.datetime.now()
        self.current_phase_idx = start_from
        stop_at = stop_at or len(self.phases)
        
        if self.verbose:
            print(f"\n🧪 Starting experiment: {self.name}")
            print(f"📁 Save directory: {self.save_dir}")
            if start_from > 0:
                print(f"🔄 Resuming from phase {start_from} (phases 0-{start_from-1} already done)")
            else:
                print(f"🔄 Running phases {start_from} to {stop_at-1}")
            print("=" * 50)
        
        success = True
        
        for i in range(start_from, min(stop_at, len(self.phases))):
            phase_or_group = self.phases[i]
            
            # Handle phase groups
            if isinstance(phase_or_group, PhaseGroup):
                success = self._run_phase_group(phase_or_group, i)
            else:
                success = self._run_single_phase(phase_or_group, i)
            
            if not success:
                break
            
            self.current_phase_idx = i + 1
        
        # Final cleanup
        self._cleanup_experiment()
        
        # Save summary
        self.save_summary()
        
        if self.verbose:
            total_time = (datetime.datetime.now() - self.start_time).total_seconds()
            print("\n" + "=" * 50)
            print(f"🏁 Experiment {'completed' if success else 'failed'} in {total_time:.1f}s")
            print(f"📊 Phases completed: {self.current_phase_idx}/{len(self.phases)}")
        
        return success
    
    def _execute_phase(self, phase):
        """Execute scientific work; runners may wrap this for controls/observation."""
        return phase.run(self)

    def _run_single_phase(self, phase: PhaseV3, phase_idx: int,
                          checkpoint_boundary: bool = True) -> bool:
        """Run a single phase."""
        if self.verbose:
            print(f"\n⚡ Phase {phase_idx+1}/{len(self.phases)}: {phase.name}")
        
        try:
            phase_start = time.perf_counter()
            phase.start_time = phase_start
            try:
                self._setup_phase(phase)
                phase.validate_requirements(self.data)
                result = self._execute_phase(phase)
                if result:
                    self._phase_output_keys.update(result)
                    self.data.update(result)
                    for key in result:
                        self.data_tracker.record_access(phase.name, key, 'write')
            finally:
                # Recording close/flush is part of completion, not an action
                # performed after the success checkpoint has been published.
                self._cleanup_phase(phase)

            persisted_outputs = self._persist_phase_outputs(result)
            phase_time = time.perf_counter() - phase_start
            self._record_phase_completion(phase, phase_idx, True, phase_time, result,
                                          checkpoint_boundary=checkpoint_boundary,
                                          persisted_outputs=persisted_outputs)
            
            if self.verbose:
                print(f"✅ Completed in {phase_time:.1f}s")
                if result:
                    print(f"   Provided: {list(result.keys())}")
            
            return True
            
        except Exception as e:
            self._handle_phase_failure(phase, phase_idx, e,
                                       checkpoint_boundary=checkpoint_boundary)
            return False
    
    def _run_phase_group(self, group: PhaseGroup, group_idx: int) -> bool:
        """Run a group of phases sharing an environment."""
        # TODO: This works, yet we do not CONFIGURE for each separately
        # There is an edge case where you want to have different stim electrodes,
        # and thus we'd have to configure/change params for each...
        if self.verbose:
            print(f"\n📦 Phase Group {group_idx+1}/{len(self.phases)}: {group.name}")
            print(f"   Contains {len(group)} phases")
        
        group_start = time.perf_counter()
        # Mark group start before any hardware or child work. An interruption
        # even before the first child completes must not look like a fresh group.
        self._record_phase_completion(group, group_idx, False, 0.0, {},
                                      checkpoint_boundary=True, event='started')
        try:
            if group.needs_environment():
                self._create_environment_for_phase(group.phases[0])
            for i, phase in enumerate(group):
                if self.verbose:
                    print(f"\n   ⚡ Sub-phase {i+1}/{len(group)}: {phase.name}")
                if not self._run_single_phase(phase, group_idx, checkpoint_boundary=False):
                    return False
        finally:
            if group.close_environment_after() and self.current_env:
                self._close_environment()

        group_time = time.perf_counter() - group_start
        self._record_phase_completion(group, group_idx, True, group_time, {},
                                      checkpoint_boundary=True)
        if self.verbose:
            print(f"\n📦 Group completed in {group_time:.1f}s")
        
        return True
    
    # ========== Phase Setup/Cleanup ==========
    
    def _setup_phase(self, phase: PhaseV3):
        """Setup phase before execution."""
        # Set references
        phase.set_experiment(self)

        # Auto-configure from experiment
        phase.configure_from_experiment(self)
        
        # Create/set environment if needed
        if phase.needs_environment():
            if self.current_env is None:
                self._create_environment_for_phase(phase)
            phase.set_env(self.current_env)
        

        
        # Track data reads
        for req in phase.inputs:
            self.data_tracker.record_access(phase.name, req, 'read')
    
    def _cleanup_phase(self, phase: PhaseV3):
        """Cleanup after phase execution."""
        # Phase cleanup
        phase.cleanup()
        
        # Close environment if requested
        if phase.close_environment_after() and self.current_env:
            self._close_environment()
    
    def _cleanup_experiment(self):
        """Final experiment cleanup."""
        if self.current_env:
            self._close_environment()
    
    # ========== Environment Management ==========
    
    def _create_environment_for_phase(self, phase: PhaseV3):
        """Create Maxwell environment for phase.

        If the phase has a ``recording_tag``, a new recording subdirectory is
        created and Maxwell's ``save_dir`` is pointed there.
        """
        from braindance.core.maxwell_env import MaxwellEnv
        from braindance.core.params import maxwell_params

        # Create a recording subdirectory when the phase declares one
        if getattr(phase, 'recording_tag', None):
            self._create_recording_dir(tag=phase.recording_tag)

        # Start with defaults
        env_params = maxwell_params.copy()
        configured_env_params = self.params.get('maxwell_env', {})
        if configured_env_params is None:
            configured_env_params = {}
        if not isinstance(configured_env_params, dict):
            raise TypeError("Experiment param 'maxwell_env' must be a dict")
        env_params.update(configured_env_params)

        # Point Maxwell at the recording dir (or fall back to experiment dir)
        env_save_dir = str(self.current_recording_dir or self.save_dir)

        # Update from experiment params
        env_params.update({
            'save_dir': env_save_dir,
            'name': self.current_recording_count or f"{self.name}_{phase.name.lower()}",
            'config': self.params.get(
                'config', None if env_params.get('replay') is not None else 'config.cfg'
            ),
            'stim_electrodes': self.params.get('stim_electrodes', []),
            'observation_type': 'raw',
            'max_time_sec': 3600
        })
        
        # Let phase customize
        env_params = phase.customize_environment_params(env_params)
        
        # Create environment
        self.current_env = MaxwellEnv(**env_params)
        
        if self.verbose:
            print(f"   🔧 Created Maxwell environment: {env_params['name']}")
    
    def _close_environment(self):
        """Close current Maxwell environment."""
        if self.current_env:
            self.current_env.close()
            self.current_env = None
            if self.verbose:
                print("   🔧 Closed Maxwell environment")
    
    # ========== Result Tracking ==========
    
    @staticmethod
    def _summarize_result(result: Optional[Dict]) -> Dict:
        """Create a JSON-safe summary of a phase result dict.

        Raw objects are replaced with type/shape descriptions so the
        experiment log and metadata never contain unpicklable references.
        """
        if not result:
            return {}
        summary = {}
        for k, v in result.items():
            try:
                json.dumps(v)
                summary[k] = v
            except (TypeError, ValueError):
                desc: Dict[str, Any] = {'_type': type(v).__name__}
                if hasattr(v, 'shape'):
                    desc['shape'] = str(v.shape)
                elif hasattr(v, '__len__'):
                    desc['length'] = len(v)
                summary[k] = desc
        return summary

    def _record_phase_completion(self, phase: PhaseV3, phase_idx: int, 
                                success: bool, duration: float, result: Dict,
                                checkpoint_boundary: bool = True,
                                persisted_outputs: Optional[List[Dict]] = None,
                                event: str = 'completed'):
        """Record phase completion and update the experiment log on disk."""
        record = {
            'phase_idx': phase_idx,
            'phase_name': phase.name,
            'phase_class': phase.__class__.__name__,
            'success': success,
            'checkpoint_boundary': checkpoint_boundary,
            'event': event,
            'persisted_outputs': persisted_outputs or [],
            'duration': round(duration, 3),
            'timestamp': datetime.datetime.now().isoformat(),
            'provided_keys': list(result.keys()) if result else [],
            'result_summary': self._summarize_result(result),
        }
        if isinstance(phase, PhaseGroup):
            record['group_phases'] = [{'name': child.name, 'class': type(child).__name__}
                                     for child in phase]

        if self.current_recording_dir:
            record['recording_dir'] = self.current_recording_dir.name
        
        self.results.append(record)
        self.metadata['completed_phases'].append(record)

        # Save recording metadata when a recording-producing phase completes
        if (success and self.current_recording_dir
                and getattr(phase, 'recording_tag', None)):
            self._save_recording_metadata(phase, self.current_recording_dir,
                                           duration=duration)

        # Write incremental experiment log
        self._save_experiment_log()
    
    def _handle_phase_failure(self, phase: PhaseV3, phase_idx: int, error: Exception,
                              checkpoint_boundary: bool = True):
        """Handle phase failure."""
        error_record = {
            'phase_idx': phase_idx,
            'phase_name': phase.name,
            'phase_class': phase.__class__.__name__,
            'success': False,
            'checkpoint_boundary': checkpoint_boundary,
            'event': 'failed',
            'error': str(error),
            'traceback': traceback.format_exc(),
            'timestamp': datetime.datetime.now().isoformat()
        }

        if self.current_recording_dir:
            error_record['recording_dir'] = self.current_recording_dir.name
        
        self.results.append(error_record)
        self.metadata['completed_phases'].append(error_record)
        
        print(f"❌ Phase failed: {error}")
        if self.verbose:
            print(traceback.format_exc())

        self._save_experiment_log()

    def _save_experiment_log(self):
        """Write / update ``experiment_log.json`` at the experiment level.

        Called after every phase so the log is always up-to-date even if the
        process crashes later.
        """
        log_path = self.save_dir / 'experiment_log.json'
        log = {
            'checkpoint_schema_version': 1,
            'experiment': self.name,
            'project_id': self.project_id,
            'chip_id': self.chip_id,
            'started': self.start_time.isoformat() if self.start_time else None,
            'last_updated': datetime.datetime.now().isoformat(),
            'phases_completed': self.current_phase_idx,
            'phases_total': len(self.phases),
            'recording_dirs': [
                d.name for d in self._list_recording_dirs()
            ],
            'phase_log': self.results,
        }
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                                             dir=self.save_dir, prefix='.experiment_log.',
                                             suffix='.tmp', delete=False) as handle:
                temporary_path = Path(handle.name)
                json.dump(log, handle, indent=2, default=str)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, log_path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
    
    # ========== Data Access ==========
    
    def get_result(self, phase_name: str = None, phase_idx: int = None) -> Optional[Dict]:
        """Get result from a specific phase."""
        for result in reversed(self.results):
            if phase_name and result.get('phase_name') == phase_name:
                return result
            if phase_idx is not None and result.get('phase_idx') == phase_idx:
                return result
        return None
    
    def get_last_result(self, key: str = None) -> Any:
        """Get the last result or specific key from it."""
        if not self.results:
            return None
        
        last = self.results[-1]
        if key:
            return last.get('result', {}).get(key)
        return last
    
    # ========== Save/Load ==========
    
    def save_summary(self, path: Optional[Path] = None) -> Path:
        """Save a JSON experiment summary at the experiment level.

        Phase results are already saved incrementally into each recording's
        ``results/`` folder, so this method only writes a lightweight overview
        (no large data re-dump).
        """
        if path is None:
            path = self.save_dir / f"{self.name}_summary.json"

        # Build per-recording overview
        recording_info = []
        for rec_dir in self._list_recording_dirs():
            entry: Dict[str, Any] = {'recording_id': rec_dir.name}
            meta_files = list(rec_dir.glob('*_metadata.json'))
            if meta_files:
                try:
                    with open(meta_files[0], 'r') as f:
                        entry['metadata'] = json.load(f)
                except Exception:
                    pass
            # List result files
            results_dir = rec_dir / self.results_dir_name
            if results_dir.exists():
                entry['result_files'] = [
                    f.name for f in results_dir.iterdir() if f.is_file()
                ]
            recording_info.append(entry)

        # Build a lightweight data-keys summary (types only, no values)
        data_keys_summary = {}
        for key in self.data.keys():
            value = self.data.get(key)
            desc: Dict[str, Any] = {'type': type(value).__name__}
            if hasattr(value, 'shape'):
                desc['shape'] = str(value.shape)
            elif hasattr(value, '__len__'):
                desc['length'] = len(value)
            data_keys_summary[key] = desc

        total_time = None
        if self.start_time:
            total_time = round(
                (datetime.datetime.now() - self.start_time).total_seconds(), 2
            )

        summary = {
            'name': self.name,
            'project_id': self.project_id,
            'chip_id': self.chip_id,
            'save_dir': str(self.save_dir),
            'started': self.start_time.isoformat() if self.start_time else None,
            'total_time_s': total_time,
            'params': self.params,
            'phases_completed': self.current_phase_idx,
            'phases_total': len(self.phases),
            'phase_log': self.results,
            'recordings': recording_info,
            'data_keys': data_keys_summary,
        }
        
        with open(path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        if self.verbose:
            print(f"📄 Saved experiment summary to: {path}")
        
        return path
    
    def save_data(self, path: Optional[Path] = None, overwrite_files: bool | None = None) -> Path:
        """
        Save experiment data context.
        
        Args:
            path: Directory to save data. Defaults to
                ``current_recording_dir/<results_dir_name>`` when a recording
                is active, otherwise ``save_dir/<results_dir_name>``.
            overwrite_files: Whether to overwrite existing files (defaults to instance policy)
        """
        if path is None:
            if self.current_recording_dir:
                path = self.current_recording_dir / self.results_dir_name
            else:
                path = self.save_dir / self.results_dir_name
        
        self.data.save(path, overwrite_files=overwrite_files)
        return path
    
    def load_data(self, path: Optional[Path] = None, keys: Optional[List[str]] = None):
        """Load experiment data context.

        When no *path* is given the method checks
        ``current_recording_dir/<results_dir_name>`` first, then falls back to
        ``save_dir/<results_dir_name>``.
        """
        if path is None:
            if self.current_recording_dir:
                candidate = self.current_recording_dir / self.results_dir_name
                if candidate.exists():
                    path = candidate
            if path is None:
                path = self.save_dir / self.results_dir_name
        
        self.data.load(path, keys)
    
    def load_data_from_experiment(self, experiment_name: str, experiment_dir: str = None, keys: Optional[List[str]] = None):
        """
        Load data from another experiment.
        
        Args:
            experiment_name: Name of the source experiment
            experiment_dir: Directory containing experiments (defaults to ./experiments)
            keys: Specific keys to load (None = load all)
        """
        if experiment_dir is None:
            experiment_dir = "./experiments"
        
        exp_root = Path(experiment_dir) / experiment_name

        # Try results/ first, then legacy data/
        source_path = None
        for subdir in (self.results_dir_name, "data"):
            candidate = exp_root / subdir
            if candidate.exists():
                source_path = candidate
                break

        if source_path is None:
            raise FileNotFoundError(
                f"No data found for experiment '{experiment_name}' in {exp_root}"
            )
        
        self.data.load(source_path, keys)
        
        if self.verbose:
            loaded_keys = list(self.data.keys())
            print(f"📂 Loaded data from experiment '{experiment_name}': {loaded_keys}")
    
    def get_data_index_info(self) -> Optional[Dict]:
        """
        Get information about the data index file.
        
        Returns:
            Dictionary with index information or None if no index exists
        """
        # Check recording-level results dir first, then experiment-level
        candidates = []
        if self.current_recording_dir:
            candidates.append(
                self.current_recording_dir / self.results_dir_name / 'data_index.json'
            )
        candidates.append(self.save_dir / self.results_dir_name / 'data_index.json')
        candidates.append(self.save_dir / 'data' / 'data_index.json')  # legacy

        for index_path in candidates:
            if index_path.exists():
                try:
                    with open(index_path, 'r') as f:
                        return json.load(f)
                except Exception as e:
                    if self.verbose:
                        print(f"⚠️  Warning: Could not read data index: {e}")

        return None
    
    def print_data_summary(self):
        """Print a summary of current data and saved data."""
        print(f"\n📊 Data Summary for experiment '{self.name}'")
        print("=" * 50)
        
        # Current data in memory
        if self.data.has_data():
            print(f"📋 Current data in memory:")
            for key, summary in self.data.data_summary().items():
                print(f"   {key}: {summary}")
        else:
            print("📋 No data currently in memory")
        
        # Saved data index
        index_info = self.get_data_index_info()
        if index_info:
            print(f"\n💾 Saved data on disk:")
            saved_files = index_info.get('saved_files', {})
            for key, file_type in saved_files.items():
                print(f"   {key}: {file_type}")
            
            print(f"\n📈 Save history:")
            print(f"   Created: {index_info.get('created_time', 'Unknown')}")
            print(f"   Last saved: {index_info.get('last_save_time', 'Unknown')}")
            print(f"   Total save operations: {len(index_info.get('save_history', []))}")
            
            if self.verbose and index_info.get('save_history'):
                print(f"\n🔍 Recent save operations:")
                for i, save_op in enumerate(index_info['save_history'][-3:], 1):  # Show last 3
                    files = ', '.join(save_op.get('files_saved', []))
                    overwrite = save_op.get('overwrite_policy', False)
                    timestamp = save_op.get('timestamp', 'Unknown')
                    print(f"   {i}. {timestamp}: {files} (overwrite={overwrite})")
        else:
            print(f"\n💾 No saved data found")
    
    def __repr__(self) -> str:
        phase_names = [p.name if hasattr(p, 'name') else str(p) for p in self.phases]
        return (f"SimpleExperiment('{self.name}', phases={phase_names}, "
                f"completed={self.current_phase_idx}/{len(self.phases)})")
