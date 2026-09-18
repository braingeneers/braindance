"""Run editable V3 phase objects with the workshop's acquisition setup."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import time

from braindance.core.phases_v3.phase_base_v3 import PhaseV3
from braindance.core.phases_v3.phases_binned import (
    BinnedRecordingPhaseV3 as RecordingPhase,
    ResponseProbePhaseV3 as ResponseProbePhase,
    MappedEnvironmentPhaseV3,
)
from .custom_analysis import CustomAnalysisPhase


class CartPolePhase(MappedEnvironmentPhaseV3):
    environment = 'cartpole'


class FoodLandPhase(MappedEnvironmentPhaseV3):
    environment = 'foodland'


class AntPhase(MappedEnvironmentPhaseV3):
    environment = 'ant'


class PythonExperiment:
    """Collect actual phase objects and execute them without reconstructing them.

    ``settings`` on add_phase supplies runtime mappings for streaming phases or
    scoped input/parameter overrides for native phases. Constructor attributes
    are authoritative; there is no second saved list of constructor arguments.
    """

    def __init__(self, settings=None, native=False):
        self.settings = deepcopy(settings or {})
        self.settings.pop('phases', None)
        self.settings.pop('phase_plan', None)
        self.native = native
        self.phases, self.phase_settings = [], []

    def add_phase(self, phase, *, name=None, settings=None):
        if not isinstance(phase, PhaseV3):
            raise TypeError('add_phase expects a PhaseV3 object')
        if name is not None:
            phase.name = name
        self.phases.append(phase)
        self.phase_settings.append(deepcopy(settings or {}))
        return self

    def to_config(self):
        """Derive streaming preflight settings from the current phase objects.

        Native objects need only names and scoped settings, not serialized
        constructors. Their real inputs/outputs are checked by PhaseValidator.
        """
        specs = []
        for phase, local in zip(self.phases, self.phase_settings):
            if self.native:
                specs.append(dict(id=phase.name, settings=deepcopy(local)))
                continue
            params = deepcopy(local)
            if isinstance(phase, CustomAnalysisPhase):
                kind, params = 'custom_analysis', deepcopy(phase.options)
            elif isinstance(phase, RecordingPhase):
                kind = 'recording'
                params['record_seconds'] = phase.duration
            elif isinstance(phase, ResponseProbePhase):
                kind = 'causal'
                params['causal_repeats'] = phase.repeats
            elif isinstance(phase, MappedEnvironmentPhaseV3):
                kind = getattr(phase, 'environment', 'environment')
                params['environment_seconds'] = phase.duration
            else:
                raise TypeError('Use native=True for scientific V3 phase objects')
            specs.append(dict(id=phase.name, type=kind, params=params))
        return {**deepcopy(self.settings), 'phases': specs}

    def run(self, output_dir=None, verify=False):
        from braindance.config import get_output_dir
        config = self.to_config()
        if not self.phases or len({p.name for p in self.phases}) != len(self.phases):
            raise ValueError('Add phases with unique names before running')
        output = Path(output_dir) if output_dir else get_output_dir() / 'streaming_workshop'
        if not self.native:
            from .session import WorkshopSession
            session = WorkshopSession(output, config, phases=self.phases)
            session.run(verify_only=verify)
            return session
        from .native_runner import run_phases
        run_dir = output / ('native_' + str(time.time_ns()))
        try:
            result = run_phases(config, self.phases, run_dir, verify=verify)
            status = ('verified' if verify else 'completed') if result == 0 else 'error'
            error = '' if result == 0 else 'Experiment failed; inspect phase logs'
        except Exception as exc:
            status, error = 'error', f'{type(exc).__name__}: {exc}'
        return SimpleNamespace(status=status, error=error, snapshot=dict(
            status=status, error=error, output=str(run_dir), phases=[p.name for p in self.phases]))
