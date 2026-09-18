"""Editable Python phase plans using the workshop's acquisition-aware runners."""
from copy import deepcopy
from pathlib import Path


class ExperimentPlan:
    """Build an ordered specification with calls, then run or verify it.

    This is the workshop plan API. ``add_phase`` takes a catalog type, rather
    than a core PhaseV3 instance, so native phases can be built in their worker.
    """

    def __init__(self, settings=None):
        self.settings = deepcopy(settings or {})
        self.settings.pop('phases', None)
        self.settings.pop('phase_plan', None)
        self.phases = []

    def add_phase(self, phase_type, *, name, params=None, settings=None):
        """Append a phase; return this plan for chaining."""
        phase = dict(id=name, type=phase_type, params=deepcopy(params or {}))
        if settings is not None:
            phase['settings'] = deepcopy(settings)
        self.phases.append(phase)
        return self

    def to_config(self):
        return {**deepcopy(self.settings), 'phases': deepcopy(self.phases)}

    def run(self, output_dir=None, verify=False):
        from braindance.config import get_output_dir
        from .experiment_spec import uses_native_runner
        from .native_runner import NativeSession
        from .session import WorkshopSession

        config = self.to_config()
        runner = NativeSession if uses_native_runner(config) else WorkshopSession
        session = runner(Path(output_dir) if output_dir else get_output_dir() / 'streaming_workshop', config)
        session.run(verify_only=verify)
        return session
