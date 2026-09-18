"""
Enhanced Phase Base Classes for Experiment Framework V3
"""
from typing import Dict, List, Any, Optional, Set, TYPE_CHECKING
from abc import ABC, ABCMeta, abstractmethod
import time
import inspect
import ast
from typing import Callable

if TYPE_CHECKING:
    from .data_context import DataContext


class _PhaseContractMeta(ABCMeta):
    """Keep historical contract names as aliases, including on subclasses."""

    def __new__(mcls, name, bases, namespace, **kwargs):
        for canonical, legacy in (("inputs", "requires"), ("outputs", "provides")):
            if canonical in namespace and legacy in namespace:
                if namespace[canonical] != namespace[legacy]:
                    raise ValueError(f"{name}: conflicting {canonical} and {legacy}")
            if canonical in namespace or legacy in namespace:
                contract = list(namespace.get(canonical, namespace.get(legacy)))
                namespace[canonical] = namespace[legacy] = contract
        return super().__new__(mcls, name, bases, namespace, **kwargs)

    def __setattr__(cls, name, value):
        aliases = {"inputs": "requires", "requires": "inputs",
                   "outputs": "provides", "provides": "outputs"}
        if name in aliases:
            value = list(value)
            super().__setattr__(aliases[name], value)
        super().__setattr__(name, value)


class PhaseV3(ABC, metaclass=_PhaseContractMeta):
    """
    Enhanced base phase class with dependency management and lifecycle hooks.
    
    Key features:
    - Declarative data dependencies (inputs/outputs)
    - Automatic configuration from experiment
    - Environment lifecycle management
    - Built-in timing and metadata tracking
    """
    
    # Data dependencies - override in subclasses
    inputs: List[str] = []  # Data keys this phase needs
    outputs: List[str] = []  # Data keys this phase produces
    
    def __init__(self, name: str = None, suffix: str = "",
                 recording_tag: str = None):
        self.inputs = self.inputs
        self.outputs = self.outputs
        self.name = name or self.__class__.__name__
        self.suffix = suffix
        self.recording_tag = recording_tag
        self.experiment = None
        self.env = None
        self.start_time = None
        self._env_time_origin = None
        self.metadata = {}
        
    def __setattr__(self, name, value):
        aliases = {"inputs": "requires", "requires": "inputs",
                   "outputs": "provides", "provides": "outputs"}
        if name in aliases:
            value = list(value)
            object.__setattr__(self, aliases[name], value)
        object.__setattr__(self, name, value)

    def needs_environment(self) -> bool:
        """Override to specify if this phase needs a Maxwell environment."""
        return True
    
    def close_environment_after(self) -> bool:
        """Override to specify if environment should be closed after this phase."""
        return True
    
    def set_experiment(self, experiment):
        """Set experiment reference."""
        self.experiment = experiment
        
    def set_env(self, env):
        """Set environment reference."""
        self.env = env
        self._env_time_origin = (
            env.time_elapsed() if getattr(env, "is_replay", False) else None
        )
        
    def validate_requirements(self, data_context) -> bool:
        """Validate that all required data is available."""
        missing = []
        for req in self.inputs:
            if not hasattr(data_context, req):
                missing.append(req)
                
        if missing:
            raise ValueError(f"{self.name} requires data keys: {missing}")
        return True
    
    def configure_from_experiment(self, experiment):
        """
        Auto-configure phase from experiment data and config.
        Override to add custom configuration logic.
        """
        # Auto-populate from experiment data
        for req in self.inputs:
            if hasattr(experiment.data, req):
                setattr(self, req, getattr(experiment.data, req))
    
    def customize_environment_params(self, env_params: dict) -> dict:
        """
        Customize Maxwell environment parameters.
        Override to modify environment creation.
        """
        return env_params
    
    @abstractmethod
    def run(self, experiment) -> Dict[str, Any]:
        """
        Execute the phase logic.
        
        Args:
            experiment: The parent experiment
            
        Returns:
            Dictionary with keys matching self.outputs
        """
        pass
    
    def cleanup(self):
        """Cleanup after phase execution. Override if needed."""
        pass
    
    def time_elapsed(self) -> float:
        """Get elapsed time since phase started."""
        if self._env_time_origin is not None:
            return max(0.0, self.env.time_elapsed() - self._env_time_origin)
        if self.start_time is None:
            return 0.0
        return time.perf_counter() - self.start_time
    
    def info(self) -> Dict[str, Any]:
        """Get phase information. Override to add custom info."""
        return {
            'name': self.name,
            'inputs': self.inputs,
            'requires': self.inputs,
            'outputs': self.outputs,
            'provides': self.outputs,
            'metadata': self.metadata
        }


class AnalysisPhaseV3(PhaseV3):
    """
    Base class for analysis phases that don't need Maxwell environments.
    """
    
    def needs_environment(self) -> bool:
        return False
    
    def close_environment_after(self) -> bool:
        return False


class PhaseGroup:
    """
    Container for phases that should share the same environment/save file.
    """
    
    def __init__(self, phases: List[PhaseV3], name: str = None):
        self.phases = phases
        self.name = name or "phase_group"
        self._is_group = True  # Marker for experiment runner
        
    def __iter__(self):
        return iter(self.phases)
    
    def __len__(self):
        return len(self.phases)
    
    def needs_environment(self) -> bool:
        """Group needs environment if any phase needs it."""
        return any(phase.needs_environment() for phase in self.phases)
    
    def close_environment_after(self) -> bool:
        """Close environment after group completes."""
        return True


class ValidationError(Exception):
    """Raised when phase validation fails."""
    pass


class PhaseValidator:
    """
    Validates phase dependencies and data flow.
    """
    
    @staticmethod
    def validate_pipeline(phases: List[PhaseV3], existing_data: Optional['DataContext'] = None) -> bool:
        """
        Validate that all phase dependencies can be satisfied.
        
        Args:
            phases: List of phases in execution order
            existing_data: Optional DataContext with pre-existing data (e.g., loaded from previous experiment)
            
        Returns:
            True if valid
            
        Raises:
            ValidationError if dependencies cannot be satisfied
        """
        # Start with existing data if provided
        available_data: Set[str] = set()
        if existing_data is not None:
            available_data.update(existing_data.keys())
        
        # Flatten phase groups
        flat_phases = []
        for phase in phases:
            if hasattr(phase, '_is_group'):
                flat_phases.extend(phase.phases)
            else:
                flat_phases.append(phase)
        
        # Check each phase
        for i, phase in enumerate(flat_phases):
            # Check requirements
            missing = set(phase.inputs) - available_data
            if missing:
                raise ValidationError(
                    f"Phase {i} ({phase.name}) requires {missing} "
                    f"but only {available_data} is available"
                )
            
            # Add provides to available data
            available_data.update(phase.outputs)
            
        return True
    





class QuickPhaseV3(AnalysisPhaseV3):
    """
    A phase that executes a lambda/function inline with automatic dependency inference.
    
    Examples:
        # Simple data selection
        exp.add_phase(LambdaPhaseV3(
            lambda exp: {"selected_pair": exp.data.connectivity_matrix[4,2]},
            name="SelectPair"
        ))
        
        # More complex processing
        exp.add_phase(LambdaPhaseV3(
            lambda exp: {
                "top_neurons": sorted(exp.data.neurons, 
                                    key=lambda n: len(exp.data.spike_trains[n]), 
                                    reverse=True)[:5]
            },
            requires=["neurons", "spike_trains"],  # Optional explicit requirements
            provides=["top_neurons"],
            name="FindTopNeurons"
        ))
        
        # Using experiment context
        exp.add_phase(LambdaPhaseV3(
            lambda exp: {
                "recording_info": {
                    "file": exp.data.recording_file,
                    "duration": exp.config.get("record_duration", 0),
                    "n_neurons": len(exp.data.neurons)
                }
            },
            name="RecordingInfo"
        ))
    """
    
    def __init__(self, 
                 func: Callable[[Any], Dict[str, Any]], 
                 requires: Optional[List[str]] = None,
                 provides: Optional[List[str]] = None,
                 name: str = None, *,
                 inputs: Optional[List[str]] = None,
                 outputs: Optional[List[str]] = None):
        """
        Initialize lambda phase.
        
        Args:
            func: Function that takes experiment and returns dict of results
            inputs: Data keys needed (auto-inferred if None)
            outputs: Data keys returned (auto-inferred if None)
            requires: Historical alias for inputs
            provides: Historical alias for outputs
            name: Phase name (auto-generated if None)
        """
        # Auto-generate name if not provided
        if name is None:
            name = f"Quick_{id(func) % 10000}"
        
        super().__init__(name)
        self.func = func
        
        for canonical, legacy, label in ((inputs, requires, "inputs/requires"),
                                         (outputs, provides, "outputs/provides")):
            if canonical is not None and legacy is not None and canonical != legacy:
                raise ValueError(f"Conflicting {label}")
        requires = inputs if inputs is not None else requires
        provides = outputs if outputs is not None else provides

        # Try to auto-infer dependencies
        if requires is None or provides is None:
            inferred_req, inferred_prov = self._infer_dependencies(func)
            print(f"   Inferred inputs: {inferred_req}")
            print(f"   Inferred outputs: {inferred_prov}")
            
        self.inputs = inferred_req if requires is None else requires
        self.outputs = inferred_prov if provides is None else provides
        
        # Store function info for debugging
        self.func_source = self._get_function_source(func)
    
    def _infer_dependencies(self, func: Callable) -> tuple[List[str], List[str]]:
        """
        Try to infer what the function requires and provides by analyzing its source.
        """
        requires = set()
        provides = set()
        
        try:
            # Get function source
            source = inspect.getsource(func)
            
            # Parse AST to find exp.data.X accesses
            tree = ast.parse(source)
            
            for node in ast.walk(tree):
                # Look for exp.data.something patterns
                if (isinstance(node, ast.Attribute) and 
                    isinstance(node.value, ast.Attribute) and
                    isinstance(node.value.value, ast.Name)):
                    
                    if (node.value.value.id == 'exp' and 
                        node.value.attr == 'data'):
                        # This is exp.data.something - it's a requirement
                        requires.add(node.attr)
                
                # Look for dictionary returns to infer provides
                if isinstance(node, ast.Dict):
                    for key in node.keys:
                        if isinstance(key, ast.Constant) and isinstance(key.value, str):
                            provides.add(key.value)
                        elif isinstance(key, ast.Str):  # Python < 3.8 compatibility
                            provides.add(key.s)
        
        except (OSError, TypeError, SyntaxError):
            # If we can't parse, return empty lists
            pass
        
        return list(requires), list(provides)
    
    def _get_function_source(self, func: Callable) -> str:
        """Get function source for debugging."""
        try:
            return inspect.getsource(func).strip()
        except OSError:
            return f"<function {func.__name__}>"
    
    def run(self, experiment) -> Dict[str, Any]:
        """Execute the lambda function."""
        print(f"   Executing lambda: {self.func_source[:50]}...")
        
        try:
            result = self.func(experiment)
            
            if not isinstance(result, dict):
                raise ValueError(f"Lambda function must return a dict, got {type(result)}")
            
            print(f"   Lambda produced: {list(result.keys())}")
            return result
            
        except Exception as e:
            print(f"   Lambda failed: {e}")
            raise


# Convenience function for creating inline phases
def quick_phase(func: Callable[[Any], Dict[str, Any]], 
                 requires: Optional[List[str]] = None,
                 provides: Optional[List[str]] = None,
                 name: str = None, *,
                 inputs: Optional[List[str]] = None,
                 outputs: Optional[List[str]] = None) -> QuickPhaseV3:
    """
    Convenience function to create an inline phase.
    
    Examples:
        # With lambda
        exp.add_phase(inline_phase(
            lambda exp: {"selected_pair": exp.data.connectivity_matrix[4,2]}
        ))
        
        # With function reference
        def my_processor(exp):
            return {"result": exp.data.some_data * 2}
            
        exp.add_phase(inline_phase(my_processor))
    """
    return QuickPhaseV3(func, requires, provides, name, inputs=inputs, outputs=outputs)

# Decorator approach for even simpler usage
def phase(name: str = None, 
          requires: Optional[List[str]] = None, 
          provides: Optional[List[str]] = None, *,
          inputs: Optional[List[str]] = None,
          outputs: Optional[List[str]] = None):
    """
    Decorator to turn a function into a phase.
    
    Example:
        @phase("SelectPair")
        def select_pair(exp):
            return {"selected_pair": exp.data.connectivity_matrix[4,2]}
        
        exp.add_phase(select_pair)
    """
    def decorator(func):
        return QuickPhaseV3(func, requires, provides, name, inputs=inputs, outputs=outputs)
    return decorator
