"""
Analysis Phases for Experiment Framework V3

These phases handle data analysis and don't require Maxwell environments.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import pickle
import os
from spikelab import SpikeData
from .phase_base_v3 import AnalysisPhaseV3
from braindance.core.phases_v3.experiment_v3 import Experiment as Experiment


class ActivityPhaseV3(AnalysisPhaseV3):
    """
    Analyze activity from a recording.
    
    Requires:
        - recording_file: Path to recording file
    
    Provides:
        - activity_map: DataFrame with activity metrics per electrode
        - active_electrodes: List of electrodes with activity
    """
    
    inputs = ['recording_file']
    outputs = ['activity_map', 'active_electrodes']
    
    def __init__(self, threshold: float = 5.0, name: str = None):
        super().__init__(name or "ActivityPhase")
        self.threshold = threshold
        self.recording_file = None  # Will be auto-populated
    
    def run(self, experiment) -> Dict[str, Any]:
        """Analyze recording for activity."""
        from braindance.analysis.data_loader import load_data_maxwell
        
        print(f"   Analyzing activity in: {self.recording_file}")
        
        # Load data
        data = load_data_maxwell(self.recording_file)
        
        # Calculate activity metrics
        activity_map = pd.DataFrame({
            'electrode': data['mapping']['electrode'],
            'channel': data['mapping']['channel'],
            'spike_rate': np.random.rand(len(data['mapping'])) * 10  # Placeholder
        })
        
        # Find active electrodes
        active_electrodes = activity_map[
            activity_map['spike_rate'] > self.threshold
        ]['electrode'].tolist()
        
        print(f"   Found {len(active_electrodes)} active electrodes")
        
        return {
            'activity_map': activity_map,
            'active_electrodes': active_electrodes
        }


class RTSortPhaseV3(AnalysisPhaseV3):
    """
    Real-time spike sorting phase.
    
    Requires:
        - recording_file: Path to recording file
    
    Provides:
        - neurons: List of sorted neuron IDs
        - channels_per_neuron: Dict mapping neuron ID to channels
        - electrodes_per_neuron: Dict mapping neuron ID to electrodes
        - spike_trains: Dict of spike times per neuron
        - templates: Spike templates for each neuron
        - spike_data: Spike data object
        - rt_sort_object: The RT sort object (saved separately)
    """
    
    inputs = ['recording_file']
    outputs = ['neurons', 'channels_per_neuron', 'electrodes_per_neuron',
                # 'spike_trains', 'templates', 
                'spike_data', 'rt_sort_object',
                'positions_per_neuron']
    
    def __init__(self, sorter: str = 'rt_sort', 
                 min_spikes: int = 100,
                 detection_model_path: str = None,
                 inter_path: str = None,
                 recording_window_ms: Tuple[int, int] = (0, 60000),
                 artifact_removal_params: Dict = None,
                 art_rem_N: int = 60,
                 force_redo: bool = True,
                 make_plots: bool = False,
                 save_plots: bool = False,
                 delete_inter: bool = True,
                 save_rt_sort: bool = True,
                 verbose: bool = True,
                 name: str = None):
        super().__init__(name or "RTSortPhase")
        self.sorter = sorter
        self.min_spikes = min_spikes
        self.recording_file = None  # Will be auto-populated
        
        # RT sort specific parameters
        self.detection_model_path = detection_model_path
        self.inter_path = inter_path
        self.recording_window_ms = recording_window_ms
        self.artifact_removal_params = artifact_removal_params
        self.art_rem_N = art_rem_N
        self.force_redo = force_redo
        self.make_plots = make_plots
        self.save_plots = save_plots
        self.delete_inter = delete_inter
        self.save_rt_sort = save_rt_sort
        self.verbose = verbose
        self.rt_sort = None
    
    def configure_from_experiment(self, experiment: Experiment):
        """Auto-configure from experiment."""
        super().configure_from_experiment(experiment)
        
        # Can use mapping if available
        if hasattr(experiment, 'mapping'):
            self.mapping = experiment.mapping

        # If rt_sort_path is provided, use it
        if hasattr(experiment, 'rt_sort_path'):
            self.rt_sort_path = experiment.rt_sort_path
            from braindance.core.spikesorter.rt_sort import RTSort
            # Load the rt_sort object
            self.rt_sort = RTSort.load_from_file(self.rt_sort_path, model=self.detection_model_path)
            print(f"   Loaded RT sort object from {self.rt_sort_path}")
        else:
            self.rt_sort = None
            
        # Set inter_path if not provided
        if self.inter_path is None:
            self.inter_path = experiment.save_dir / 'rt_sort_inter'
    
    def run(self, experiment) -> Dict[str, Any]:
        """Run spike sorting."""
        print(f"   Running {self.sorter} on: {self.recording_file}")

        # spike_data_dir is only created when a sorter actually needs it
        spike_data_dir = experiment.save_dir / 'spike_data'
        
        if self.sorter == 'rt_sort':
            if self.rt_sort is None:
                if self.verbose:
                    print("   Initializing RT sort")
                results = self._run_rt_sort_init(experiment, spike_data_dir)
                if self.verbose:
                    print("   Running offline RT sorting")
                extra_results = self._run_rt_sort(experiment, spike_data_dir)
                results.update(extra_results)
            else:
                results = self._run_rt_sort(experiment, spike_data_dir)
        elif self.sorter == 'kilosort2':
            spike_data_dir.mkdir(exist_ok=True)
            results = self._run_kilosort2(experiment, spike_data_dir)
        else:
            spike_data_dir.mkdir(exist_ok=True)
            results = self._run_mock_sort(experiment, spike_data_dir)

        # Save [count]_spike_data.pkl at the recording level and an rt-tagged copy at experiment level
        rec_dir = getattr(experiment, 'current_recording_dir', None)
        count = getattr(experiment, 'current_recording_count', None) or 'rec'
        sd = results.get('spike_data')
        if sd is not None:
            if rec_dir is not None:
                spike_data_path = rec_dir / f'{count}_spike_data.pkl'
                with open(spike_data_path, 'wb') as f:
                    pickle.dump(sd, f)
                if self.verbose:
                    print(f"   Saved SpikeData to: {spike_data_path}")

            # Also save at experiment level with _rt tag indicating realtime sorting
            exp_spike_data_path = experiment.save_dir / f'{count}_rt_spike_data.pkl'
            with open(exp_spike_data_path, 'wb') as f:
                pickle.dump(sd, f)
            if self.verbose:
                print(f"   Saved RT SpikeData to: {exp_spike_data_path}")
        
        return results
    
    def _run_rt_sort_init(self, experiment, spike_data_dir: Path) -> Dict[str, Any]:
        """Run actual RT sort implementation."""
        try:
            from braindance.core.spikesorter.rt_sort import detect_sequences
            from braindance import get_rt_sort_path
            from spikeinterface.extractors import MaxwellRecordingExtractor
            
            # Get detection model path
            if self.detection_model_path is None:
                self.detection_model_path = get_rt_sort_path()
            
            # Extract raw_h5 file path
            if not self.recording_file.endswith('.raw.h5'):
                recording_file = self.recording_file + '.raw.h5'
            else:
                recording_file = self.recording_file
            recording = MaxwellRecordingExtractor(recording_file)

            rt_sort = detect_sequences(
                recording, self.inter_path, self.detection_model_path, 
                recording_window_ms=self.recording_window_ms,
                return_spikes=False, delete_inter=self.delete_inter,
                verbose=self.verbose
            )

            sequence_spike_trains = rt_sort.seq_spike_trains
        
            # Get channels per neuron - this is already a list of lists
            channels_per_neuron_list = rt_sort.seq_comp_elecs
            
            # Extract data from RT sort
            neurons = list(range(len(sequence_spike_trains)))
            channels_per_neuron = {}
            electrodes_per_neuron = {}
            spike_trains = {}
            templates = {}
            positions_per_neuron = {}
            
            # Convert to electrodes using mapping if available
            electrodes_per_neuron_list = []
            
            # Load mapping for channel to electrode conversion
            mapping_df = None
            try:
                from braindance.analysis import data_loader
                mapping_df = data_loader.load_mapping_maxwell(self.recording_file)
                
                for i, channels in enumerate(channels_per_neuron_list):
                    # Filter mapping to only include channels in the sequence
                    ch_mapping = mapping_df[mapping_df.index.isin(channels)]
                    electrodes = ch_mapping['electrode'].tolist()
                    electrodes_per_neuron_list.append(electrodes)
            except Exception as e:
                print(f"   Warning: Could not load mapping or convert channels to electrodes: {e}")
                # Use channels as electrodes as fallback
                electrodes_per_neuron_list = channels_per_neuron_list
            
            for i, spike_train in enumerate(sequence_spike_trains):
                # Store channels and electrodes
                channels_per_neuron[i] = channels_per_neuron_list[i] if i < len(channels_per_neuron_list) else []
                electrodes_per_neuron[i] = electrodes_per_neuron_list[i] if i < len(electrodes_per_neuron_list) else []
                
                # Calculate positions if we have mapping and electrodes
                if mapping_df is not None and electrodes_per_neuron[i]:
                    try:
                        elec_mapping = mapping_df[mapping_df['electrode'].isin(electrodes_per_neuron[i])]
                        positions = elec_mapping[['x', 'y']].values.tolist()
                        positions_per_neuron[i] = positions
                    except:
                        positions_per_neuron[i] = []
                else:
                    positions_per_neuron[i] = []
                
                # Store spike train - spike_train should already be an array
                spike_trains[i] = np.array(spike_train) if hasattr(spike_train, '__len__') else np.array([])
                
                # Generate placeholder template based on number of channels
                # n_channels = len(channels_per_neuron[i]) if channels_per_neuron[i] else 1
                # templates[i] = np.random.randn(60, n_channels)

            
            print(f"   Found {len(neurons)} neurons using RT sort")
            
            # Plot results if requested
            if self.make_plots:
                import matplotlib.pyplot as plt
                
                # Plot neuron positions if we have positions
                if any(len(pos) > 0 for pos in positions_per_neuron.values()):
                    plt.figure(figsize=(10, 10))
                    for i, positions in positions_per_neuron.items():
                        if len(positions) > 0:
                            positions = np.array(positions)
                            plt.scatter(positions[:, 0], positions[:, 1], 
                                      label=f'Neuron {i}', alpha=0.6, s=3)
                    
                    plt.xlabel('X Position (μm)')
                    plt.ylabel('Y Position (μm)')
                    plt.title('Spatial Distribution of Electrodes per Neuron')
                    plt.legend()
                    plt.grid(True)
                    plt.axis('equal')
                    
                    if self.save_plots:
                        plt.savefig(experiment.save_dir / f"{self.recording_file}_neurons.png")
                    else:
                        plt.show()
                
                # Plot spike trains
                plt.figure(figsize=(12, 8))
                for i, spike_train in enumerate(sequence_spike_trains):
                    if len(spike_train) > 0:
                        plt.plot(spike_train, i*np.ones_like(spike_train), 'k|')
                
                max_time = max([max(st) if len(st) > 0 else 0 for st in sequence_spike_trains])
                plt.xlim(0, min(10000, max_time + 1000) if max_time > 0 else 10000)
                plt.xlabel('Time (ms)')
                plt.ylabel('Neuron')
                plt.title('Spike Trains')
                
                if self.save_plots:
                    plt.savefig(experiment.save_dir / f"{self.recording_file}_spike_trains.png")
                else:
                    plt.show()
            
            # Save RT sort object if requested (at experiment level)
            if self.save_rt_sort:
                if isinstance(self.save_rt_sort, (str, Path)):
                    rt_sort_path = Path(self.save_rt_sort)
                else:
                    rt_sort_path = experiment.save_dir / "RT_sort.pkl"
                
                rt_sort.save(rt_sort_path)
                
                print(f"   Saved RT sort object to: {rt_sort_path}")

            # Can't pickle the model, so we need to remove it
            rt_sort.model = None
            self.rt_sort = rt_sort
            self.rt_sort_path = rt_sort_path
            
            # Return paths and JSON-serializable data
            return {
                'neurons': neurons,
                'channels_per_neuron': channels_per_neuron,
                'electrodes_per_neuron': electrodes_per_neuron,
                'positions_per_neuron': positions_per_neuron,
                # 'spike_trains': spike_trains,  # Will be saved by DataContext
                # 'templates': templates,  # Will be saved by DataContext
                'rt_sort_object': rt_sort  # Will be saved by DataContext
            }
            
        except ImportError as e:
            print(f"   Warning: Could not import RT sort modules: {e}")
            raise e
            # print("   Falling back to mock data")
            # return self._run_mock_sort(experiment, spike_data_dir)
    

    def _run_rt_sort(self, experiment, spike_data_dir: Path) -> Dict[str, Any]:
        """Run RT sort with existing RT sort object."""
        from braindance.core.spikesorter.rt_sort import RTSort
        from braindance import get_rt_sort_path
        from spikeinterface.extractors import MaxwellRecordingExtractor

        # Get detection model path
        if self.detection_model_path is None:
            self.detection_model_path = get_rt_sort_path()

        self.rt_sort = RTSort.load_from_file(self.rt_sort_path, model=self.detection_model_path)
        
        # Extract raw_h5 file path
        if not self.recording_file.endswith('.raw.h5'):
            recording_file = self.recording_file + '.raw.h5'
        else:
            recording_file = self.recording_file
        recording = MaxwellRecordingExtractor(recording_file)

        # Run RT sort
        rt_sort = self.rt_sort.sort_offline(recording, reset=True, verbose=self.verbose, inter_path=self.inter_path)
        sd = SpikeData(rt_sort)

        # Remove the model from the rt_sort object for pickling
        self.rt_sort.model = None

        # Only return spike data, since we already have the other info!
        return {
            'spike_data': sd
        }



        # Get electrodes per neuron

    def _run_kilosort2(self, experiment, spike_data_dir: Path) -> Dict[str, Any]:
        """Run Kilosort2 spike sorting."""
        try:
            from braindance.core.spikesorter.kilosort2 import RunKilosort
            
            # Implementation would go here
            print("   Kilosort2 sorting not fully implemented yet")
            return self._run_mock_sort(experiment, spike_data_dir)
            
        except ImportError:
            print("   Warning: Kilosort2 not available")
            return self._run_mock_sort(experiment, spike_data_dir)
    
    def _run_mock_sort(self, experiment, spike_data_dir: Path) -> Dict[str, Any]:
        """Generate mock sorting data for testing."""
        print("   Generating mock spike sorting data")
        
        n_neurons = 10
        neurons = list(range(n_neurons))
        
        # Generate mock data
        channels_per_neuron = {}
        electrodes_per_neuron = {}
        spike_trains = {}
        templates = {}
        
        for neuron_id in neurons:
            # Assign 1-3 channels per neuron
            n_channels = np.random.randint(1, 4)
            channels = np.random.choice(range(100), n_channels, replace=False)
            channels_per_neuron[neuron_id] = channels.tolist()
            
            # Convert to electrodes if mapping available
            if hasattr(self, 'mapping') and self.mapping:
                electrodes = [self.mapping.get_electrode(ch) for ch in channels]
                electrodes_per_neuron[neuron_id] = electrodes
            else:
                electrodes_per_neuron[neuron_id] = channels.tolist()
            
            # Generate mock spike times
            n_spikes = np.random.randint(self.min_spikes, 1000)
            spike_trains[neuron_id] = np.sort(np.random.rand(n_spikes) * 
                                             (self.recording_window_ms[1] / 1000))
            
            # Generate mock template
            templates[neuron_id] = np.random.randn(60, n_channels)
        
        # Save spike data
        spike_data = {
            'spike_trains': spike_trains,
            'templates': templates,
            'mock_data': True
        }
        spike_data_file = spike_data_dir / 'spike_data_mock.pkl'
        with open(spike_data_file, 'wb') as f:
            pickle.dump(spike_data, f)
        
        print(f"   Generated {len(neurons)} mock neurons")
        
        # Return paths and JSON-serializable data
        return {
            'neurons': neurons,
            'channels_per_neuron': channels_per_neuron,
            'electrodes_per_neuron': electrodes_per_neuron,
            'spike_trains': spike_trains,
            'templates': templates,
            'spike_data_file': {
                'absolute': str(spike_data_file.absolute()),
                'relative': os.path.relpath(spike_data_file, experiment.save_dir)
            },
            'rt_sort_object': None  # No RT sort object for mock data
        }


class FootprintPhaseV3(AnalysisPhaseV3):
    """
    Calculate spatial footprints of neurons.
    
    Requires:
        - neurons: List of neuron IDs
        - templates: Spike templates
        - electrodes_per_neuron: Electrode assignments
    
    Provides:
        - footprints: Spatial footprint for each neuron
        - footprint_metrics: Metrics about footprints
    """
    
    inputs = ['neurons', 'templates', 'electrodes_per_neuron']
    outputs = ['footprints', 'footprint_metrics']
    
    def __init__(self, threshold: float = 0.1, name: str = None):
        super().__init__(name or "FootprintPhase")
        self.threshold = threshold
    
    def run(self, experiment) -> Dict[str, Any]:
        """Calculate footprints."""
        print(f"   Calculating footprints for {len(self.neurons)} neurons")
        
        footprints = {}
        footprint_metrics = {}
        
        for neuron_id in self.neurons:
            template = self.templates[neuron_id]
            electrodes = self.electrodes_per_neuron[neuron_id]
            
            # Calculate footprint (placeholder)
            footprint = {
                'electrodes': electrodes,
                'amplitudes': np.max(np.abs(template), axis=0).tolist(),
                'center': np.mean(electrodes) if electrodes else 0
            }
            
            footprints[neuron_id] = footprint
            
            # Calculate metrics
            footprint_metrics[neuron_id] = {
                'n_electrodes': len(electrodes),
                'max_amplitude': np.max(np.abs(template)),
                'spatial_extent': np.std(electrodes) if len(electrodes) > 1 else 0
            }
        
        return {
            'footprints': footprints,
            'footprint_metrics': pd.DataFrame(footprint_metrics).T
        }


class ConnectivityPhaseV3(AnalysisPhaseV3):
    """
    Analyze connectivity between neurons.
    
    Requires:
        - neurons: List of neuron IDs
        - spike_trains: Spike times for each neuron
    
    Provides:
        - connectivity_matrix: Pairwise connectivity strengths
    """
    
    inputs = ['spike_data']
    outputs = ['connectivity_matrix']
    
    def __init__(self, method: str = 'spike_time_tiling',
                 window_ms: float = 20.0,
                 significance_threshold: float = 0.1,
                 name: str = None):
        super().__init__(name or "ConnectivityPhase")
        self.method = method
        self.window_ms = window_ms
        self.significance_threshold = significance_threshold
        self.spike_data: SpikeData | None= None
        self.neurons = None

    def configure_from_experiment(self, experiment: Experiment):
        """Auto-configure from experiment."""
        super().configure_from_experiment(experiment)
        # self.spike_data = experiment.data.spike_data # Should auto-populate
        # self.neurons = list(range(self.spike_data.N))
    
    def run(self, experiment) -> Dict[str, Any]:
        """Analyze connectivity."""
        print(f"   Analyzing connectivity using {self.method}")

        if self.method == 'spike_time_tiling':
            connectivity_matrix = self.spike_data.spike_time_tilings(self.window_ms).matrix
        else:
            raise ValueError(f"Invalid method: {self.method}")
        

        return {
            'connectivity_matrix': connectivity_matrix
        }


class CausalAnalysisV3(AnalysisPhaseV3):
    """
    Analyze causal relationships between neurons.
    
    Requires:
        - neurons: List of neuron IDs
        - spike_trains: Spike times
        - connectivity_matrix: Prior connectivity analysis
    
    Provides:
        - causal_matrix: Directed causal strengths
        - causal_graph: NetworkX directed graph
    """
    
    inputs = ['neurons', 'spike_trains', 'connectivity_matrix']
    outputs = ['causal_matrix', 'causal_graph']
    
    def __init__(self, method: str = 'granger',
                 max_lag_ms: float = 100.0,
                 name: str = None):
        super().__init__(name or "CausalAnalysis")
        self.method = method
        self.max_lag_ms = max_lag_ms
    
    def run(self, experiment) -> Dict[str, Any]:
        """Run causal analysis."""
        import networkx as nx
        
        print(f"   Running {self.method} causality analysis")
        
        n_neurons = len(self.neurons)
        
        # Start with connectivity matrix and add directionality
        causal_matrix = self.connectivity_matrix.copy()
        
        # Make it asymmetric to represent directionality (placeholder)
        for i in range(n_neurons):
            for j in range(i+1, n_neurons):
                if np.random.rand() > 0.5:
                    causal_matrix[i, j] *= 1.5
                    causal_matrix[j, i] *= 0.5
                else:
                    causal_matrix[i, j] *= 0.5
                    causal_matrix[j, i] *= 1.5
        
        # Create directed graph
        causal_graph = nx.DiGraph()
        for i, neuron_i in enumerate(self.neurons):
            causal_graph.add_node(neuron_i)
            
        for i in range(n_neurons):
            for j in range(n_neurons):
                if i != j and causal_matrix[i, j] > 0.1:
                    causal_graph.add_edge(
                        self.neurons[i], 
                        self.neurons[j],
                        weight=causal_matrix[i, j]
                    )
        
        print(f"   Causal graph: {causal_graph.number_of_nodes()} nodes, "
              f"{causal_graph.number_of_edges()} edges")
        
        return {
            'causal_matrix': causal_matrix,
            'causal_graph': causal_graph
        }
