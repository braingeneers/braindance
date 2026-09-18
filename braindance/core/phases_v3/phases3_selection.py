"""
Selection Phases for Experiment Framework V3

These phases help select neurons or electrode pairs based on various criteria.
"""
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

from .phase_base_v3 import AnalysisPhaseV3


class SelectionPhaseV3(AnalysisPhaseV3):
    """
    Select neuron pairs based on connectivity criteria.
    
    Requires:
        - connectivity_matrix: Pairwise connectivity strengths
        - neurons: List of neuron IDs
        - electrodes_per_neuron: Electrode assignments (optional)
    
    Provides:
        - selected_neuron_pair: Tuple of (input_neuron, output_neuron)
        - selection_candidates: List of all candidate pairs
        - selection_scores: Scores for each candidate
    """
    
    inputs = ['connectivity_matrix', 'neurons']
    outputs = ['selected_neuron_pair', 'selection_candidates', 'selection_scores']
    
    def __init__(self, connectivity_range: Tuple[float, float] = (0.3, 0.6),
                 min_distance: Optional[float] = None,
                 max_distance: Optional[float] = None,
                 selection_method: str = 'best_in_range',
                 name: str = None):
        """
        Initialize selection phase.
        
        Args:
            connectivity_range: (min, max) connectivity strength to consider
            min_distance: Minimum spatial distance between neurons (if electrodes available)
            max_distance: Maximum spatial distance between neurons
            selection_method: How to select from candidates:
                - 'best_in_range': Select pair with highest connectivity in range
                - 'median_in_range': Select pair closest to median connectivity
                - 'random_in_range': Random selection from valid pairs
        """
        super().__init__(name or "SelectionPhase")
        self.connectivity_range = connectivity_range
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.selection_method = selection_method
        
        # Will be auto-populated
        self.connectivity_matrix = None
        self.neurons = None
        self.electrodes_per_neuron = None
    
    def calculate_distance(self, neuron1: int, neuron2: int) -> Optional[float]:
        """Calculate spatial distance between neurons if electrode info available."""
        if not hasattr(self, 'electrodes_per_neuron') or self.electrodes_per_neuron is None:
            return None
            
        elecs1 = self.electrodes_per_neuron.get(neuron1, [])
        elecs2 = self.electrodes_per_neuron.get(neuron2, [])
        
        if not elecs1 or not elecs2:
            return None
            
        # TODO: Implement actual spatial distance calculation based on mapping/position
        return abs(np.mean(elecs1) - np.mean(elecs2))
    
    def run(self, experiment) -> Dict[str, Any]:
        """Select neuron pairs based on criteria."""
        print(f"   Selecting neuron pairs with connectivity in range {self.connectivity_range}")
        
        candidates = []
        n_neurons = len(self.neurons)
        
        # Find all pairs within connectivity range
        for i in range(n_neurons):
            for j in range(n_neurons):
                if i == j:
                    continue
                    
                conn_strength = self.connectivity_matrix[i, j]
                
                # Check connectivity range
                if not (self.connectivity_range[0] <= conn_strength <= self.connectivity_range[1]):
                    continue
                
                # Check spatial constraints if applicable
                if self.min_distance is not None or self.max_distance is not None:
                    distance = self.calculate_distance(self.neurons[i], self.neurons[j])
                    if distance is not None:
                        if self.min_distance and distance < self.min_distance:
                            continue
                        if self.max_distance and distance > self.max_distance:
                            continue
                
                candidates.append({
                    'source': self.neurons[i],
                    'target': self.neurons[j],
                    'source_idx': i,
                    'target_idx': j,
                    'connectivity': conn_strength,
                    'distance': self.calculate_distance(self.neurons[i], self.neurons[j])
                })
        
        print(f"   Found {len(candidates)} candidate pairs")
        
        if not candidates:
            raise ValueError(f"No neuron pairs found with connectivity in range {self.connectivity_range}")
        
        # Select based on method
        if self.selection_method == 'best_in_range':
            selected = max(candidates, key=lambda x: x['connectivity'])
        elif self.selection_method == 'median_in_range':
            # Sort by connectivity and pick middle
            sorted_candidates = sorted(candidates, key=lambda x: x['connectivity'])
            selected = sorted_candidates[len(sorted_candidates) // 2]
        elif self.selection_method == 'random_in_range':
            selected = np.random.choice(candidates)
        else:
            raise ValueError(f"Unknown selection method: {self.selection_method}")
        
        selected_pair = (selected['source'], selected['target'])
        
        print(f"   Selected pair: {selected_pair[0]} -> {selected_pair[1]} "
              f"(connectivity: {selected['connectivity']:.3f})")
        
        # Extract scores for all candidates
        selection_scores = [c['connectivity'] for c in candidates]
        
        return {
            'selected_neuron_pair': selected_pair,
            'selection_candidates': candidates,
            'selection_scores': selection_scores
        }


class SpatialSelectionPhaseV3(AnalysisPhaseV3):
    """
    Select neurons based on spatial criteria.
    
    Requires:
        - neurons: List of neuron IDs
        - footprints: Spatial footprints from FootprintPhase
        - electrodes_per_neuron: Electrode assignments
    
    Provides:
        - selected_neurons: List of selected neuron IDs
        - spatial_clusters: Groupings of spatially related neurons
    """
    
    inputs = ['neurons', 'footprints', 'electrodes_per_neuron']
    outputs = ['selected_neurons', 'spatial_clusters']
    
    def __init__(self, selection_criteria: str = 'distributed',
                 n_select: Optional[int] = None,
                 min_separation: float = 100.0,
                 name: str = None):
        """
        Initialize spatial selection.
        
        Args:
            selection_criteria: How to select neurons:
                - 'distributed': Maximize spatial coverage
                - 'clustered': Select from same region
                - 'edges': Select from array edges
                - 'center': Select from array center
            n_select: Number of neurons to select (None = all meeting criteria)
            min_separation: Minimum separation for 'distributed' mode
        """
        super().__init__(name or "SpatialSelectionPhase")
        self.selection_criteria = selection_criteria
        self.n_select = n_select
        self.min_separation = min_separation
    
    def run(self, experiment) -> Dict[str, Any]:
        """Select neurons based on spatial criteria."""
        print(f"   Selecting neurons using '{self.selection_criteria}' criteria")
        
        # Calculate spatial positions (placeholder - use footprint centers)
        positions = {}
        for neuron_id in self.neurons:
            if neuron_id in self.footprints:
                positions[neuron_id] = self.footprints[neuron_id].get('center', 0)
        
        selected_neurons = []
        
        if self.selection_criteria == 'distributed':
            # Select neurons that are well-separated
            remaining = list(positions.keys())
            while remaining and (self.n_select is None or len(selected_neurons) < self.n_select):
                # Pick one
                if not selected_neurons:
                    selected = remaining.pop(0)
                else:
                    # Find neuron furthest from already selected
                    max_min_dist = 0
                    best_candidate = None
                    for candidate in remaining:
                        min_dist = min(abs(positions[candidate] - positions[s]) 
                                     for s in selected_neurons)
                        if min_dist > max_min_dist:
                            max_min_dist = min_dist
                            best_candidate = candidate
                    
                    if best_candidate and max_min_dist >= self.min_separation:
                        selected = best_candidate
                        remaining.remove(selected)
                    else:
                        break
                
                selected_neurons.append(selected)
        
        elif self.selection_criteria == 'clustered':
            # Select neurons that are close together
            if positions:
                center = np.mean(list(positions.values()))
                # Sort by distance to center
                sorted_neurons = sorted(positions.keys(), 
                                      key=lambda n: abs(positions[n] - center))
                n_to_select = self.n_select or len(sorted_neurons)
                selected_neurons = sorted_neurons[:n_to_select]
        
        # Create spatial clusters (simple grouping by position)
        spatial_clusters = []
        if selected_neurons:
            # Simple clustering - group nearby neurons
            cluster_threshold = 50.0
            clusters = []
            for neuron in selected_neurons:
                pos = positions.get(neuron, 0)
                # Find existing cluster
                added = False
                for cluster in clusters:
                    cluster_positions = [positions.get(n, 0) for n in cluster]
                    if any(abs(pos - cp) < cluster_threshold for cp in cluster_positions):
                        cluster.append(neuron)
                        added = True
                        break
                if not added:
                    clusters.append([neuron])
            
            spatial_clusters = clusters
        
        print(f"   Selected {len(selected_neurons)} neurons in {len(spatial_clusters)} clusters")
        
        return {
            'selected_neurons': selected_neurons,
            'spatial_clusters': spatial_clusters
        }


class ActivitySelectionPhaseV3(AnalysisPhaseV3):
    """
    Select neurons based on activity patterns.
    
    Requires:
        - neurons: List of neuron IDs
        - spike_trains: Spike times for each neuron
        - recording_duration: Duration of recording (optional)
    
    Provides:
        - selected_neurons: List of selected neuron IDs
        - activity_metrics: Metrics used for selection
    """
    
    inputs = ['neurons', 'spike_trains']
    outputs = ['selected_neurons', 'activity_metrics']
    
    def __init__(self, min_rate: float = 0.1,
                 max_rate: float = 100.0,
                 regularity_range: Optional[Tuple[float, float]] = None,
                 n_select: Optional[int] = None,
                 name: str = None):
        """
        Initialize activity-based selection.
        
        Args:
            min_rate: Minimum firing rate (Hz)
            max_rate: Maximum firing rate (Hz)
            regularity_range: (min, max) CV of ISI for regularity filtering
            n_select: Number to select (None = all meeting criteria)
        """
        super().__init__(name or "ActivitySelectionPhase")
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.regularity_range = regularity_range
        self.n_select = n_select
        
        # Optional - will try to get from data
        self.recording_duration = None
    
    def configure_from_experiment(self, experiment):
        """Auto-configure from experiment."""
        super().configure_from_experiment(experiment)
        
        # Try to get recording duration if available
        if 'recording_duration' in experiment.data:
            self.recording_duration = experiment.data.recording_duration
    
    def calculate_metrics(self, spike_times: np.ndarray, duration: float) -> Dict[str, float]:
        """Calculate activity metrics for a neuron."""
        n_spikes = len(spike_times)
        
        metrics = {
            'n_spikes': n_spikes,
            'firing_rate': n_spikes / duration if duration > 0 else 0
        }
        
        # Calculate ISI regularity if enough spikes
        if n_spikes > 10:
            isis = np.diff(spike_times)
            metrics['cv_isi'] = np.std(isis) / np.mean(isis) if np.mean(isis) > 0 else np.inf
            metrics['burst_index'] = np.sum(isis < 0.01) / len(isis)  # Fraction of short ISIs
        else:
            metrics['cv_isi'] = np.nan
            metrics['burst_index'] = np.nan
        
        return metrics
    
    def run(self, experiment) -> Dict[str, Any]:
        """Select neurons based on activity."""
        print(f"   Selecting neurons by activity (rate: {self.min_rate}-{self.max_rate} Hz)")
        
        # Determine recording duration
        duration = self.recording_duration
        if duration is None:
            # Estimate from spike times
            max_times = [np.max(times) if len(times) > 0 else 0 
                        for times in self.spike_trains.values()]
            duration = max(max_times) if max_times else 300.0  # Default 5 min
        
        # Calculate metrics for all neurons
        all_metrics = {}
        candidates = []
        
        for neuron_id in self.neurons:
            spike_times = self.spike_trains.get(neuron_id, np.array([]))
            metrics = self.calculate_metrics(spike_times, duration)
            all_metrics[neuron_id] = metrics
            
            # Check criteria
            if self.min_rate <= metrics['firing_rate'] <= self.max_rate:
                if self.regularity_range is None or (
                    not np.isnan(metrics['cv_isi']) and
                    self.regularity_range[0] <= metrics['cv_isi'] <= self.regularity_range[1]
                ):
                    candidates.append(neuron_id)
        
        # Select subset if requested
        if self.n_select and len(candidates) > self.n_select:
            # Sort by firing rate and take middle neurons (most typical)
            candidates.sort(key=lambda n: all_metrics[n]['firing_rate'])
            start_idx = (len(candidates) - self.n_select) // 2
            selected_neurons = candidates[start_idx:start_idx + self.n_select]
        else:
            selected_neurons = candidates
        
        print(f"   Selected {len(selected_neurons)} neurons from {len(candidates)} candidates")
        
        return {
            'selected_neurons': selected_neurons,
            'activity_metrics': all_metrics
        } 