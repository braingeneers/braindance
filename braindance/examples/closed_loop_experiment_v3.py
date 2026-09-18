"""
Closed-Loop Plasticity Experiment using Framework V3

This example demonstrates:
1. Recording baseline activity
2. Running RT-Sort to find neurons
3. Analyzing connectivity
4. Selecting a neuron pair with moderate connectivity
5. Running closed-loop stimulation
6. Recording post-activity and measuring connectivity changes
"""
from braindance.core.phases_v3.experiment_v3 import Experiment
from braindance.core.phases_v3.phases3 import RecordPhaseV3
from braindance.core.phases_v3.phases_analysis_3 import (
    RTSortPhaseV3, ConnectivityPhaseV3, FootprintPhaseV3
)
from braindance.core.phases_v3.phases3_selection import SelectionPhaseV3
from braindance.core.phases_v3.phases3_loop import ClosedLoopPhaseV3
from braindance.core.phases_v3.phase_base_v3 import PhaseGroup


def run_closed_loop_plasticity_experiment(config):
    """
    Run a complete closed-loop plasticity experiment.
    
    This experiment tests whether closed-loop stimulation can modify
    connectivity between neuron pairs.
    """
    

    # Build experiment pipeline
    exp = (Experiment("closed_loop_plasticity", config=config)
           
           # Phase 1: Baseline recording and analysis
           .add_phase_group([
               RecordPhaseV3(duration=300, name="baseline"),
               RTSortPhaseV3(min_spikes=100),
               ConnectivityPhaseV3(
                   method='correlation',
                   window_ms=50.0,
                   significance_threshold=0.05
               )
           ], name="baseline_analysis")
           
           # Phase 2: Select neuron pair for closed-loop
           .add_phase(SelectionPhaseV3(
               connectivity_range=(0.3, 0.6),  # Moderate connectivity
               selection_method='best_in_range',
               min_distance=100.0  # Ensure spatial separation
           ))
           
           # Phase 3: Closed-loop stimulation
           .add_phase(ClosedLoopPhaseV3(
               duration=600,  # 10 minutes
               amp_mv=400,
               detection_threshold=-3.5,
               delay_ms=1.0  # 1ms delay for synaptic effect
           ))
           
           # Phase 4: Post-stimulation recording and analysis
           .add_phase_group([
               RecordPhaseV3(duration=300, name="post_stim"),
               ConnectivityPhaseV3(
                   method='correlation',
                   window_ms=50.0,
                   significance_threshold=0.05,
                   name="post_connectivity"
               )
           ], name="post_analysis"))
    
    # Run the experiment
    print("\n" + "="*60)
    print("Closed-Loop Plasticity Experiment")
    print("="*60)
    
    success = exp.run()
    
    if success:
        print("\n✅ Experiment completed successfully!")
        
        # Analyze results
        analyze_plasticity_results(exp)
        
        # Save all data
        data_path = exp.save_data()
        summary_path = exp.save_summary()
        
        print(f"\n💾 Data saved to: {data_path}")
        print(f"📄 Summary saved to: {summary_path}")
        
    else:
        print("\n❌ Experiment failed!")
        check_failure_point(exp)
    
    return exp


def analyze_plasticity_results(exp):
    """Analyze the plasticity induced by closed-loop stimulation."""
    print("\n📊 Analyzing Plasticity Results")
    print("="*40)
    
    # Get selected neuron pair
    selected_pair = exp.data.selected_neuron_pair
    print(f"\nSelected neuron pair: {selected_pair[0]} → {selected_pair[1]}")
    
    # Get connectivity matrices
    pre_connectivity = exp.data.connectivity_matrix
    post_result = exp.get_result('post_connectivity')
    
    if post_result and 'result' in post_result:
        post_connectivity = post_result['result'].get('connectivity_matrix')
        
        if post_connectivity is not None:
            # Find indices of selected neurons
            neurons = exp.data.neurons
            idx_0 = neurons.index(selected_pair[0])
            idx_1 = neurons.index(selected_pair[1])
            
            # Compare connectivity
            pre_strength = pre_connectivity[idx_0, idx_1]
            post_strength = post_connectivity[idx_0, idx_1]
            change = post_strength - pre_strength
            percent_change = (change / pre_strength) * 100 if pre_strength > 0 else 0
            
            print(f"\nConnectivity Analysis:")
            print(f"  Pre-stimulation:  {pre_strength:.3f}")
            print(f"  Post-stimulation: {post_strength:.3f}")
            print(f"  Change:           {change:+.3f} ({percent_change:+.1f}%)")
            
            # Check if significant
            if abs(percent_change) > 10:
                print(f"\n🎯 Significant connectivity change detected!")
            else:
                print(f"\n📊 No significant connectivity change")
            
            # Closed-loop statistics
            print(f"\nClosed-Loop Statistics:")
            print(f"  Spikes detected: {exp.data.spike_count}")
            print(f"  Stimulations:    {exp.data.stim_count}")
            if exp.data.spike_count > 0:
                efficiency = (exp.data.stim_count / exp.data.spike_count) * 100
                print(f"  Efficiency:      {efficiency:.1f}%")
    
    # Additional analysis could include:
    # - Changes in other neuron pairs
    # - Network-wide connectivity changes
    # - Spatial patterns of plasticity
    
    return exp


def check_failure_point(exp):
    """Diagnose where the experiment failed."""
    print("\n🔍 Checking failure point...")
    
    for i, result in enumerate(exp.results):
        if not result.get('success', True):
            print(f"\nFailed at phase {i+1}: {result['phase_name']}")
            print(f"Error: {result.get('error', 'Unknown error')}")
            
            # Check common issues
            if 'SelectionPhase' in result['phase_name']:
                print("\n💡 Possible issues:")
                print("  - No neuron pairs found in connectivity range")
                print("  - Try adjusting connectivity_range parameters")
                print("  - Check if RT-Sort found enough neurons")
            
            elif 'ClosedLoopPhase' in result['phase_name']:
                print("\n💡 Possible issues:")
                print("  - Selected neurons not in stim_electrodes")
                print("  - Detection threshold too strict")
                print("  - Check electrode mappings")
            
            break


def run_simple_test(config):
    """Run a simpler test version with shorter durations."""

    
    exp = (Experiment("test_closed_loop", config=config)
           .add_phase(RecordPhaseV3(duration=60))  # 1 minute
           .add_phase(RTSortPhaseV3(min_spikes=10))  # Lower threshold
           .add_phase(ConnectivityPhaseV3())
           .add_phase(SelectionPhaseV3(connectivity_range=(0.1, 0.8)))  # Wider range
           .add_phase(ClosedLoopPhaseV3(duration=120)))  # 2 minutes
    
    success = exp.run()
    
    if success:
        print(f"\n✅ Test completed!")
        print(f"Selected pair: {exp.data.selected_neuron_pair}")
        print(f"Closed-loop stats: {exp.data.spike_count} spikes, {exp.data.stim_count} stims")
    
    return exp


if __name__ == "__main__":

    # argparse config
    import argparse
    parser = argparse.ArgumentParser(description="Closed-Loop Plasticity Experiment")
    parser.add_argument("--config", type=str, default=None, help="Maxwell configuration file")
    parser.add_argument("--stim_electrodes", type=list, default=[1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008], help="List of stimulation electrodes")
    parser.add_argument("--verbose", type=bool, default=True, help="Verbose mode")
    parser.add_argument("--record_duration", type=int, default=300, help="Recording duration in seconds")
    parser.add_argument("--simple_test", type=bool, default=False, help="Run simple test")
    args = parser.parse_args()

    config = {
        "config": args.config,
        "stim_electrodes": args.stim_electrodes,
        "verbose": args.verbose,
        "record_duration": args.record_duration
    }

    # Run the full experiment
    if args.simple_test:
        exp = run_simple_test(config)
    else:
        exp = run_closed_loop_plasticity_experiment(config)
    
    