"""
Simple Example of Experiment Framework V3

This example demonstrates:
- Basic experiment setup
- Method chaining API
- Automatic data passing between phases
- Phase groups
"""
from braindance.core.phases_v3.experiment_v3 import Experiment
from braindance.core.phases_v3.phases3 import RecordPhaseV3, NeuralSweepPhaseV3, FrequencyStimPhaseV3
from braindance.core.phases_v3.phase_base_v3 import PhaseGroup


def main():
    """Run a simple recording and stimulation experiment."""
    
    # Create experiment configuration
    config = {
        "config": "config.cfg",  # Maxwell config file
        "stim_electrodes": [1001, 1002, 1003, 1004],  # Example electrodes
        "record_duration": 300,  # 5 minutes baseline
        "verbose": True
    }
    
    # Build and run experiment using method chaining
    exp = (Experiment("simple_demo", params=config)
           # First record baseline activity
           .add_phase(RecordPhaseV3(duration=300, name="baseline"))
           
           # Then do amplitude sweep to find thresholds
           .add_phase(NeuralSweepPhaseV3(
               amp_bounds=(100, 400, 10),  # 100-400 mV in 10 steps
               stim_freq=0.5,  # 2 second between stims
               replicates=5,
               order='ran'
           ))
           
           # Group of frequency stimulation phases
           .add_phase_group([
               FrequencyStimPhaseV3(
                   stim_command=([0], 250, 100),  # Stim first electrode
                   stim_freq=1.0,
                   duration=60,
                   tag='freq_1hz'
               ),
               FrequencyStimPhaseV3(
                   stim_command=([1], 250, 100),  # Stim second electrode
                   stim_freq=2.0,
                   duration=60,
                   tag='freq_2hz'
               ),
               FrequencyStimPhaseV3(
                   stim_command=([0, 1], 250, 100),  # Stim both
                   stim_freq=0.5,
                   duration=60,
                   tag='freq_paired'
               )
           ], name="frequency_tests")
           
           # Final recording to see changes
           .add_phase(RecordPhaseV3(duration=300, name="post_stim")))
    
    # Run the experiment
    print("\n" + "="*60)
    print("Starting Simple Experiment Demo")
    print("="*60)
    
    success = exp.run()
    
    if success:
        print("\n✅ Experiment completed successfully!")
        
        # Access data from phases
        print("\n📊 Results:")
        print(f"  Baseline recording: {exp.data.recording_file}")
        print(f"  Sweep results: {len(exp.data.sweep_results)} stimulations")
        print(f"  Stimulation count: {exp.data.stim_count}")
        
        # Save experiment data
        data_path = exp.save_data()
        print(f"\n💾 Data saved to: {data_path}")
        
        # Save summary
        summary_path = exp.save_summary()
        print(f"📄 Summary saved to: {summary_path}")
        
    else:
        print("\n❌ Experiment failed!")
        
        # Check which phase failed
        for result in exp.results:
            if not result.get('success', True):
                print(f"\nFailed at: {result['phase_name']}")
                print(f"Error: {result.get('error', 'Unknown')}")
                break


def advanced_example():
    """More advanced example with custom configuration."""
    
    # Can also load config from file
    exp = Experiment("advanced_demo", params={
        "verbose": True,
        "record_duration": 600,
    })
    
    # Load mapping if available
    # exp.load_mapping("baseline_recording.h5")
    
    # Add phases with dependencies
    exp.add_phase(RecordPhaseV3())  # Provides: recording_file
    
    # This would be an analysis phase that requires recording_file
    # exp.add_phase(ActivityAnalysisPhase())  # Requires: recording_file
    
    # Run with validation
    success = exp.run(validate=True)
    
    # Can also run specific phases
    # exp.run(start_from=1, stop_at=3)  # Run phases 1 and 2 only
    
    return exp


if __name__ == "__main__":
    # Run simple demo
    main()
    
    # Uncomment to run advanced example
    # advanced_example() 
