"""Preserved GUI causal stage from proj/cartpole_v1/causal_analysis.py.

Requires live Maxwell and the historical analysis dependencies.
"""

def main():
    from braindance.core.maxwell_env import MaxwellEnv
    from braindance.core.params import maxwell_params
    from braindance.core.phases import PhaseManager, NeuralSweepPhase, RecordPhase
    from braindance.core.phases_analysis import HeatmapPhase, FootprintPhase, CausalAnalysis
    from braindance.analysis.data_loader import AnalysisDAO
    
    import numpy as np
    import json
    import argparse
    import os
    
    from matplotlib import pyplot as plt
    
    
    parser = argparse.ArgumentParser(description='JSON config file for neural config analysis')
    parser.add_argument('--json',"-j", type=str, default='cartpole_params.json')
    parser.add_argument('--do_recording',"-d", type=bool, default=True)
    
    args = parser.parse_args()
    json_file = args.json
    do_recording = args.do_recording
    json_params = json.load(open(json_file, 'r'))
    
    
    
    params = maxwell_params.copy()
    params['save_dir'] = f'{json_params["save_dir"]}' # Path to the data directory, will be created if it doesn't exist
    params['name'] = json_params['name'] + '_causal' # Name of the experiment
    
    params['max_time_sec'] = 60*30 # 30 min
    
    params['config'] = json_params['config']#'config.cfg'# Path to the config file
    params['observation_type'] = 'raw'
    
    params['stim_electrodes'] = json_params['stim_electrodes']
    
    
    
    env = MaxwellEnv(**params)
    
    # Run causal phase
    # This sweeps every stim electrode n_replicates times at a given frequency/amplitude
    causal_freq = 2 # Hz
    n_replicates = 50 
    neuron_list = np.arange(len(params['stim_electrodes']))
    causal_phase = NeuralSweepPhase(env, neuron_list, amp_bounds=400,stim_freq=causal_freq,
                                     tag="causal", replicates=n_replicates, order='rna', verbose=True,
                                     single_connect=True, phase_length=200)
    
    # This phase is strictly to get the filepath
    heatmap_phase = HeatmapPhase("Heatmap1", verbose=True,  make_gif=False, make_plots=False)
    
    # Causal analysis
    causal_analysis = CausalAnalysis(wind_ms = 300, verbose=True, tag='causal', clean_data_path=True)
    
    
    
    
    
    # Build experiment
    phase_manager = PhaseManager(env, verbose=True)
    
    
    phase_manager.add_phase(causal_phase)
    phase_manager.add_phase(heatmap_phase)
    phase_manager.add_phase(causal_analysis)
    
    
    phase_manager.summary()
    phase_manager.run()
    
    # Get the analysis object
    analysis = phase_manager.analysis_dao
    
    from braindance.analysis import plot_helper as ph
    # Plot heatmap of the reactivity matrix
    ph.plot_reactivity(analysis)
    # Save reactivity
    name = analysis.name
    
    # make plots dir
    plot_dir = params['save_dir'] + '/plots/'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    plt.savefig(f'{plot_dir}{name}_reactivity.png')
    
    # Make all of the reactivity traces
    print("Making time scatter")
    ph.plot_reactivity_traces(analysis, save_dir = plot_dir, time_scatter=True)
    print("Making raw traces")
    ph.plot_reactivity_traces(analysis, save_dir = plot_dir)
    
    ph.plot_causal_connectivity(analysis, save_dir = plot_dir)
    
    
    # plt.show()
    
    # Now plot the reactions on index 2 of reactivity channels to
    # each of the stim electrodes
    # plt.figure()
    # for i in range(len(analysis.stim_electrodes)):
    #     plt.plot(analysis.reactivity[i,3,:], label = analysis.stim_electrodes[i])
    # plt.legend()
    # plt.show()
    
    


if __name__ == "__main__":
    main()
