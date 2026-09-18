"""
BraindanceExperiment Class

Represents a single BrainDance experiment and provides methods for
extracting metadata like stimulation counts, frequencies, and experiment type.
"""

import os
import pandas as pd

from .s3_helpers import construct_log_path, load_log_from_s3
from .validators import calculate_stim_frequency, categorize_frequency, calculate_freq_from_stim_count

from braindance.io import open_file


class BraindanceExperiment:
    """
    Represents a single Braindance experiment across different data structures.
    
    This class handles experiments whether they are stored in the original
    structure (freqs/freqs_cont_N) or newer structures (exp1/exp1_cont_N).
    
    Attributes:
        base_path: S3 base path
        chip: Chip identifier
        exp: Experiment identifier
        sub_exp: Sub-experiment identifier (optional)
        exp_path: Full path to the experiment directory
    """
    
    def __init__(self, base_path, chip, exp, path_manager, sub_exp=None):
        """
        Initialize a Braindance experiment.
        
        Args:
            base_path: S3 base path
            chip: Chip identifier
            exp: Experiment identifier
            path_manager: DataPathManager instance for S3 operations
            sub_exp: Optional sub-experiment identifier (e.g., 'freqs_cont_1')
        """
        self.base_path = base_path
        self.chip = chip
        self.exp = exp
        self.path_manager = path_manager
        self.exp_path = path_manager.get_experiment_path(base_path, chip, exp)
        self.sub_exp = sub_exp
        self.log_files = None
        self.raw_files = None
    
    def get_logs(self, log_suffixes=None):
        """
        Get log files for the experiment.
        
        Args:
            log_suffixes: List of log file suffixes to filter by
            
        Returns:
            tuple: (list of log DataFrames, list of log type names)
        """
        if log_suffixes is None:
            log_suffixes = ['_log.csv', '_game_log.csv', '_pattern_log.csv', 
                           '_reward_log.csv', '_causal_log.csv']
        
        # If this is a sub-experiment, find its specific log file
        if self.sub_exp is not None:
            log_path = self.path_manager.get_log_file_for_sub_experiment(
                self.exp_path, self.sub_exp, log_suffixes
            )
            
            if log_path:
                try:
                    with open_file(log_path) as f:
                        log_name = os.path.basename(log_path)
                        log_df = pd.read_csv(f)
                        
                        # Extract log type from filename
                        log_type = None
                        for suffix in log_suffixes:
                            if log_name.endswith(suffix):
                                log_type = suffix.split('.')[0].split('_')[1]
                                break
                        
                        return [log_df], [log_type] if log_type else ["log"]
                except Exception as e:
                    print(f"Error reading sub-experiment log file {log_path}: {e}")
                    return [], []
            else:
                return [], []
        
        # Otherwise, get all logs for the main experiment
        log_paths = self.path_manager.list_log_files(self.exp_path, log_suffixes)
        
        log_objs = []
        log_names = []
        
        for log_path in log_paths:
            try:
                with open_file(log_path) as f:
                    log_name = os.path.basename(log_path)
                    log_df = pd.read_csv(f)
                    log_objs.append(log_df)
                    
                    # Extract log type from filename
                    for suffix in log_suffixes:
                        if log_name.endswith(suffix):
                            log_type = suffix.split('.')[0].split('_')[1]
                            log_names.append(log_type)
                            break
            except Exception as e:
                print(f"Error reading log file {log_path}: {e}")
        
        return log_objs, log_names
    
    def get_raw_data(self):
        """
        Get raw data files for the experiment.
        
        Returns:
            list: List of raw data file paths (.raw.h5 files)
        """
        if self.raw_files is None:
            self.raw_files = self.path_manager.get_raw_data_files(self.exp_path)
        return self.raw_files
    
    def determine_experiment_type(self):
        """
        Determine the type of experiment based on stimulation presence.
        
        Returns:
            str: 'stim' if stimulations found, 'spontaneous' otherwise
        """
        n_stims = self.count_stimulations()
        return 'stim' if n_stims > 0 else 'spontaneous'
    
    def count_stimulations(self):
        """
        Count the number of stimulations in the experiment.
        
        Returns:
            int: Number of stimulations (rows in log file), or 0 if not found
        """
        logs, _ = self.get_logs(['_log.csv', '_causal_log.csv'])
        
        if logs and len(logs) > 0:
            return logs[0].shape[0]
        
        return 0
    
    def get_experiment_info(self):
        """
        Get comprehensive information about the experiment.
        
        Returns:
            dict: Dictionary containing:
                - base_path: S3 base path
                - chip: Chip identifier
                - experiment: Full experiment path
                - full_path: Full S3 path to the experiment
                - n_stims: Number of stimulations
                - freq: Categorized frequency (Hz)
                - calculated_freq: Raw calculated frequency
                - log_available: Whether log file was found
                - type: Experiment type ('stim' or 'spontaneous')
                - n_raw_files: Number of raw data files
                - success: Whether experiment has valid data
        """
        # Count stimulations first
        n_stims = self.count_stimulations()
        
        # Determine experiment type
        exp_type = 'stim' if n_stims > 0 else 'spontaneous'
        
        # Try to calculate frequency from actual stimulus logs
        freq = None
        calculated_freq = None
        log_available = False
        
        if n_stims > 0:
            # Try to get the log path for frequency calculation
            if self.sub_exp:
                full_experiment = f"{self.exp}/{self.sub_exp}"
            else:
                full_experiment = self.exp
            
            log_path = construct_log_path(self.base_path, self.chip, full_experiment)
            
            if log_path:
                stim_log = load_log_from_s3(log_path)
                if stim_log is not None:
                    log_available = True
                    calculated_freq = calculate_stim_frequency(stim_log)
                    if calculated_freq is not None:
                        freq = categorize_frequency(calculated_freq)
            
            # Fallback to count-based frequency calculation
            if freq is None:
                freq = calculate_freq_from_stim_count(n_stims)
        else:
            # No stimulations = spontaneous
            freq = 0.0
        
        # Count raw files
        raw_files = self.get_raw_data()
        n_raw_files = len(raw_files)
        
        # Check if this is a valid experiment
        success = (n_raw_files > 0 or n_stims > 0)
        
        # Construct the full experiment name and path
        if self.sub_exp:
            full_experiment = f"{self.exp}/{self.sub_exp}"
            full_path = f"{self.exp_path}{self.sub_exp}"
        else:
            full_experiment = self.exp
            full_path = self.exp_path
        
        return {
            'base_path': self.base_path,
            'chip': self.chip,
            'experiment': full_experiment,
            'full_path': full_path,
            'n_stims': n_stims,
            'freq': freq,
            'calculated_freq': calculated_freq,
            'log_available': log_available,
            'type': exp_type,
            'n_raw_files': n_raw_files,
            'success': success
        }
