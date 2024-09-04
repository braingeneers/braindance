from pathlib import Path

import numpy as np
from spikeinterface.extractors import BaseRecording
import torch
from torch.utils.data import DataLoader

from braindance.core.spikedetector import data, utils
from braindance.core.spikedetector.model2 import ModelSpikeSorter

def train_detection_model(recordings: list,
                          dl_folder_name="dl_folder",
                          validation_recording=None,
                        
                          # Dataset parameters
                          thresh_amp=18.88275, 
                          thresh_std=0.6,
                          sample_size_ms=10,
                          recording_spike_before_ms=2, recording_spike_after_ms=2,
                        
                          # Training parameters
                          samples_per_waveform=2, num_wfs_probs=[0.5, 0.3, 0.12, 0.06, 0.02],
                          isi_wf_min_ms=0.2, isi_wf_max_ms=None,
                          
                          learning_rate=7.76e-4, 
                          
                          momentum=0.85, 
                          training_thresh=0.01, learning_rate_patience=5, learning_rate_decay=0.4,
                          epoch_patience=10, max_num_epochs=200,
                          batch_size=1, num_workers=0, shuffle=True, 
                          training_random_seed=231,
                        
                          # Detection model parameters
                          input_scale=0.01, input_scale_decay=0.1,
                        
                          # General parameters
                          device="cuda", dtype=torch.float16,
                        
                          # run_kilosort2 parameters
                          **run_kilosort2_kwargs):
    """
    Train a DL detection model with :recording_files:
    
    Params:
        General parameters:
            recordings:
                A list containing the recordings to use for training the detection model. The length recordings must be at least two. 
                Each element can be one of the following:
                    - Path to a recording file in .h5 or .nwb format
                    - Path to a folder containing “sorted.npz” and “scaled_traces.npy”. 
                        This is used if the recordings have already been sorted, and you do not want to re-sort them. 
                    - A recording object loaded with SpikeInterface. See SpikeInterface's Extractor Module (https://spikeinterface.readthedocs.io/en/latest/modules/extractors.html) for details
            dl_folder_name:
                For each recording that needs to be sorted, the results will be stored in the same folder as the recording in the folder dl_folder_name 
        
        Dataset parameters:
            thresh_amp:
                Only waveforms whose amplitude is at least thresh_amp (microvolts) will be selected to train and test the detection model
            thresh_std:
                Only waveforms in which the standard deviation of the trough divided by the amplitude is at most thresh_std will be selected to train and test the detection model
            sample_size_ms:
                The size of the input samples fed into the detection model (milliseconds)
            recording_spike_before_ms:
                When extracting noise from the recording, do not extract if there was a spike within recording_spike_before_ms milliseconds
            recording_spike_after_ms:
                When extracting noise from the recording, do not extract if there is a spike within recording_spike_after_ms milliseconds
                
        Training parameters:
            samples_per_waveform:
                The number of samples in an epoch equals samples_per_waveform * the total number of waveformsrms
            num_wfs_probs:
                The ith element refers to the probability (decimal) that i waveforms will appear in a training or validation sample.
            isi_wf_min_ms:
                If multiple waveforms are pasted into a sample, they must be at least isi_wf_min_ms milliseconds apart
            isi_wf_max_ms:
                If multiple waveforms are pasted into a sample, one waveform will be at most isi_wf_max_ms milliseconds apart from another waveform
                If None, there is no limit
            learning_rate:
                The learning rate to use for training 
            momentum:
                The momentum value to use for training. See https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
            training_thresh:
                If the validation loss does not decrease by training_thresh after an epoch, the patience counter increases by 1
                If it does, the patience counter resets to 0
            learning_rate_decay and learning_rate_patience:
                If the patience counter reaches learning_rate_patience, the learning rate decreases by a factor of learning_rate_decay
            epoch_patience:
                If the patience counter reacehs epoch_patience, training stops for the recording
            max_num_epochs:
                Training stops after max_num_epochs even if the validation loss continues to decrease
            batch_size:
                The batch size used for training
            num_workers:
                The number of workers used to load data. See https://pytorch.org/docs/stable/data.html#single-and-multi-process-data-loading
            shuffle:
                Whether to randomly shuffle the training and validation data for each epoch
            training_random_seed
                The random seed set before trainings starts for reproducibility
            
        Detection model parameters:
            input_scale:
                The recording traces (μV) are multiplied by input_scale before being inputted into the detection model. 
                If the training process indicates “nan” for the loss, then decrease input_scale because the traces are too large for the detection model. 
            input_scale_decay:
                If loss is np.nan, reduce input_scale by factor of input_scale_decay
                
        General parameters:
            device
                The device to use. "cuda" for GPU and "cpu" for CPU
            dtype
                The data type to use
        
        run_kilosort2 parameters
            See braindance.core.spikesorter.kilosort2.py

    """    
    
    # Setup DL folders
    if validation_recording is not None:
        recordings.append(validation_recording)
        
    dl_folders = []
    for rec in recordings:
        if isinstance(rec, BaseRecording):
            if 'file_path' in rec._kwargs:
                file = rec._kwargs['file_path']
            else:
                file = rec._kwargs['file_paths'][0]
        else:
            file = rec
        file = Path(file)
        if file.is_dir():
            if data.is_dl_folder(file):
                dl_folders.append(file)
            else:
                raise ValueError(f"Folder '{file}' does not have the correct files. In 'recording_files', replace it with the corresponding recording file")
        else:
            folder = file.parent / dl_folder_name
            data.setup_dl_folders([rec], [folder], **run_kilosort2_kwargs)
            dl_folders.append(folder)
            
    if validation_recording is not None:
        val_folder = dl_folders[-1]
        dl_folders = dl_folders[:-1]

    # Setup parameters
    ## Model architecture needs to be adjusted if these are changed
    front_buffer_ms=2
    end_buffer_ms=2
    if sample_size_ms <= front_buffer_ms + end_buffer_ms:
        raise ValueError(f"Argument 'sample_size_ms' must be greater than {front_buffer_ms+end_buffer_ms}")
    
    samp_freq = round(np.load(dl_folders[0] / "sorted.npz", allow_pickle=True)['fs'] / 1000)  # kHz 
    sample_size = round(sample_size_ms * samp_freq)
    isi_wf_min = round(isi_wf_min_ms * samp_freq) if isi_wf_min_ms is not None else None
    isi_wf_max = round(isi_wf_max_ms * samp_freq) if isi_wf_max_ms is not None else None
    front_buffer = round(front_buffer_ms * samp_freq)
    end_buffer = round(end_buffer_ms * samp_freq)

    train = data.MultiRecordingDataset(
        rec_paths=dl_folders,
        samples_per_waveform=samples_per_waveform, front_buffer=front_buffer, end_buffer=end_buffer,
        num_wfs_probs=num_wfs_probs,
        isi_wf_min=isi_wf_min, isi_wf_max=isi_wf_max,
        thresh_amp=thresh_amp, thresh_std=thresh_std,
        sample_size=sample_size, ms_before=recording_spike_before_ms, ms_after=recording_spike_after_ms,
        device=device, dtype=dtype, mmap_mode="r",
    )
    num_train = len(train)
    train = DataLoader(train, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    if validation_recording is None:
        print(f"Num. training samples per epoch: {num_train}")
        val = None
    else:
        val = data.MultiRecordingDataset(
            rec_paths=[val_folder],
            samples_per_waveform=samples_per_waveform, front_buffer=front_buffer, end_buffer=end_buffer,
            num_wfs_probs=num_wfs_probs,
            isi_wf_min=isi_wf_min, isi_wf_max=isi_wf_max,
            thresh_amp=thresh_amp, thresh_std=thresh_std,
            sample_size=sample_size, ms_before=recording_spike_before_ms, ms_after=recording_spike_after_ms,
            device=device, dtype=dtype,
        )
        num_val = len(val)
        val = DataLoader(val, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        print(f"Train: {num_train:.0f} samples -- Validation: {num_val:.0f} -- Train/Total: {num_train/(num_train+num_val)*100:.1f}%")

    # Train
    ## Set random seed for reproducibility
    utils.random_seed(training_random_seed)
    
    # Init model
    model = ModelSpikeSorter(num_channels_in=1,
                             sample_size=sample_size, buffer_front_sample=front_buffer, buffer_end_sample=end_buffer,
                             loc_prob_thresh=20, buffer_front_loc=0, buffer_end_loc=0,
                             input_scale=input_scale, samp_freq=samp_freq,
                             device=device, dtype=dtype,
                             architecture_params=(str(samp_freq), 50, "relu", 0, 0, 0, 0))
    
    # Fit
    while True:
        model.init_weights_and_biases("xavier", prelu_init=0)
        model.model.init_final_bias(model.num_output_locs, train.dataset.num_wfs_probs)
    
        train_loss = model.fit(train, val, optim='momentum',
                               num_epochs=max_num_epochs, epoch_patience=epoch_patience, training_thresh=training_thresh,
                               lr=learning_rate, momentum=momentum, 
                               lr_patience=learning_rate_patience, lr_factor=learning_rate_decay, 
                               tune_thresh_every=10, save_best=True)
        if not np.isnan(train_loss):
            break
        
        print(f"\nRestarting training, reducing input_scale: {input_scale} --> {input_scale * input_scale_decay}")
        input_scale *= input_scale_decay
        model.input_scale = input_scale
    
    # Save model
    model.save(rec, verbose=True)
        
    return model
                
def main():
    train_detection_model(
        ["/data/MEAprojects/organoid/intrinsic/dl/{rec}/dl_folder".format(rec=rec)
         for rec in (2950, 2953, 2954, 2957, 5116, 5118)],
        kilosort_path='/home/mea/SpikeSorting/kilosort/Kilosort2',
    )


if __name__ == "__main__":
    main()
    