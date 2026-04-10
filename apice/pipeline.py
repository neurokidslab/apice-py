import json

import numpy as np
import pandas as pd
import sys
import time
from pathlib import Path
from datetime import datetime
from tabulate import tabulate
from datetime import timedelta
import matplotlib.pyplot as plt
from datetime import datetime, timezone


import mne
from mne import BaseEpochs


from apice.data_structures import RawAPICE
from apice.io import load_rawapice
from apice.utils import (get_onset_and_duration, get_cfg)
from apice.filter import Filter

# %% CLASSES DEFINITIONS
class Summary():

    def __init__(self, 
                 output_folder,
                 output_file,
                 columns=['file_id', 'step', 'length', 'corrected_data', 'bad_data', 'bad_channels', 'bad_times'],
                 try_loading=True):
        
        if 'file_id' not in columns or 'step' not in columns:
            raise ValueError("Columns must include 'file_id' and 'step'")
        
        self.output_folder = Path(output_folder)
        self.output_file = output_file
        self.output_full_path = self.output_folder / output_file
        if try_loading and self.output_full_path.exists():
            self.load()
        else:
            self.summary_df = pd.DataFrame(columns=columns)
        
    def load(self):
        if self.output_full_path.exists():
            self.summary_df = pd.read_csv(self.output_full_path)
        else:
            print(f"File {self.output_full_path} does not exist. Summary could not be loaded.")

    def save(self):
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.summary_df.to_csv(self.output_full_path, index=False)
    
    def remove_file_from_summary(self, file_id):
        self.summary_df = self.summary_df[self.summary_df['file_id'] != file_id]

    def remove_file_step_from_summary(self, file_id, step):
        self.summary_df = self.summary_df[~((self.summary_df['file_id'] == file_id) & (self.summary_df['step'] == step))]


class SummaryPreprocessing(Summary):

    def __init__(self, 
                 output_folder, 
                 file_name,
                 file_id=None,
                 outputfile_subfix="-summary-preproc.csv", 
                 try_loading=True,
                 ):
        print("Initializing SummaryPreprocessing object")
        file_name = Path(file_name).stem
        outputfile = file_name + outputfile_subfix
        super().__init__(output_folder=output_folder, 
                         output_file=outputfile, 
                         columns=[
                             "file_id", 
                             "step", 
                             "length", 
                             "%_corrected_data", 
                             "%_bad_data", 
                             "%_bad_channels", 
                             "%_bad_times",
                             ],
                         try_loading=try_loading)
        if file_id is None:
            file_id = file_name
        self.file_id = file_id

    def add_to_summary(self, step, raw, overwrite=False):
        
        if not isinstance(raw, mne.io.BaseRaw):
            raise TypeError("raw must be an instance of mne.io.Raw")
        
        if any(((self.summary_df['file_id'] == self.file_id) & (self.summary_df['step'] == step))) and not overwrite:
            print(f"File {self.file_id} and step {step} already exist in the summary. Use overwrite=True to overwrite the existing entry.")
            return
        
        if any(((self.summary_df['file_id'] == self.file_id) & (self.summary_df['step'] == step))) and overwrite:
            self.remove_file_step_from_summary(self.file_id, step)
        
        length = raw.times.max()
        if hasattr(raw, 'artifacts'):
            corrected_data = np.round(np.sum(raw.artifacts.CCT) / np.size(raw.artifacts.CCT) * 100, 2)
            bad_data = np.round(np.sum(raw.artifacts.BCT) / np.size(raw.artifacts.BCT) * 100, 2)
            bad_channels = np.round(np.sum(raw.artifacts.BC) / np.size(raw.artifacts.BC) * 100, 2)
            bad_times = np.round(np.sum(raw.artifacts.BT) / np.size(raw.artifacts.BT) * 100, 2)
        else:
            corrected_data = np.nan
            bad_data = np.nan
            bad_channels = np.nan
            bad_times = np.nan

        self.summary_df.loc[len(self.summary_df)] = [self.file_id, step, length, corrected_data, bad_data, bad_channels, bad_times]

class SummaryEpochs(Summary):

    def __init__(self, 
                 output_folder, 
                 file_name,
                 file_id=None,
                 outputfile_subfix="summary-epo.csv", 
                 try_loading=True,
                 ):
        print("Initializing SummaryEpochs object")
        file_name = Path(file_name).stem
        outputfile = file_name + outputfile_subfix
        super().__init__(output_folder=output_folder, 
                         output_file=outputfile, 
                         columns=[
                             "file_id",
                             "step",
                             "n_epochs",
                             "n_remaining_epochs",
                             "length", 
                             "%_corrected_data", 
                             "%_bad_data", 
                             "%_bad_channels", 
                             "%_bad_times",
                             "%_bad_epochs",
                             ],
                         try_loading=try_loading)
        if file_id is None:
            file_id = file_name
        self.file_id = file_id

    def add_to_summary(self, step, epochs, overwrite=False):
        
        if not isinstance(epochs, BaseEpochs):
            raise TypeError("epochs must be an instance of mne.BaseEpochs")
        
        if any(((self.summary_df['file_id'] == self.file_id) & (self.summary_df['step'] == step))) and not overwrite:
            print(f"File {self.file_id} and step {step} already exist in the summary. Use overwrite=True to overwrite the existing entry.")
            return
        
        if any(((self.summary_df['file_id'] == self.file_id) & (self.summary_df['step'] == step))) and overwrite:
            self.remove_file_step_from_summary(self.file_id, step)
        
        length = epochs.times.max() - epochs.times.min()
        drop_log = np.asarray(epochs.drop_log, dtype=list)
        no_of_epochs = np.shape(drop_log)[0]
        no_of_remaining_epochs = np.shape(epochs._data)[0]
        if hasattr(epochs, 'artifacts'):
            corrected_data = np.round(np.sum(epochs.artifacts.CCT) / np.size(epochs.artifacts.CCT) * 100, 2)
            bad_data = np.round(np.sum(epochs.artifacts.BCT) / np.size(epochs.artifacts.BCT) * 100, 2)
            bad_channels = np.round(np.sum(epochs.artifacts.BC) / np.size(epochs.artifacts.BC) * 100, 2)
            bad_times = np.round(np.sum(epochs.artifacts.BT) / np.size(epochs.artifacts.BT) * 100, 2)
            bad_epochs = np.round(np.sum(epochs.artifacts.BE) / np.size(epochs.artifacts.BE) * 100, 2)
        else:
            corrected_data = np.nan
            bad_data = np.nan
            bad_channels = np.nan
            bad_times = np.nan
            bad_epochs = np.nan
        
        self.summary_df.loc[len(self.summary_df)] = [self.file_id, step, no_of_epochs, no_of_remaining_epochs,
                                            length, corrected_data, bad_data, bad_channels, bad_times, bad_epochs]
        

class StdOutLogger():

    def __init__(self, output_folder, file_name):
        file_name = Path(file_name).stem
        self.output_folder = Path(output_folder)
        self.output_file = f"{file_name}_log.txt"
        self.output_full_path = self.output_folder / self.output_file

    def restore_stdout(self):
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.output_full_path.write_text("")

    def redirect_stdout_to_file(self, restore=False):
        self.output_folder.mkdir(parents=True, exist_ok=True)
        if restore:
            self.restore_stdout()
        sys.stdout = open(self.output_full_path, "w")

    def close(self):
        sys.stdout.close()



# %% Function to run the full APICE for multiple data files

def run_preprocessing(input_dir, 
                      output_dir, 
                      input_dir_bids=False,
                      bids_session=None,
                      bids_task=None,
                      bids_run=None,
                      bids_subject=None,
                      bids_extension='.vhdr',
                      bids_datatype='eeg',
                      bids_suffix='eeg',
                      processed_file_pattern='*-preproc.fif',
                      data_selection_method="all",
                      drop_electrodes=None,
                      picks='eeg',
                      reference_channels=None,
                      crop_times=None,
                      crop_from_beginnning=None,
                      crop_from_end=None,
                      resample_freq=None,
                      stim_channels_to_annotations=True,
                      montage=None,
                      save_log=True,
                      save_report=True,
                      save_summary=True,
                      l_freq=0.10,
                      h_freq=40,
                      l_trans_bandwidth=0.1,
                      h_trans_bandwidth=10,
                      cfg_bad_channels_detection=None,
                      cfg_glitches_detection=None,
                      cfg_target_pca=None,
                      cfg_artifacts_detection=None,
                      cfg_spline_segments=None,
                      cfg_spline_channels=None,
                      n_jobs=-1,
                      ):
    
    # Initialize output folders
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Get all files to process
    if not input_dir_bids:
        from apice.io import get_files_to_process
        files = get_files_to_process(input_dir=input_dir,
                                    output_dir=output_dir,
                                    processed_file_pattern=processed_file_pattern,
                                    data_selection_method=data_selection_method,
                                    )
    else:
        import mne_bids
        from apice.io import get_bids_files_to_process
        files = get_bids_files_to_process(bids_root=input_dir, 
                                          session=bids_session,
                                          task=bids_task,
                                          run=bids_run,
                                          subject=bids_subject,
                                          datatype=bids_datatype,
                                          suffix=bids_suffix,
                                          extension=bids_extension,
                                          output_dir=output_dir,
                                          processed_file_pattern=processed_file_pattern,
                                          data_selection_method=data_selection_method,
                                          )
    print(f"\nNumber of files to process: {len(files)}\n")

    # Loop over the subjects for data processing
    for file in files:

        print(f"Processing file: {file}")

        try:
        
            # Load raw data
            if not input_dir_bids:
                raw = mne.io.read_raw(file, preload=False, verbose=False)
                file_name = file.stem
            else:
                raw = mne_bids.read_raw_bids(file)
                file_name = file.basename.replace(bids_extension, '')

            # Run initial preprocessing steps
            raw = preprocess_initial_steps(
                                        raw,
                                        drop_electrodes=drop_electrodes,
                                        picks=picks,
                                        crop_times=crop_times,
                                        crop_from_beginnning=crop_from_beginnning,
                                        crop_from_end=crop_from_end,
                                        resample_freq=resample_freq,
                                        stim_channels_to_annotations=stim_channels_to_annotations,
                                        montage=montage,
                                        )
            
            # Run APICE default preprocessing pipeline
            _ = preprocess_apice_default(
                                        raw,
                                        file_name=file_name,
                                        output_dir=output_dir,
                                        create_report=save_report,
                                        save_log=save_log,
                                        save_data=True,
                                        save_report=save_report,
                                        save_summary=save_summary,
                                        reference_channels=reference_channels,
                                        l_freq=l_freq,
                                        h_freq=h_freq,
                                        l_trans_bandwidth=l_trans_bandwidth,
                                        h_trans_bandwidth=h_trans_bandwidth,
                                        cfg_bad_channels_detection=cfg_bad_channels_detection,
                                        cfg_glitches_detection=cfg_glitches_detection,
                                        cfg_target_pca=cfg_target_pca,
                                        cfg_artifacts_detection=cfg_artifacts_detection,
                                        cfg_spline_segments=cfg_spline_segments,
                                        cfg_spline_channels=cfg_spline_channels,
                                        n_jobs=n_jobs,
                                        )
            
            plt.close('all')
            
        except Exception as e:
            print(f"Error processing file {file}: {e}")
            continue
            

def run_segmentation(input_dir, 
                     output_dir, 
                     kwargs_events_from_annotations_for_segmentation, 
                     event_time_window,
                     processed_file_pattern='*-epo.fif',
                     data_selection_method="all",
                     l_freq=None,
                     h_freq=None,
                     l_trans_bandwidth=0.1,
                     h_trans_bandwidth=10,
                     baseline=None, 
                     kwargs_events_from_annotations_for_metadata=None,
                     kwargs_make_metadata=None,                             
                     evoked_by=True,
                     save_log=True,
                     save_epochs=True,
                     save_only_good_epochs=False,
                     save_evoked=True,
                     save_report=True,
                     save_summary=True,
                     save_cfg=True,
                     set_reference=None,
                     cfg_define_bcbt_epochs=None,
                     cfg_spline_channels=None,  
                     cfg_bad_epochs=None,              
                     n_jobs=-1,
                     ):

    # Initialize output folders
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Get all files to process
    from apice.io import get_files_to_process
    files = get_files_to_process(input_dir=input_dir,
                                output_dir=output_dir,
                                processed_file_pattern=processed_file_pattern,
                                data_selection_method=data_selection_method,
                                )

    print(f"\nNumber of files to process: {len(files)}\n")

    for file in files:

        print(f"Processing file: {file}")

        try:
        
            # Load raw data
            raw = load_rawapice(file)

            # Run segmentation pipeline
            _ = segment_default_pipeline(raw, 
                        kwargs_events_from_annotations_for_segmentation, 
                        event_time_window,
                        file_name=file.stem,
                        l_freq=l_freq,
                        h_freq=h_freq,
                        l_trans_bandwidth=l_trans_bandwidth,
                        h_trans_bandwidth=h_trans_bandwidth,
                        baseline=baseline, 
                        kwargs_events_from_annotations_for_metadata=kwargs_events_from_annotations_for_metadata,
                        kwargs_make_metadata=kwargs_make_metadata,                             
                        evoked_by=evoked_by,
                        output_dir=output_dir,
                        save_log=save_log,
                        save_epochs=save_epochs,
                        save_only_good_epochs=save_only_good_epochs,
                        save_evoked=save_evoked,
                        save_report=save_report,
                        save_summary=save_summary,
                        save_cfg=save_cfg,
                        set_reference=set_reference,
                        cfg_define_bcbt_epochs=cfg_define_bcbt_epochs,
                        cfg_spline_channels=cfg_spline_channels,  
                        cfg_bad_epochs=cfg_bad_epochs,              
                        n_jobs=n_jobs,
                        )
            
            plt.close('all')
            
        except Exception as e:
            print(f"Error processing file {file}: {e}")
            continue


# %% Sub Functions


def preprocess_initial_steps(raw,
                          drop_electrodes=None,
                          picks='eeg',
                          crop_times=None,
                          crop_from_beginnning=None,
                          crop_from_end=None,
                          resample_freq=None,
                          stim_channels_to_annotations=True,
                          montage=None,
                          head_size=None,
                            ):

    # INITIALIZATION -----------------------------------------------------------------------------------------------

    # Check if raw is an instance of mne.io.Raw
    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError("raw must be an instance of mne.io.Raw")

    # Preprocessing start time
    sim_time_start = time.time()
    print('=============================================')
    print('Starting initial preprocessing steps\n')
    print(f"Processing date and time: {datetime.now()}\n\n")

    if raw.info['meas_date'] is None:
        print("raw.info['meas_date'] is None. Setting it to the current date and time.")
        raw.set_meas_date(datetime.now(tz=timezone.utc))

    # STIM CHANNELS TO ANNOTATIONS ------------------------------------------------------------------------------------------
    if stim_channels_to_annotations and "stim" in np.unique(raw.get_channel_types()):
        convert_stim_channels_to_annotations(raw)

    # DROPPING ELECTRODES -----------------------------------------------------------------------------------------------
    if drop_electrodes is not None:
        raw.drop_channels(list(drop_electrodes), on_missing='warn')

    # PICKING ELECTRODES -----------------------------------------------------------------------------------------------
    if picks is not None:
        raw.pick(picks)

    # CROP TIMES -----------------------------------------------------------------------------------------------
    if crop_times is not None:
        raw.crop(tmin=crop_times[0], tmax=crop_times[1])
    if crop_from_beginnning is not None:
            raw.crop(tmin=crop_from_beginnning, tmax=None)
    if crop_from_end is not None:
            raw.crop(tmin=0, tmax=raw.times[-1]-crop_from_end)    

    # RESAMPLE -----------------------------------------------------------------------------------------------
    if resample_freq is not None:
        raw.resample(resample_freq)

    # SET THE MONTAGE -----------------------------------------------------------------------------------------------
    if montage is not None:
        if isinstance(montage, mne.channels.DigMontage):
            raw.set_montage(montage)
        elif isinstance(montage, (str, Path)):
            if Path(montage).exists() and Path(montage).suffix is not None:
                try:
                    montage = mne.channels.read_custom_montage(montage, head_size=head_size)
                    raw.set_montage(montage)
                except Exception as e:
                    raise ValueError(f"Could not read montage from {montage}: {e}")
            elif montage in mne.channels.get_builtin_montages():
                montage = mne.channels.make_standard_montage(montage, head_size=head_size)
                raw.set_montage(montage)
            else:
                raise ValueError(f"Montage {montage} not recognized as a built-in montage or a valid file path.")
    
    if raw.get_montage() is None:
        raise ValueError("raw must have a montage. Please set the montage before preprocessing.")
    
    # END TIME ------------------------------------------------------------------------------------------------
    sim_time_end = timedelta(seconds=np.round(time.time() - sim_time_start))
    print(f"\nInitial preprocessing steps completed in: {sim_time_end}, in hh:mm:ss")
    print('=============================================\n')
        

    return raw
    

def preprocess_apice_default(raw, 
                             preprocessed_data_suffix='-preproc',
                             output_dir=None,
                             file_name=None,
                             create_report=True,
                             save_log=True,
                             save_data=True,
                             save_report=True,
                             save_summary=True,
                             save_cfg=True,
                             reference_channels=None,
                             l_freq=0.10,
                             h_freq=40,
                             l_trans_bandwidth=0.1,
                             h_trans_bandwidth=10,
                             cfg_define_bcbt_raw=None,
                             cfg_bad_channels_detection=None,
                             cfg_glitches_detection=None,
                             cfg_target_pca=None,
                             cfg_artifacts_detection=None,
                             cfg_spline_segments=None,
                             cfg_spline_channels=None,
                             n_jobs=-1,
                             ):
    
    '''
    Preprocess raw EEG data using the APICE pipeline.

    Args:
        - raw (mne.io.Raw): The raw EEG data to preprocess.
        - output_dir (str or Path): The directory where the preprocessed data, report, and summary will be saved. Required if any of the saving options is True.
        - save_log (bool): Whether to save the log of the preprocessing steps (default: True).
        - save_data (bool): Whether to save the preprocessed raw data (default: True).
        - save_report (bool): Whether to save the preprocessing report (default: True).
        - save_summary (bool): Whether to save the summary of artifacts detected and corrected during preprocessing (default: True).
        - l_freq (float): The high-pass filter frequency in Hz. If None, the default value from Filters.l_freq will be used.
        - h_freq (float): The low-pass filter frequency in Hz. If None, the default value from Filters.h_freq will be used.
        - drop_electrodes (list of str): List of electrode names to remove from the data (default: None). 
        - picks (str or array_like or slice): Input for raw.pick(). If picks is provided, only the specified electrodes will be kept and all others will be dropped.
        - crop_times (tuple of float): Tuple specifying the start and end times (in seconds) to crop the raw data (default: None).
        - crop_from_beginnning (float): Time in seconds to crop from the beginning of the raw data (default: None).
        - crop_from_end (float): Time in seconds to crop from the end of the raw data (default: None).
        - n_jobs (int): The number of parallel jobs to run for computationally intensive steps (default: -1, which means using all available cores).

    Returns:
        - raw (mne.io.Raw): The preprocessed raw EEG data.
        - summary (SummaryPreprocessing): The summary of artifacts detected and corrected during preprocessing.
        - report (mne.Report): The preprocessing report containing visualizations and information about the preprocessing steps.

    Note:
        - The raw data must have a montage set before preprocessing. Please set the montage using raw.set_montage() before calling this function.
        - If any of the saving options (save_log, save_data, save_report, save_summary) is True, the output_dir must be provided to specify where the files will be saved.
        - If the reference channel is included in raw, please provide the name of the reference channel in the drop_electrodes list to ensure it is removed during preprocessing.
    '''

        
    # INITIALIZATION -----------------------------------------------------------------------------------------------

    # Check if raw is an instance of mne.io.Raw
    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError("raw must be an instance of mne.io.Raw")
    
    # Check that raw has a montage
    if raw.get_montage() is None:
        raise ValueError("raw must have a montage. Please set the montage before preprocessing.")
    
    # Check that output_dir is provided if any of the saving options is True
    if output_dir is None and (save_log or save_data or save_report or save_summary):
        raise ValueError("output_dir must be provided if any of the saving options is True")
    
    # Check that file_name is provided if any of the saving options is True, to use as part of the file name for the saved files
    if file_name is None and (save_log or save_data or save_report or save_summary):
        raise ValueError("file_name must be provided if any of the saving options is True, to use as part of the file name for the saved files")
    
    # Check the filter frequencies
    if l_freq is None:
        l_freq=0.1
        print(f"Warning: High pass frequency not provided. Using default value: {l_freq} Hz")
        
    if isinstance(reference_channels, str):
        reference_channels = [reference_channels]
    # check that reference channels are in raw
    if reference_channels is not None:
        for ch in reference_channels:
            if ch not in raw.ch_names:
                print(f"Warning:Reference channel {ch} not found in raw data channels")

    # Get default configurations if not provided or load it if provided as path to a json file
    cfg_define_bcbt_raw = get_cfg(cfg_define_bcbt_raw, 'define_bcbt_raw_config.json')
    cfg_bad_channels_detection = get_cfg(cfg_bad_channels_detection, 'detect_bad_channels_config.json')
    cfg_glitches_detection = get_cfg(cfg_glitches_detection, 'detect_artifacts_glitches_config.json')
    cfg_target_pca = get_cfg(cfg_target_pca, 'correction_target_pca_config.json')
    cfg_artifacts_detection = get_cfg(cfg_artifacts_detection, 'detect_artifacts_all_config.json')
    cfg_spline_segments = get_cfg(cfg_spline_segments, 'correction_spline_segments_config.json')
    cfg_spline_channels = get_cfg(cfg_spline_channels, 'correction_spline_channels_config.json')    
    
    # for each configuration for correction set n_jobs to the value provided in the function argument
    cfg_spline_segments['n_jobs'] = n_jobs
    cfg_spline_channels['n_jobs'] = n_jobs
    
    # Create output folder if it does not exist
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        if save_summary:
            output_dir_summary = output_dir / "summary"
            output_dir_summary.mkdir(exist_ok=True)
        if save_report:
            output_dir_reports = output_dir / "reports"
            output_dir_reports.mkdir(exist_ok=True)
        if save_cfg:
            output_dir_cfgs = output_dir / "cfgs"
            output_dir_cfgs.mkdir(exist_ok=True)
        if save_log:
            output_dir_logs = output_dir / "logs"
            output_dir_logs.mkdir(exist_ok=True)
    
    # Initialize object for logging
    if save_log: logger = StdOutLogger(output_dir_logs, file_name)
        
    # Preprocessing start time
    sim_time_start = time.time()
    print('=============================================')
    print('Starting APICE default preprocessing pipeline')
    print(f"Processing date and time: {datetime.now()}\n\n")

    # Initialize object tracking the summary of artifacts
    if save_summary:
        summary = SummaryPreprocessing(output_dir_summary, file_name, try_loading=False)

    # Initialize reports
    if create_report:
        report = mne.Report(title=file_name)
    else:
        report = None

    # Save log if True
    if save_log: logger.redirect_stdout_to_file(restore=True)

    # FILTER -----------------------------------------------------------------------------------------------
    Filter(raw, l_freq=l_freq, h_freq=h_freq, l_trans_bandwidth=l_trans_bandwidth, h_trans_bandwidth=h_trans_bandwidth, n_jobs=n_jobs)


    # ARTIFACT DETECTION AND CORRECTION -----------------------------------------------------------------------------

    # Initialize artifacts structure
    raw = RawAPICE(raw, **cfg_define_bcbt_raw)

    if create_report:
        try:
            report.add_raw(raw, 
                            title="Raw Data", 
                            psd=False, 
                            butterfly=False, 
                            replace=True, 
                            )
        except Exception as e:
            print(f"Warning: Could not add raw data to report: {e}")
        
        # Add PSD
        if h_freq is None:
            fmax = raw.info['sfreq'] / 2
        else:
            fmax = h_freq
        if reference_channels is not None:
            exclude = reference_channels
        else:
            exclude = []
        try:
            fig = raw.compute_psd(method='welch',fmax=fmax,exclude=exclude).plot()
            report.add_figure(fig, "PSD", section="Raw Data", replace=True)
        except Exception as e:
            print(f"Warning: Could not add raw PSD to report: {e}")

        # Add events
        try:
            events, event_id = mne.events_from_annotations(raw)
            if (len(events) > 0) and (len(events)<30):  # only add events to the report if there are events and if there are not too many events (otherwise the report gets too heavy and does not open properly)
                report.add_events(events=events, title='Events from "annotations"', sfreq=raw.info['sfreq'], section="Raw Data", replace=True)
            else:                
                print(f"Warning: Not adding events to report because there are {len(events)} events, which is more than the threshold of 30 events.")
        except Exception as e:
            print(f"Warning: Could not add events to report: {e}")


    # Detect bad channels
    raw.detect_bad_channels(cfg_bad_channels_detection=cfg_bad_channels_detection)
    raw.deal_with_reference_channels(reference_channels)
    if save_summary:
        summary.add_to_summary('artifacts_detection_BadElectrodes', raw, overwrite=True)

    # Detect glitches
    raw.detect_glitches(cfg_glitches_detection=cfg_glitches_detection)
    raw.deal_with_reference_channels(reference_channels)
    if save_summary:
        summary.add_to_summary('artifacts_detection_Glitches', raw, overwrite=True)

    # Correct glitches
    raw.correct_target_pca(cfg_target_pca=cfg_target_pca)
    Filter(raw, l_freq=l_freq, h_freq=None, l_trans_bandwidth=l_trans_bandwidth, h_trans_bandwidth=h_trans_bandwidth, n_jobs=n_jobs)
    if save_summary:
        summary.add_to_summary('artifacts_correction_TargetPCA', raw, overwrite=True)

    # Detect artifacts
    raw.detect_artifacts(cfg_artifacts_detection=cfg_artifacts_detection)
    raw.deal_with_reference_channels(reference_channels)
    if save_summary:
        summary.add_to_summary('artifacts_detection_Artifacts', raw, overwrite=True)

    # Create a figure to visualize the artifact structure
    if create_report:
        try:
            fig = raw.plot_artifact_structure(color_scheme='jet')    
            report.add_figure(fig, "Artifacts Matrix", section="Raw Data", replace=True)
        except Exception as e:
            print(f"Warning: Could not add raw artifacts matrix to report: {e}")

    # Add topomap of bad electrodes
    if create_report:
        try:
            fig = raw.plot_percentage_of_bad_data_across_sensors()
            report.add_figure(fig, "Bad data across electrodes", section="Raw Data", replace=True)
        except Exception as e:
            print(f"Warning: Could not add raw bad-data topomap to report: {e}")
    
    # Correct artifacts using spherical spline interpolation per segment
    raw.correct_spline_segments(cfg_spline_segments=cfg_spline_segments)
    Filter(raw, l_freq=l_freq, h_freq=None, l_trans_bandwidth=l_trans_bandwidth, h_trans_bandwidth=h_trans_bandwidth, n_jobs=n_jobs)
    if save_summary:
        summary.add_to_summary('artifacts_correction_Segments', raw, overwrite=True)

    # Correct channels using spherical spline interpolation
    raw.correct_spline_channels(cfg_spline_channels=cfg_spline_channels)
    if save_summary:
        summary.add_to_summary('artifacts_correction_BadChannels', raw, overwrite=True)

    # Re-detect bad data after correction to check if there are still bad channels or time segments that need to be marked as bad after the correction
    raw.detect_artifacts(cfg_artifacts_detection=cfg_artifacts_detection)
    raw.deal_with_reference_channels(reference_channels)    
    if save_summary:
        summary.add_to_summary('artifacts_detection_ArtifactsPostCorrection', raw, overwrite=True)

    if create_report:
        try:
            report.add_raw(raw, 
                            title="Preprocessed Raw Data", 
                            psd=False, 
                            butterfly=True, 
                            scalings=50e-6, 
                            replace=True,
                            topomap_kwargs={"color_scheme": "jet"}
                            )
        except Exception as e:
            print(f"Warning: Could not add preprocessed raw data to report: {e}")
    
        # Add PSD
        if h_freq is None:
            fmax = raw.info['sfreq'] / 2
        else:
            fmax = h_freq
        if reference_channels is not None:
            exclude = reference_channels
        else:
            exclude = []
        try:
            fig = raw.compute_psd(method='welch',fmax=fmax,exclude=exclude).plot()
            report.add_figure(fig, "PSD", section="Preprocessed Raw Data", replace=True)
        except Exception as e:
            print(f"Warning: Could not add preprocessed PSD to report: {e}")
    
    # Create a figure to visualize the artifact structure
    if create_report:
        try:
            fig = raw.plot_artifact_structure(color_scheme='jet')
            report.add_figure(fig, "Artifacts Matrix", section="Preprocessed Raw Data", replace=True)
        except Exception as e:
            print(f"Warning: Could not add preprocessed artifacts matrix to report: {e}")

    # Add topomap of bad electrodes
    if create_report:
        try:
            fig = raw.plot_percentage_of_bad_data_across_sensors()
            report.add_figure(fig, "Bad data across electrodes", section="Preprocessed Raw Data", replace=True)
        except Exception as e:
            print(f"Warning: Could not add preprocessed bad-data topomap to report: {e}")


    # EXPORT DATA -----------------------------------------------------------------------------------------------
    
    # Save preprocessed raw
    if save_data:
        raw.export(file_name, output_dir, data_suffix=preprocessed_data_suffix)

    # Save summary file
    if save_summary:
        output_dir_summary.mkdir(exist_ok=True)
        summary.save()

    # Save report
    if create_report and save_report:
        output_dir_reports.mkdir(exist_ok=True)
        print("Saving report")
        full_path = output_dir_reports / (file_name + preprocessed_data_suffix + ".html")
        report.save(fname=full_path, open_browser=False, overwrite=True)

    # Save the configurations used for preprocessing
    if save_cfg:
        cfg_to_save = {
            "cfg_bad_channels_detection": cfg_bad_channels_detection,
            "cfg_glitches_detection": cfg_glitches_detection,
            "cfg_target_pca": cfg_target_pca,
            "cfg_artifacts_detection": cfg_artifacts_detection,
            "cfg_spline_segments": cfg_spline_segments,
            "cfg_spline_channels": cfg_spline_channels,
            }
        for cfg_name, cfg in cfg_to_save.items():
            with open(output_dir_cfgs / f"{file_name}_{cfg_name}.json", 'w') as f:
                json.dump(cfg, f, indent=4)

    # Preprocessing end time
    sim_time_end = timedelta(seconds=np.round(time.time() - sim_time_start))
    print(f"\nAPICE default preprocessing pipeline completed in: {sim_time_end}, in hh:mm:ss")
    print('=============================================\n')
    
    if save_log:
        logger.close()

    return raw, summary, report


def segment_default_pipeline(raw, 
                   kwargs_events_from_annotations_for_segmentation, 
                   event_time_window,
                   file_name=None,
                   l_freq=None,
                   h_freq=None,
                   l_trans_bandwidth=0.1,
                   h_trans_bandwidth=10,
                   baseline=None, 
                   kwargs_events_from_annotations_for_metadata=None,
                   kwargs_make_metadata=None,                             
                   evoked_by="all",
                   output_dir=None,
                   create_report=True,
                   save_log=True,
                   save_epochs=True,
                   save_only_good_epochs=False,
                   save_evoked=True,
                   save_report=True,
                   save_summary=True,
                   save_cfg=True,
                   set_reference=None,
                   cfg_define_bcbt_epochs=None,
                   cfg_spline_channels=None,  
                   cfg_bad_epochs=None,              
                   n_jobs=-1,
                   ):

    # INITIALIZATION -----------------------------------------------------------------------------------------------

    # Check if raw is an instance of mne.io.Raw
    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError("raw must be an instance of mne.io.Raw")
    
    # Check that raw is an instance of RawAPICE, which is required for the segmentation pipeline
    if not isinstance(raw, RawAPICE):
        raise TypeError("raw must be an instance of RawAPICE. Please run the preprocess_apice_default function before segmentation to ensure raw is an instance of RawAPICE and has the necessary attributes for segmentation.")    
    
    # Check that raw has a montage
    if raw.get_montage() is None:
        raise ValueError("raw must have a montage. Please set the montage before preprocessing.")
    
    # Check that output_dir is provided if any of the saving options is True
    if output_dir is None and (save_log or save_epochs or save_evoked or save_report or save_summary):
        raise ValueError("output_dir must be provided if any of the saving options is True")

    # Check that file_name is provided if any of the saving options is True, to use as part of the file name for the saved files
    if file_name is None and (save_log or save_epochs or save_evoked or save_report or save_summary):
        raise ValueError("file_name must be provided if any of the saving options is True, to use as part of the file name for the saved files")

    # Get the default configurations if not provided or load it if provided as path to a json file
    cfg_define_bcbt_epochs = get_cfg(cfg_define_bcbt_epochs, 'define_bcbt_epochs_config.json')
    cfg_spline_channels = get_cfg(cfg_spline_channels, 'correction_spline_channels_config.json')
    cfg_bad_epochs = get_cfg(cfg_bad_epochs, 'detect_bad_epochs_config.json')
    
    # for each configuration for correction set n_jobs to the value provided in the function argument
    cfg_spline_channels['n_jobs'] = n_jobs

    
    # Create output folder if it does not exist
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        if save_cfg:
            output_dir_cfgs = output_dir / "cfgs"
            output_dir_cfgs.mkdir(exist_ok=True)
        if save_report:
            output_dir_reports = output_dir / "reports"
            output_dir_reports.mkdir(exist_ok=True)
        if save_log:
            output_dir_logs = output_dir / "logs"
            output_dir_logs.mkdir(exist_ok=True)
        if save_summary:
            output_dir_summary = output_dir / "summary"
            output_dir_summary.mkdir(exist_ok=True)

    # Initialize object tracking the summary of artifacts
    if save_summary:
        summary = SummaryEpochs(output_dir_summary, file_name, try_loading=False)

    # Initialize object for logging
    if save_log: 
        logger = StdOutLogger(output_dir_logs, file_name)
        
    # Initialize reports
    if create_report:
        report = mne.Report(title=file_name)
    else:
        report = None

    # Save log if True
    if save_log: logger.redirect_stdout_to_file(restore=True)

    # Segmentation start time
    sim_time_start = time.time()
    print('=============================================')
    print('Starting APICE default segmentation pipeline')
    print(f"Segmentation date and time: {datetime.now()}\n\n")


    # FILTER -----------------------------------------------------------------------------------------------
    Filter(raw, l_freq=l_freq, h_freq=h_freq, l_trans_bandwidth=l_trans_bandwidth, h_trans_bandwidth=h_trans_bandwidth, n_jobs=n_jobs)

    # SEGMENTATION ----------------------------------------------------------------------------------------------
    
    # raw_=raw.copy()
    # raw_.annotate_bads(data=False, corrected=False)
    # raw_.plot(n_channels=raw_.info['nchan'], duration=100)
    
    if kwargs_events_from_annotations_for_metadata:
        if kwargs_make_metadata is None:
            kwargs_make_metadata = {}
        tmin_metadata = kwargs_make_metadata.get('tmin',event_time_window[0]) 
        tmax_metadata = kwargs_make_metadata.get('tmax',event_time_window[1]) 
        keep_first = kwargs_make_metadata.get('keep_first',None) 
        keep_last = kwargs_make_metadata.get('keep_last',None) 
        columns_events_to_keep = kwargs_make_metadata.get('columns_events_to_keep', None) 
        metadata = generate_metadata_for_epochs(raw, 
                                                kwargs_events_from_annotations_for_metadata=kwargs_events_from_annotations_for_metadata,
                                                kwargs_events_from_annotations_for_segmentation=kwargs_events_from_annotations_for_segmentation,
                                                tmin=tmin_metadata, 
                                                tmax=tmax_metadata, 
                                                keep_first=keep_first, 
                                                keep_last=keep_last,
                                                columns_events_to_keep=columns_events_to_keep,
                                                )
    else:
        metadata = None
    
    # Segment data into epochs, apply artifact correction, and compute evoked responses
    epochs, evokeds, summary = compute_epochs_and_evoked(raw, 
                                   kwargs_events_from_annotations=kwargs_events_from_annotations_for_segmentation,
                                   tmin=event_time_window[0], 
                                   tmax=event_time_window[1],
                                   n_jobs=n_jobs,
                                   baseline=baseline,
                                   metadata=metadata,
                                   evoked_by=evoked_by,
                                   set_reference=set_reference,
                                   summary=summary,
                                   cfg_define_bcbt_epochs=cfg_define_bcbt_epochs,
                                   cfg_spline_channels=cfg_spline_channels,
                                   cfg_bad_epochs=cfg_bad_epochs,
                                   )
    
    # n_channels = epochs.info['nchan']
    # epoch_colors = [['black' for i in range(n_channels)] for i in range(len(epochs))]
    # for i in range(len(epochs)):
    #     if epochs.artifacts.BE[i]:
    #         epoch_colors[i] = ['red' for i in range(n_channels)]
    #     else:
    #         epoch_colors[i] = ['black' if not epochs.artifacts.BC[i][j,0] else 'blue' for j in range(n_channels)]
    # epochs.plot(epoch_colors=epoch_colors)

    # Add epochs in report
    if create_report:
        try:
            report.add_epochs(epochs, "Epochs", psd=True, replace=True)
        except Exception as e:
            print(f"Warning: Could not add epochs to report: {e}")
    
    # Add epochs artifacts matrix
    if create_report:
        try:
            fig = epochs.plot_artifact_structure(color_scheme='jet')
            report.add_figure(fig, "Artifacts Matrix", section="Epochs", replace=True)
        except Exception as e:
            print(f"Warning: Could not add epochs artifacts matrix to report: {e}")

    # Add topomap of bad electrodes
    if create_report:
        try:
            fig = epochs.plot_percentage_of_bad_data_across_sensors()
            report.add_figure(fig, "Bad data across electrodes", section="Epochs", replace=True)
        except Exception as e:
            print(f"Warning: Could not add epochs bad-data topomap to report: {e}")

    # Add evokeds in the report
    if create_report:
        if evoked_by is None:
            evokeds_to_add = []
        elif evoked_by == "all":
            evokeds_to_add = dict(all=evokeds)
        else:                
            evokeds_to_add = evokeds
        for key, evoked in evokeds_to_add.items():
            try:
                report.add_evokeds(evoked, titles=key, replace=True)
            except Exception as e:
                print(f"Warning: Could not add evoked responses {key} to report: {e}")
                try:
                    fig = evoked.plot()
                    report.add_figure(fig, f"Evoked responses {key}", section="Evoked responses", replace=True)
                except Exception as fig_err:
                    print(f"Warning: Could not add fallback evoked figure {key} to report: {fig_err}")


    # EXPORT DATA -----------------------------------------------------------------------------------------------
    
    # Save epochs 
    if save_epochs:
        file_name_epochs = (file_name + '-epo.fif')
        if save_only_good_epochs:
            epochs_good = epochs.copy()
            epochs_good.remove_bad_epochs()
            epochs_good.export(file_name_epochs, output_dir)
        else:
            epochs.export(file_name_epochs, output_dir)

    # Save evoked responses
    if save_evoked:

        if evoked_by is None:
            print(f"No evoked responses to save for {file_name} since evoked_by is set to None.")
        elif evoked_by == "all":
            file_name_evoked = (file_name + '-ave.fif')
            folder_path = output_dir / 'erp'
            folder_path.mkdir(parents=True, exist_ok=True)
            full_path = folder_path / file_name_evoked
            print(f"Writing {full_path}")
            evokeds.save(full_path, overwrite=True)
            print(f"Closing {full_path}")
        else:
            for key, ev in evokeds.items():
                folder_path = output_dir / 'erp'
                folder_path.mkdir(parents=True, exist_ok=True)
                # make the key name a valid file name by removing or replacing characters that are not allowed in file names
                key_ = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in key)
                full_path = folder_path / f"{file_name}_{key_}-ave.fif"
                print(f"Writing {full_path}")
                ev.save(full_path, overwrite=True)
                print(f"Closing {full_path}")
            print('[done]')

    # Save summary file
    if save_summary:
        output_dir_summary.mkdir(exist_ok=True)
        summary.save()

    # Save report
    if save_report:
        output_dir_reports.mkdir(exist_ok=True)
        print("Saving report")
        full_path = output_dir_reports / (file_name + "-epo.html")
        report.save(fname=full_path, open_browser=False, overwrite=True)

    # Save configurations used for segmentation
    if save_cfg:    
        cfg_to_save = {
            "cfg_define_bcbt_epochs": cfg_define_bcbt_epochs,
            "cfg_spline_channels": cfg_spline_channels,
            "cfg_bad_epochs": cfg_bad_epochs,
            }
        for cfg_name, cfg in cfg_to_save.items():
            with open(output_dir_cfgs / f"{file_name}_{cfg_name}.json", 'w') as f:
                json.dump(cfg, f, indent=4)
    
    # Preprocessing end time
    sim_time_end = timedelta(seconds=np.round(time.time() - sim_time_start))
    print(f"\nAPICE default segmentation pipeline completed in: {sim_time_end}, in hh:mm:ss")
    print('=============================================\n')
    
    if save_log:
        logger.close()

    return epochs, evokeds, summary, report


def get_stim_duration(raw, threshold=0.01):

    stim_data = raw.copy().pick('stim').get_data()    
    above_baseline = np.where(stim_data > threshold, 1, 0)
    onsets, durations = get_onset_and_duration(above_baseline)

    return np.max(durations)
    
def convert_stim_channels_to_annotations(raw):
    # Get a copy of the original annotations
    df_annotations = raw.annotations.to_data_frame(time_format=None)
    
    # Convert stim channels to annotations
    print("\nConverting STIMs to annotations...")
    
    stims = raw.copy().pick('stim')
    
    if stims:

        # Detect all events
        from mne import find_events
        events = find_events(raw, stim_channel=stims.ch_names, verbose=False)
        
        # Assuming event IDs are directly the data values in the stim channel
        event_ids = np.unique(events[:, 2])  # Unique event identifiers
        
        onsets = events[:, 0] / raw.info['sfreq']  # Convert sample indices to times
        
        ch_names = [()] * len(events)
        
        durations = [get_stim_duration(raw)] * len(events) 
        
        event_map = {event_id: f"{stims.ch_names[event_id - 1]}" for event_id in event_ids if event_id < len(raw.ch_names)}
        
        descriptions = [event_map[event_id] for event_id in events[:, 2]]

        # Create Annotations object
        df_annotations_from_stims = pd.DataFrame(dict(onset=onsets, duration=durations, description=descriptions, ch_names=ch_names))
        
        # List of DataFrames to combine
        dfs = [df_annotations, df_annotations_from_stims]
        
        # Filter out empty DataFrames
        dfs = [df for df in dfs if not df.empty]
        
        if dfs:
            # Concatenating DataFrames
            df_combined = pd.concat(dfs, ignore_index=True)

            # Dropping duplicates and sorting by 'onset'
            df_final = df_combined.drop_duplicates().sort_values(by="onset").reset_index(drop=True)

            from mne import Annotations 
            annotations = Annotations(onset=list(df_final["onset"]),
                                        duration=list(df_final["duration"]),
                                        description=list(df_final["description"]),
                                        ch_names=list(df_final["ch_names"]))
            
            raw.set_annotations(annotations)
    

def generate_metadata_for_epochs(raw, 
                 kwargs_events_from_annotations_for_metadata={},
                 kwargs_events_from_annotations_for_segmentation={},
                 columns_events_to_keep=None,
                 tmin=-0.5, 
                 tmax=0.5, 
                 keep_first=None, 
                 keep_last=None,
                 ):
    
    # Get events, and event ids for segmentation
    events_segm, event_id_segm = mne.events_from_annotations(raw, **kwargs_events_from_annotations_for_segmentation)
    
    # get events and event ids for metadata
    events_metadata, event_ids_metadata = mne.events_from_annotations(raw, **kwargs_events_from_annotations_for_metadata)
    
    # make metadata
    metadata_, events, event_id = mne.epochs.make_metadata(
                                                    events_metadata,
                                                    event_ids_metadata,
                                                    tmin,
                                                    tmax,
                                                    raw.info["sfreq"],
                                                    row_events=list(event_id_segm.keys()),
                                                    keep_first=keep_first, 
                                                    keep_last=keep_last
                                                    )
    # keep relevant columns
    if columns_events_to_keep is not None:
        col = [c for c in columns_events_to_keep if c in metadata_.columns]
        metadata = metadata_[col]
    metadata = metadata.reset_index(drop=True)
    
    return metadata
    

def compute_epochs_and_evoked(raw, 
                 kwargs_events_from_annotations={},
                 metadata=None,
                 tmin=-0.2, 
                 tmax=0.5, 
                 n_jobs=-1, 
                 baseline=None, 
                 evoked_by="all", 
                 set_reference=None,
                 cfg_define_bcbt_epochs=None,
                 cfg_spline_channels=None,
                 cfg_bad_epochs=None,
                 summary=None):
    """
    Segments continuous EEG data, applies artifact correction, and computes evoked responses.

    This function performs several steps on EEG data: it segments the data into epochs based on specified events, 
    applies artifact correction, defines and removes bad epochs, re-references the data, and computes the 
    evoked responses.

    Parameters:
    raw : Raw EEG object
        Continuous EEG data to be processed.
    event_keys : list
        Keys identifying the events around which to segment the data.
    tmin : float
        Start time before the event in seconds.
    tmax : float
        End time after the event in seconds.
    baseline : tuple, optional
        Time window for baseline correction (start, end) in seconds. Defaults to None.
    evoked_by : str, list or None, optional
        Specifies how to compute evoked responses. Can be "all" to compute a single averaged response,
        a string or list of strings specifying an event type to compute evoked responses for that event, or None to skip evoked computation.
        Defaults to "all".
    summary : Summary object, optional
        The summary object to update with segmentation information. Defaults to None.
    set_reference : None or dictionary, optional
        The parameters for the mne.io.Epochs.set_eeg_reference method. Defaults to None.

    Returns:
    epochs : mne.Epochs object
        The processed epochs after segmentation, artifact correction, and bad epoch removal.
    evoked : mne.Evoked or list of mne.Evoked
        The evoked response(s), either as a single averaged response or separated by event type.
    """
    # get the configurations 
    cfg_define_bcbt_epochs = get_cfg(cfg_define_bcbt_epochs, 'define_bcbt_epochs_config.json')
    cfg_spline_channels = get_cfg(cfg_spline_channels, 'correction_spline_channels_config.json')
    cfg_bad_epochs = get_cfg(cfg_bad_epochs, 'detect_bad_epochs_config.json')

    # Get events, and event ids
    events_segm, event_id_segm = mne.events_from_annotations(raw, **kwargs_events_from_annotations)
    
    # Segment the continuous data into epochs
    karg = dict(reject_by_annotation=False,
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            preload=True,
            metadata=metadata,
    )
    epochs = raw.segment_continuous_data(events_segm, event_id_segm, karg)
    epochs.update_artifacts_params(**cfg_define_bcbt_epochs)
    
    # Define BadTimes and BadChannels for the segmented data
    epochs.define_bcbt()
    if summary is not None:
        summary.add_to_summary('segmentation_Initial', epochs, overwrite=True)

    # Apply spherical spline interpolation for artifact correction and re-define BT and BC after interpolation
    epochs.correct_spline_channels(cfg_spline_channels=cfg_spline_channels)
    
    # Identify and define bad epochs
    epochs.define_bad_epochs(bad_data=cfg_bad_epochs['bad_data'], 
                             bad_time=cfg_bad_epochs['bad_time'], 
                             bad_channel=cfg_bad_epochs['bad_channel'], 
                             lim_dist=cfg_bad_epochs['lim_dist'], 
                             lim_gfp=cfg_bad_epochs['lim_gfp'])
    

    # Re-reference the data if specified by the user
    if set_reference is not None:
        epochs.set_eeg_reference(**set_reference)
    if summary is not None:
        summary.add_to_summary('segmentation_Final', epochs, overwrite=True)
    
    # Print summary of artifacts and processing
    print(f"\nSummary: {print(epochs.artifacts.print_summary())}\n")

    # Compute the evoked responses
    print(
        f"\nGetting evoked responses...",
        f"-\n\t-- by event type: {evoked_by}"
        )
    # Remove bad epochs from the data 
    epochs_good = epochs.copy()
    epochs_good.remove_bad_epochs()
    if evoked_by is not None:
        if evoked_by == "all":
            evokeds = epochs_good.average()
        else:
            evokeds = {}
            for ev in evoked_by:
                try:
                   evokeds[ev] = epochs_good[ev].average()
                except Exception as e:
                    print(f"Warning: Could not compute evoked response for event type {ev}: {e}")
    else:
        evokeds = None

    if summary is not None:
        return epochs, evokeds, summary
    else:
        return epochs, evokeds, None





