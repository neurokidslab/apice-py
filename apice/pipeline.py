import numpy as np
import pandas as pd
import sys
import time
from pathlib import Path
from datetime import datetime
from tabulate import tabulate
from datetime import timedelta

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
        
        if not isinstance(epochs, mne.Epochs):
            raise TypeError("epochs must be an instance of mne.Epochs")
        
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
                      preprocessed_file_pattern='*-preproc.fif',
                      data_selection_method="all",
                      drop_electrodes=None,
                      picks='eeg',
                      crop_times=None,
                      crop_from_begignning=None,
                      crop_from_end=None,
                      resample_freq=None,
                      stim_channels_to_annotations=True,
                      montage=None,
                      create_report=True,
                      save_log=True,
                      save_data=True,
                      save_report=True,
                      save_summary=True,
                      high_pass_freq=None,
                      low_pass_freq=None,
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
                                    preprocessed_file_pattern=preprocessed_file_pattern,
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
                                          preprocessed_file_pattern=preprocessed_file_pattern,
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
            else:
                raw = mne_bids.read_raw_bids(file)

            # Run initial preprocessing steps
            raw = preprocess_initial_steps(
                                        raw,
                                        drop_electrodes=drop_electrodes,
                                        picks=picks,
                                        crop_times=crop_times,
                                        crop_from_begignning=crop_from_begignning,
                                        crop_from_end=crop_from_end,
                                        resample_freq=resample_freq,
                                        stim_channels_to_annotations=stim_channels_to_annotations,
                                        montage=montage,
                                        )
            
            # Run APICE default preprocessing pipeline
            _ = preprocess_apice_default(
                                        raw,
                                        output_dir=output_dir,
                                        create_report=create_report,
                                        save_log=save_log,
                                        save_data=save_data,
                                        save_report=save_report,
                                        save_summary=save_summary,
                                        high_pass_freq=high_pass_freq,
                                        low_pass_freq=low_pass_freq,
                                        n_jobs=n_jobs,
                                        )
            
        except Exception as e:
            print(f"Error processing file {file}: {e}")
            continue
            



# %% Sub Functions

# Libraries and dependencies
import mne
import apice
from apice.artifacts_rejection import (BadElectrodes, Motion, Jump)
from apice.artifacts_structure import (DefineBTBC, Artifacts, annotations_to_rejection_matrix, plot_percentage_of_bad_data_across_sensors)
from apice.artifacts_correction import (TargetPCA, SegmentSphericalSplineInterpolation, ChannelsSphericalSplineInterpolation)
from apice.filter import Filter
from apice.parameters import Filters

def preprocess_initial_steps(raw,
                          drop_electrodes=None,
                          picks='eeg',
                          crop_times=None,
                          crop_from_begignning=None,
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

    # STIM CHANNELS TO ANNOTATIONS ------------------------------------------------------------------------------------------
    if stim_channels_to_annotations and "stim" in np.unique(raw.get_channel_types()):
        apice.io.Raw.stim_channels_to_annotations(raw)

    # DROPPING ELECTRODES -----------------------------------------------------------------------------------------------
    if drop_electrodes is not None:
        raw.drop_channels(list(drop_electrodes), on_missing='warn')

    # PICKING ELECTRODES -----------------------------------------------------------------------------------------------
    if picks is not None:
        raw.pick(picks)

    # CROP TIMES -----------------------------------------------------------------------------------------------
    if crop_times is not None:
        raw.crop(tmin=crop_times[0], tmax=crop_times[1])
    if crop_from_begignning is not None:
            raw.crop(tmin=crop_from_begignning, tmax=None)
    if crop_from_end is not None:
            raw.crop(tmin=None, tmax=raw.times[-1]-crop_from_end)    

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
    print('\nTotal processing time :', str(sim_time_end), 'in hh:mm:ss')
    print('=============================================\n')
        

    return raw
    

def preprocess_apice_default(raw, 
                             preprocessed_data_suffix='-preproc',
                             output_dir=None,
                             create_report=True,
                             save_log=True,
                             save_data=True,
                             save_report=True,
                             save_summary=True,
                             high_pass_freq=None,
                             low_pass_freq=None,
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
        - high_pass_freq (float): The high-pass filter frequency in Hz. If None, the default value from Filters.high_pass_freq will be used.
        - low_pass_freq (float): The low-pass filter frequency in Hz. If None, the default value from Filters.low_pass_freq will be used.
        - drop_electrodes (list of str): List of electrode names to remove from the data (default: None). 
        - picks (str or array_like or slice): Input for raw.pick(). If picks is provided, only the specified electrodes will be kept and all others will be dropped.
        - crop_times (tuple of float): Tuple specifying the start and end times (in seconds) to crop the raw data (default: None).
        - crop_from_begignning (float): Time in seconds to crop from the beginning of the raw data (default: None).
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
    
    # Check the filter frequencies
    if high_pass_freq is None:
        high_pass_freq=Filters.high_pass_freq
        print(f"Warning: High pass frequency not provided. Using default value: {high_pass_freq} Hz")
        
    # Get file name without extension to use as file_id in summary and report title 
    file_name = Path(raw.filenames[0]).stem
    
    # Create output folder if it does not exist
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        output_dir_reports = output_dir / "reports"

    # Initialize object for logging
    logger = StdOutLogger(output_dir_reports, file_name)
        
    # Preprocessing start time
    sim_time_start = time.time()
    print('=============================================')
    print('Starting APICE default preprocessing pipeline')
    print(f"Processing date and time: {datetime.now()}\n\n")

    # Initialize object tracking the summary of artifacts
    summary = SummaryPreprocessing(output_dir_reports, file_name, try_loading=False)

    # Initialize reports
    if create_report:
        report = mne.Report(title=file_name)

    # Save log if True
    if save_log: logger.redirect_stdout_to_file(restore=True)

    # FILTER -----------------------------------------------------------------------------------------------
    Filter(raw,
            high_pass_freq=high_pass_freq,
            low_pass_freq=low_pass_freq, 
            n_jobs=n_jobs)

    # add notch filtering

    # ARTIFACT DETECTION ------------------------------------------------------------------------------

    # Initialize artifacts structure
    raw.artifacts = Artifacts(raw)
    annotations_to_rejection_matrix(raw)
    if create_report:
        report.add_raw(raw, 
                        title="Raw Data", 
                        psd=True, 
                        butterfly=False, 
                        replace=True, 
                        )

    # Detect artifacts
    raw, summary = detect_artifacts(raw, summary=summary)

    # Create a figure to visualize the artifact structure
    fig = DefineBTBC.plot_artifact_structure(raw, color_scheme='jet')    
    if create_report:
        report.add_figure(fig, "Artifacts Matrix", section="Raw Data", replace=True)

    # Add topomap of bad electrodes
    fig = plot_percentage_of_bad_data_across_sensors(raw)
    if create_report:
        report.add_figure(fig, "Bad data across electrodes", section="Raw Data", replace=True)
    
    # ARTIFACT CORRECTION ------------------------------------------------------------------------------
    raw, summary = correct_artifacts(raw, n_jobs=n_jobs, summary=summary)
    if create_report:
        report.add_raw(raw, 
                        title="Preprocessed Raw Data", 
                        psd=False, 
                        butterfly=True, 
                        scalings=50e-6, 
                        replace=True,
                        topomap_kwargs={"color_scheme": "jet"}
                        )
    
    # Add PSD
    fig = mne.viz.plot_raw_psd(raw, 
                                fmax=Filters.low_pass_freq, 
                                show=False)
    if create_report:
        report.add_figure(fig, "PSD", section="Preprocessed Raw Data", replace=True)
    
    # Create a figure to visualize the artifact structure
    fig = DefineBTBC.plot_artifact_structure(raw, color_scheme='jet')
    if create_report:
        report.add_figure(fig, "Artifacts Matrix", section="Preprocessed Raw Data", replace=True)

    # Add topomap of bad electrodes
    fig = plot_percentage_of_bad_data_across_sensors(raw)
    if create_report:
        report.add_figure(fig, "Bad data across electrodes", section="Preprocessed Raw Data", replace=True)


    # EXPORT DATA -----------------------------------------------------------------------------------------------
    
    # Save preprocessed raw
    if save_data:
        apice.io.Raw.export(raw, file_name, output_dir, data_suffix=preprocessed_data_suffix)

    # Save summary file
    if save_summary:
        output_dir_reports.mkdir(exist_ok=True)
        summary.save()

    # Save report
    if create_report and save_report:
        output_dir_reports.mkdir(exist_ok=True)
        print("Saving report")
        full_path = output_dir_reports / (file_name + preprocessed_data_suffix + ".html")
        report.save(fname=full_path, open_browser=False, overwrite=True)
    
    # Preprocessing end time
    sim_time_end = timedelta(seconds=np.round(time.time() - sim_time_start))
    print('\nTotal processing time :', str(sim_time_end), 'in hh:mm:ss')
    print('=============================================\n')
    
    if save_log:
        logger.close()

    return raw, summary, report


def segment_default_pipeline(raw, 
                   kwargs_events_from_annotations_for_segmentation, 
                   event_time_window,
                   high_pass_freq=None,
                   low_pass_freq=None,
                   baseline=None, 
                   kwargs_events_from_annotations_for_metadata=None,
                   kwargs_make_metadata=None,                             
                   evoked_by_event_type=True,
                   output_dir=None,
                   save_log=True,
                   save_epochs=True,
                   save_evoked=True,
                   save_report=True,
                   save_summary=True,
                   set_reference=None,
                   n_jobs=-1,
                   ):

    # INITIALIZATION -----------------------------------------------------------------------------------------------

    # Check if raw is an instance of mne.io.Raw
    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError("raw must be an instance of mne.io.Raw")
    
    # Check that raw has a montage
    if raw.get_montage() is None:
        raise ValueError("raw must have a montage. Please set the montage before preprocessing.")
    
    # Check that output_dir is provided if any of the saving options is True
    if output_dir is None and (save_log or save_epochs or save_evoked or save_report or save_summary):
        raise ValueError("output_dir must be provided if any of the saving options is True")
    
    # Get file name without extension to use as file_id in summary and report title
    file_name = Path(raw.filenames[0]).stem
    
    # Create output folder if it does not exist
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        output_dir_reports = output_dir / "reports"

    # Initialize object tracking the summary of artifacts
    summary = SummaryEpochs(output_dir_reports, file_name, try_loading=False)

    # Initialize object for logging
    logger = StdOutLogger(output_dir_reports, file_name)
        
    # Initialize reports
    report = mne.Report(title=file_name)

    # Save log if True
    if save_log: logger.redirect_stdout_to_file(restore=True)

    # Segmentation start time
    sim_time_start = time.time()
    print('=============================================')
    print('Starting APICE default segmentation pipeline')
    print(f"Segmentation date and time: {datetime.now()}\n\n")

    # Initialize artifacts structure
    if not hasattr(raw, 'artifacts'):
        raw.artifacts = Artifacts(raw)
        annotations_to_rejection_matrix(raw)

    # FILTER -----------------------------------------------------------------------------------------------
    Filter(raw,
            high_pass_freq=high_pass_freq,
            low_pass_freq=low_pass_freq, 
            n_jobs=n_jobs)

    # SEGMENTATION ----------------------------------------------------------------------------------------------
    
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
                                   evoked_by_event_type=evoked_by_event_type,
                                   set_reference=set_reference,
                                   summary=summary,
                                   )
    
    # Add epochs in report
    report.add_epochs(epochs, "Epochs", psd=True, replace=True)
    
    # Add epochs artifacts matrix
    fig = DefineBTBC.plot_artifact_structure(epochs, color_scheme='jet')
    report.add_figure(fig, "Artifacts Matrix", section="Epochs", replace=True)

    # Add topomap of bad electrodes
    fig = plot_percentage_of_bad_data_across_sensors(epochs)
    report.add_figure(fig, "Bad data across electrodes", section="Epochs", replace=True)

    # Add evokeds in the report
    report.add_evokeds(evokeds, titles=None, replace=True)


    # EXPORT DATA -----------------------------------------------------------------------------------------------
    
    # Save epochs 
    if save_epochs:
        apice.io.Epochs.export(epochs, file_name, output_dir)

    # Save evoked responses
    if save_evoked:
        file_name_evoked = (file_name + '-erp.fif')

        if not evoked_by_event_type:
            file_name_evoked = (file_name + '-erp.fif')
            folder_path = output_dir / 'erp'
            folder_path.mkdir(parents=True, exist_ok=True)
            full_path = folder_path / file_name_evoked
            print(f"Writing {full_path}")
            evokeds.save(full_path, overwrite=True)
            print(f"Closing {full_path}")
        else:
            for i in np.arange(len(evokeds)):
                full_path = output_dir / 'erp' / f"{file_name}_{evokeds[i].comment}.fif"
                print(f"Writing {full_path}")
                evokeds[i].save(full_path, overwrite=True)
                print(f"Closing {full_path}")
            print('[done]')

    # Save summary file
    if save_summary:
        output_dir_reports.mkdir(exist_ok=True)
        summary.save()

    # Save report
    if save_report:
        output_dir_reports.mkdir(exist_ok=True)
        print("Saving report")
        full_path = output_dir_reports / (file_name + "-epo.html")
        report.save(fname=full_path, open_browser=False, overwrite=True)
    
    # Preprocessing end time
    sim_time_end = timedelta(seconds=np.round(time.time() - sim_time_start))
    print('\nTotal processing time :', str(sim_time_end), 'in hh:mm:ss')
    print('=============================================\n')
    
    if save_log:
        logger.close()

    return epochs, evokeds, summary, report






def detect_artifacts(raw, summary=None):
    """
    Detects various artifacts in EEG data using specified detection algorithms.

    This function applies a series of artifact detection algorithms to the raw EEG data.
    Each algorithm is configured to use user-defined parameters. The algorithms look for bad electrodes,
    motion artifacts, jumps in signal, and defines the bad time segments for EEG correction.

    Args:
        raw (mne.io.Raw): The raw EEG data object that contains the EEG signal and metadata.
        summary (SummaryPreprocessing, optional): An instance of the SummaryPreprocessing class to track the summary of detected artifacts. Defaults to None.

    Returns:
        None: This function does not return a value but modifies the raw data object in place
            to annotate the detected artifacts.
    """
    
    # Detects bad electrodes based on user-configured parameters
    BadElectrodes(raw, config=True)
    if summary is not None:
        summary.add_to_summary('artifacts_detection_BadElectrodes', raw, overwrite=True)

    # Detects motion artifacts with a specific type set by user-defined parameters
    Motion(raw, type=1, config=True)
    if summary is not None:
        summary.add_to_summary('artifacts_detection_Motion', raw, overwrite=True)
    
    # Detects jumps in the EEG signal using user-configured parameters
    Jump(raw, config=True)
    if summary is not None:
        summary.add_to_summary('artifacts_detection_Jump', raw, overwrite=True)
    
    # Defines bad time segments in the EEG data for further correction, using user-configured parameters
    DefineBTBC(raw, config=True)
    if summary is not None:
        summary.add_to_summary('artifacts_detection_DefineBTBC', raw, overwrite=True)

    if summary is not None:
        return raw, summary
    else:
        return raw

def correct_artifacts(raw, 
                      n_jobs=-1, 
                      summary=None,
                      apply_targetPCA=True,
                      apply_segment_spherical_spline_interpolation=True,
                      apply_channels_spherical_spline_interpolation=True,
                      apply_motion_correction=True,):
    """
    Corrects artifacts in EEG data using a series of processing steps.

    This function applies multiple artifact correction techniques including Target PCA, 
    Spherical Spline Interpolation, and motion artifact correction. It also includes 
    filtering and baseline rescaling as part of the artifact correction process.

    Parameters:
    raw : Raw EEG object
        The raw EEG data to be processed for artifact correction.
    n_jobs : int
        Number of core used for the parallel computation. -1 to get all the available cores.
    summary (SummaryPreprocessing, optional): An instance of the SummaryPreprocessing class to track the summary of corrected artifacts. Defaults to None.

    Returns:
    None
    """
 
    # Apply Target PCA per electrode
    if apply_targetPCA:
        TargetPCA(raw, config=True)
        Filter(raw,
            high_pass_freq=Filters.high_pass_freq, 
            low_pass_freq=[], 
            n_jobs=n_jobs) 
        DefineBTBC(raw, config=True)
        if summary is not None:
            summary.add_to_summary('artifacts_correction_targetPCA', raw, overwrite=True)

     # Apply Spherical Spline Interpolation fro segmets for artifact correction
    if apply_segment_spherical_spline_interpolation:
        SegmentSphericalSplineInterpolation(raw, n_jobs, config=True)
        Filter(raw, high_pass_freq=Filters.high_pass_freq, low_pass_freq=[], n_jobs=n_jobs)
        DefineBTBC(raw, config=True)
        if summary is not None:
            summary.add_to_summary('artifacts_correction_SegmentsSphericalSpline', raw, overwrite=True)

    # Apply Spherical Spline Interpolation for whole channels
    if apply_channels_spherical_spline_interpolation:
        ChannelsSphericalSplineInterpolation(raw, n_jobs, config=True)
        if summary is not None:
            summary.add_to_summary('artifacts_correction_ChannelsSphericalSpline', raw, overwrite=True)
    
    # Detect motion artifacts again after correction
    if apply_motion_correction:
        Motion(raw, type=2, keep_rejected_previous=True, config=True)
        DefineBTBC(raw, config=True)
        if summary is not None:
            summary.add_to_summary('artifacts_correction_Motion', raw, overwrite=True)

    if summary is not None:
        return raw, summary
    else:
        return raw
    

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
    events_segm, event_id_segm = apice.io.Raw.events_from_annotations(raw, **kwargs_events_from_annotations_for_segmentation)
    
    # get events and event ids for metadata
    events_metadata, event_ids_metadata = apice.io.Raw.events_from_annotations(raw, **kwargs_events_from_annotations_for_metadata)
    
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
                 evoked_by_event_type=True, 
                 set_reference=None,
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
    evoked_by_event_type : bool, optional
        Flag to compute evoked responses by event type. Defaults to True.
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
    
    # Get events, and event ids
    events_segm, event_id_segm = apice.io.Raw.events_from_annotations(raw, **kwargs_events_from_annotations)
    
    # Segment the continuous data into epochs
    karg = dict(reject_by_annotation=False,
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            preload=True,
            metadata=metadata,
    )
    epochs = apice.io.Epochs.segment_continuous_data(raw, 
                                            events_segm, 
                                            event_id_segm, 
                                            karg,
                                            )
    
    # Define BadTimes and BadChannels for the segmented data
    DefineBTBC(epochs, segmented=True, config=True)
    if summary is not None:
        summary.add_to_summary('segmentation_Initial', epochs, overwrite=True)

    # Apply spherical spline interpolation for artifact correction and re-define BT and BC after interpolation
    ChannelsSphericalSplineInterpolation(epochs, n_jobs, config=True)
    DefineBTBC(epochs, segmented=True, config=True)
    
    # Identify and define bad epochs
    apice.io.Epochs.define_bad_epochs(epochs, config=True)
    
    # Remove bad epochs from the data 
    apice.io.Epochs.remove_bad_epochs(epochs)

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
        f"-\n\t-- by event type: {evoked_by_event_type}"
        )
    if evoked_by_event_type:
        evokeds = epochs.average(by_event_type=evoked_by_event_type)
    else:
        evokeds = epochs.average()

    if summary is not None:
        return epochs, evokeds, summary
    else:
        return epochs, evokeds


def get_summary(subject_no, subject_name, raw, df_summary, option='preprocessing'):
    """
    Generates a summary of preprocessing, correction, or segmentation for EEG data.

    Parameters:
    subject_no : int
        The subject number.
    subject_name : str
        The name of the subject.
    raw : mne.io.Raw or mne.Epochs
        The Raw or Epochs object containing EEG data and artifacts.
    df_summary : pandas.DataFrame
        The DataFrame where the summary will be appended.
    option : str
        The type of summary to generate. Options are 'preprocessing', 'correction', and 'segmentation'.

    Returns:
    df_summary : pandas.DataFrame
        Updated DataFrame with the new summary information.
    """

    # Summarize preprocessing steps
    if option == 'preprocessing':
        length = raw.times.max()
        bad_data = np.round(np.sum(raw.artifacts.BCT) / np.size(raw.artifacts.BCT) * 100, 2)
        bad_channels = np.round(np.sum(raw.artifacts.BC) / np.size(raw.artifacts.BC) * 100, 2)
        bad_times = np.round(np.sum(raw.artifacts.BT) / np.size(raw.artifacts.BT) * 100, 2)
        # Append the summary to the DataFrame
        df_summary.loc[len(df_summary)] = [subject_no, subject_name, length, bad_data, bad_channels, bad_times]

    # Summarize artifact correction steps
    elif option == 'correction':
        length = raw.times.max()
        corrected_data = np.round(np.sum(raw.artifacts.CCT) / np.size(raw.artifacts.CCT) * 100, 2)
        bad_data = np.round(np.sum(raw.artifacts.BCT) / np.size(raw.artifacts.BCT) * 100, 2)
        bad_channels = np.round(np.sum(raw.artifacts.BC) / np.size(raw.artifacts.BC) * 100, 2)
        bad_times = np.round(np.sum(raw.artifacts.BT) / np.size(raw.artifacts.BT) * 100, 2)
        # Append the summary to the DataFrame
        df_summary.loc[len(df_summary)] = [subject_no, subject_name, length, corrected_data, bad_data, bad_channels, bad_times]
                
    # Summarize segmentation steps
    elif option == 'segmentation':
        length = raw.times.max() - raw.times.min()
        drop_log = np.asarray(raw.drop_log, dtype=list)
        no_of_epochs = np.shape(drop_log)[0]
        no_of_remaining_epochs = np.shape(raw._data)[0]
        
        corrected_data = np.round(np.sum(raw.artifacts.CCT) / np.size(raw.artifacts.CCT) * 100, 2)
        bad_data = np.round(np.sum(raw.artifacts.BCT) / np.size(raw.artifacts.BCT) * 100, 2)
        bad_channels = np.round(np.sum(raw.artifacts.BC) / np.size(raw.artifacts.BC) * 100, 2)
        bad_times = np.round(np.sum(raw.artifacts.BT) / np.size(raw.artifacts.BT) * 100, 2)
        bad_epochs = np.round(np.sum(raw.artifacts.BE) / np.size(raw.artifacts.BE) * 100, 2)
        # Append the summary to the DataFrame
        df_summary.loc[len(df_summary)] = [subject_no, subject_name, no_of_epochs, no_of_remaining_epochs,
                                            length, corrected_data, bad_data, bad_channels, bad_times, bad_epochs]
    
    return df_summary


