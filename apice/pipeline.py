"""High-level preprocessing and segmentation workflows for APICE.

This module orchestrates the end-to-end APICE pipelines for continuous EEG
preprocessing, epoch segmentation, artifact summaries, logging, and report
generation.
"""

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
import seaborn as sns

from datetime import datetime, timezone


import mne
from mne import BaseEpochs


from apice.data_structures import RawAPICE
from apice.io import load_rawapice
from apice.utils import (get_onset_and_duration, get_cfg)
from apice.filter import Filter, ZapLine

# %% CLASSES DEFINITIONS
class Summary():
    """Base helper for CSV-backed processing summaries.

    Parameters
    ----------
    output_folder : str | pathlib.Path
        Directory where the summary CSV will be stored.
    output_file : str
        Summary filename.
    columns : list of str, default=['file_id', 'step', 'length', 'corrected_data', 'bad_data', 'bad_channels', 'bad_times']
        Columns used when initializing a new summary dataframe.
    try_loading : bool, default=True
        If True, load an existing summary file when present.
    """

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
        """Load summary data from disk if the CSV exists.

        Returns
        -------
        None
        """
        if self.output_full_path.exists():
            self.summary_df = pd.read_csv(self.output_full_path)
        else:
            print(f"File {self.output_full_path} does not exist. Summary could not be loaded.")

    def save(self):
        """Persist the current summary dataframe to disk.

        Returns
        -------
        None
        """
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.summary_df.to_csv(self.output_full_path, index=False)
    
    def remove_file_from_summary(self, file_id):
        """Remove all rows associated with one file identifier.

        Parameters
        ----------
        file_id : str
            File identifier to remove.

        Returns
        -------
        None
        """
        self.summary_df = self.summary_df[self.summary_df['file_id'] != file_id]

    def remove_file_step_from_summary(self, file_id, step):
        """Remove rows associated with a specific file and step.

        Parameters
        ----------
        file_id : str
            File identifier to filter.
        step : str
            Processing step label to remove.

        Returns
        -------
        None
        """
        self.summary_df = self.summary_df[~((self.summary_df['file_id'] == file_id) & (self.summary_df['step'] == step))]


class SummaryPreprocessing(Summary):
    """Summary table specialized for continuous preprocessing outputs."""

    def __init__(self, 
                 output_folder, 
                 file_name,
                 file_id=None,
                 outputfile_subfix="-summary-preproc.csv", 
                 try_loading=True,
                 ):
        """Initialize preprocessing summary storage.

        Parameters
        ----------
        output_folder : str | pathlib.Path
            Directory where the summary CSV is stored.
        file_name : str | pathlib.Path
            Source filename used to derive the summary filename.
        file_id : str | None, default=None
            Custom file identifier. If None, derived from ``file_name``.
        outputfile_subfix : str, default='-summary-preproc.csv'
            Filename suffix for the summary CSV.
        try_loading : bool, default=True
            If True, load an existing summary file if available.

        Returns
        -------
        None
        """
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
        """Append one preprocessing summary row.

        Parameters
        ----------
        step : str
            Processing step label.
        raw : mne.io.BaseRaw
            Raw object whose artifact metrics will be summarized.
        overwrite : bool, default=False
            If True, replace an existing row for the same file and step.

        Returns
        -------
        None
        """
        
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
    """Summary table specialized for segmented/epoched outputs."""

    def __init__(self, 
                 output_folder, 
                 file_name,
                 file_id=None,
                 outputfile_subfix="summary-epo.csv", 
                 try_loading=True,
                 ):
        """Initialize epochs summary storage.

        Parameters
        ----------
        output_folder : str | pathlib.Path
            Directory where the summary CSV is stored.
        file_name : str | pathlib.Path
            Source filename used to derive the summary filename.
        file_id : str | None, default=None
            Custom file identifier. If None, derived from ``file_name``.
        outputfile_subfix : str, default='summary-epo.csv'
            Filename suffix for the summary CSV.
        try_loading : bool, default=True
            If True, load an existing summary file if available.

        Returns
        -------
        None
        """
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
                             "n_rejected_epochs",
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
        """Append one epochs summary row.

        Parameters
        ----------
        step : str
            Processing step label.
        epochs : mne.BaseEpochs
            Epochs object whose artifact metrics will be summarized.
        overwrite : bool, default=False
            If True, replace an existing row for the same file and step.

        Returns
        -------
        None
        """
        
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
            no_of_rejected_epochs = np.sum(epochs.artifacts.BE)
        else:
            corrected_data = np.nan
            bad_data = np.nan
            bad_channels = np.nan
            bad_times = np.nan
            bad_epochs = np.nan
            no_of_rejected_epochs = np.nan
        self.summary_df.loc[len(self.summary_df)] = [self.file_id, step, no_of_epochs, no_of_remaining_epochs, no_of_rejected_epochs,
                                            length, corrected_data, bad_data, bad_channels, bad_times, bad_epochs]
        

class StdOutLogger():
    """Redirect standard output to a log file during pipeline execution.

    Parameters
    ----------
    output_folder : str | pathlib.Path
        Directory that will store the log file.
    file_name : str | pathlib.Path
        Base filename used to derive the log filename.
    """

    def __init__(self, output_folder, file_name):
        """Initialize logger paths.

        Parameters
        ----------
        output_folder : str | pathlib.Path
            Destination folder for log output.
        file_name : str | pathlib.Path
            Base name used to create the log filename.

        Returns
        -------
        None
        """
        file_name = Path(file_name).stem
        self.output_folder = Path(output_folder)
        self.output_file = f"{file_name}_log.txt"
        self.output_full_path = self.output_folder / self.output_file

    def restore_stdout(self):
        """Reset the log file contents before a new run.

        Returns
        -------
        None
        """
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.output_full_path.write_text("")

    def redirect_stdout_to_file(self, restore=False):
        """Redirect ``sys.stdout`` to the configured log file.

        Parameters
        ----------
        restore : bool, default=False
            If True, clear the existing log file before redirecting.

        Returns
        -------
        None
        """
        self.output_folder.mkdir(parents=True, exist_ok=True)
        if restore:
            self.restore_stdout()
        sys.stdout = open(self.output_full_path, "w")

    def close(self):
        """Close the redirected standard-output stream.

        Returns
        -------
        None
        """
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
                      show_figures=False,
                      l_freq=0.10,
                      h_freq=40,
                      l_trans_bandwidth=0.1,
                      h_trans_bandwidth=10,
                      line_noise_filt=True,
                      cfg_bad_channels_detection=None,
                      cfg_glitches_detection=None,
                      cfg_target_pca=None,
                      cfg_artifacts_detection=None,
                      cfg_spline_segments=None,
                      cfg_spline_channels=None,
                      n_jobs=-1,
                      ):
    """Run the default APICE preprocessing workflow over multiple files.

    Parameters
    ----------
    input_dir : str | pathlib.Path
        Input folder containing raw EEG files, or the BIDS root when
        ``input_dir_bids`` is True.
    output_dir : str | pathlib.Path
        Output folder for preprocessed data, reports, logs, and summaries.
    input_dir_bids : bool, default=False
        If True, discover input files as BIDS entries instead of plain files.
    bids_session : str | None, default=None
        Optional BIDS session filter.
    bids_task : str | None, default=None
        Optional BIDS task filter.
    bids_run : str | None, default=None
        Optional BIDS run filter.
    bids_subject : str | None, default=None
        Optional BIDS subject filter.
    bids_extension : str, default='.vhdr'
        BIDS file extension to select.
    bids_datatype : str, default='eeg'
        BIDS datatype selector.
    bids_suffix : str, default='eeg'
        BIDS suffix selector.
    processed_file_pattern : str, default='*-preproc.fif'
        Pattern used to detect already processed outputs.
    data_selection_method : str | list[str], default='all'
        File-selection strategy forwarded to APICE I/O helpers.
    drop_electrodes : list[str] | None, default=None
        Channels to remove before preprocessing.
    picks : str | list | slice, default='eeg'
        Channels to keep during preprocessing.
    reference_channels : list[str] | str | None, default=None
        Reference channels to preserve in artifact handling.
    crop_times : tuple[float, float] | None, default=None
        Optional ``(tmin, tmax)`` crop window in seconds.
    crop_from_beginnning : float | None, default=None
        Seconds to trim from the start of the recording.
    crop_from_end : float | None, default=None
        Seconds to trim from the end of the recording.
    resample_freq : float | None, default=None
        Target sampling frequency in Hz.
    stim_channels_to_annotations : bool, default=True
        If True, convert stim-channel events to annotations before processing.
    montage : mne.channels.DigMontage | str | pathlib.Path | None, default=None
        Montage object, built-in montage name, or path to a montage file.
    save_log : bool, default=True
        If True, write one log file per processed recording.
    save_report : bool, default=True
        If True, write an HTML report per processed recording.
    save_summary : bool, default=True
        If True, write a preprocessing summary CSV.
    show_figures : bool, default=False
        If False, disable interactive Matplotlib rendering while each file is
        processed. This prevents figure windows from popping up during
        multi-file runs while still allowing report figures to be created.
    l_freq : float, default=0.10
        High-pass cutoff frequency in Hz.
    h_freq : float | None, default=40
        Low-pass cutoff frequency in Hz.
    l_trans_bandwidth : float, default=0.1
        High-pass transition bandwidth.
    h_trans_bandwidth : float, default=10
        Low-pass transition bandwidth.
    line_noise_filt : bool, default=True
        If True, apply the line noise filtering.
    cfg_bad_channels_detection : None | str | pathlib.Path | dict, default=None
        Configuration source for bad-channel detection.
    cfg_glitches_detection : None | str | pathlib.Path | dict, default=None
        Configuration source for glitch detection.
    cfg_target_pca : None | str | pathlib.Path | dict, default=None
        Configuration source for target-PCA correction.
    cfg_artifacts_detection : None | str | pathlib.Path | dict, default=None
        Configuration source for artifact detection.
    cfg_spline_segments : None | str | pathlib.Path | dict, default=None
        Configuration source for segment-wise spline correction.
    cfg_spline_channels : None | str | pathlib.Path | dict, default=None
        Configuration source for channel-wise spline correction.
    n_jobs : int, default=-1
        Number of parallel jobs for compute-intensive steps.

    Returns
    -------
    None
        Writes preprocessing outputs for each selected file.
    """
    
    # Initialize output folders
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
            was_interactive = plt.isinteractive()
            original_backend = plt.get_backend()
            if not show_figures:
                plt.close('all')
                if original_backend.lower() != 'agg':
                    plt.switch_backend('Agg')
                plt.ioff()
            try:
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
                                            line_noise_filt=line_noise_filt,
                                            cfg_bad_channels_detection=cfg_bad_channels_detection,
                                            cfg_glitches_detection=cfg_glitches_detection,
                                            cfg_target_pca=cfg_target_pca,
                                            cfg_artifacts_detection=cfg_artifacts_detection,
                                            cfg_spline_segments=cfg_spline_segments,
                                            cfg_spline_channels=cfg_spline_channels,
                                            n_jobs=n_jobs,
                                            )
            finally:
                plt.close('all')
                if not show_figures and original_backend.lower() != 'agg':
                    plt.switch_backend(original_backend)
                if not show_figures and was_interactive:
                    plt.ion()
            
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
                     show_figures=False,
                     save_cfg=True,
                     set_reference=None,
                     cfg_define_bcbt_epochs=None,
                     cfg_spline_channels=None,  
                     cfg_bad_epochs=None,              
                     n_jobs=-1,
                     ):
    """Run the default APICE segmentation workflow over multiple files.

    Parameters
    ----------
    input_dir : str | pathlib.Path
        Directory containing preprocessed raw FIF files.
    output_dir : str | pathlib.Path
        Output folder for epoch files, ERP files, reports, and summaries.
    kwargs_events_from_annotations_for_segmentation : dict
        Keyword arguments passed to ``mne.events_from_annotations`` for epoching.
    event_time_window : tuple[float, float]
        Epoch time window ``(tmin, tmax)`` in seconds.
    processed_file_pattern : str, default='*-epo.fif'
        Pattern used to detect existing segmentation outputs.
    data_selection_method : str | list[str], default='all'
        File-selection strategy forwarded to APICE I/O helpers.
    l_freq : float | None, default=None
        Optional high-pass cutoff used before segmentation.
    h_freq : float | None, default=None
        Optional low-pass cutoff used before segmentation.
    l_trans_bandwidth : float, default=0.1
        High-pass transition bandwidth.
    h_trans_bandwidth : float, default=10
        Low-pass transition bandwidth.
    baseline : tuple[float, float] | None, default=None
        Baseline window passed to MNE epoching.
    kwargs_events_from_annotations_for_metadata : dict | None, default=None
        Keyword arguments used to derive metadata events.
    kwargs_make_metadata : dict | None, default=None
        Keyword arguments used when creating epoch metadata.
    evoked_by : bool | str | list[str], default=True
        Event grouping used when computing evoked responses.
    save_log : bool, default=True
        If True, write one segmentation log per processed recording.
    save_epochs : bool, default=True
        If True, export epochs to FIF.
    save_only_good_epochs : bool, default=False
        If True, export only epochs that survive rejection.
    save_evoked : bool, default=True
        If True, export evoked responses.
    save_report : bool, default=True
        If True, write an HTML segmentation report.
    save_summary : bool, default=True
        If True, write a segmentation summary CSV.
    show_figures : bool, default=False
        If False, disable interactive Matplotlib rendering while each file is
        processed. This prevents figure windows from popping up during
        multi-file runs while still allowing report figures to be created.
    save_cfg : bool, default=True
        If True, save the effective segmentation configurations to JSON.
    set_reference : dict | None, default=None
        Reference parameters passed to ``epochs.set_eeg_reference``.
    cfg_define_bcbt_epochs : None | str | pathlib.Path | dict, default=None
        Configuration source for epoch-level BC/BT derivation.
    cfg_spline_channels : None | str | pathlib.Path | dict, default=None
        Configuration source for spline interpolation of epochs.
    cfg_bad_epochs : None | str | pathlib.Path | dict, default=None
        Configuration source for bad-epoch detection.
    n_jobs : int, default=-1
        Number of parallel jobs for compute-intensive steps.

    Returns
    -------
    None
        Writes segmentation outputs for each selected file.
    """

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
            was_interactive = plt.isinteractive()
            original_backend = plt.get_backend()
            if not show_figures:
                plt.close('all')
                if original_backend.lower() != 'agg':
                    plt.switch_backend('Agg')
                plt.ioff()
            try:
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
            finally:
                plt.close('all')
                if not show_figures and original_backend.lower() != 'agg':
                    plt.switch_backend(original_backend)
                if not show_figures and was_interactive:
                    plt.ion()
            
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
    """Apply structural preprocessing steps before APICE artifact handling.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Raw object to process.
    drop_electrodes : list[str] | None, default=None
        Channels to drop before further processing.
    picks : str | list | slice, default='eeg'
        Channels to keep using MNE picking semantics.
    crop_times : tuple[float, float] | None, default=None
        Optional ``(tmin, tmax)`` crop window in seconds.
    crop_from_beginnning : float | None, default=None
        Seconds to trim from the beginning of the recording.
    crop_from_end : float | None, default=None
        Seconds to trim from the end of the recording.
    resample_freq : float | None, default=None
        Target sampling frequency in Hz.
    stim_channels_to_annotations : bool, default=True
        If True, convert stim-channel events to annotations.
    montage : mne.channels.DigMontage | str | pathlib.Path | None, default=None
        Montage object, built-in montage name, or path to a montage file.
    head_size : float | None, default=None
        Head-size parameter forwarded to MNE montage creation when relevant.

    Returns
    -------
    raw : mne.io.BaseRaw
        Updated raw object after structural preprocessing.
    """

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
                             preprocessed_data_suffix='-preproc-raw',
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
                             line_noise_filt=True,
                             cfg_define_bcbt_raw=None,
                             cfg_bad_channels_detection=None,
                             cfg_glitches_detection=None,
                             cfg_target_pca=None,
                             cfg_artifacts_detection=None,
                             cfg_spline_segments=None,
                             cfg_spline_channels=None,
                             n_jobs=-1,
                             ):
    """Run the default APICE artifact-detection and correction pipeline.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Raw object to preprocess. It must already contain a montage.
    preprocessed_data_suffix : str, default='-preproc-raw'
        Suffix appended to exported preprocessed raw files.
    output_dir : str | pathlib.Path | None, default=None
        Destination folder for exported artifacts, reports, logs, and summaries.
    file_name : str | None, default=None
        Base name used for output files.
    create_report : bool, default=True
        If True, build an in-memory MNE report during processing.
    save_log : bool, default=True
        If True, write pipeline logs to disk.
    save_data : bool, default=True
        If True, export the preprocessed raw file.
    save_report : bool, default=True
        If True, save the generated HTML report.
    save_summary : bool, default=True
        If True, save preprocessing summary metrics.
    save_cfg : bool, default=True
        If True, save the effective preprocessing configurations to JSON.
    reference_channels : list[str] | str | None, default=None
        Reference channels to preserve during bad-channel handling.
    l_freq : float, default=0.10
        High-pass cutoff frequency in Hz.
    h_freq : float | None, default=40
        Low-pass cutoff frequency in Hz.
    l_trans_bandwidth : float, default=0.1
        High-pass transition bandwidth.
    h_trans_bandwidth : float, default=10
        Low-pass transition bandwidth.
    line_noise_filt : bool, default=True
        If True, apply the line noise filtering.
    cfg_define_bcbt_raw : None | str | pathlib.Path | dict, default=None
        Configuration source for deriving BC/BT masks from raw BCT masks.
    cfg_bad_channels_detection : None | str | pathlib.Path | dict, default=None
        Configuration source for bad-channel detection.
    cfg_glitches_detection : None | str | pathlib.Path | dict, default=None
        Configuration source for glitch detection.
    cfg_target_pca : None | str | pathlib.Path | dict, default=None
        Configuration source for target-PCA correction.
    cfg_artifacts_detection : None | str | pathlib.Path | dict, default=None
        Configuration source for artifact detection.
    cfg_spline_segments : None | str | pathlib.Path | dict, default=None
        Configuration source for segment-wise spline correction.
    cfg_spline_channels : None | str | pathlib.Path | dict, default=None
        Configuration source for channel-wise spline correction.
    n_jobs : int, default=-1
        Number of parallel jobs for compute-intensive steps.

    Returns
    -------
    raw : RawAPICE
        Preprocessed raw object with artifact masks and corrections applied.
    summary : SummaryPreprocessing
        Summary object tracking preprocessing metrics.
    report : mne.Report | None
        Generated report, or None when ``create_report`` is False.
    """

        
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
    summary = SummaryPreprocessing(output_dir_summary, file_name, try_loading=False)

    # Initialize reports
    if create_report:
        report = mne.Report(title=file_name)
    else:
        report = None

    # Save log if True
    if save_log: logger.redirect_stdout_to_file(restore=True)

    # FILTER -----------------------------------------------------------------------------------------------
    # High pass filter to remove the slow drifts
    Filter(raw, l_freq=l_freq, h_freq=None, l_trans_bandwidth=l_trans_bandwidth, h_trans_bandwidth=h_trans_bandwidth, n_jobs=n_jobs)    

    # Optional line noise filter
    if line_noise_filt: 
        zap_worker = ZapLine(raw, fline=50, chunk_duration=30, n_jobs=n_jobs)
        raw, zap_fig = zap_worker.apply(raw)
        if create_report:
            report.add_figure(zap_fig, "ZapLine Spectral Power", section="Raw Data", replace=True)
            plt.close(zap_fig)

    # Low pass filter
    Filter(raw, l_freq=None, h_freq=h_freq, l_trans_bandwidth=l_trans_bandwidth, h_trans_bandwidth=h_trans_bandwidth, n_jobs=n_jobs)

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
            fig = raw.compute_psd(method='welch',fmax=fmax,exclude=exclude).plot(show=False)
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
    raw.detect_bad_channels(cfg=cfg_bad_channels_detection)
    raw.deal_with_reference_channels(reference_channels)
    summary.add_to_summary('artifacts_detection_BadElectrodes', raw, overwrite=True)

    # Detect glitches
    raw.detect_glitches(cfg=cfg_glitches_detection)
    raw.deal_with_reference_channels(reference_channels)
    summary.add_to_summary('artifacts_detection_Glitches', raw, overwrite=True)

    # Correct glitches
    raw.correct_target_pca(cfg=cfg_target_pca)
    Filter(raw, l_freq=l_freq, h_freq=None, l_trans_bandwidth=l_trans_bandwidth, h_trans_bandwidth=h_trans_bandwidth, n_jobs=n_jobs)
    summary.add_to_summary('artifacts_correction_TargetPCA', raw, overwrite=True)

    # Detect artifacts
    raw.detect_artifacts(cfg=cfg_artifacts_detection)
    raw.deal_with_reference_channels(reference_channels)
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
    raw.correct_spline_segments(cfg=cfg_spline_segments)
    Filter(raw, l_freq=l_freq, h_freq=None, l_trans_bandwidth=l_trans_bandwidth, h_trans_bandwidth=h_trans_bandwidth, n_jobs=n_jobs)
    summary.add_to_summary('artifacts_correction_Segments', raw, overwrite=True)

    # Correct channels using spherical spline interpolation
    raw.correct_spline_channels(cfg=cfg_spline_channels)
    summary.add_to_summary('artifacts_correction_BadChannels', raw, overwrite=True)

    # Re-detect bad data after correction to check if there are still bad channels or time segments that need to be marked as bad after the correction
    raw.detect_artifacts(cfg=cfg_artifacts_detection)
    raw.deal_with_reference_channels(reference_channels)    
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
            fig = raw.compute_psd(method='welch',fmax=fmax,exclude=exclude).plot(show=False)
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

    # Add a plot showing the amount of rejected data
    if create_report:
        try:
            fig = plot_summary(summary.summary_df, metrics = ["%_corrected_data", "%_bad_data", "%_bad_channels", "%_bad_times"])
            report.add_figure(fig, "Rejected Data Summary", section="Preprocessed Raw Data", replace=True)
        except Exception as e:
            print(f"Warning: Could not add preprocessed rejected-data summary to report: {e}")

    # EXPORT DATA -----------------------------------------------------------------------------------------------
    
    # Save preprocessed raw
    if save_data:
        print("Saving preprocessed raw data")
        full_path = output_dir / (file_name + preprocessed_data_suffix + ".fif")
        raw.export(full_path, overwrite=True)

    # Save summary file
    if save_summary:
        print("Saving preprocessing summary")
        output_dir_summary.mkdir(exist_ok=True)
        summary.save()

    # Save report
    if create_report and save_report:
        output_dir_reports.mkdir(exist_ok=True)
        print("Saving report")
        full_path = output_dir_reports / (file_name + preprocessed_data_suffix + "-report.html")
        report.save(fname=full_path, open_browser=False, overwrite=True)

    # Save the configurations used for preprocessing
    if save_cfg:
        print("Saving preprocessing configurations")
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
                   epoch_data_suffix='-epo',
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
    """Run epoching, interpolation, bad-epoch marking, and ERP computation.

    Parameters
    ----------
    raw : RawAPICE
        Preprocessed raw object ready for segmentation.
    kwargs_events_from_annotations_for_segmentation : dict
        Keyword arguments passed to ``mne.events_from_annotations`` for epoching.
    event_time_window : tuple[float, float]
        Epoch time window ``(tmin, tmax)`` in seconds.
    file_name : str | None, default=None
        Base name used for exported files.
    l_freq : float | None, default=None
        Optional high-pass cutoff before segmentation.
    h_freq : float | None, default=None
        Optional low-pass cutoff before segmentation.
    l_trans_bandwidth : float, default=0.1
        High-pass transition bandwidth.
    h_trans_bandwidth : float, default=10
        Low-pass transition bandwidth.
    baseline : tuple[float, float] | None, default=None
        Baseline correction window passed to MNE epoching.
    kwargs_events_from_annotations_for_metadata : dict | None, default=None
        Event selection kwargs used to build metadata.
    kwargs_make_metadata : dict | None, default=None
        Keyword arguments forwarded to metadata generation.
    evoked_by : str | list[str] | None, default='all'
        Grouping used when computing evoked responses.
    output_dir : str | pathlib.Path | None, default=None
        Destination folder for exported segmentation outputs.
    create_report : bool, default=True
        If True, build an in-memory MNE report during processing.
    save_log : bool, default=True
        If True, write pipeline logs to disk.
    save_epochs : bool, default=True
        If True, export epochs to FIF.
    save_only_good_epochs : bool, default=False
        If True, export only epochs that remain after rejection.
    save_evoked : bool, default=True
        If True, export evoked responses.
    save_report : bool, default=True
        If True, save the generated HTML report.
    save_summary : bool, default=True
        If True, save segmentation summary metrics.
    save_cfg : bool, default=True
        If True, save the effective segmentation configurations to JSON.
    set_reference : dict | None, default=None
        Reference parameters passed to ``epochs.set_eeg_reference``.
    cfg_define_bcbt_epochs : None | str | pathlib.Path | dict, default=None
        Configuration source for epoch-level BC/BT derivation.
    cfg_spline_channels : None | str | pathlib.Path | dict, default=None
        Configuration source for channel-wise spline interpolation.
    cfg_bad_epochs : None | str | pathlib.Path | dict, default=None
        Configuration source for bad-epoch detection.
    n_jobs : int, default=-1
        Number of parallel jobs for compute-intensive steps.

    Returns
    -------
    epochs : EpochsAPICE
        Segmented data with updated artifact masks.
    evokeds : mne.Evoked | dict | None
        Evoked response object(s) computed from good epochs.
    summary : SummaryEpochs
        Summary object tracking segmentation metrics.
    report : mne.Report | None
        Generated report, or None when ``create_report`` is False.
    """

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

    # Add a plot showing the amount of rejected data
    if create_report:
        try:
            fig = plot_summary(summary.summary_df, metrics = ["%_corrected_data", "%_bad_data", "%_bad_channels", "%_bad_times", "%_bad_epochs"])
            report.add_figure(fig, "Rejected Data Summary", section="Epochs", replace=True)
        except Exception as e:
            print(f"Warning: Could not add rejected data summary to report: {e}")

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
                    fig = evoked.plot(show=False)
                    report.add_figure(fig, f"Evoked responses {key}", section="Evoked responses", replace=True)
                except Exception as fig_err:
                    print(f"Warning: Could not add fallback evoked figure {key} to report: {fig_err}")


    # EXPORT DATA -----------------------------------------------------------------------------------------------
    
    # Save epochs 
    if save_epochs:
        fullpath_epochs = output_dir / (file_name + epoch_data_suffix + '.fif')
        if save_only_good_epochs:
            epochs_good = epochs.copy()
            epochs_good.remove_bad_epochs()
            epochs_good.export(fullpath_epochs, overwrite=True)
        else:
            epochs.export(fullpath_epochs, overwrite=True)

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
    """Estimate a default annotation duration from stim channels.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Raw object containing one or more stim channels.
    threshold : float, default=0.01
        Minimum stim amplitude considered active.

    Returns
    -------
    duration : float
        Maximum detected stim duration in seconds.
    """

    stim_data = raw.copy().pick('stim').get_data()    
    above_baseline = np.where(stim_data > threshold, 1, 0)
    onsets, durations = get_onset_and_duration(above_baseline)

    return np.max(durations)
    
def convert_stim_channels_to_annotations(raw):
    """Convert stim-channel events into MNE annotations.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Raw object whose stim channels will be translated into annotations.

    Returns
    -------
    None
        Updates ``raw.annotations`` in place.
    """
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
    """Generate an epoch metadata dataframe from raw annotations.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Raw object providing annotations and sampling information.
    kwargs_events_from_annotations_for_metadata : dict, default={}
        Keyword arguments used to derive metadata events.
    kwargs_events_from_annotations_for_segmentation : dict, default={}
        Keyword arguments used to derive segmentation events.
    columns_events_to_keep : list[str] | None, default=None
        Optional subset of metadata columns to retain.
    tmin : float, default=-0.5
        Metadata window start in seconds.
    tmax : float, default=0.5
        Metadata window end in seconds.
    keep_first : str | None, default=None
        Forwarded to ``mne.epochs.make_metadata``.
    keep_last : str | None, default=None
        Forwarded to ``mne.epochs.make_metadata``.

    Returns
    -------
    metadata : pandas.DataFrame
        Metadata table aligned with segmentation events.
    """
    
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
                 baseline=None, 
                 evoked_by="all", 
                 set_reference=None,
                 cfg_define_bcbt_epochs=None,
                 cfg_spline_channels=None,
                 cfg_bad_epochs=None,
                 summary=None):
    """Segment continuous data, mark bad epochs, and compute evoked responses.

    Parameters
    ----------
    raw : RawAPICE
        Preprocessed raw object containing annotations and artifact masks.
    kwargs_events_from_annotations : dict, default={}
        Keyword arguments passed to ``mne.events_from_annotations``.
    metadata : pandas.DataFrame | None, default=None
        Optional metadata aligned with the requested epochs.
    tmin : float, default=-0.2
        Epoch start relative to event onset in seconds.
    tmax : float, default=0.5
        Epoch end relative to event onset in seconds.
    baseline : tuple[float, float] | None, default=None
        Baseline correction window for MNE epoching.
    evoked_by : str | list[str] | None, default='all'
        Grouping used when computing evoked responses.
    set_reference : dict | None, default=None
        Reference parameters passed to ``epochs.set_eeg_reference``.
    cfg_define_bcbt_epochs : None | str | pathlib.Path | dict, default=None
        Configuration source for epoch-level BC/BT derivation.
    cfg_spline_channels : None | str | pathlib.Path | dict, default=None
        Configuration source for channel-wise spline interpolation.
    cfg_bad_epochs : None | str | pathlib.Path | dict, default=None
        Configuration source for bad-epoch detection.
    summary : SummaryEpochs | None, default=None
        Summary object updated with segmentation metrics.

    Returns
    -------
    epochs : EpochsAPICE
        Segmented epochs with updated artifact masks.
    evokeds : mne.Evoked | dict | None
        Evoked response object(s) computed from good epochs.
    summary : SummaryEpochs | None
        Updated summary object, or None when no summary was provided.
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
    epochs.correct_spline_channels(cfg=cfg_spline_channels)
    
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
    if summary is not None:
        summary.add_to_summary('segmentation_GoodEpochs', epochs_good, overwrite=True)
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




def plot_summary(summary_df, metrics = ["%_corrected_data", "%_bad_data", "%_bad_channels", "%_bad_times"]):

    preproc_df = summary_df.melt(id_vars='step', value_vars=metrics, var_name='metric', value_name='value')

    # Do a bar plot using the metrics as hues for each preprocessing step
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=preproc_df, x='step', y='value', hue='metric', ax=ax)
    # draw vertical line to separate steps  
    for i in range(len(preproc_df['step'].unique()) - 1):
        ax.axvline(x=i + 0.5, color='gray', linestyle='--')
    ax.set_title("Preprocessing Metrics by Step")
    ax.set_ylabel("Percentage")
    ax.set_xlabel("Preprocessing Step")
    ax.legend(title="Metric")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    return fig
