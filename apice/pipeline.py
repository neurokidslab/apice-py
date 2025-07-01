import os

import mne.baseline
import numpy as np
from prettytable import PrettyTable

import apice.artifacts_correction 
from apice.artifacts_structure import annotate_bads, set_reference, Artifacts, annotations_to_rejection_matrix, plot_percentage_of_bad_data_across_sensors
import apice.parameters
import apice.segmentation
import apice.artifacts_rejection
import apice.io 
import apice.filter
import sys
from datetime import date, datetime
from tabulate import tabulate
from datetime import timedelta

import mne
import matplotlib.pyplot as plt
import pandas as pd
import time

from apice.io import Raw, export_epoch
from apice.filter import Filter


# %% FUNCTIONS

def create_summary_dataframes(output_dir):
    """
    Create empty DataFrames with predefined columns for summarizing data processing results.

    Returns:
        pd.DataFrame, pd.DataFrame, pd.DataFrame: Three empty DataFrames for preprocessing, correction, and epochs summaries.
    """
    
    csv_file_path = os.path.join(output_dir, 'summary_of_artifacts_detected_in_raw.csv')
    if os.path.exists(csv_file_path):
        # If the file exists, import it as a DataFrame
        df_preprocessing_summary = pd.read_csv(csv_file_path)
    else:
        # If the file does not exist, create a new DataFrame
        df_preprocessing_summary = pd.DataFrame(
            columns=[
                "No.",
                "Subject",
                "Length (s)",
                "Bad Data (%)",
                "Bad Channels (%)",
                "Bad Times (%)",
            ]
        )

    csv_file_path = os.path.join(output_dir, 'summary_of_corrected_artifacts_in_raw.csv')
    if os.path.exists(csv_file_path):
        # If the file exists, import it as a DataFrame
        df_correction_summary = pd.read_csv(csv_file_path)
    else:
        # If the file does not exist, create a new DataFrame
        df_correction_summary = pd.DataFrame(
            columns=[
                "No.",
                "Subject",
                "Length (s)",
                "Corrected Data (%)",
                "Bad Data (%)",
                "Bad Channels (%)",
                "Bad Times (%)",
            ]
        )
    
    csv_file_path = os.path.join(output_dir, 'summary_of_corrected_artifacts_in_epochs.csv')
    if os.path.exists(csv_file_path):
        # If the file exists, import it as a DataFrame
        df_epochs_summary = pd.read_csv(csv_file_path)
    else:
        # If the file does not exist, create a new DataFrame
        df_epochs_summary = pd.DataFrame(
                columns=[
                "No.",
                "Subject",
                "No Of Epochs",
                "No of Remaining Epochs",
                "Length of epoch (s)",
                "Corrected Data (%)",
                "Bad Data (%)",
                "Bad Channels (%)",
                "Bad Times (%)",
                "Bad Epochs (%)",
            ]
        )
    
    return df_preprocessing_summary, df_correction_summary, df_epochs_summary


def redirect_stdout_to_file(output_dir, subject_name):
    """
    Redirects the standard output to a log file.

    Args:
        output_dir (str): The directory where the log file will be saved.
        subject_name (str): The name of the subject used to create the log file name.

    Returns:
        None
    """
    # Create the file_name based on output_dir and subject_name
    folder_dir = os.path.join(output_dir, 'reports')
    file_name = os.path.join(folder_dir, subject_name.split(sep=".")[0] + "_log.txt")
    
    # Create the folder if it does not exist
    if not os.path.exists(folder_dir):
        os.makedirs(folder_dir)
    
    # Redirect sys.stdout to the specified file
    sys.stdout = open(file_name, "w")


def run(input_dir, output_dir,
        data_selection_method, event_keys_for_segmentation, event_time_window,
        baseline_time_window, montage, by_event_type=True,
        save_preprocessed_raw=True,
        save_segmented_data=True, save_evoked_response=True,
        save_log=False, n_jobs=-1): 

    # Initialize output folders
    folder_names = ['preprocessed_raw', 'epochs', 'erp', 'reports']
    for folder_name in folder_names:
        folder = os.path.join(output_dir, folder_name)
        if not os.path.exists(folder):
            os.makedirs(folder)
    
    # Get all files to process
    from apice.io import get_files_to_process
    subjects = get_files_to_process(
        input_dir=input_dir,
        output_dir=os.path.join(output_dir, 'preprocessed_raw'),
        data_selection_method=data_selection_method,
    )

    # Initialize data frames for summary of artifacts
    (df_preprocessing_summary, df_correction_summary, df_epochs_summary) = create_summary_dataframes(output_dir)
    
    print(f"\nNumber of files to process: {len(subjects)}\n")


    from apice.parameters import Filters, Filters_epochs

    # Loop over the subjects for data processing
    for i in np.arange(len(subjects)):
        
        # IMPORT RAW DATA ---------------------------------------------------------------------------------------------
        
        # Get subject name and no
        full_path = subjects[i]
        subject_no = i + 1
        subject_name = os.path.basename(full_path)
        
        # Initialize reports
        report = mne.Report(title=subject_name)

        # Save log if True
        if save_log: redirect_stdout_to_file(output_dir, subject_name)

        # Preprocessing start time
        sim_time_start = time.time()
        print('=============================================\n')
        print(f"Processing date and time: {datetime.now()}\n\n")
        
        raw = Raw.import_raw(full_path, montage=montage)

        raw.artifacts = Artifacts(raw)

        annotations_to_rejection_matrix(raw)

        # Add Raw to report
        report.add_raw(raw, 
                       title="Raw Data", 
                       psd=True, 
                       butterfly=False, 
                       replace=True, 
                       )

        # PREPROCESSING PIPELINE ----------------------------------------------------------------------------------------------

        # FILTER
        Filter(raw,
               high_pass_freq=Filters.high_pass_freq,
               low_pass_freq=Filters.low_pass_freq, 
               n_jobs=n_jobs)

        raw._data = mne.baseline.rescale(raw._data, raw.times, baseline=(None, None), mode='mean', copy=False)
        
        # ARTIFACT DETECTION
        detect_artifacts(raw)
        df_preprocessing_summary = get_summary(subject_no, subject_name, raw, df_preprocessing_summary, option='preprocessing')

        # Create a figure to visualize the artifact structure
        fig = DefineBTBC.plot_artifact_structure(raw, color_scheme='jet')
        
        # Add artifacts to reports
        report.add_figure(fig, "Artifacts Matrix", section="Raw Data", replace=True)

        # Add topomap of bad electrodes
        fig = plot_percentage_of_bad_data_across_sensors(raw)
        report.add_figure(fig, "Bad data across electrodes", section="Raw Data", replace=True)
        
        # ARTIFACT CORRECTION
        correct_artifacts(raw, n_jobs)
        df_correction_summary = get_summary(subject_no, subject_name, raw, df_correction_summary, option='correction')
        
        # Add preprocessed raw to report
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
        report.add_figure(fig, "PSD", section="Preprocessed Raw Data", replace=True)
        
        # REJECTION MATRIX TO ANNOTATIONS
        annotate_bads(raw, channels=True, times=True, data=True, corrected=True)
        
        # Create a figure to visualize the artifact structure
        fig = DefineBTBC.plot_artifact_structure(raw, color_scheme='jet')
        
        # Add artifacts to reports
        report.add_figure(fig, "Artifacts Matrix", section="Preprocessed Raw Data", replace=True)

        # Add topomap of bad electrodes
        fig = plot_percentage_of_bad_data_across_sensors(raw)
        report.add_figure(fig, "Bad data across electrodes", section="Preprocessed Raw Data", replace=True)

        # SEGMENTATION ----------------------------------------------------------------------------------------------
        if event_keys_for_segmentation:
            epochs, evokeds = segment_data(raw, event_keys=event_keys_for_segmentation,
                                                        tmin=event_time_window[0], tmax=event_time_window[1],
                                                        n_jobs=n_jobs,
                                                        baseline_time_window=baseline_time_window,
                                                        by_event_type=by_event_type)
            df_epochs_summary = get_summary(subject_no, subject_name, epochs, df_epochs_summary, option='segmentation')
            
            # Add epochs in report
            report.add_epochs(epochs, "Epochs", psd=True, replace=True)
            
            # Add epochs artifacts matrix
            fig = DefineBTBC.plot_artifact_structure(epochs, color_scheme='jet')
            report.add_figure(fig, "Artifacts Matrix", section="Epochs", replace=True)

            # Add topomap of bad electrodes
            fig = plot_percentage_of_bad_data_across_sensors(epochs)
            report.add_figure(fig, "Bad data across electrodes", section="Epochs", replace=True)

        # EXPORT DATA -----------------------------------------------------------------------------------------------
        if save_preprocessed_raw:
            file_name = subject_name.split(sep='.')[0]
            if file_name.endswith('-raw'):
                file_name = file_name.replace('-raw', '-prp')
            else:
                file_name = file_name+'-prp'
            file_name = file_name+'.fif'
            Raw.export_raw(raw, file_name, output_path=os.path.join(output_dir, 'preprocessed_raw'))

        if save_segmented_data and event_keys_for_segmentation:
            file_name = subject_name.split(sep='.')[0]
            if file_name.endswith('-raw'):
                file_name = file_name.replace('-raw', '-epo')
            else:
                file_name = file_name+'-epo'
            file_name = file_name+'.fif'
            
            # Export epochs
            export_epoch(epochs, file_name, output_dir)

        if save_evoked_response and event_keys_for_segmentation:
            print('\nSaving evoked response...')
            
            # Set file name
            file_name = subject_name.split(sep='.')[0]
            if file_name.endswith('-raw'):
                file_name = file_name.replace('-raw', '-erp')
            else:
                file_name = file_name+'-erp'
                    
            if not by_event_type:
                file_name = file_name+'.fif'

                try:
                    evokeds.set_montage(montage=os.path.basename(montage).split('.')[0])
                except:
                    evokeds.set_montage(montage=montage)

                full_path = os.path.join(output_dir, 'erp', file_name)
                print(f"Writing {full_path}")
                evokeds.save(full_path, overwrite=True)
                print(f"Closing {full_path}")

            else:
                for i in np.arange(len(evokeds)):
                    full_path = os.path.join(output_dir, 'erp', f"{file_name}_{evokeds[i].comment}.fif")
                    try:
                        evokeds[i].set_montage(montage=os.path.basename(montage).split('.')[0])
                    except:
                        evokeds[i].set_montage(montage=montage)

                    print(f"Writing {full_path}")
                    evokeds[i].save(full_path, overwrite=True)
                    print(f"Closing {full_path}")
                print('[done]')
                        
            # Add evokeds in the report
            report.add_evokeds(evokeds, titles=None, replace=True)

        # Remove duplicate rows
        df_preprocessing_summary = df_preprocessing_summary.drop_duplicates()
        df_correction_summary = df_correction_summary.drop_duplicates()
        df_epochs_summary = df_epochs_summary.drop_duplicates()
                    
        # Save summary file
        file_name = subject_name.split(sep='.')[0]
        csv_file_path = os.path.join(output_dir, 'reports', f'{file_name}_summary_of_artifacts_detected_in_raw.csv')
        df_preprocessing_summary.to_csv(csv_file_path, index=False)    
        csv_file_path = os.path.join(output_dir, 'reports', f'{file_name}_summary_of_corrected_artifacts_in_raw.csv')
        df_correction_summary.to_csv(csv_file_path, index=False)    
        csv_file_path = os.path.join(output_dir, 'reports', f'{file_name}_summary_of_corrected_artifacts_in_epochs.csv')
        df_epochs_summary.to_csv(csv_file_path, index=False)
        
        # Save report
        print("Saving report")
        file_name = subject_name.split(sep='.')[0] + ".html"
        report.save(fname=os.path.join(output_dir, "reports", file_name), open_browser=False, overwrite=True)
        
        # Preprocessing end time
        print('\n---------------------------------------------')
        sim_time_end = timedelta(seconds=np.round(time.time() - sim_time_start))
        print('\nTotal processing time :', str(sim_time_end), 'in hh:mm:ss')
        print('=============================================\n')
        
        if save_log:
            sys.stdout.close()
    

# %% Sub Functions

# Libraries and dependencies
from apice.artifacts_rejection import BadElectrodes, Motion, Jump
from apice.artifacts_structure import DefineBTBC
from apice.artifacts_correction import TargetPCA, SegmentSphericalSplineInterpolation, ChannelsSphericalSplineInterpolation
from apice.filter import Filter
from apice.segmentation import Epochs


def detect_artifacts(raw):
    """
    Detects various artifacts in EEG data using specified detection algorithms.

    This function applies a series of artifact detection algorithms to the raw EEG data.
    Each algorithm is configured to use user-defined parameters. The algorithms look for bad electrodes,
    motion artifacts, jumps in signal, and defines the bad time segments for EEG correction.

    Args:
        raw (mne.io.Raw): The raw EEG data object that contains the EEG signal and metadata.

    Returns:
        None: This function does not return a value but modifies the raw data object in place
            to annotate the detected artifacts.
    """
    
    # Detects bad electrodes based on user-configured parameters
    BadElectrodes(raw, config=True)

    # Detects motion artifacts with a specific type set by user-defined parameters
    Motion(raw, type=1, config=True)
    
    # Detects jumps in the EEG signal using user-configured parameters
    Jump(raw, config=True)
    
    # Defines bad time segments in the EEG data for further correction, using user-configured parameters
    DefineBTBC(raw, config=True)


def correct_artifacts(raw, n_jobs):
    """
    Corrects artifacts in EEG data using a series of processing steps.

    This function applies multiple artifact correction techniques including Target PCA, 
    Spherical Spline Interpolation, and motion artifact correction. It also includes 
    filtering and baseline rescaling as part of the artifact correction process.

    Parameters:
    EEG : EEG object
        The EEG data structure to be processed for artifact correction.
    n_jobs : int
        Number of core used for the parallel computation. -1 to get all the available cores.

    Returns:
    None
    """
    
    from apice.parameters import Filters
    # Apply Target PCA per electrode and rescale baseline
    TargetPCA(raw, config=True)
    
    # Rescale the EEG data baseline to mean
    raw._data = mne.baseline.rescale(raw._data, raw.times, baseline=(None, None), mode='mean', copy=False)
    
    # Apply high-pass filtering to the EEG data
    Filter(raw,
           high_pass_freq=Filters.high_pass_freq, 
           low_pass_freq=[], 
           n_jobs=n_jobs) 
    
    # Define bad times and bad channels in the EEG data
    DefineBTBC(raw, config=True)

    # Apply Spherical Spline Interpolation for artifact correction
    SegmentSphericalSplineInterpolation(raw, n_jobs, config=True)
    
    # Rescale the EEG data baseline again post-interpolation
    raw._data = mne.baseline.rescale(raw._data, raw.times, baseline=(None, None), mode='mean', copy=False)
    
    # Re-apply high-pass filtering post-interpolation
    Filter(raw, high_pass_freq=Filters.high_pass_freq, low_pass_freq=[], n_jobs=n_jobs)
    
    # Re-define bad times and bad channels post-interpolation
    DefineBTBC(raw, config=True)

    # Apply Spherical Spline Interpolation for whole channels
    ChannelsSphericalSplineInterpolation(raw, n_jobs, config=True)
    
    # Motion artifact correction
    Motion(raw, type=2, keep_rejected_previous=True, config=True) 
    
    # Final definition of bad times and bad channels after all corrections
    DefineBTBC(raw, keep_rejected_previous=True, config=True)


def segment_data(raw, event_keys, tmin, tmax, n_jobs=-1, baseline_time_window=None, by_event_type=True):
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
    baseline_time_window : tuple, optional
        Time window for baseline correction (start, end) in seconds. Defaults to (None, None).
    by_event_type : bool, optional
        Flag to compute evoked responses by event type. Defaults to True.

    Returns:
    epochs : mne.Epochs object
        The processed epochs after segmentation, artifact correction, and bad epoch removal.
    evoked : mne.Evoked or list of mne.Evoked
        The evoked response(s), either as a single averaged response or separated by event type.
    """
    
    # Set default baseline correction window if not provided
    if baseline_time_window is None:
        baseline_time_window = (None, None)
    
    from apice.parameters import Filters_epochs

    # Apply filtering to the raw EEG data
    Filter(raw,
           high_pass_freq=Filters_epochs.high_pass_freq,
           low_pass_freq=Filters_epochs.low_pass_freq,
           n_jobs=n_jobs)
    
    # Segment the continuous data into epochs
    epochs = Epochs.segment_continuous_data(raw, event_keys=event_keys, tmin=tmin, tmax=tmax)
    
    # Define and correct artifacts in the segmented data
    DefineBTBC(epochs, segmented=True, config=True)

    # Apply spherical spline interpolation for artifact correction
    ChannelsSphericalSplineInterpolation(epochs, n_jobs, config=True)

    # Re-define artifacts after interpolation
    DefineBTBC(epochs, segmented=True, config=True)
    
    # Identify and define bad epochs
    Epochs.define_bad_epochs(epochs, config=True)

    # Remove bad epochs from the data 
    Epochs.remove_bad_epochs(epochs)

    if len(epochs.ch_names) > 30:
        # Re-reference the raw EEG data to the average and correct the baseline
        print('\n')
        epochs.set_eeg_reference(ref_channels='average')
    
    # Correct baseline
    epochs._data = mne.baseline.rescale(epochs._data, epochs.times, baseline=baseline_time_window, mode='mean', copy=False)

    # Print summary of artifacts and processing
    print(f"\nSummary: {print(epochs.artifacts.print_summary())}\n")

    # Compute the evoked responses
    print(
        f"\nGetting evoked responses...",
        f"-\n\t-- by event type: {by_event_type}"
        )
    if by_event_type:
        evokeds = epochs.average(by_event_type=by_event_type)
    else:
        evokeds = epochs.average()

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

