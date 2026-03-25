
# %% LIBRARIES
import os
from pathlib import Path
import numpy as np
import mne.io
import pandas as pd 

from apice.artifacts_structure import (Artifacts, annotations_to_rejection_matrix, calculate_event_onsets_and_durations, dataframe_to_rejection_matrix, rejection_matrix_to_data_frame, remove_artifacts_annotations)
from apice.utils import print_header, get_data_size
        

# %% FUNCTIONS

def is_valid_extension(filename, valid_extensions):
    """
    Check if a filename has a valid extension.

    Parameters:
    - filename (str): The filename to check.
    - valid_extensions (list): A list of valid file extensions.

    Returns:
    - bool: True if the filename has a valid extension, False otherwise.
    """
    return Path(filename).suffix.lstrip(".").lower() in valid_extensions


def get_files_to_process(input_dir, 
                         output_dir=None, 
                         data_selection_method: str ='all',
                         preprocessed_file_pattern='*-preproc.fif',
                         ):
    """
    Get a list of files to process based on the specified data selection method.

    Parameters:
    - input_dir (str): The input directory containing raw data files.
    - output_dir (str, optional): The output directory where preprocessed files will be saved.
    - data_selection_method (str, optional): The method to select files:
        'all' - All valid files in the input directory.
        'new' - Files not already processed in the input directory and output directory.
        'manual' - Manually input filenames.
    - preprocessed_file_pattern (str, optional): The pattern to identify preprocessed files in the output directory (default: '*-preproc.fif').

    Returns:
    - list: A list of file paths to process.
    """

    if data_selection_method in [1, 2, 3]:
        # Backward-compatible support for legacy integer options.
        data_selection_method = {1: "all", 2: "new", 3: "manual"}[data_selection_method]

    if data_selection_method not in ['all', 'new', 'manual']:
        raise ValueError("Invalid data selection method. Choose from 'all', 'new', or 'manual'.")
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir) if output_dir is not None else None

    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"input_dir does not exist or is not a directory: {input_dir}")
    
    # List of valid extensions
    valid_extensions = {
        "fif",
        "mat",
        "vhdr",
        "bdf",
        "cnt",
        "edf",
        "set",
        "egi",
        "mff",
        "nxe",
        "gdf",
        "data",
        "lay",
        "raw",
        }

    # Get all files in the input directory
    valid_files = sorted(
        [
            f
            for f in input_dir.iterdir()
            if f.is_file() and f.suffix.lstrip(".").lower() in valid_extensions
        ],
        key=lambda p: p.name.lower(),
    )

    if data_selection_method == 'all':
        # Method 'all': Include all valid files in the input directory
        files_to_process = valid_files

    if data_selection_method == 'new':
        # Method 'new': Include files not already processed in the output directory.
        if output_dir is None:
            raise ValueError("output_dir must be provided when data_selection_method='new'.")

        if not output_dir.exists():
            files_to_process = valid_files
        else:
            output_folder_files = {
                f.stem for f in output_dir.glob(preprocessed_file_pattern) if f.is_file()
            }
            files_to_process = []
            for file in valid_files:
                expected_output_stem = Path(
                    preprocessed_file_pattern.replace("*", file.stem)
                ).stem
                if expected_output_stem not in output_folder_files:
                    files_to_process.append(file)

    if data_selection_method == 'manual':
        # Method 'manual': Manually input filenames
        files_to_process = []
        while True:
            # input names space separated
            input_name = input("File names, SPACE separated (press Enter to finish): ")
            # If the user presses Enter without typing anything, break the loop
            if not input_name.strip():
                break
            for name in input_name.split():
                if is_valid_extension(name, valid_extensions):
                    if (input_dir / name).is_file():
                        files_to_process.append(input_dir / name)
                    else:
                        print(f"File {name} does not exist in the input directory. Skipping this file.")
                else:
                    print(
                        f"Invalid file extension for file {name}. Valid extensions are: {', '.join(valid_extensions)}. Skipping this file."
                    )

    return files_to_process

def get_bids_files_to_process(bids_root, 
                              session=None,
                              task=None,
                              run=None,
                              subject=None,
                              datatype='eeg',
                              suffix='eeg',
                              extension='.vhdr',
                              output_dir=None, 
                              data_selection_method: str ='all', 
                              preprocessed_file_pattern='*-preproc.fif'):
    """
    Get a list of BIDS files to process based on the specified data selection method.

    Parameters:
    - bids_root (str): The root directory containing BIDS data.
    - output_dir (str, optional): The output directory where preprocessed files will be saved.
    - data_selection_method (str, optional): The method to select files:
        'all' - All valid BIDS files in the input directory.
        'new' - BIDS files not already processed in the input directory and output directory.
        'manual' - Manually input BIDS file paths.
    - preprocessed_file_pattern (str, optional): The pattern to identify preprocessed files in the output directory (default: '*-preproc.fif').
    """

    if data_selection_method not in ['all', 'new']:
        raise ValueError("Invalid data selection method. Choose from 'all' or 'new'.")
    
    output_dir = Path(output_dir) if output_dir is not None else None

    from mne_bids import BIDSPath

    bids_path =  BIDSPath(subject=subject, 
                          session=session, 
                          task=task, 
                          run=run, 
                          datatype=datatype, 
                          root=bids_root, 
                          suffix=suffix, 
                          extension=extension,
                          )
    list_files = bids_path.match()

    if data_selection_method == 'all':
        # Method 'all': Include all valid files in the input directory
        files_to_process = list_files

    if data_selection_method == 'new':
        # Method 'new': Include files not already processed in the output directory.
        if output_dir is None:
            raise ValueError("output_dir must be provided when data_selection_method='new'.")

        if not output_dir.exists():
            files_to_process = list_files
        else:
            output_folder_files = {
                f.stem for f in output_dir.glob(preprocessed_file_pattern) if f.is_file()
            }
            files_to_process = []
            for file in list_files:
                expected_output_stem = Path(
                    preprocessed_file_pattern.replace("*", file.basename)
                ).stem
                if expected_output_stem not in output_folder_files:
                    files_to_process.append(file)


    return files_to_process


# %% CLASSES

from apice.artifacts_structure import annotate_bads

class Raw:

    @staticmethod
    def load(fname):
        # load the data
        raw = mne.io.read_raw(fname, preload=False, verbose=None)
        # Initialize artifacts structure
        raw.artifacts = Artifacts(raw)
        # convert artifacts annotations to rejection matrix
        annotations_to_rejection_matrix(raw)
        return raw

    @staticmethod
    def export(raw, file_name, output_dir, data_suffix='-preproc'):
        # rejection matrix to annotations
        annotate_bads(raw, channels=True, times=True, data=True, corrected=True)
        # save preprocessed raw
        full_path = output_dir / (file_name + data_suffix + '.fif')
        raw.save(full_path, overwrite=True)

    @staticmethod
    def events_from_annotations(raw, **kwargs):
        """
        Process annotations in a raw MNE object.

        This function extracts annotations, separates them based on descriptions ('artifact', 'corrected', 'badtime'),
        and sets the event annotations accordingly.

        Args:
            - raw (mne.io.Raw): The MNE raw object containing EEG data with annotations.
            - kwargs: Additional keyword arguments to be passed to mne.events_from_annotations().

        Returns:
            - raw (mne.io.Raw): The processed MNE raw object.
        """

        # Extract annotations and create event_ids
        annotations_df = raw.annotations.to_data_frame(time_format=None)

        # Remove artifact annotations from the raw object
        remove_artifacts_annotations(raw)

        # Create events and event_ids attributes
        events, event_ids = mne.events_from_annotations(raw, **kwargs)

        # Put all original annotations back
        annotations = mne.Annotations(
            onset=list(annotations_df["onset"]),
            duration=list(annotations_df["duration"]),
            description=list(annotations_df["description"]),
            ch_names=list(annotations_df["ch_names"]),
        )

        # Set raw with original annotations
        raw.set_annotations(annotations)

        return events, event_ids

    @staticmethod    
    def stim_channels_to_annotations(raw):
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
            
            durations = [Raw.get_stim_duration(raw)] * len(events) 
            
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
    
    @staticmethod
    def get_stim_duration(raw):

        # Load data from the specified stimulus channel
        stim_data = raw.copy().pick('stim').get_data()
        
        above_baseline = np.where(stim_data > 0.01, 1, 0)
        diff = np.diff(above_baseline, prepend=0)

        # Event onsets (where diff == 1) and offsets (where diff == -1)
        onsets = np.where(diff == 1)[1]
        offsets = np.where(diff == -1)[1]

        # Check for the case where the last event doesn't have an offset
        if len(onsets) > len(offsets):
            if len(offsets) == 0 or onsets[-1] > offsets[-1]:
                offsets = np.append(offsets, len(stim_data[0])) 

        # Calculate durations in seconds
        durations = (offsets - onsets) / raw.info['sfreq']

        return np.max(durations)
    

class Epochs:
    """
    A class for managing and processing EEG epoch data.

    This class includes methods for segmenting continuous EEG data into epochs, defining bad epochs based on various criteria, and removing bad epochs from the dataset.
    """
    
    @staticmethod
    def segment_continuous_data(raw, 
                                events,
                                event_id,
                                epoching_kwargs={}
                                ):
        """
        Segments continuous EEG data into epochs based on specified events.

        Parameters:
        raw : Raw EEG object
            Continuous EEG data to be segmented.
        epoching_kwargs : dict, optional
            Additional arguments to pass to the `mne.Epochs` constructor.
        
        Returns:
        epochs : mne.Epochs object
            The segmented epochs.
        """
        
        # Print a header for the segmentation process
        print_header('SEGMENTING CONTINUOUS DATA', separator='=')

        # Create epochs from the continuous raw data using the extracted events
        epochs = mne.Epochs(raw, 
                            events=events, 
                            event_id=event_id, 
                            **epoching_kwargs,
                            )
        
        # # Rename event ids
        # new_event_id = {}
        # keys = list(epochs.event_id.keys())
        # for i in range(len(keys)):
        #     new_event_id[event_keys[i]] = epochs.event_id[keys[i]]
        # epochs.event_id = new_event_id
        
        # Additional code to handle the rejection matrix and update artifacts in the epochs
        # Calculate left and right limits for the time window
        time_window_start = epochs.times[0]/1000
        time_window_end = epochs.times[-1]/1000
        # time_window_start = (np.abs(epoching_kwargs.get('tmin', 0)) * raw.info['sfreq'])
        # time_window_end = (np.abs(epoching_kwargs.get('tmax', 0.5)) * raw.info['sfreq'])

        # Extract the stimulus times from the events
        stimulus_times = events[:, 0]

        # Identify the events to be dropped based on the drop log of epochs
        stimulus_events_to_drop = []
        for ep in np.arange(len(epochs.drop_log)):
            if len(epochs.drop_log[ep]) > 0:
                stimulus_events_to_drop.append(ep)

        # Update stimulus times by removing the dropped events
        stimulus_times = np.delete(stimulus_times, stimulus_events_to_drop).astype(int)

        # Initialize artifact structures in the epochs
        n_epochs, _, _ = np.shape(epochs)
        epochs.artifacts = Artifacts(epochs)

        # Update the artifact structures with information from the raw data
        # if not hasattr(epochs.artifacts, 'CCT'):
        #     epochs.artifacts.CCT = np.full(np.shape(Epochs), False)

        for ep in np.arange(n_epochs):
            epoch_start_time = (stimulus_times[ep] - time_window_start).astype(int)
            epoch_end_time = (stimulus_times[ep] + time_window_end).astype(int)
            
            time_range = list(np.arange(epoch_start_time, epoch_end_time + 1).astype(int))
            
            epochs.artifacts.BCT[ep] = raw.artifacts.BCT[0, :, time_range].T
            epochs.artifacts.BT[ep] = raw.artifacts.BT[0, :, time_range].T
            epochs.artifacts.BC[ep] = raw.artifacts.BC[0]
            epochs.artifacts.CCT[ep] = raw.artifacts.CCT[0, :, time_range].T

        return epochs

    @staticmethod
    def define_bad_epochs(epochs, bad_data = 1.00, bad_time = 0.00, bad_channel = 0.30, corrected_data = 0.50,
                            max_iterations=1, tmin=[], tmax=[], keep_rejected_previous=False, log=False, config=False):
        """
        Identifies bad epochs in the segmented EEG data.

        Parameters:
        epochs : mne.Epochs object
            The epochs to analyze for bad data.
        bad_data : float, optional
            Threshold for marking an epoch as bad based on data quality (default: 1.00).
        bad_time : float, optional
            Threshold for marking an epoch as bad based on time quality (default: 0.00).
        bad_channel : float, optional
            Threshold for marking an epoch as bad based on channel quality (default: 0.30).
        corrected_data : float, optional
            Threshold for marking an epoch as bad based on corrected data quality (default: 0.50).
        max_iterations : int, optional
            Maximum number of iterations for defining bad epochs (default: 1).
        tmin : list, optional
            Start time range for considering data in epochs (default: []).
        tmax : list, optional
            End time range for considering data in epochs (default: []).
        keep_rejected_previous : bool, optional
            Flag to keep previously rejected epochs marked as bad (default: False).
        log : bool, optional
            Flag to log the thresholding process (default: False).
        config : bool, optional
            Flag to use a custom configuration for defining bad epochs (default: False).

        Returns:
        None
        """

        # Set up the parameters for identifying bad epochs, with an option to update from a config
        params = {
            'bad_data':bad_data, 
            'bad_time':bad_time, 
            'bad_channel':bad_channel,
            'corrected_data':corrected_data, 
            'max_iterations':max_iterations, 
            'tmin':tmin, 
            'tmax':tmax,
            'keep_rejected_previous':keep_rejected_previous,
            'log':log
        }
        
        if config:
            # Load custom configurations if specified
            from apice.parameters import BE_Definition
            for keys in list(params.keys()):
                if hasattr(BE_Definition, keys):
                    params[keys] = BE_Definition.__dict__.get(keys)

        print('\nIdentifying bad epochs...\n')

        # Assign threshold limits for different artifact types
        limit_BCT, limit_BT, limit_BC, limit_CCT = params['bad_data'], params['bad_time'], params['bad_channel'], params['corrected_data']
        limit_BCT_relative, limit_BT_relative, limit_BC_relative, limit_CCT_relative = [], [], [], []

        # Lists to store relative and absolute limits for comparison
        limit_relative = [limit_BCT_relative, limit_BT_relative, limit_BC_relative, limit_CCT_relative]
        limit_absolute = [limit_BCT, limit_BT, limit_BC, limit_CCT]

        # Gather size information about the epochs
        from apice.io import Raw
        n_electrodes, n_samples, n_epochs = get_data_size(epochs)

        # Keep track of previously rejected epochs if specified
        if keep_rejected_previous & hasattr(epochs.artifacts, 'BE'):
            initial_bad_epochs = epochs.artifacts.BE.copy()
        else:
            initial_bad_epochs = np.full((n_epochs, 1, 1), False)

        # Initialize or load various artifact matrices           
        artifact_attributes = {
                                'BEmanual': (n_epochs, 1, 1),
                                'BCT': (n_epochs, n_electrodes, n_samples),
                                'BT': (n_epochs, 1, n_electrodes),
                                'BC': (n_epochs, n_electrodes, 1),
                                'CCT': (n_epochs, n_electrodes, n_samples)
                            }
        for attr, shape in artifact_attributes.items():
            if not hasattr(epochs.artifacts, attr):
                setattr(epochs.artifacts, attr, np.full(shape, False))

        # Set default times to consider if not specified in params
        params['tmin'] = params['tmin'] or epochs.times[0]
        params['tmax'] = params['tmax'] or epochs.times[-1]

        # Determine the time indices to consider based on provided time range
        time_range_mask = (epochs.times >= params['tmin']) & (epochs.times <= params['tmax'])

        # Calculate the total number of samples within the specified time range
        n_samples = np.sum(time_range_mask)

        # Find bad epochs
        epoch_quality_scores = np.empty((n_epochs, 4))
        epoch_quality_scores[:] = np.nan
        epoch_quality_scores[:, 0] = np.sum(np.sum(epochs.artifacts.BCT[:, :, time_range_mask], axis=1), axis=1) / (n_samples * n_electrodes)
        epoch_quality_scores[:, 1] = np.squeeze(np.sum(epochs.artifacts.BT[:, :, time_range_mask], axis=2) / n_samples)
        epoch_quality_scores[:, 2] = np.squeeze(np.sum(epochs.artifacts.BC, axis=1) / n_electrodes)
        epoch_quality_scores[:, 3] = np.sum(np.sum(epochs.artifacts.CCT[:, :, time_range_mask], axis=1), axis=1) / (n_samples * n_electrodes)

        # Apply logarithmic transformation if logging is enabled
        if log:
            # Replace zero scores with a minimal value to avoid undefined log(0) during transformation
            # This minimal value is inversely proportional to the number of samples and electrodes
            epoch_quality_scores[epoch_quality_scores[:, 0] == 0, 0] = 1 / (n_samples * n_electrodes)
            epoch_quality_scores[epoch_quality_scores[:, 1] == 0, 1] = 1 / n_samples
            epoch_quality_scores[epoch_quality_scores[:, 2] == 0, 2] = 1 / n_electrodes
            epoch_quality_scores[epoch_quality_scores[:, 3] == 0, 3] = 1 / (n_samples * n_electrodes)
            
            # Replace zero scores with a minimal value to avoid undefined log(0) during transformation
            # This minimal value is inversely proportional to the number of samples and electrodes
            for i in np.arange(4):
                limit_absolute[i] = np.log(limit_absolute[i])

        # Initialize matrices to keep track of bad epochs
        BE = initial_bad_epochs[:] | epochs.artifacts.BEmanual[:]
        newly_detected_bad_epochs = np.full(np.shape(BE), False)

        # Thresholding loop to identify bad epochs
        end_thresholding = False
        iteration_count = 1

        # Loop to identify bad epochs based on quality thresholds
        while (not end_thresholding) & (iteration_count <= params['max_iterations']):
            # Initialize an array to hold the quality thresholds for each metric
            quality_thresholds = np.ones(4)

            # Calculate quality thresholds for each of the four metrics (bad data, time, channel, corrected data)
            for i in np.arange(4):
                # If relative limits are set, calculate thresholds based on the percentile approach
                if limit_relative[i]:
                    # Compute 75th and 25th percentiles for the current metric
                    P75 = np.percentile(epoch_quality_scores[~BE[:, 0, 0], i], 75, interpolation='midpoint')
                    P25 = np.percentile(epoch_quality_scores[~BE[:, 0, 0], i], 25, interpolation='midpoint')
                    # Set the quality threshold based on the IQR method
                    quality_thresholds[i] = P75 + limit_relative[i] * (P75 - P25)

                    # Check and apply the absolute limits if they are set
                    if limit_absolute[i] & np.size(limit_absolute[i]) == 2:
                        # Ensure the calculated threshold is within the specified absolute range
                        quality_thresholds[i] = max(min(quality_thresholds[i], limit_absolute[i][1]), limit_absolute[i][0])
                else:
                    # Use the absolute threshold if relative is not set
                    quality_thresholds[i] = limit_absolute[i]

            # Compare each epoch's quality scores against the thresholds to identify bad epochs
            R = epoch_quality_scores > np.tile(quality_thresholds, [n_epochs, 1])

            # Check if any new data was identified as bad in this iteration
            if np.all((np.any(R, axis=1) | BE) == BE):
                # If no new bad data is found, end the thresholding process
                end_thresholding = True

            # Update the record of newly detected bad epochs
            newly_detected_bad_epochs = np.squeeze(newly_detected_bad_epochs) | np.any(R, axis=1)
            # Update the overall record of bad epochs
            BE = np.squeeze(BE) | np.any(R, axis=1)

            # Increment the iteration count
            iteration_count += 1

        
        # Print details about the rejected epochs

        BE_new = BE & np.squeeze(~initial_bad_epochs) & np.squeeze(~epochs.artifacts.BEmanual)

        print(
                f"Rejected epochs by this algorithm: \t {np.sum(newly_detected_bad_epochs)}",
                f"out of {n_epochs} ({np.round(np.sum(newly_detected_bad_epochs) / n_epochs * 100, 2)} %)",
                f"{np.where(newly_detected_bad_epochs)[0]})"
            )
        print(
                f"--> BCT threshold {quality_thresholds[0]}",
                f"trials {np.sum(R[:, 0])} ({np.round(np.sum(R[:, 0]) / n_epochs * 100, 2)}) : ",
                f"{np.where(R[:, 0])[0]}"
            )
        print(
                f"--> BT threshold {quality_thresholds[1]} ",
                f"trials {np.sum(R[:, 1])} ({np.round(np.sum(R[:, 1]) / n_epochs * 100, 2)}) : ",
                f"{np.where(R[:, 1])[0]}"
            )
        print(
                f"--> BC threshold {quality_thresholds[2]} ",
                f"trials {np.sum(R[:, 2])} ({np.round(np.sum(R[:, 2]) / n_epochs * 100, 2)}) : ",
                f"{np.where(R[:, 2])[0]}"
            )
        print(
                f"--> CCT threshold {quality_thresholds[3]} ",
                f"trials {np.sum(R[:, 3])} ({np.round(np.sum(R[:, 3]) / n_epochs * 100, 2)}) : "
                f"{np.where(R[:, 3])[0]}"
            )
        print('\n')
        print(
                f"Total rejected epochs: \t {np.sum(BE)} ",
                f"out of {n_epochs} ({np.round(np.sum(BE) / n_epochs * 100, 2)} %)",
                f"{np.where(BE)[0]}"
            )
        print(
                f"New rejected epochs: \t {np.sum(BE_new)} ",
                f"out of {n_epochs} ({np.round(np.sum(BE_new) / n_epochs * 100, 2)} %) ",
                f"{np.where(BE_new)[0]}"
            )
        print('\n')

        # Update the rejection matrix in the epochs object
        epochs.artifacts.BE = np.reshape(BE, (n_epochs, 1, 1))
        epochs.artifacts.print_summary()

    @staticmethod
    def remove_bad_epochs(epochs):
        """
        Removes bad epochs from the EEG data.

        Parameters:
        epochs : mne.Epochs object
            The epochs from which bad epochs will be removed.

        Returns:
        None
        """

        # Identify the bad epochs from the artifacts attribute
        bad_epochs = epochs.artifacts.BE[:, 0, 0]
        
        # Identify the good epochs as the inverse of bad epochs
        good_epochs = ~epochs.artifacts.BE[:, 0, 0]

        # Ensure that the epochs will only be dropped once
        # Get the number of epochs in the data and in the BCT artifact matrix
        n_epochs, _, _ = np.shape(epochs)
        n_epochs_bct, _, _ = np.shape(epochs.artifacts.BCT)

        # Check if the number of epochs in the data matches the number in the BCT artifact matrix
        if n_epochs == n_epochs_bct:
            # Drop the bad epochs from the epochs dat
            epochs.drop(bad_epochs, reason='bad epoch')

            # Update the artifacts matrices to reflect the removal of bad epochs
            # Check and update each artifact attribute if it exists
            if hasattr(epochs, 'artifacts'):
                if hasattr(epochs.artifacts, 'BE'):
                    epochs.artifacts.BE = epochs.artifacts.BE[good_epochs, :, :]
                if hasattr(epochs.artifacts, 'BEmanual'):
                    epochs.artifacts.BEmanual = epochs.artifacts.BEmanual[good_epochs, :, :]
                if hasattr(epochs.artifacts, 'BCT'):
                    epochs.artifacts.BCT = epochs.artifacts.BCT[good_epochs, :, :]
                if hasattr(epochs.artifacts, 'BT'):
                    epochs.artifacts.BT = epochs.artifacts.BT[good_epochs, :, :]
                if hasattr(epochs.artifacts, 'BC'):
                    epochs.artifacts.BC = epochs.artifacts.BC[good_epochs, :, :]
                if hasattr(epochs.artifacts, 'CCT'):
                    epochs.artifacts.CCT = epochs.artifacts.CCT[good_epochs, :, :]

            # Print a summary of the current state of the artifacts
            epochs.artifacts.print_summary()

        else:
            # Notify if bad epochs have already been removed
            print('\nBad epochs already dropped.')
    

    @staticmethod
    def export(epochs, file_name, output_dir, data_suffix='-epo'):

        # # Remove the artifact annotation because it contains the raw annotations
        # annotations = mne.Annotations(ch_names=[], description=[], duration=[], onset=[])
        # epochs.set_annotations(annotations)

        # get the artifacts ina dataframe
        artifacts_df = rejection_matrix_to_data_frame(epochs)
        delattr(epochs, 'artifacts')
                
        # Save the epochs and the artifacts information in a csv file in the output directory
        print('\nExporting epochs...')
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        epochs_fullpath = output_dir / (file_name + data_suffix + '.fif')
        epochs.save(epochs_fullpath, overwrite=True)
        print()(f"Epochs saved at {epochs_fullpath}.")
        
        art_fullpath = output_dir / (file_name + data_suffix + '-artifacts.csv')
        artifacts_df.to_csv(art_fullpath, index=False)
        print(f"\nEpochs artifacts information saved at {art_fullpath}.")
        
       
    @staticmethod
    def load(fname):
        # load the epochs
        epochs = mne.read_epochs(fname)

        # get the path to the parent folder
        folder_path = Path(fname).parent
        art_file = folder_path / (Path(fname).stem + '-artifacts.csv')

        # read the artifacts information from the csv file and add it to the epochs object
        if art_file.is_file():
            artifacts_df = pd.read_csv(art_file)
            epochs = dataframe_to_rejection_matrix(epochs, artifacts_df)
        else:
            print(f"No artifacts file found at {art_file}. Returning epochs without artifacts information.")

        return epochs

