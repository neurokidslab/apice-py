# %% LIBRARIES

import numpy as np
import mne
import pandas as pd
from apice.artifacts_structure import Artifacts

# %% FUNCTIONS

def print_header(header, separator="="):
    """
    Print a header with separator lines of the same length.

    Args:
        header (str): The header text to be printed.
        separator (str, optional): The character used to create separator lines. Defaults to "-".
    """
    # Calculate the length of the header text
    header_length = len(header)

    # Create separator lines of the same size as the header text
    separator = separator * header_length

    # Print the separator line
    print(separator)

    # Print the header text
    print(header)

    # Print the separator line below the header
    print(separator + "\n")

# %% CLASSES

class Epochs:
    """
    A class for managing and processing EEG epoch data.

    This class includes methods for segmenting continuous EEG data into epochs, defining bad epochs based on various criteria, and removing bad epochs from the dataset.
    """

    @staticmethod
    def segment_continuous_data(raw, event_keys=None, tmin=-0.2, tmax=0.7):
        """
        Segments continuous EEG data into epochs based on specified events.

        Parameters:
        raw : Raw EEG object
            Continuous EEG data to be segmented.
        event_keys : list, optional
            Keys identifying the events around which to segment the data.
        tmin : float, optional
            Start time before the event in seconds (default: -0.2s).
        tmax : float, optional
            End time after the event in seconds (default: 0.7s).

        Returns:
        epochs : mne.Epochs object
            The segmented epochs.
        """
        
        # Print a header for the segmentation process
        print_header('SEGMENTING CONTINUOUS DATA', separator='=')

        # Extract event IDs from the raw data based on provided event keys
        event_id = []
        for i in event_keys:
            event_id.append(raw.event_ids[i])
        
        # Create an array to store events used for creating epochs
        events = np.zeros(0, dtype=int)
        
        # Populate the events array with events that match the event IDs
        for i in np.arange(np.shape(raw.events)[0]):
            if raw.events[i, 2] in event_id:
                events = np.append(events, raw.events[i, :])
        
        # Reshape the events array for compatibility with MNE's Epochs structure
        events = np.reshape(events, (int(np.size(events) / 3), 3))

        # Create epochs from the continuous raw data using the extracted events
        epochs = mne.Epochs(raw, events=events, event_id=event_id, tmin=tmin, tmax=tmax,
                            reject_by_annotation=False, preload=True, baseline=None)
        
        # Rename event ids
        new_event_id = {}
        keys = list(epochs.event_id.keys())
        for i in range(len(keys)):
            new_event_id[event_keys[i]] = epochs.event_id[keys[i]]
        epochs.event_id = new_event_id
        
        # Additional code to handle the rejection matrix and update artifacts in the epochs
        # Calculate left and right limits for the time window
        time_window_start = (np.abs(tmin) * raw.info['sfreq'])
        time_window_end = (np.abs(tmax) * raw.info['sfreq'])

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
        if not hasattr(epochs.artifacts, 'CCT'):
            epochs.artifacts.CCT = np.full(np.shape(Epochs), False)

        if not hasattr(epochs.artifacts, 'CCT'):
            epochs.artifacts.CCT = np.full(np.shape(Epochs), False)
    
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
        n_electrodes, n_samples, n_epochs = Raw.get_data_size(epochs)

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
    


