# %% LIBRARIES

# Import necessary modules
import mne  
import numpy as np 
from prettytable import PrettyTable 
import matplotlib.pyplot as plt 

# Import specific modules from your project's modules
from apice.parameters import *  
import pandas as pd  
import apice.io  


# %% FUNCTIONS

def remove_bad_data(raw, bad_data='none', artifact_type='all', silent=False):
    """
    Replaces or retains bad data in raw EEG data based on the artifacts matrix.

    This function allows for different strategies to handle bad data within EEG records, such as
    replacing bad data with zeros, NaNs, or the mean value across epochs. The user can specify
    the type of artifacts to consider and whether to output a warning message about the percentage
    of bad data.

    Parameters:
    - raw (Raw object): An object containing the EEG data.
    - bad_data (str): Strategy for handling bad data:
                        'none' (default) - bad data is retained,
                        'replace by zero' - bad data is replaced by 0,
                        'replace by nan' - bad data is replaced by NaNs,
                        'replace by mean' - bad data is replaced by the mean over all epochs.
    - artifact_type (str): Type of artifact to remove. Defaults to 'all'.
    - silent (bool): If False, prints a warning message about the percentage of bad data.

    Returns:
    - ndarray: The EEG data array with bad data handled according to the specified strategy.
    """
    
    # Retrieve the data size for dimensions setup
    n_electrodes, n_samples, n_epochs = apice.io.Raw.get_data_size(raw)

    # Initialize a boolean array for indexing data to remove
    data_to_remove = np.full((n_epochs, n_electrodes, n_samples), False)
    
    # Mark bad channels, times, and conditions based on artifact type specified
    if hasattr(raw.artifacts, 'BCT') and artifact_type in ['all', 'BCT']:
        data_to_remove[raw.artifacts.BCT] = True
    if hasattr(raw.artifacts, 'BT') and artifact_type in ['all', 'BTBC', 'BT']:
        BT = np.tile(raw.artifacts.BT, (1, n_electrodes, 1))
        data_to_remove[BT] = True
    if hasattr(raw.artifacts, 'BC') and artifact_type in ['all', 'BTBC', 'BC']:
        BC = np.tile(raw.artifacts.BC, (1, 1, n_samples))
        data_to_remove[BC] = True

    # Calculate the number of bad data points
    n_bad_data = np.sum(data_to_remove)
    
    # Reshape the EEG data for manipulation
    raw_data = np.reshape(raw._data.copy(), (n_epochs, n_electrodes, n_samples))
    
    # Process bad data if present
    if n_bad_data > 0:
        
        # Display warning message if silent mode is off
        if not silent:
            total_data_points = np.size(data_to_remove)
            percentage_bad_data = np.round(n_bad_data / total_data_points * 100, 2)
            print(
                f'\nPercentage of bad data from overall data: {n_bad_data} samples out of '
                f'{total_data_points} ({percentage_bad_data}%)'
                )
            
            messages = {
                'none': '--> Bad data will be retained',
                'replace by nan': '--> Bad data will be replaced by NaNs',
                'replace by zero': '--> Bad data will be replaced by zeros',
                'replace by mean': '--> Bad data will be replaced by the mean over all epochs'
            }
            
            print(messages.get(bad_data, '--> Unknown bad data handling strategy'))
            
        # Execute the replacement strategy
        if bad_data == 'replace by nan':
            raw_data[data_to_remove] = np.nan
        if bad_data == 'replace by zero':
            raw_data[data_to_remove] = 0
        if bad_data == 'replace by mean':
            # Calculate the mean, adjusting for the standard deviation
            mean_values = np.nanmean(raw_data, axis=0)
            std_deviation_adjustment = np.nanstd(mean_values) / np.nanstd(raw_data)
            mean_values = mean_values * std_deviation_adjustment
            mean_values = np.tile(mean_values, (n_epochs, 1, 1))
            
            # remove bad data
            raw_data[data_to_remove] = mean_values[data_to_remove]
        
        # Reshape raw data to original dimensions
        raw_data = np.reshape(raw_data, np.shape(raw._data))
        
    return raw_data


def set_reference(raw, bad_data='none', save_reference=False):
    """
    Sets the reference for the EEG data.

    This function defines the reference for the EEG data. The reference can be a specific electrode or an
    average reference. If bad data is specified, it will handle it accordingly before setting the reference.
    Optionally, the reference can be saved for later use.

    Parameters:
    - raw (mne.io.Raw or similar object): The EEG data object.
    - bad_data (str): Strategy for handling bad data prior to setting the reference. Options are:
                        'none' : the bad data will be retained
                        'replace by zero': the bad data will be replaced by 0
                        'replace by nan': the bad data will be replaced by 'NaNs
                        'replace by mean': the bad data will be replaced by the mean over all epochs
                        This parameter can be extended to handle other methods of dealing with bad data.
    - save_reference (bool): If True, the reference used will be saved in the EEG object for future reference.
                            This is useful for consistency in post-processing.

    Returns:
    - None: The function modifies the EEG data object in place and does not return anything.
    """

    # Remove bad data
    good_data = remove_bad_data(raw, bad_data=bad_data)

    # Reference to the mean
    n_electrodes, n_samples, n_epochs = apice.io.Raw.get_data_size(raw)

    # Calculate the average reference for the EEG data. The check is on the shape of the good_data variable.
    if n_epochs == 1: 
        # For a single epoch, compute the mean signal across all electrodes (axis=0).
        reference = np.nanmean(np.squeeze(good_data), axis=0) # Squeezing to be sure that the array don't have the epoch dimension
    else:
        # For multiple epochs, compute the mean signal across each epoch (axis=1).
        reference = np.nanmean(good_data, axis=1)

    # Check for NaN values within the 'good_data' array.
    if np.any(np.isnan(good_data)):
        reference[np.isnan(reference)] = 0

    # Apply mean reference to the EEG data
    if n_epochs == 1: 
        # If there is only one epoch, subtract the reference from each electrode's data.
        raw_data = raw._data.copy() - np.tile(reference, (n_electrodes, 1))
    else:
        # If there are multiple epochs, subtract the reference from each electrode's data in all epochs.
        raw_data = raw._data.copy() - np.tile(reference, (n_epochs, n_electrodes, 1))
    
    # Store the calculated reference in the raw data object for future reference.
    raw.mean_reference = reference.copy()
    
    # Update the raw data object with the re-referenced data.
    raw._data = raw_data.copy()

    # If the flag 'save_reference' is True, append the reference to the raw data.
    if save_reference:
        # Append the reference as a new channel to the raw data.
        raw._data = np.r_[raw, [reference]]
        
        # Check if the 'artifacts' attribute exists in the 'raw' object.
        if hasattr(raw, 'artifacts'):
            if hasattr(raw.artifacts, 'BCT'):
                for ep in np.arange(n_epochs):
                    raw.artifacts.BCT[ep] = np.r_[raw.artifacts.BCT[ep], np.full((1, n_samples), False)]
            if hasattr(raw.artifacts, 'CCT'):
                for ep in np.arange(n_epochs):
                    raw.artifacts.CCT[ep] = np.r_[raw.artifacts.CCT[ep], np.full((1, n_samples), False)]
            if hasattr(raw.artifacts, 'BC'):
                for ep in np.arange(n_epochs):
                    raw.artifacts.BC[ep] = np.r_[raw.artifacts.BC[ep], np.full((1, 1), False)]

    return


def compute_z_score(raw):
    """
    Computes the z-score normalization of EEG data.

    Artifacts in the data are marked as NaNs. The z-score is computed using the mean and
    standard deviation of the non-artifact data. This normalization can be used for
    statistical analysis and comparison between EEG signals.

    Parameters:
    - raw (Raw object): An object containing the EEG data and artifacts information.

    Returns:
    - raw (np.ndarray): The EEG data after z-score normalization.
    - mu (np.ndarray): The computed mean of the EEG data, used for z-score normalization.
    - sd (np.ndarray): The computed standard deviation of the EEG data, used for z-score normalization.
    """
    
    import warnings
    warnings.filterwarnings("ignore")

    # Retrieve the dimensions of the data.
    n_electrodes, n_samples, n_epochs = apice.io.Raw.get_data_size(raw)

    # Copy EEG data to prevent modification of the original.<
    raw_data = raw._data.copy()

    # If 'BCT' (bad channel times) artifacts are marked in the data, replace their values with NaN.
    if not hasattr(raw.artifacts, 'BCT'):
        raw_data = np.reshape(raw_data, (n_epochs, n_electrodes, n_samples))
        raw_data[raw.artifacts.BCT] = np.nan
        idx = np.tile(raw.artifacts.BT, (1, n_electrodes, 1))
        raw_data[idx] = np.nan

    # Reshape the data for the computation of z-score.
    raw_data = np.reshape(raw_data, (n_electrodes, n_samples * n_epochs))

    # Compute the mean (mu) and standard deviation (sd) across the second axis, ignoring NaNs.
    mu = np.nanmean(raw_data, axis=1)
    sd = np.nanstd(raw_data, axis=1)
    
    # Replace NaNs in mu and sd with 0 and 1, respectively, to avoid division by zero or NaNs.
    mu[np.isnan(mu)] = 0
    sd[np.isnan(sd)] = 1

    # Prepare the mean and sd for element-wise operations with raw_data.
    sd2 = np.reshape(sd, (n_electrodes, 1))
    sd2 = np.tile(sd2, (1, n_samples))
    mu2 = np.reshape(mu, (n_electrodes, 1))
    mu2 = np.tile(mu2, (1, n_samples))

    # Compute the z-score normalized EEG data.
    raw_data = np.divide(np.subtract(raw_data, mu2), sd2)
    
    return raw_data, mu, sd


def calculate_event_onsets_and_durations(event_array, time_vector, sampling_frequency):
    """
    Calculates the onsets and durations of events in a binary array.

    This function analyzes a binary array where occurrences of an event are marked with 1s.
    It uses the corresponding time vector to determine the precise onset times of these events
    and computes their durations based on the provided sampling frequency.

    Parameters:
    - event_array (np.ndarray): A binary array with 1s indicating the occurrence of events.
    - time_vector (np.ndarray): The timestamps for each sample in the event_array.
    - sampling_frequency (float): The sampling frequency of the data (in Hz).

    Returns:
    - event_onsets (np.ndarray): An array of onset times for the events.
    - event_durations (np.ndarray): An array of durations for the events (in seconds).
    """
    
    # Append zeros at both ends to identify changes at the first and last positions.
    padded_event_array = np.concatenate(([0], event_array, [0]))
    
    # Detect the indices where changes occur (from 0 to 1 or 1 to 0).
    change_indices = np.flatnonzero(padded_event_array[1:] != padded_event_array[:-1])
    
    # Compute the number of samples between consecutive changes; every second count corresponds to an event.
    sample_counts = change_indices[1:] - change_indices[:-1]
    
    # Select indices that correspond to the start of events.
    onset_indices = change_indices[::2]
    
    # Event durations are every second element in the sample counts.
    event_sample_counts = sample_counts[::2]
    
    # Convert the onset indices to times by indexing the time vector.
    event_onsets = time_vector[onset_indices]
    
    # Calculate the duration in seconds by dividing the sample counts by the sampling frequency.
    event_durations = event_sample_counts / sampling_frequency
    
    return event_onsets, event_durations


def annotate_bads(raw, channels=True, times=True, data=True, corrected=True):
    """
    Annotates bad channels, times, and artifacts in an EEG raw data structure.
    
    Parameters:
    - raw: The raw EEG data structure (usually an instance of mne.io.Raw or similar).
    - channels (bool): If True, annotate bad channels based on the 'BC' (bad channels) artifact flag.
    - times (bool): If True, annotate bad times based on the 'BT' (bad times) artifact flag.
    - data (bool): If True, annotate bad data based on the 'BCT' (bad channel times) artifact flag.
    - corrected (bool): If True, annotate data that has been corrected based on the 'CCT' (corrected channel times) artifact flag.
    
    Modifies the raw data structure by adding annotations for any identified bad data.
    """
    
    # Extract raw data dimensions
    n_electrodes, n_samples, n_epochs = apice.io.Raw.get_data_size(raw)
    
    # Initialize annotations
    annotations = mne.Annotations(onset=[], duration=[], description=[])

    # Annotate bad channels if specified and the artifact attribute exists
    if channels and hasattr(raw.artifacts, 'BC') and np.sum(raw.artifacts.BC):
        # Loop through each epoch to find bad channels
        for ep in np.arange(n_epochs):
            # Identify which channels are bad for this epoch
            bad_channels = np.where(raw.artifacts.BC[ep, :, :])[0].astype(int)
            
            # Add the names of the bad channels to the 'info' attribute
            for i in bad_channels:
                raw.info['bads'].append(raw.ch_names[i])
                
            # Remove duplicates from the bad channels list
            bad_channels = np.asarray(raw.info['bads'])
            raw.info['bads'] = list(np.unique(bad_channels))
    
    # Annotate bad times if specified and the artifact attribute exists
    if times and hasattr(raw.artifacts, 'BT') and np.sum(raw.artifacts.BT):
        # Iterate through each epoch to annotate bad times
        for ep in np.arange(n_epochs):
            # Extract the binary time series indicating bad times for the current epoch
            BT = np.asarray(raw.artifacts.BT[ep, 0, :], dtype=int)

            # Calculate the onset times and durations of bad segments from the binary time series
            onset, duration = calculate_event_onsets_and_durations(BT, raw.times, raw.info['sfreq'])

            # Define a description label for these annotations
            description = 'badtime'

            # Append the annotations for bad times to the raw object
            annotations.append(onset=onset, duration=duration, description=description)

            # Update the raw object with these new annotations
            raw.set_annotations(annotations)


    # Annotate bad data if specified and the artifact attribute exists
    if data and hasattr(raw.artifacts, 'BCT') and np.sum(raw.artifacts.BCT):
        # Iterate through each epoch in the EEG data
        for ep in np.arange(n_epochs):
            # Iterate through each electrode
            for el in np.arange(n_electrodes):
                # Retrieve the binary series for bad channel times (BCT) for the current electrode and epoch
                BCT = np.asarray(raw.artifacts.BCT[ep, el, :], dtype=int)

                # Calculate the onset times and durations for bad data segments
                onset, duration = calculate_event_onsets_and_durations(BCT, raw.times, raw.info['sfreq'])

                # Create a description label for these annotations and associate them with the current channel
                description = ['artifact'] * len(onset)
                ch_names = [[raw.ch_names[el]]] * len(onset)

                # Append the annotations for bad data to the raw object
                annotations.append(onset=onset, duration=duration, description=description, ch_names=ch_names)

            # After annotating all electrodes for the current epoch, update the raw object with these new annotations
            raw.set_annotations(annotations)


    # Annotate corrected artifacts if specified and the artifact attribute exists
    if corrected and hasattr(raw.artifacts, 'CCT') and np.sum(raw.artifacts.CCT):
        # Iterate through each epoch to process corrected artifacts
        for ep in np.arange(n_epochs):
            # Iterate through each electrode within the epoch
            for el in np.arange(n_electrodes):
                # Extract the binary series indicating corrected artifacts for the current electrode in the epoch
                CCT = np.asarray(raw.artifacts.CCT[ep, el, :], dtype=int)

                # Calculate the onset times and durations of corrected segments from the binary series
                onset, duration = calculate_event_onsets_and_durations(CCT, raw.times, raw.info['sfreq'])

                # Define a description label for these annotations and assign the corresponding channel names
                description = ['corrected'] * len(onset)
                ch_names = [[raw.ch_names[el]]] * len(onset)

                # Append the annotations for corrected artifacts to the raw object
                annotations.append(onset=onset, duration=duration, description=description, ch_names=ch_names)

            # Update the raw object with these new annotations for each epoch
            raw.set_annotations(annotations)


def find_nearest_element_and_index(array, value):
    """
    Find the nearest element in an array to a given value and return both the element and its index.

    Parameters:
    - array: A one-dimensional numpy array or list where the nearest value will be searched.
    - value: The value to which the nearest element in the array will be found.

    Returns:
    - nearest_element: The element in the array that is closest to the given value.
    - index: The index of the nearest element in the array.

    Example:
    >>> array = np.array([0, 3, 6, 9])
    >>> value = 4
    >>> find_nearest_element_and_index(array, value)
    (3, 1)
    """
    
    # Ensure the input is a numpy array
    array = np.asarray(array)
    
    # Compute the absolute differences and find the index of the smallest difference
    idx = (np.abs(array - value)).argmin()
    
    # Return the nearest element and its index
    return array[idx], idx


def extract_annotations(raw) -> pd.DataFrame:
    """
    Extracts annotations from an MNE-Python EEG object and returns them as a pandas DataFrame.
    
    Parameters:
    - raw: Raw object
        The EEG object from MNE-Python containing annotations.
    
    Returns:
    - DataFrame
        A pandas DataFrame with the annotations data, including channels, descriptions, onsets, and durations.
        
    Notes:
    - The function assumes that the annotations in the EEG object are structured with ch_names, description, onset, and duration attributes.
    - The function will fail if the EEG object does not contain annotations or if the annotations do not have the expected attributes.
    """
    
    # Check if the EEG object has the attribute 'annotations'
    if hasattr(raw, 'annotations') and raw.annotations:
        # Create a DataFrame to hold the annotations
        annotations = pd.DataFrame(data=[], columns=['Channel', 'Description', 'Onset', 'Duration'])
        
        # Assign each column in the DataFrame by extracting corresponding data from the annotations
        annotations['Channel'] = raw.annotations.ch_names if hasattr(raw.annotations, 'ch_names') else ['N/A'] * len(raw.annotations.onset)
        annotations['Description'] = raw.annotations.description
        annotations['Onset'] = raw.annotations.onset
        annotations['Duration'] = raw.annotations.duration
        
        return annotations
    else:
        # If there are no annotations, return an empty DataFrame with the same structure
        return pd.DataFrame(columns=['Channel', 'Description', 'Onset', 'Duration'])

def annotations_to_rejection_matrix(raw) -> None:
    """
    Converts annotations in an EEG raw data structure to a rejection matrix format.

    Parameters:
    - raw: BaseRaw
        The raw EEG data structure with annotations that need to be converted.

    Notes:
    - This function modifies the raw object in place, adding artifacts rejection information
        for bad channels, bad times, bad data, and corrected data as specified by annotations.
    - The annotations are expected to be in a specific format with a description field indicating
        the type of artifact (bad channel, bad time, artifact, corrected).
    """

    print("Converting annotations to artifacts matrix")
    
    # Extract annotations using the provided helper function
    annotations = extract_annotations(raw)

    # Get time vector and channel list from the raw data structure
    t = raw.times
    ch_names = np.asarray(raw.ch_names)

    # Get data size information from the custom Raw object
    n_electrodes, n_samples, n_epochs = apice.io.Raw.get_data_size(raw)

    # Create a rejection matrix for bad channels (BC)
    # Get indices of bad channels and ensure they are integers
    bad_channel_indices = np.array([np.where(ch_names == el)[0] for el in raw.info['bads']], dtype=int).flatten()

    # Apply the bad channel mask efficiently
    raw.artifacts.BC[:, bad_channel_indices, :] = True

    # Create a rejection matrix for bad times (BT)
    bad_time = annotations[annotations['Description'] == 'badtime'].reset_index(drop=True)

    # Vectorize search for nearest indices
    onset_indices = np.searchsorted(t, bad_time['Onset'])
    end_indices = np.searchsorted(t, bad_time['Onset'] + bad_time['Duration'])

    # Efficiently apply the artifacts mask
    for start, end in zip(onset_indices, end_indices):
        raw.artifacts.BT[:, :, start:end] = True

    # Create a rejection matrix for bad data (BCT)
    bad_data = annotations[annotations['Description'] == 'artifact'].reset_index(drop=True)

    # Vectorized search for nearest indices
    onset_indices = np.searchsorted(t, bad_data['Onset'])
    end_indices = np.searchsorted(t, bad_data['Onset'] + bad_data['Duration'])

    # Precompute channel indices
    channel_indices = np.array([np.where(ch_names == ch)[0][0] for ch in bad_data['Channel']])

    # Efficiently apply the artifacts mask
    for ep in range(n_epochs):
        for el, start, end in zip(channel_indices, onset_indices, end_indices):
            raw.artifacts.BCT[ep, el, start:end] = True  

    # Create a rejection matrix for corrected data (CCT)
    corrected_data = annotations[annotations['Description'] == 'corrected'].reset_index(drop=True)

    # Initialize CCT matrix if it doesn't exist
    if not hasattr(raw.artifacts, 'CCT'):
        raw.artifacts.CCT = np.full((n_epochs, n_electrodes, n_samples), False)

    # Vectorized search for nearest indices
    onset_indices = np.searchsorted(t, corrected_data['Onset'])
    end_indices = np.searchsorted(t, corrected_data['Onset'] + corrected_data['Duration'])

    # Precompute channel indices
    channel_indices = np.array([np.where(ch_names == ch)[0][0] for ch in corrected_data['Channel']])

    # Efficiently apply the corrected artifacts mask
    for ep in range(n_epochs):
        for el, start, end in zip(channel_indices, onset_indices, end_indices):
            raw.artifacts.CCT[ep, el, start:end] = True 

# %% CLASSES

class Artifacts:
    """
    Class representing the artifacts in raw EEG data.
    
    This class holds and manages an artifact rejection matrix for an EEG dataset,
    providing facilities to access and update information about various types of artifacts.
    
    Attributes:
    - DETECTION_ALGORITHMS: List of string names representing various artifact detection algorithms.
    - POSTDETECTION_ALGORITHMS: List of string names for algorithms used after detection.
    
    The artifact rejection matrix contains several fields that denote different aspects of data quality:
    - BCT: Bad Channel Time - samples that are bad for specific channels over time.
    - BC: Bad Channel - channels that are bad throughout the recording.
    - BCmanual: Bad Channel (manual) - manually specified bad channels.
    - BT: Bad Time - time segments that are bad across all channels.
    - BE: Bad Epoch - entire epochs that are bad.
    - BS: Bad Segment - bad segments of data.
    - CCT: Corrected Channel Time - samples that have been corrected over time.
    """

    DETECTION_ALGORITHMS = [
        'Power', 'ChannelCorr', 'TimeVariance', 'TimeVariance_AverageReferenced',
        'Amplitude', 'AmplitudeAbsoluteThresh', 'AmplitudeRelativeThresh',
        'AmplitudeRelativeThresh_AverageReferenced', 'RunningAverage',
        'RunningAverage_AverageReferenced', 'FastChange',
        'FastChange_AverageReferenced', 'Derivative', 'AmplitudeVariance'
    ]
    POSTDETECTION_ALGORITHMS = [
        'ChannelPerSample', 'SamplePerChannel', 'ShortBadSegments', 'ShortGoodSegments', 'Mask'
    ]

    def __init__(self, raw, *required_artifacts):
        """
        Initializes the Artifacts object with artifact rejection matrices based on the EEG data.
        
        Parameters:
        - raw: An object containing the EEG data.
        - required_artifacts: Variable length argument list representing the artifacts
                                that are required for the rejection algorithm.
        """

        # Get data size information from the EEG object.
        n_electrodes, n_samples, n_epochs = apice.io.Raw.get_data_size(raw)
        
        # Initialize all possible types of artifact rejection matrices.
        artifacts_types = {
            'BCT': np.full((n_epochs, n_electrodes, n_samples), False),  # Bad Channel Time
            'BC': np.full((n_epochs, n_electrodes, 1), False),  # Bad Channel
            'BCmanual': [],  # Bad Channel (manual input)
            'BT': np.full((n_epochs, 1, n_samples), False),  # Bad Time
            'BE': np.full((n_epochs, 1, 1), False),  # Bad Epoch
            'BS': np.full(1, False),  # Bad Segment
            'CCT': np.full((n_epochs, n_electrodes, n_samples), False)  # Corrected Channel Time
        }

        # If the EEG object does not already contain an artifacts attribute, set it up.
        if not hasattr(raw, 'artifacts'):
            print('\nSetting-up artifacts rejection matrix. . .')
            self.algorithm = {
                'params': [],
                'step_name': [],
                'rejection_step': []
            }
            # Assign the initialized matrices to the attributes.
            for artifact_type in artifacts_types.keys():
                setattr(self, artifact_type, artifacts_types[artifact_type])
                
        else:
            # If raw already has an artifacts attribute, update the current object's attributes.
            for attribute in raw.artifacts.__dict__.keys():
                setattr(self, attribute, getattr(raw.artifacts, attribute))
            for artifact_type in artifacts_types.keys():
                if not hasattr(self, artifact_type):
                    setattr(self, artifact_type, artifacts_types[artifact_type])

    def print_summary(self):
        """
        Prints a summary of the bad data in the rejection matrices as a percentage of the total data.
        
        Returns:
        - summary: PrettyTable object representing the percentage of bad data for each artifact type.
        """
        
        # Initialize table
        summary = PrettyTable()

        # Define readable names for the artifact types to be used in the summary.
        artifact_types = {
            'BCT': 'Bad Channel Time',
            'BC': 'Bad Channels',
            'BT': 'Bad Times',
            'BE': 'Bad Epochs',
            'CCT': 'Corrected Channel Time'
        }
        
        # Calculate and add data to the summary table.
        for artifact_key, artifact_name in artifact_types.items():
            if hasattr(self, artifact_key):
                total_elements = np.size(getattr(self, artifact_key))
                total_true_elements = np.sum(getattr(self, artifact_key))
                percentage = np.round(total_true_elements / total_elements * 100, 2)
                summary.add_column(artifact_name, [f"{percentage}%"])

        return summary


class DefineBTBC:
    """
    Class for defining Bad Times (BT) and Bad Channels (BC) in EEG data.
    
    This class provides methods to identify and categorize portions of the EEG data
    that should be considered 'bad' either in terms of specific time segments (BT)
    or channels (BC). It supports configuring thresholds, handling masks, and options
    for plotting and segmenting data.
    
    Attributes:
    -----------
        - params (dict): Dictionary containing the parameters.
    
    Args:
    -----
        - raw (Raw object): The EEG dataset to be processed.
        - thresh_bad_channels (list, optional): Threshold values to classify bad channels.
        - thresh_bad_times (list, optional): Threshold values to classify bad times.
        - min_good_time (float, optional): Minimum duration to consider a time segment as good.
        - min_bad_time (float, optional): Minimum duration to consider a time segment as bad.
        - mask_time (float, optional): Time to mask before and after a bad segment.
        - keep_rejected_previous (bool, optional): Whether to retain previously marked bad data.
        - plot_rejection_matrix (bool, optional): Whether to plot the rejection matrix.
        - segmented (bool, optional): Whether the data is segmented (e.g., into epochs).
        - config (bool/dict, optional): Configuration parameters or use default if False.
    
    Methods:
    --------
        __init__(self, raw, thresh_bad_channels=None, thresh_bad_times=None, min_good_time=0, min_bad_time=0,
                mask_time=0, keep_rejected_previous=False, plot_rejection_matrix=False, segmented=False, config=False):
            Initializes the DefineBTBC class.
        
        reject_samples_based_on_rejection_matrix: Main method reject samples based on the artifacts rejection matrix.
    """
    
    def __init__(self, raw, thresh_bad_channels=None, thresh_bad_times=None, min_good_time=0, min_bad_time=0,
                mask_time=0, keep_rejected_previous=False, plot_rejection_matrix=False, segmented=False, config=False):
        """
        Initializes the DefineBTBC object with default or specified parameters.
        """
        
        # Set default thresholds if not provided
        if thresh_bad_times is None:
            thresh_bad_times = [0.7, 0.5, 0.3]
        if thresh_bad_channels is None:
            thresh_bad_channels = [0.7, 0.5, 0.3]

        # Store parameters
        self.params = {
            'thresh_bad_channels':thresh_bad_channels, 
            'thresh_bad_times':thresh_bad_times,
            'min_good_time':min_good_time, 
            'min_bad_time':min_bad_time, 
            'mask_time':mask_time,
            'keep_rejected_previous':keep_rejected_previous, 
            'plot_rejection_matrix':plot_rejection_matrix,
            'segmented':segmented
        }

        # Get configuration (user-input parameters)
        if config:
            config_source = BTBC_Definition_Epochs if segmented else BTBC_Definition
            for key in self.params:
                if hasattr(config_source, key):
                    self.params[key] = getattr(config_source, key)

        print('\nIdentifying bad samples and channels...')

        # Define artifacts
        BC_pre, BT_pre = self.define_artifacts(raw, keep_rejected=self.params['keep_rejected_previous'])

        # Update rejection matrix
        BC, BT, BCBT = self.update_rejection_matrix(raw, BC_pre, BT_pre,
                                                    thresh_bad_times=self.params['thresh_bad_times'],
                                                    thresh_bad_channels=self.params['thresh_bad_channels'])

        # Sample rejection based on the rejection matrix
        BT = self.reject_samples_based_on_rejection_matrix(BT, data_size=np.size(BT), sfreq=raw.info['sfreq'])

        # Display rejected data
        self.display_rejected_data(BC, BT, BC_pre, BT_pre)

        # Update the rejection matrix for bad times and bad channels in the EEG data
        raw.artifacts.BT = BT  # Set the bad times rejection matrix
        raw.artifacts.BC = BC  # Set the bad channels rejection matrix

        # Print a summary of artifacts detected in the EEG data
        print(
            f"\n\nSUMMARY: Artifacts\n",
            f"{raw.artifacts.print_summary()}\n"
            )
        
        # Check if the plotting of the rejection matrix is enabled in the parameters
        if self.params['plot_rejection_matrix']:
            # Plot the artifact structure for Bad Channel Time (BCT)
            self.plot_artifact_structure(raw, artifact='BCT')
            
            # Plot the artifact structure for all artifact types combined
            self.plot_artifact_structure(raw, artifact='all')

    @staticmethod
    def define_artifacts(raw, keep_rejected):
        """
        Static method to define the initial bad channels and bad times artifacts matrices.
        
        This method initializes or retains the bad channel (BC) and bad time (BT) artifacts 
        matrices based on the 'keep_rejected' parameter, which dictates whether to start 
        with a clean slate or keep the previously detected artifacts.

        Parameters:
        - raw (Raw): The raw EEG object containing the data and previous artifacts information.
        - keep_rejected (bool|int|str): A flag or string specifying how to handle 
                                        previously detected artifacts. If boolean and True, retains all; if False, 
                                        resets all. If string, specifies which specific artifact type to retain 
                                        ('BC' for bad channels, 'BT' for bad times).

        Returns:
        - BC_old (np.ndarray): A matrix representing the previously marked bad channels.
        - BT_old (np.ndarray): A matrix representing the previously marked bad times.
        """
        
        # Extract data dimensions
        n_electrodes, n_samples, n_epochs = apice.io.Raw.get_data_size(raw)
        
        # Determine the initial state of bad channel and bad time matrices
        # based on the keep_rejected parameter
        if isinstance(keep_rejected, (bool, int)):
            if keep_rejected:
                if hasattr(raw.artifacts, 'BC'):
                    BC_old = raw.artifacts.BC.copy()
                else:
                    BC_old = np.full((n_epochs, n_electrodes, 1), False)
                if hasattr(raw.artifacts, 'BT'):
                    BT_old = raw.artifacts.BT.copy()
                else:
                    BT_old = np.full((n_epochs, 1, n_samples))
            else:
                BC_old = np.full((n_epochs, n_electrodes, 1), False)
                BT_old = np.full((n_epochs, 1, n_samples), False)
                
        elif isinstance(keep_rejected, str):
            # Handling string input to selectively keep certain types of rejected data
            if keep_rejected == 'BT':
                if hasattr(raw.artifacts, 'BT'):
                    BT_old = raw.artifacts.BT.copy()
                    BC_old = np.full((n_epochs, n_electrodes, 1), False)
                else:
                    BT_old = np.full((n_epochs, 1, n_samples), False)
                    BC_old = np.full((n_epochs, n_electrodes, 1), False)
            if keep_rejected == 'BC':
                if hasattr(raw.artifacts, 'BC'):
                    BC_old = raw.artifacts.BC.copy()
                    BT_old = np.full((n_epochs, 1, n_samples), False)
                else:
                    BC_old = np.full((n_epochs, n_electrodes, 1), False)
                    BT_old = np.full((n_epochs, 1, n_samples), False)
                    
        return BC_old, BT_old

    @staticmethod
    def update_rejection_matrix(raw, BC_old, BT_old, thresh_bad_times, thresh_bad_channels):
        """
        Updates the rejection matrix based on newly computed thresholds for bad channels and times.
        
        This method compares the new thresholds against the EEG artifact data to determine
        which channels and time periods should be marked as bad. The rejection matrix is
        updated accordingly.
        
        Parameters:
        - raw: The Raw EEG object containing artifact data and EEG signal times.
        - BC_old (numpy.ndarray): The matrix representing previously marked bad channels.
        - BT_old (numpy.ndarray): The matrix representing previously marked bad times.
        - thresh_bad_times (list): Thresholds used to define bad times based on artifact data.
        - thresh_bad_channels (list): Thresholds used to define bad channels based on artifact data.
        
        Returns:
        - BC (numpy.ndarray): The updated matrix of bad channels.
        - BT (numpy.ndarray): The updated matrix of bad times.
        - BCBT_new (numpy.ndarray): A combined matrix of bad channels and bad times.
        """
        
        # Extract necessary data size information from the EEG object
        n_electrodes, n_samples, n_epochs = apice.io.Raw.get_data_size(raw)
        
        # Copy thresholds for internal manipulation
        DefBT = thresh_bad_times
        DefBC = thresh_bad_channels.copy()
        DefBC_all = thresh_bad_channels.copy()
        
        # Copy the old bad channel and time data
        BC_new = BC_old.copy()
        BT_new = BT_old.copy()
        
        # For multi-epoch data, adjust the threshold for the last position
        if n_epochs > 1:
            DefBC_all[-1] = 0.1 / np.diff([raw.times[0], raw.times[-1]])
            
        # Determine the number of cycles for iteration based on the unique length of threshold arrays
        n_cycle = np.unique([len(DefBT), len(DefBC), len(DefBC_all)])

        # Initialize a matrix to track changes
        BCBT_new = np.zeros((n_epochs, n_electrodes, n_samples))
        
        # Mark the initially known bad channels and times in the new matrix
        BCBT_new[np.tile(BC_new.copy(), [1, 1, n_samples])] = True
        BCBT_new[np.tile(BT_new.copy(), [1, n_electrodes, 1])] = True
        
        # Track the channels that are bad throughout the whole dataset
        BC_all = np.all(BC_new.copy(), axis=2)
        
        # If there are manually marked bad channels, incorporate them
        if hasattr(raw.artifacts, 'BCmanual'):
            BC_all[raw.artifacts.BCmanual] = True
            BC_new[:, raw.artifacts.BCmanual, :] = True

        # Initialize a temporary matrix to track updates per cycle
        bcbt = np.zeros((n_epochs, n_electrodes, n_samples))
        
        # Loop through each cycle and epoch to update based on thresholds
        for i in np.arange(n_cycle):
            for ep in np.arange(n_epochs):
                bct = raw.artifacts.BCT[ep, :, :].copy()
                
                # Define BAD SAMPLES based on absolute threshold
                # Number of bad channels per sampled
                thresh_bad_channels = DefBT[i]
                bct_ = bct.copy()
                BC_new_ = BC_new[ep, :, :].copy()
                bct_[np.tile(BC_new_, [1, n_samples])] = False
                n_bad_channels = np.sum(bct_, axis=0)
                p_bad_channels = n_bad_channels / np.tile(np.sum(~BC_new_), [1, n_samples])

                # Define BAD CHANNELS during the whole recording on ABSOLUTE THRESHOLD
                # Number of bad samples per channel
                thresh_bad_samples_all = DefBC_all[i]
                bct_ = bct.copy()
                BT_new_ = BT_new[ep, :, :].copy()
                bct_[np.tile(BT_new_, (n_electrodes, 1))] = False
                n_bad_samples_all = np.sum(bct_, axis=1)
                p_bad_samples_all = n_bad_samples_all / np.sum(~BT_new_)

                # Define BAD CHANNELS per epoch on ABSOLUTE THRESHOLD
                # Number of bad samples per channel
                thresh_bad_samples = DefBC[i]
                bct_ = bct.copy()
                BT_new_ = BT_new[ep, :, :].copy()
                bct_[np.tile(BT_new_, (n_electrodes, 1))] = False
                n_bad_samples = np.sum(bct_, axis=1)
                p_bad_samples = np.divide(n_bad_samples, np.tile(np.sum(~BT_new_), [n_electrodes]))

                # Reject bad data
                BT_new[ep, :, :] = BT_new[ep, :, :].copy() | (p_bad_channels > thresh_bad_channels)
                BC_all[ep, :] = BC_all[ep, :].copy() | (p_bad_samples_all > thresh_bad_samples_all)
                BC_new[ep, :, :] = BC_new[ep, :, :].copy() | np.reshape(BC_all[ep, :], [n_electrodes, 1])
                BC_new[ep, :, :] = BC_new[ep, :, :].copy() | np.reshape((p_bad_samples > thresh_bad_samples),
                                                                        [n_electrodes, 1])

            # Test if the definition changes

            bcbt[np.tile(BC_new, [1, 1, n_samples])] = True
            bcbt[np.tile(BT_new, [1, n_electrodes, 1])] = True
            change_in_def = np.not_equal(bcbt, BCBT_new)
            BCBT_new = bcbt.copy()
            print('Cycle ', str(i),
                  ': new rejected data ', np.round(np.sum(change_in_def) / np.size(change_in_def) * 100, 2), '%')

        # Update
        BT = np.logical_or(BT_old, BT_new)
        BC = np.logical_or(BC_old, BC_new)

        return BC, BT, BCBT_new

    def reject_samples_based_on_rejection_matrix(self, BT, data_size, sfreq):
        """
        This method processes an artifact rejection matrix to remove undesirable samples.
        
        Args:
            BT (numpy.ndarray): The initial matrix representing marked bad times (artifacts).
            data_size (numpy.ndarray): An array that defines the size of the data.
            sfreq (int): The sampling frequency of the data in Hz.

        Returns:
            numpy.ndarray: The updated rejection matrix with certain bad times removed based on the criteria.
        """
        
        # Remove artifacts that are shorter than the minimum bad time specified in parameters
        # This step ensures that very brief artifacts do not lead to the rejection of data
        BT = self.remove_short_artifacts(BT, min_bad_time=self.params['min_bad_time'], sfreq=sfreq)

        # Apply a mask around the artifacts that are short but significant enough to require a buffer zone
        # This creates a safe margin around artifacts to avoid including any contaminated data
        BT = self.mask_around_short_artifacts(BT, mask_time=self.params['mask_time'], sfreq=sfreq)

        # Remove periods that are artifact-free but shorter than the minimum good time
        # This step is to ensure that data segments are of sufficient length to be considered reliable
        BT = self.remove_short_periods_without_artifacts(BT, data_size=data_size,
                                                        min_good_time=self.params['min_good_time'],
                                                        sfreq=sfreq)

        return BT

    @staticmethod
    def remove_short_artifacts(BT, min_bad_time, sfreq):
        """
        Static method to remove artifact periods that are shorter than a specified minimum duration.

        This method identifies the start and end of artifact periods and compares their duration to
        a minimum bad time threshold. If the duration is less than this threshold, the period is
        marked as non-artifact.

        Args:
            BT (numpy.ndarray): A binary matrix with shape (n_epochs, 1, n_samples) indicating bad samples.
            min_bad_time (float): The minimum duration (in seconds) for a bad period to be considered significant.
            sfreq (float): The sampling frequency of the data.

        Returns:
            numpy.ndarray: The updated binary matrix with very short artifacts removed.
        """
        
        # Get the number of epochs (data segments) to process
        n_epochs = np.shape(BT)[0]

        if min_bad_time != []:
            # Convert the minimum bad time from seconds to samples
            min_bad_time = min_bad_time * sfreq
            
            if min_bad_time != 0:
                # Process each epoch independently
                for ep in np.arange(n_epochs):
                    # Convert to integer for processing
                    bad_samples = BT[ep, 0, :].astype(int)
                    bad_time = bad_samples.copy().astype(int)

                    # Identify the start of bad periods
                    bad_time_initial = np.where(np.diff(bad_time, prepend=0) == 1)[0]
                    
                    # Identify the end of bad periods
                    bad_time_final = np.where(np.diff(bad_time, append=0) == -1)[0]

                    # Pair up the start and end points of bad periods
                    time_limit = np.asarray([bad_time_initial, bad_time_final]).T
                    
                    if np.size(time_limit) > 0:
                        # Calculate the duration of each bad period
                        duration = time_limit[:, 1] - time_limit[:, 0]
                        
                        # Find periods shorter than the minimum bad time
                        idx = np.where(duration < min_bad_time)[0]
                        if idx.size > 0:
                            # Create an array of all the indices that should be marked as non-artifact
                            indices_to_reset = np.hstack([np.arange(bad_time_initial[i], bad_time_final[i] + 1) for i in idx])
                            # Mark these indices as non-artifact in one operation
                            bad_samples[indices_to_reset] = 0
                            # Update the binary matrix to reflect these changes
                            BT[ep, :, :] = bad_samples.astype(bool)

        return BT

    @staticmethod
    def mask_around_short_artifacts(BT, mask_time, sfreq):
        """
        Apply a mask around short artifacts to create a buffer zone in the binary matrix.

        This method 'masks' (marks as bad) additional samples surrounding the identified artifacts,
        extending the bad periods by a specified buffer time on both sides. This is often done to 
        ensure that the data immediately around an artifact is not used in analysis due to potential
        contamination.

        Args:
            BT (numpy.ndarray): Binary matrix indicating bad samples (artifacts).
                                Shape is (n_epochs, 1, n_samples).
            mask_time (float): Time in seconds to extend the mask around each artifact.
            sfreq (int): Sampling frequency of the data in Hz.

        Returns:
            numpy.ndarray: Updated binary matrix with additional samples marked as bad around each artifact.
        """
        
        # Parameters
        n_epochs, _, n_samples = np.shape(BT)
        
        # Only proceed if mask_time is provided and non-zero
        if mask_time != [] and mask_time != 0:
            # Convert mask_time from seconds to number of samples
            mask_time = mask_time * sfreq
            # Round buffer to nearest sample to avoid fractional indices
            buffer = np.round(mask_time)
            
            # Process each epoch to mask around artifacts
            for ep in np.arange(n_epochs):
                bt = BT[ep, :, :].copy() # Copy the artifact data for this epoch
                bad = bt.copy()
                bad_idx = np.asarray(np.where(bad)[1], dtype=int) # Find indices of artifacts
                
                # Check if there are any artifacts to mask around
                if len(bad_idx) > 0:
                    # Generate indices for masking around each artifact
                    temp1 = np.tile(bad_idx, [int(2 * buffer + 1), 1]).T
                    temp2 = np.tile(np.arange(-buffer, buffer + 1), [len(bad_idx), 1])  # buffer + 1
                    bad_idx = temp1 + temp2
                    
                    # Ensure indices are within the sample range and unique
                    bad_idx = np.unique(bad_idx)
                    bad_idx = np.asarray(bad_idx[np.logical_and((bad_idx > 0), (bad_idx <= n_samples - 1))], dtype=int)
                
                # Apply the mask if any indices are to be masked
                if np.size(bad_idx) > 0:
                    bt[0, bad_idx] = True
                    
                # Update the binary matrix for this epoch
                BT[ep, :, :] = bt.astype(bool)
                
        return BT

    @staticmethod
    def remove_short_periods_without_artifacts(BT, data_size, min_good_time, sfreq):
        """Removes short periods without artifacts between longer artifact segments.

        This identifies any short segments of "good" data in between longer segments 
        of artifacts, and marks those short good segments as bad. The goal is to avoid 
        keeping small snippets of clean data between big artifacts.

        Args:
            BT (numpy.ndarray): 2D array of epochs x timepoints indicating artifact segments
            data_size (int): Total size of the data in timepoints
            min_good_time (int): Minimum duration of a good segment to keep in seconds 
            sfreq (int): The sampling frequency of the data in Hz

        Returns:
            numpy.ndarray: Updated BT array with short good sections now marked as bad
        
        """
        # Extract the shape of the bad times matrix
        n_epochs = np.shape(BT)[0]
        
        if min_good_time != []:
            # Calculate min good time in samples
            min_good_time = int(min_good_time * sfreq)
            
            if min_good_time < (data_size - 2):
                
                # Iterate through epochs
                for ep in np.arange(n_epochs):
                    # Get a copy of the bad samples for this epoch.
                    bad_samples = BT[ep, 0, :]
                                            
                    # Identify good sections
                    good_time = np.logical_not(bad_samples.copy()).astype(int)

                    # Add zeros before and after good times
                    padded_good_times = np.concatenate(([0], good_time, [0]))
                    
                    # Find indices where padded array changes 
                    change_indices = np.flatnonzero(np.diff(padded_good_times))
                    
                    # The even indices are start of good sections
                    good_start_indices = change_indices[::2]  

                    # The odd indices are ends of good sections
                    good_end_indices = change_indices[1::2]

                    # Duration is difference between end and start 
                    durations = good_end_indices - good_start_indices

                    # Calculate good section end points from start point and duration
                    good_end_points = good_start_indices + durations - 1

                    # Filter to only durations less than min_good_time
                    short_durations = durations[durations < min_good_time]

                    # Get corresponding start and end indices
                    short_start_indices = good_start_indices[durations < min_good_time]
                    short_end_indices = good_end_points[durations < min_good_time]

                    # Mark those sections as bad 
                    for start, end in zip(short_start_indices, short_end_indices):
                        bad_samples[start:end+1] = True

                    # Update epoch with new bad sections
                    BT[ep] = np.logical_or(BT[ep], bad_samples) 

        return BT

    @staticmethod
    def display_rejected_data(BC, BT, BC_pre, BT_pre):
        """
        Display summary statistics of rejected data.
        
        Prints the percentage of new and total rejected data for 
        bad times, bad channels per epoch, and bad channels.

        Args:
            BC: 3D array of bad channels
            BT: 3D array of bad times 
            BC_pre: 3D array of previous bad channels
            BT_pre: 3D array of previous bad times
        
        Returns:
            None
        """

        # Calculate total samples
        n_epochs = np.shape(BC)[0]
        n_samples = np.shape(BT)[2]
        n_electrodes = np.shape(BC)[1]
        
        # Calculate new and total bad times
        d = n_epochs * n_samples
        new_d = np.sum(np.logical_and(BT, np.logical_not(BT_pre)))
        all_d = np.sum(BT)
        
        # Print bad times summary
        print('\nTotal new BAD TIMES ______________________________ ', np.round(new_d / d * 100, 2), '%')
        print('Total BAD TIMES __________________________________ ', np.round(all_d / d * 100, 2), '%')

        # Calculate new and total bad channels per epoch
        d = n_epochs * n_electrodes
        new_d = np.sum(np.logical_and(BC[:], np.logical_not(BC_pre[:])))
        all_d = np.sum(BC[:])
        
        # Print bad channels per epoch summary
        print('\nTotal new BAD CHANNELS per epoch _________________ ', np.round(new_d / d * 100, 2), '%')
        print('Total BAD CHANNELS per epoch _____________________ ', np.round(all_d / d * 100, 2), '%')

        # Calculate new and total bad channels
        new_d = np.sum(np.logical_and(np.all(BC, axis=0), np.all(np.logical_not(BC_pre), axis=0)))
        all_d = np.sum(np.all(BC, axis=0))
        
        # Print bad channels summary
        print('\nTotal new BAD CHANNELS ___________________________ ', new_d)
        print('Total BAD CHANNELS _______________________________ ', all_d)
        
        return

    @staticmethod
    def plot_artifact_structure(raw, artifact='all', time_step=50, color_scheme='gnuplot', figsize=(12, 6)):
        """
        This function plots a visual representation of the artifact structure within EEG data.
        It allows visualization of different types of artifacts and their occurrences over time or epochs.

        Args:
            raw (mne.io.Raw): The MNE Raw object containing EEG data.
            artifact (str): Specifies the type of artifact to plot ('all', 'BCT', 'BT', 'BC', 'BE'). Defaults to 'all'.
            time_step (int): Time step for x-axis ticks, in seconds. Defaults to 50.
            color_scheme (str): The color scheme for plotting. Defaults to 'gnuplot'.
            figsize (tuple): Tuple specifying the figure size (width, height) in inches. Defaults to (8, 6).

        Returns:
            matplotlib.figure.Figure: The figure object containing the artifact plot.
        """

        # Import necessary modules
        import numpy as np
        import matplotlib.pyplot as plt
        from apice.io import Raw
        
        # Functions
        def prepare_cmap(ax, data, artifact, color_scheme='gnuplot'):
            """
            Prepare colormap and colorbar for artifact matrix visualization.

            Args:
                ax (matplotlib.axes.Axes): The subplot where the artifact matrix will be displayed.
                data (numpy.ndarray): The artifact matrix to be visualized.
                artifact (str): Specifies the type of artifact ('all', 'BCT', 'BT', 'BC', 'BE').
                color_scheme (str): The color scheme for plotting. Defaults to 'gnuplot'.

            Returns:
                None
            """
            if artifact == 'all':
                # Define tick labels and colormap for all artifact types
                # tick_labels = ['good', 'bad', 'BT', 'BC', 'BE']
                tick_labels = ['Good Data', 'Bad Data', 'Bad Time Point', 'Bad Channel', 'Bad Epoch']
                cmap = plt.get_cmap(color_scheme, len(tick_labels))
                mat = ax.imshow(data, cmap=cmap, vmin=-0.5, vmax=4.5, aspect='auto')
                cax = plt.colorbar(mat, ticks=np.arange(5))
                cax.set_ticklabels(tick_labels)
            else:
                # Define tick labels and colormap for all artifact types
                cmap = plt.get_cmap(color_scheme, len(np.unique(data)))
                mat = ax.imshow(data, cmap=cmap, vmin=np.min(data) - 0.5, vmax=np.max(data) + 0.5, aspect='auto')
                cax = plt.colorbar(mat, ticks=np.asarray(np.unique(data), dtype=int))
                colorbar_ticks = np.unique(data)
                # labels = ['good', 'bad', 'BT', 'BC', 'BE']
                labels = ['Good Data', 'Bad Data', 'Bad Time Point', 'Bad Channel', 'Bad Epoch']
                tick_labels = []
                for i in colorbar_ticks:
                    tick_labels.append(labels[i])
                cax.set_ticklabels(tick_labels)

        def set_ticks(ax, data, t, time_step, sfreq, n_epochs, n_electrodes, n_samples, ch_names):
            """
            Set ticks and labels for the x and y axes of a subplot.

            Args:
                ax (matplotlib.axes.Axes): The subplot where ticks and labels will be set.
                data (numpy.ndarray): The data matrix being displayed.
                time_step (int): Time step for x-axis ticks, in seconds.
                sfreq (float): The sampling frequency of the data.
                n_electrodes (int): Number of electrode channels.
                ch_names (list): List of channel names.
                x_label (str): Label for the x-axis. Defaults to 'Time (s)'.
                artifact (str): Specifies the type of artifact ('all' or specific type).

            Returns:
                None
            """
            
            # Set x-axis ticks and labels
            ax.tick_params(axis="x", bottom=True, top=False, labeltop=False, labelbottom=True)
            if n_epochs > 1:
                ax.set_xticks(np.arange(0, n_epochs * n_samples, n_samples * 5))
                ax.set_xticklabels(np.arange(0, n_epochs, 5))
                ax.set_xlabel('Epoch #')
            else:
                ax.set_xticks(np.arange(0, np.shape(data)[1], time_step * sfreq))
                xticks = np.asarray(ax.get_xticks(), dtype=int)
                ax.set_xticklabels(t[xticks])   
                ax.set_xlabel('Time (s)')    
            
            # Set y-axis ticks and labels
            # if len(ch_names) > 40:
            #     yticks = np.arange(0, n_electrodes, 5)
            # else:
            #     yticks = np.arange(n_electrodes)
            yticks = np.arange(n_electrodes)
            ax.set_yticks(yticks)
            ax.set_yticklabels([ch_names[i] for i in yticks], fontsize=5)  # Use channel names for labels
            
            # Set subplot title, x-axis label, and y-axis label
            ax.set_title(artifact)
            ax.set_ylabel('Channel #')


        # Extract the dimensions of the EEG data for plotting
        n_electrodes, n_samples, n_epochs = Raw.get_data_size(raw)

        # Initialize a matrix to store artifact occurrence information
        M = np.zeros(np.shape(raw.artifacts.BCT))

        # Populate the matrix 'M' based on the specified artifacts
        if artifact in ['BCT', 'all']:
            M[raw.artifacts.BCT] = 1
        if artifact in ['BT', 'all']:
            M[np.tile(raw.artifacts.BT, [1, n_electrodes, 1])] = 2
        if artifact in ['BC', 'all']:
            M[np.tile(raw.artifacts.BC, [1, 1, n_samples])] = 3
        if artifact in ['BE', 'all']:
            M[np.tile(raw.artifacts.BE, [1, n_electrodes, n_samples])] = 4

        # Convert the artifact matrix to an integer data type for consistent plotting
        M = np.asarray(M, dtype=int)
        # Extract time information from the EEG object for the x-axis
        t = raw.times
        # Create a figure object with the specified figsize
        fig = plt.figure(figsize=figsize)
        # Get sampling frequency
        sfreq = raw.info['sfreq']
        # Retrieve channel names from EEG info for y-axis labels
        ch_names = raw.info['ch_names']

        # Plotting routine for a single epoch
        if n_epochs == 1:
            ax = fig.add_subplot(111)
            data = M[0, :, :]
            prepare_cmap(ax, data, artifact, color_scheme=color_scheme)
            set_ticks(ax, data, t, time_step, sfreq, n_epochs, n_electrodes, n_samples, ch_names)
        # Plotting routine for multiple epochs
        else:
            N = M[0, :, :]
            for ep in np.arange(1, n_epochs):
                N = np.concatenate((N.copy(), M[ep, :, :]), axis=1)
            ax = fig.add_subplot(111)
            data = N
            prepare_cmap(ax, data, artifact, color_scheme=color_scheme)
            set_ticks(ax, data, t, time_step, sfreq, n_epochs, n_electrodes, n_samples, ch_names)

        return fig

