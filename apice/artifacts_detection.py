# %% LIBRARIES
# Importing necessary libraries for EEG data processing
import numpy as np
import apice
from apice.artifacts_structure import Artifacts, set_reference, compute_z_score
from apice.io import Raw

# Configuration to ignore divide by zero errors in numpy
np.seterr(divide='ignore')

# Suppressing specific numpy warnings related to NaN values and runtime issues
import warnings
warnings.filterwarnings(action='ignore', message='All-NaN slice encountered')
warnings.simplefilter("ignore", category=RuntimeWarning)


# %% FUNCTIONS

def insert_initial_and_final_samples(t_initial, t_final):
    """
    Insert a single 'False' data point at the beginning and end of a time window.

    When processing time-based data, particularly for artifact rejection in signal processing,
    it is sometimes necessary to add markers at the start and end of a time series to
    indicate the beginning and ending of a non-valid data segment. This function adds a
    'False' value to the start and end of boolean arrays that represent such time windows.

    Args:
        - t_initial (numpy.ndarray): The array of initial time points of the segments.
        - t_final (numpy.ndarray): The array of final time points of the segments.

    Returns:
        tuple: A tuple containing the updated arrays of initial and final time points.

    Example:
        >>> t_initial, t_final = insert_initial_and_final_samples(t_initial, t_final)
    """

    # Compute the boolean condition for intermediate samples
    samples = (t_final[:-1] + 1 - t_initial[1:]) >= 0

    # Prepare a temporary array to insert 'False' at the beginning
    temp = np.full(len(samples) + 1, False)
    temp[1:] = samples
    
    # Remove the initial time points where the condition is not met
    t_initial = np.delete(t_initial, temp)

    # Prepare another temporary array to insert 'False' at the end
    temp = np.full(len(samples) + 1, False)
    temp[:-1] = samples
    
    # Remove the final time points where the condition is not met
    t_final = np.delete(t_final, temp)

    return t_initial, t_final


def define_time_window(raw, time_window, time_window_step):
    """
    Divides raw EEG data into overlapping time segments based on a specified time window and step size.

    Args:
        - raw (object): An object containing the raw EEG data.
        - time_window (float): The duration of the time window in seconds.
        - time_window_step (float): The step length for sliding the window, in seconds.

    Returns:
        - bct_time_windows (np.ndarray): A matrix indicating the bad time windows.
        - i_t (np.ndarray): The indices representing the start and end of each time window.
        - n_time_window (int): The total number of time windows created.
    """
    
    # Retrieve the dimensions of the raw EEG data:
    n_electrodes, n_samples, n_epochs = Raw.get_data_size(raw)

    # Check if the time window and step are valid and not infinite
    if time_window and time_window != np.inf:
        # Convert time window duration from seconds to number of data samples
        time_window_samples = int(np.round(time_window * raw.info['sfreq']))
        time_window_step_samples = int(np.round(time_window_step * raw.info['sfreq']))
        
        # Calculate the number of time windows that fit into the raw EEG data
        n_time_window = int(np.round((n_samples - time_window_samples + 1) / time_window_step_samples)) + 1
        
        # If the calculated number of time windows is not positive, warn the user
        if n_time_window <= 0:
            warnings.warn('The time window is too long.')
        
        # Compute the indices that represent the start of each time window
        start_indices = np.round(np.linspace(0, n_samples - time_window_samples, n_time_window)).astype(int)
        
        # Create a 2D array where each column represents the indices for one time window
        i_t = (np.tile(start_indices, (time_window_samples, 1)) +
            np.tile(np.arange(time_window_samples), (len(start_indices), 1)).T).astype(int)
    else:
        # If no valid time window is defined, use the entire sample length
        n_time_window = 1
        i_t = np.arange(n_samples)[np.newaxis].T  # Transpose to get a column vector

    # Initialize a 3D array to mark bad time windows for all epochs and electrodes
    bct_time_windows = np.full((n_epochs, n_electrodes, n_time_window), False)

    # Loop over each time window
    for i in range(n_time_window):
        # Extract the indices for the current time window
        idx = i_t[:, i]
        # Sum the artifact markers within the current time window across all epochs and electrodes
        # Compare the sum against the total length of the window to check for bad data
        # If the number of artifacts equals or exceeds the window length, mark it as a bad window
        bct_time_windows[:, :, i] = np.sum(raw.artifacts.BCT[:, :, idx], axis=2) >= len(idx)

    return bct_time_windows, i_t, n_time_window


def update_rejection_matrix(artifacts, name, params, BCT):
    """
    Update the rejection matrix with new bad channel times (BCT).

    Args:
        - artifacts (dict): The object holding all the artifacts matrices.
        - name (str): The name of the current rejection algorithm being applied.
        - params (dict): Parameters that define the operation of the current algorithm.
        - BCT (ndarray): A new matrix indicating bad channel times after rejection.

    Returns:
        - None: This function updates the matrices in-place and does not return anything.
    """
    
    # If the parameters indicate that the BCT should be updated,
    # integrate the new bad channels into the existing artifacts
    if params['update_BCT']:
        if name == 'INCLUDE_ShortBadSegments':
            # If the current step is to include short bad segments,
            # the new BCT matrix is used to reset (clear) corresponding entries
            artifacts.BCT[BCT] = False
        else:
            # For other algorithms, combine the new BCT with the existing one
            # by a logical 'or', marking a sample bad if it was marked by either
            artifacts.BCT = np.logical_or(artifacts.BCT, BCT)
    
    # If the summary of bad samples should be updated, print the new summary
    if params['update_summary']:
        artifacts.summary = artifacts.print_summary()
    
    # If information about the algorithm should be updated, store the parameters
    # and information such as the step name and the number of rejected samples
    if params['update_algorithm']:
        artifacts.algorithm['params'] = params
        artifacts.algorithm['step_name'] = name
        if name == 'INCLUDE_ShortBadSegments':
            # When including short segments, store the reduction in bad samples
            artifacts.algorithm['rejection_step'] = -np.sum(BCT)
        else:
            # Otherwise, store the increase in identified bad samples
            artifacts.algorithm['rejection_step'] = np.sum(BCT)


def return_data_after_zscore(raw, sd, mu):
    """
    Reverses the z-score normalization applied to raw EEG data.

    After z-score normalization has been applied, this function restores the original
    data by reversing the z-score operation using the provided standard deviation
    and mean values.

    Args:
        - raw (Raw): The object containing the raw EEG data that has been z-scored.
        - sd (ndarray): The standard deviation(s) used during z-score normalization.
        - mu (ndarray): The mean value(s) used during z-score normalization.

    Returns:
        - None: The function updates the raw._data array in-place and does not return a value.
    """

    # Get the data size for reshaping purposes
    n_electrodes, n_samples, n_epochs = Raw.get_data_size(raw)

    # Flatten the data for the operation (n_electrodes, n_samples * n_epochs)
    temp_data = np.reshape(raw._data, (n_electrodes, n_samples * n_epochs))

    # Repeat the standard deviation and mean across all samples
    temp_sd = np.tile(sd, (n_samples * n_epochs, 1))
    temp_mu = np.tile(mu, (n_samples * n_epochs, 1))

    # Reverse the z-score: Multiply by the standard deviation and add the mean
    raw._data = np.multiply(temp_data, temp_sd.T) + temp_mu.T


def return_data_after_referencing(raw):
    """
    Resets raw EEG data to its original reference.

    This function is used when the raw EEG data has been re-referenced and you wish to restore
    the original reference. It adds back the previously calculated mean reference to each electrode's data.

    Args:
        - raw (Raw): The object containing the raw EEG data that has been re-referenced.

    Returns:
        - None: The function updates the raw._data array in-place and does not return a value.
    """

    # Retrieve the number of electrodes to determine the shape of the data
    n_electrodes, _, _ = Raw.get_data_size(raw)

    # Create an array of the mean reference values repeated across all electrodes
    temp = np.tile(raw.mean_reference, (n_electrodes, 1))

    # Add the mean reference back to the raw EEG data to revert to the original referencing
    raw._data = raw._data + temp


def create_mask_matrix(raw, mask_length=0.05):
    """
    Creates a matrix that masks artifacts in raw EEG data.

    Given raw EEG data and a specified mask length in seconds, this function generates a boolean matrix that
    indicates the regions in the data to be masked due to artifacts. It considers a buffer zone around the
    identified bad segments to ensure a cleaner signal.

    Args:
        - raw (Raw): An object containing the raw EEG data and its associated information.
        - mask_length (float): The duration in seconds for which to extend the mask around each artifact. Defaults to 0.05 seconds.

    Returns:
        - ndarray: A 3D boolean numpy array indicating the masks for artifacts. The dimensions correspond to epochs, electrodes, and samples.
    """

    # Get the size of the data to determine the structure of the mask matrix
    n_electrodes, n_samples, n_epochs = Raw.get_data_size(raw)

    # Retrieve the bad channel times (BCT) matrix from the raw data
    BCT = raw.artifacts.BCT

    # Initialize the mask matrix with the same shape as the BCT, filled with False (indicating no artifact)
    mask_matrix = np.full((n_epochs, n_electrodes, n_samples), False)

    # Calculate the number of samples that the mask length corresponds to
    buffer = int(np.round(mask_length * raw.info['sfreq']))

    # Iterate over each epoch and electrode to create the mask
    for ep in range(n_epochs):
        for el in range(n_electrodes):
            # Create an array with artifact boundaries marked
            ZA = np.concatenate(([0], BCT[ep, el, :].astype(int), [0]))
            # Find indices where the artifact status changes
            indices = np.flatnonzero(ZA[1:] != ZA[:-1])
            # Calculate the lengths of good/bad segments
            counts = indices[1:] - indices[:-1]
            # Indices of the start of bad segments
            bad_i = indices[::2]
            # Length of each bad segment
            duration = counts[::2]
            # Indices of the end of bad segments
            bad_f = bad_i + duration - 1

            # Extend mask before and after bad segments within the bounds of the data
            for start, finish in zip(bad_i, bad_f):
                start_buffer = np.arange(max(start - buffer, 0), start)
                end_buffer = np.arange(finish + 1, min(finish + buffer, n_samples))
                mask_matrix[ep, el, start_buffer] = True
                mask_matrix[ep, el, end_buffer] = True

    # Return the completed mask matrix
    return mask_matrix


def mask_around_artifacts(raw, mask_length=0.5):
    """
    Masks artifacts in raw EEG data by extending the rejection around each detected artifact.

    This function applies an additional mask to the raw EEG data where artifacts have been detected.
    It extends the rejected region by a specified duration before and after the artifact to ensure
    the data is clean. It then updates the artifact rejection matrix accordingly.

    Args:
        - raw: An object containing raw EEG data and associated artifact information.
        - mask_length (float): Duration in seconds to extend the mask around each artifact. Default is 0.5 seconds.

    Returns:
        - BCT_masked (ndarray): A boolean array that represents the updated bad channel times (BCT)
                                after applying the masking around artifacts.
    """

    print('\nMasking around artifacts...')

    # Initialize artifacts matrices
    BCT = raw.artifacts.BCT

    # Check if artifact rejection matrix exists
    raw.artifacts = Artifacts(raw)

    # Create mask matrix
    mask_matrix = create_mask_matrix(raw, mask_length=mask_length)

    # Reject bad data
    BCT_masked = np.logical_or(BCT, mask_matrix)

    # Display new rejected data
    n = np.prod(Raw.get_data_size(raw))
    new_rejected_data = np.logical_and(BCT_masked, np.logical_not(raw.artifacts.BCT))
    new_total_of_rejected_data = np.sum(new_rejected_data)
    print('\nData rejected due to masking : ', np.round(new_total_of_rejected_data / n * 100, 2), '%')
    return BCT_masked


def mask_around_artifacts_BCT(BCT, mask_length, sampling_freq):
    """
    Apply a mask around the identified bad segments within a binary classification tensor (BCT).

    Args:
        - BCT:  A 3D numpy array where the dimensions are epochs, electrodes, and samples. 
                It contains boolean values indicating the presence of artifacts.
        - mask_length: The duration in seconds around the bad segments to be masked.
        - sampling_freq: The sampling frequency of the EEG data.

    Returns:
        - BCT_masked: A 3D numpy array with the same dimensions as BCT, 
                    where additional samples around the bad segments are marked as artifacts.
    """
    
    # Obtain the dimensions of the BCT (Bad Channel Times) matrix which contains epochs, electrodes, and samples
    n_epochs, n_electrodes, n_samples = np.shape(BCT)
    
    # Create a mask matrix with the same shape as the BCT matrix, initialized with False
    # This matrix will be used to flag artifacts within the EEG data
    mask_matrix = np.full((n_epochs, n_electrodes, n_samples), False)

    # Compute for the buffer
    buffer = np.round(mask_length * sampling_freq)

    # Generate mask matrix
    for ep in np.arange(n_epochs):
        for el in np.arange(n_electrodes):
            # Identify the transitions between good and bad data
            data = BCT[ep, el, :].astype(int)
            ZA = np.concatenate(([0], data, [0]))
            indices = np.flatnonzero(ZA[1:] != ZA[:-1])
            counts = indices[1:] - indices[:-1]
            bad_i = indices[::2]
            duration = counts[::2]
            bad_f = bad_i + duration - 1

            if np.size(bad_i) > 0:
                
                # Mask before the start of each bad segment
                for i in np.arange(np.size(bad_i)):
                    bad_idx_i = np.asarray(np.arange(bad_i[i] - buffer, bad_i[i]), dtype=int)
                    bad_idx_i = np.delete(bad_idx_i, bad_idx_i < 0)
                    mask_matrix[ep, el, bad_idx_i] = True
                    
                # Mask after the end of each bad segment
                for i in np.arange(np.size(bad_f)):
                    bad_idx_f = np.asarray(np.arange(bad_f[i] + 1, bad_f[i] + buffer + 1), dtype=int)
                    bad_idx_f = np.delete(bad_idx_f, bad_idx_f >= n_samples)
                    mask_matrix[ep, el, bad_idx_f] = True
    
    # Combine the original BCT with the mask matrix to mark additional artifacts
    BCT_masked = np.logical_or(BCT, mask_matrix)
    
    return BCT_masked


def configure_threshold_parameters(params, name):
    """
    Configure threshold parameters for different artifact detection algorithms.

    This function adjusts the parameter dictionaries to ensure they have the correct size
    and structure for the specified artifact detection algorithm.

    Args:
        - params: dict, the parameter dictionary containing threshold configurations.
        - name: str, the name of the artifact detection algorithm.

    Returns:
        - dict, the updated parameter dictionary with configured threshold values.
    """

    # Define a dictionary that maps algorithm names to threshold types
    threshold_types = {
        'Amplitude': 'thresh',
        'TimeVariance': 'thresh',
        'FastChange': 'thresh',
        'RunningAverage': 'thresh_fast'
    }

    # Use the dictionary to get the threshold type, default to 'thresh' if name is not found
    threshold_type = threshold_types.get(name, 'thresh')

    # Ensure that the 'use_relative_thresh' parameter has the same size as the 'threshold_type' parameter
    if np.size(params[threshold_type]) != np.size(params['use_relative_thresh']):
        # If sizes don't match, repeat the 'use_relative_thresh' value to match the size of 'threshold_type'
        params['use_relative_thresh'] = np.repeat([params['use_relative_thresh']][0],
                                                    np.shape([params[threshold_type]])[0])
    else:
        # Ensure that 'use_relative_thresh' is a list
        params['use_relative_thresh'] = [params['use_relative_thresh']]

    # Similar check and adjustment for 'use_relative_thresh_per_electrode'
    if np.size(params[threshold_type]) != np.size(params['use_relative_thresh_per_electrode']):
        params['use_relative_thresh_per_electrode'] = np.repeat([params['use_relative_thresh_per_electrode']][0],
                                                                np.shape([params[threshold_type]])[0])
    else:
        params['use_relative_thresh_per_electrode'] = [params['use_relative_thresh_per_electrode']]

    # Reshape the threshold values to ensure they are in the correct format
    params[threshold_type] = np.reshape(params[threshold_type], (np.size(params[threshold_type])))

    # If the algorithm is 'RunningAverage', reshape 'thresh_diff' similarly
    if name == 'RunningAverage':
        params['thresh_diff'] = np.reshape(params['thresh_diff'], (np.size(params['thresh_diff'])))

    return params


# %% CLASSES


class ChannelCorr:
    """
    A class for identifying and rejecting bad raw EEG data based on channel correlations.
    
    Attributes:
        - params (dict): A dictionary containing the parameters for the artifact rejection process.
        - mu (float): Mean value used for z-scoring the data, if applicable.
        - sd (float): Standard deviation used for z-scoring the data, if applicable.
        - BCT (ndarray): A boolean matrix indicating bad data segments after rejection criteria are applied.
        - n_rejected_data (int): The number of data points rejected based on the criteria.

    Args:
        - raw: Raw EEG data object containing the data and metadata.
        - time_window (float): The duration of the window used to calculate correlation, in seconds.
        - time_window_step (float): The step size for the sliding window used in correlation calculation, in seconds.
        - top_channel_corr (int): The number of top channels to consider when computing correlation.
        - thresh (float): The threshold for correlation above which data is considered bad and rejected.
        - bad_data (str): Method for handling bad data, can be 'none', '0', or 'replace by nan'.
        - mask (int): Duration in seconds for which the data around artifacts should be masked.
        - do_reference_data (bool): Whether to set EEG reference (default False).
        - do_zscore (bool): Whether to z-score the data (default False).
        - use_relative_thresh (bool): Whether to use a relative threshold for rejection (default False).
        - update_BCT (bool): Whether to update the Bad Channel Times matrix (default True).
        - update_summary (bool): Whether to print a summary of bad data (default True).
        - update_algorithm (bool): Whether to save the current algorithm parameters (default True).
        - config (bool): Whether to update parameters based on user inputs (default False).
        - name (str): Name of the current instance or operation.
        - loop_name (list): List of names for looping constructs, if any.
        
    Methods:
        __init__(self, raw, time_window=4, time_window_step=2, top_channel_corr=5, thresh=0.4, 
                    bad_data='none', mask=0, do_reference_data=False, do_zscore=False, use_relative_thresh=False, 
                    update_BCT=True, update_summary=True, update_algorithm=True, config=False, name='ChannelCorr', 
                    loop_name=[])
        Constructs the ChannelCorr object and initializes the rejection process.
    """

    def __init__(self, raw, time_window=4, time_window_step=2, top_channel_corr=5,
                thresh=0.4, bad_data='none', mask=0, do_reference_data=False, do_zscore=False,
                use_relative_thresh=False, update_BCT=True, update_summary=True, update_algorithm=True, config=False,
                name='ChannelCorr', loop_name=[]):
        """
        Initializes the ChannelCorr object and begins the process of identifying and rejecting bad data.
        """

        # Initialize parameters for the channel correlation process
        self.params = {
            'time_window': time_window,             
            'time_window_step': time_window_step,   
            'top_channel_corr': top_channel_corr,   
            'thresh': thresh,                       
            'bad_data': bad_data,                   
            'mask': mask,                           
            'do_reference_data': do_reference_data, 
            'do_zscore': do_zscore,                 
            'use_relative_thresh': use_relative_thresh,
            'update_BCT': update_BCT,               
            'update_summary': update_summary,       
            'update_algorithm': update_algorithm,   
            'config': config,                       
            'name': name,                           
            'loop_name': loop_name                  
        }

        # Get configuration (user-input parameters)
        if config:
            from apice.artifacts_rejection import update_parameters_with_user_inputs
            self.params = update_parameters_with_user_inputs(self.params, eval('apice.parameters.' + loop_name + '.' + name))

        # Print rejection parameters
        print('\nRejecting data based on the channels correlations...')
        print('-- referenced data: ', self.params['do_reference_data'])
        print('-- z-score data: ', self.params['do_zscore'])
        print('-- relative threshold: ', self.params['use_relative_thresh'])

        # Initialize artifact rejection matrix
        raw.artifacts = Artifacts(raw, 'BCT')

        # Set reference to mean amplitude of channels
        if self.params['do_reference_data']:
            set_reference(raw, bad_data=self.params['bad_data'], save_reference=False)

        # Compute z-score for the artifacts
        if self.params['do_zscore']:
            raw._data, self.mu, self.sd = compute_z_score(raw)

        # Reject electrodes (per time window) based on correlation
        BCT = self.reject_electrodes_based_on_corr(raw, self.params['time_window'], self.params['time_window_step'],
                                                    self.params['thresh'], self.params['top_channel_corr'],
                                                    self.params['use_relative_thresh'])

        # Update rejection matrix
        update_rejection_matrix(raw.artifacts, self.__class__.__name__, self.params, BCT)

        # Get data back
        if self.params['do_zscore']:
            return_data_after_zscore(raw, self.sd, self.mu)
        if self.params['do_reference_data']:
            return_data_after_referencing(raw)

        # Mask around artifacts
        if self.params['mask']:
            BCT = mask_around_artifacts(raw, mask_length=self.params['mask'])
            # Update rejection matrix
            update_rejection_matrix(raw.artifacts, self.__class__.__name__, self.params, BCT)

        # Save a copy of BCT
        self.BCT = BCT

        # Display the total rejected data
        self.n_rejected_data = np.round(np.sum(BCT))
        # Calculate and display the percentage of rejected data
        rejected_percentage = np.round(self.n_rejected_data / np.size(BCT) * 100, 2)
        print(f'\nTotal rejected data: {rejected_percentage}%')

    @staticmethod
    def get_data_to_reject_based_on_corr(raw, time_window, time_window_step, thresh, top_channel_corr,
                                        use_relative_thresh=False):
        """
        Identify and mark artifacts based on the correlation between raw EEG channels within specified time windows.

        This method computes the correlation coefficient across all pairs of channels for each epoch within
        each time window. It then rejects any data where the average correlation across channels falls below
        the defined threshold. This can either be a fixed threshold or one based on a relative measure, such
        as a percentile of the computed correlations.

        Parameters:
        - raw: An object that contains the EEG data.
        - time_window: Duration of each time window for analysis, in seconds.
        - time_window_step: Step size between consecutive time windows, in seconds.
        - thresh: Correlation threshold below which data is considered bad and rejected.
        - top_channel_corr: Percentage representing the 'top' channels to consider when computing correlation.
        - use_relative_thresh: Boolean flag to use relative threshold based on percentile (True) or a fixed threshold (False).

        Returns:
        - data_to_reject: A boolean matrix indicating which data points are to be rejected.
        - bct_time_windows: Time windows used to segment the EEG data for artifact detection.
        - i_t: Indices representing the limits of each time window.
        """
        
        # Extract dimensions of raw EEG data
        n_electrodes, n_samples, n_epochs = Raw.get_data_size(raw)

        # Create time windows matrix
        bct_time_windows, i_t, n_time_window = define_time_window(raw, time_window=time_window,
                                                                    time_window_step=time_window_step)
        # Initialize correlation matrix
        CC = np.empty((n_epochs, n_electrodes, n_time_window))
        CC[:] = np.nan

        # Compute correlation between electrodes
        raw_data = np.reshape(raw._data, (n_epochs, n_electrodes, n_samples))
        for ep in np.arange(n_epochs):
            for itw in np.arange(n_time_window):
                # Take the data within the time window
                d = raw_data[ep, :, i_t[:, itw]]  # * 1e6
                # Compute the correlation with all channels
                channel_corr = np.abs(np.corrcoef(d.T))
                # Remove correlation with self
                channel_corr[np.identity(np.shape(channel_corr)[0], dtype=bool)] = np.nan
                # Top correlation
                ptop = np.nanpercentile(channel_corr, 100 - top_channel_corr, axis=0, interpolation='midpoint')
                channel_corr[channel_corr <= np.tile(ptop, (n_electrodes, 1))] = np.nan
                average_corr = np.nanmean(channel_corr, axis=0)
                # Store the data
                CC[ep, :, itw] = average_corr

        # Identify the threshold for rejecting data
        if use_relative_thresh:
            CCi = CC
            CCi[bct_time_windows] = np.nan
            perc = np.nanpercentile(CCi, [25, 50, 75], interpolation='midpoint')
            IQ = perc[2] - perc[0]
            t_l = perc[0] - thresh * IQ
        else:
            t_l = thresh

        # Reject data
        data_to_reject = CC < t_l
        
        return data_to_reject, bct_time_windows, i_t

    @staticmethod
    def reject_bad_data_based_on_corr(raw, data_to_reject, bct_time_windows, i_t):
        """
        Apply a binary rejection matrix to raw EEG data, marking segments to be rejected based on correlation criteria.

        This method processes the 'data_to_reject' matrix which contains boolean values indicating whether the data
        in corresponding time windows and channels are to be rejected. It updates the binary channel-time rejection
        matrix (BCT) to reflect these decisions.

        Parameters:
        - raw: An object that contains the raw EEG data.
        - data_to_reject: A boolean matrix indicating which data points are to be rejected based on correlation.
        - bct_time_windows: Time windows used for artifact detection.
        - i_t: Indices representing the limits of each time window.

        Returns:
        - BCT: A binary rejection matrix indicating bad (to be rejected) channels and times.
        """
        
        # Calculate the total number of points to be rejected
        rl_sum = np.sum(data_to_reject)

        # Display the percentage of data rejected due to low correlation
        n = np.prod(np.shape(bct_time_windows))
        print(f'\nData rejected due to low correlation: lower threshold {np.round(rl_sum / n * 100, 2)}%')

        # Determine the size of the raw EEG data for matrix initialization
        n_electrodes, n_samples, n_epochs = Raw.get_data_size(raw)

        # Initialize the binary channel-time rejection matrix (BCT) with False (indicating 'not rejected')
        BCT = np.full((n_epochs, n_electrodes, n_samples), False)
        
        # Retrieve the indices of the start and end of each time window
        Ti = i_t[0, :]  # Start times of the time windows
        Tf = i_t[-1, :]  # End times of the time windows

        # Process each epoch and electrode to update the BCT matrix with rejected segments
        for ep in np.arange(n_epochs):
            for el in np.arange(n_electrodes):
                # Get the rejection flags for the current electrode and epoch
                rl = data_to_reject[ep, el, :]
                # Determine the start and end times of the rejected segments
                til = Ti[rl]
                tfl = Tf[rl]
                # If there are multiple rejected segments, ensure they are properly delineated
                if len(tfl) > 1:
                    til, tfl = insert_initial_and_final_samples(til, tfl)
                # Mark the segments as rejected in the BCT matrix
                for j in np.arange(len(til)):
                    BCT[ep, el, til[j]:tfl[j] + 1] = True
        
        return BCT

    def reject_electrodes_based_on_corr(self, raw, time_window, time_window_step, thresh,
                                        top_channel_corr, use_relative_thresh=False):
        """
        Reject data from a raw EEG signal based on the correlation between electrodes within specified time windows.
        
        This method identifies segments of data to be rejected based on the correlation threshold criteria and 
        constructs a rejection matrix (BCT) which flags the bad data segments for potential exclusion from further analysis.

        Parameters:
        - raw: An instance of a raw EEG dataset.
        - time_window: Duration of the sliding time window in seconds.
        - time_window_step: Step size of the sliding window in seconds (stride).
        - thresh: The correlation threshold below which data is considered for rejection.
        - top_channel_corr: The percentage of top channel correlations to consider.
        - use_relative_thresh: Boolean flag to use a relative threshold (True) or a fixed threshold (False).

        Returns:
        - BCT: A boolean matrix indicating which segments of data are to be rejected (True) or retained (False).
        """

        # Step 1: Determine which data segments to reject based on electrode correlation
        data_to_reject, bct_time_windows, i_t = self.get_data_to_reject_based_on_corr(
            raw, time_window=time_window, time_window_step=time_window_step, thresh=thresh,
            top_channel_corr=top_channel_corr, use_relative_thresh=use_relative_thresh
        )

        # Step 2: Apply the rejection criteria to the data
        BCT = self.reject_bad_data_based_on_corr(raw, data_to_reject, bct_time_windows, i_t)

        return BCT


class Power:
    """
    A class for processing EEG data with an emphasis on artifact rejection based on power spectrum analysis.

    This class provides methods for EEG data processing, such as reference setting, z-score normalization,
    and artifact rejection based on frequency band power. It enables customization of processing through various
    parameters that control time window size for power calculation, artifact rejection thresholds, and masking around
    detected artifacts.

    Attributes:
        - params (dict): Dictionary containing processing parameters.
        - BCT (ndarray): Binary Correlation Threshold matrix indicating artifact locations in the EEG data.
        - mu (float or ndarray): Mean value(s) used for z-score normalization.
        - sd (float or ndarray): Standard deviation(s) used for z-score normalization.
        - n_rejected_data (int): Number of data points identified and rejected as artifacts.
        
    Args:
        - raw (object): Object containing raw EEG data.
        - time_window (int, optional): Duration of each time window for correlation calculations. Defaults to 4 seconds.
        - time_window_step (int, optional): Step size for moving the time window in correlation calculations. Defaults to 2 seconds.
        - top_channel_corr (int, optional): Percentage of top channel correlations to consider. Defaults to 5.
        - thresh (float, optional): Threshold below which data is considered bad and rejected. Defaults to 0.4.
        - bad_data (str, optional): Method for replacing bad data ('none', '0', 'nan'). Defaults to 'none'.
        - mask (int, optional): Time in seconds to mask data around detected artifacts. Defaults to 0.
        - do_reference_data (bool, optional): Flag for setting an EEG reference. Defaults to False.
        - do_zscore (bool, optional): Flag for computing z-score normalization. Defaults to False.
        - use_relative_thresh (bool, optional): Flag for using a relative threshold for artifact detection. Defaults to False.
        - update_BCT (bool, optional): Flag for updating artifacts comparison between new and previous BCTs. Defaults to True.
        - update_summary (bool, optional): Flag for printing status of bad samples in continuous data. Defaults to True.
        - update_algorithm (bool, optional): Flag for saving parameters related to the rejection matrix. Defaults to True.
        - config (bool, optional): Flag for using configuration to set parameters. Defaults to False.
        - name (str, optional): Name of the correlation-based rejection algorithm. Defaults to 'ChannelCorr'.
        - loop_name (list, optional): List of names specifying loops in configuration, if used. Defaults to an empty list.

    Methods:
        __init__(self, raw, time_window=4, time_window_step=2, top_channel_corr=5, thresh=0.4, 
            bad_data='none', mask=0, do_reference_data=False, do_zscore=False, use_relative_thresh=False,
            update_BCT=True, update_summary=True, update_algorithm=True, config=False,
            name='Power', loop_name=[]): 
        Initializes the Power object with the necessary attributes and begins data processing.
    """
    def __init__(self, raw, time_window=4, time_window_step=2, thresh=None, bad_data='none', 
                mask=0, freq_band=None, do_reference_data=False, do_zscore=True, 
                use_relative_thresh=None, update_BCT=True, update_summary=True, 
                update_algorithm=True, config=False, name='Power', loop_name=[]):
        """
        Initialize the ChannelCorr object which identifies and rejects bad raw EEG data based on 
        correlation between electrodes over specified time windows.
        """

        # Initialize a dictionary to hold parameters for the raw EEG data processing
        self.params = {
            'time_window': time_window, 
            'time_window_step': time_window_step, 
            'thresh': thresh, 
            'bad_data': bad_data, 
            'mask': mask, 
            'freq_band': freq_band,
            'do_reference_data': do_reference_data,
            'do_zscore': do_zscore, 
            'use_relative_thresh': use_relative_thresh if use_relative_thresh is not None else [True, True],
            'update_BCT': update_BCT, 
            'update_summary': update_summary, 
            'update_algorithm': update_algorithm,  
            'config': config,  
            'name': name, 
            'loop_name': loop_name 
        }

        # Set default frequency bands if not provided
        self.params['freq_band'] = self.params.get('freq_band', [[1, 10], [20, 40]])

        # Set default thresholds if not provided
        self.params['thresh'] = self.params.get('thresh', [[-3, np.inf], [-np.inf, 3]])


        # Get configuration (user-input parameters)
        if config:
            from apice.artifacts_rejection import update_parameters_with_user_inputs
            self.params = update_parameters_with_user_inputs(self.params, eval('apice.parameters.' + loop_name + '.' + name))

        # Output status of data rejection based on power spectrum
        print('\nRejecting data based on the power spectrum...')
        print('-- referenced data: ', self.params['do_reference_data'])
        print('-- z-score data: ', self.params['do_zscore'])
        print('-- relative threshold: ', self.params['use_relative_thresh'])

        # Initialize artifact rejection matrix
        raw.artifacts = Artifacts(raw)

        # Set reference to mean amplitude of channels
        if self.params['do_reference_data']:
            set_reference(raw, bad_data=self.params['bad_data'], save_reference=False)

        # Compute z-score for the artifacts
        if self.params['do_zscore']:
            raw._data, self.mu, self.sd = compute_z_score(raw)

        # Reject electrodes per (per time window) based on channels power spectrum
        BCT = self.reject_electrodes_based_on_power(raw, self.params['time_window'], self.params['time_window_step'],
                                                    self.params['freq_band'], self.params['thresh'],
                                                    self.params['use_relative_thresh'])
        # Update rejection matrix
        update_rejection_matrix(raw.artifacts, self.__class__.__name__, self.params, BCT)

        # Get data back
        if self.params['do_zscore']:
            return_data_after_zscore(raw, self.sd, self.mu)
        if self.params['do_reference_data']:
            return_data_after_referencing(raw)

        # Mask around artifacts
        if self.params['mask']:
            BCT = mask_around_artifacts(raw, mask_length=self.params['mask'])
            # Update rejection matrix
            update_rejection_matrix(raw.artifacts, self.__class__.__name__, self.params, BCT)

        # Save a copy of BCT
        self.BCT = BCT

        # Display the total rejected data
        self.n_rejected_data = np.round(np.sum(BCT))
        print(f'\nTotal rejected data: {self.n_rejected_data / np.size(BCT) * 100:.2f}%')

    @staticmethod
    def get_data_to_reject_based_on_power(band_power, bct_time_windows, thresh, use_relative_thresh=None):
        """
        Identify which data segments should be rejected based on the band power thresholds.

        Parameters
        ----------
        band_power : ndarray
            A 4D array containing the power within a specified frequency band for each time window and electrode.
        bct_time_windows : ndarray
            An array defining the time windows for which the Bad Channel Times (BCT) data segments have been calculated.
        thresh : list of tuples
            A list of tuples containing the upper and lower power thresholds for each frequency band. Each tuple is in the
            form (lower_threshold, upper_threshold).
        use_relative_thresh : list of bool, optional
            A list containing boolean values specifying whether to use a relative threshold for each frequency band when
            rejecting bad data based on power. If not provided, defaults to [True, True].

        Returns
        -------
        data_to_reject_upper_band : ndarray
            An array indicating the segments of data that exceed the upper power threshold and are considered bad.
        data_to_reject_lower_band : ndarray
            An array indicating the segments of data that fall below the lower power threshold and are considered bad.

        Notes
        -----
        The band power is computed for multiple frequency bands and the thresholding can be done using either relative
        or absolute values. If relative thresholding is used, the thresholds are calculated based on the inter-quartile
        range (IQR) of the band power distribution for each frequency band.
        """

        # Set default for `use_relative_thresh` if not provided
        use_relative_thresh = [True, True] if use_relative_thresh is None else use_relative_thresh

        # Extract the number of frequency bands from the shape of `band_power`
        n_freq_band = band_power.shape[2]

        # Create arrays filled with `False` to track upper and lower band rejections
        # The shape of the rejection arrays matches the shape of `band_power`
        R_upper = np.full(band_power.shape, False)
        R_lower = np.full(band_power.shape, False)

        # Loop through each frequency band to detect segments to reject
        for iband in range(n_freq_band):
            if use_relative_thresh:
                # Select the current frequency band and set invalid values to NaN
                RRi = band_power[:, :, iband, :].copy()
                RRi[bct_time_windows] = np.nan
                RRi[np.isinf(RRi)] = np.nan
                # Calculate the 25th, 50th, and 75th percentiles ignoring NaN values
                percentiles = np.nanpercentile(RRi, [25, 50, 75], interpolation='midpoint')
                # Compute the interquartile range (IQR)
                IQR = percentiles[2] - percentiles[0]
                # Determine lower and upper thresholds based on IQR
                lower_threshold = percentiles[0] + thresh[iband][0] * IQR
                upper_threshold = percentiles[2] + thresh[iband][1] * IQR
            else:
                # Use fixed thresholds if not using relative thresholds
                lower_threshold = thresh[iband][0]
                upper_threshold = thresh[iband][1]
            
            # Identify segments where the power exceeds the upper threshold
            R_upper[:, :, iband, :] = band_power[:, :, iband, :] > upper_threshold
            # Identify segments where the power is below the lower threshold
            R_lower[:, :, iband, :] = band_power[:, :, iband, :] < lower_threshold

        # Data to reject
        data_to_reject_upper_band = R_upper
        data_to_reject_lower_band = R_lower
        
        return data_to_reject_upper_band, data_to_reject_lower_band

    @staticmethod
    def get_power_spectrum_per_time_window(raw, time_window, i_t, freq_band):
        """
        Compute the power spectrum and average band power per time window for raw EEG data.

        Parameters
        ----------
        raw : MNE Raw object
            An object containing the raw EEG data and related information such as sampling frequency.
        time_window : float
            The length of the time window for which the power spectrum is computed, in seconds.
        i_t : ndarray
            An array with indices representing the start and end samples of each time window within the raw EEG data.
        freq_band : list of tuples
            A list where each tuple contains the lower and upper frequencies of the band of interest, in Hz.

        Returns
        -------
        Power : ndarray
            A 4D array with shape (n_epochs, n_electrodes, n_frequencies, n_time_windows) containing the power spectrum
            for each epoch, electrode, frequency, and time window.
        Power_band : ndarray
            A 4D array with shape (n_epochs, n_electrodes, n_freq_bands, n_time_windows) containing the average band power
            for each epoch, electrode, frequency band, and time window.

        Notes
        -----
        This function uses the Fast Fourier Transform (FFT) to compute the power spectrum. Power for each frequency band
        is computed by averaging the power spectrum across the frequencies within that band. Log transformation is applied
        to the power spectrum, and then it is normalized by subtracting the median power for the baseline.
        """

        # Define parameters for computing the power spectrum
        sampling_frequency = raw.info['sfreq'] 
        n_samples_for_fft = int(time_window * sampling_frequency)  
        
        # Generate frequency values for FFT output
        freq = np.linspace(0, sampling_frequency / 2, n_samples_for_fft // 2, endpoint=False)
        n_freq = len(freq) 
        n_freq_band = len(freq_band) 
        n_time_window = len(i_t[0]) 
        
        # Get the dimensions of the EEG data array
        n_electrodes, n_samples, n_epochs = Raw.get_data_size(raw) 

        # Get time series data
        raw_data = raw._data.copy()
        raw_data = np.reshape(raw_data, (n_epochs, n_electrodes, n_samples))

        # Initialize arrays to hold power spectrum data
        # 'Power' will contain the full power spectrum for each epoch, electrode, frequency, and time window
        Power = np.empty((n_epochs, n_electrodes, n_freq, n_time_window))
        Power[:] = np.nan
        
        # 'Power_band' will hold the average power within each specified frequency band
        Power_band = np.empty((n_epochs, n_electrodes, n_freq_band, n_time_window))
        Power_band[:] = np.nan

        # Compute the power spectrum
        for ep in np.arange(n_epochs):
            for el in np.arange(n_electrodes):
                # Get data per epoch and electrode
                data = raw_data[ep, el, :].copy()
                # Extract data per time window
                data_per_time_window = data[i_t]
                # Reference data to mean
                data_per_time_window_mean = np.mean(data_per_time_window, axis=0)
                mean_reference = np.tile(data_per_time_window_mean, (n_samples_for_fft, 1))
                # Fourier Transform
                data_for_fft = data_per_time_window - mean_reference
                fD = np.fft.fft(data_for_fft, n=n_samples_for_fft, axis=0, norm=None)
                # Compute for power
                P2 = np.square(np.abs(fD)) / (n_samples_for_fft * sampling_frequency)
                # Single sideband power
                Power[ep, el, :, :] = P2[0:int(np.floor(n_samples_for_fft / 2)), :]

        # Compute for the power in each frequency band
        for i in np.arange(n_freq_band):
            i_band = np.logical_and((freq > freq_band[i][0]), (freq <= freq_band[i][1]))
            p = Power[:, :, i_band, :].copy()
            p_band = np.log10(np.mean(p, axis=2))
            p_base = np.nanmedian(p_band)
            p_band = p_band - p_base
            Power_band[:, :, i, :] = p_band

        return Power, Power_band

    @staticmethod
    def reject_bad_data_based_on_power(raw, data_to_reject_upper_band, data_to_reject_lower_band, i_t, freq_band):
        """
        Reject segments of raw EEG data that exceed defined power thresholds in specified frequency bands.

        The method processes the binary arrays indicating whether data points exceed the upper or lower
        power thresholds. These arrays are used to construct a bad channel-time matrix (BCT) that identifies
        the time samples to be rejected for each channel.

        Parameters
        ----------
        raw : Raw
            The raw MNE-Python raw EEG object containing the raw EEG data and related information.
        data_to_reject_upper_band : ndarray
            A boolean array indicating data points exceeding the upper power threshold.
        data_to_reject_lower_band : ndarray
            A boolean array indicating data points falling below the lower power threshold.
        i_t : ndarray
            An array containing the indices of the initial and final time samples of each time window.
        freq_band : list of tuples
            A list of tuples specifying the frequency bands for which to evaluate the power spectrum.

        Returns
        -------
        BCT : ndarray
            A boolean array marking the bad data segments for each channel and time sample.
        """
        
        # Display rejected data information
        n_electrodes, n_samples, n_epochs = Raw.get_data_size(raw)
        n_time_window = np.shape(data_to_reject_upper_band)[3]
        n = n_electrodes * n_time_window * n_epochs
        n_freq_band = len(freq_band)
        
        # Iterate through each frequency band to report rejected data statistics
        for iband in np.arange(n_freq_band):
            # Compute the number of artifacts in the upper and lower bands
            ru_sum = np.sum(data_to_reject_upper_band[:, :, iband, :])
            rl_sum = np.sum(data_to_reject_lower_band[:, :, iband, :])
            # Print out the percentage of data rejected in each band
            print(f'\nData rejected in frequency band {freq_band[iband]}:')
            print(f'--- Upper threshold: {np.round(ru_sum / n * 100, 2)}%')
            print(f'--- Lower threshold: {np.round(rl_sum / n * 100, 2)}%')

        # Initialize bad channel-time matrices for upper and lower rejection thresholds
        BCT_upper = np.full((n_epochs, n_electrodes, n_samples, n_freq_band), False)
        BCT_lower = np.full((n_epochs, n_electrodes, n_samples, n_freq_band), False)

        # Get the initial and final times of each time window
        Ti = i_t[0, :]  # Initial times
        Tf = i_t[-1, :]  # Final times

        # Process data to reject based on upper and lower bands
        for iband in range(n_freq_band):
            for ep in range(n_epochs):
                for el in range(n_electrodes):
                    # Get the indices of time windows to reject for this band, epoch, and electrode
                    ru_indices  = data_to_reject_upper_band[ep, el, iband, :]
                    rl_indices = data_to_reject_lower_band[ep, el, iband, :]
                    # Convert indices to actual time samples
                    tiu = Ti[ru_indices]
                    tfu = Tf[ru_indices]
                    til = Ti[rl_indices]
                    tfl = Tf[rl_indices]
                    # If there is more than one sample, process to ensure continuity
                    if len(tfu) > 1:
                        tiu, tfu = insert_initial_and_final_samples(tiu, tfu)
                    if len(tfl) > 1:
                        til, tfl = insert_initial_and_final_samples(til, tfl)
                    # Set the corresponding time samples in BCT_upper and BCT_lower to True for rejection
                    for j in range(len(tiu)):
                        BCT_upper[ep, el, tiu[j]:tfu[j], iband] = True
                    for j in range(len(til)):
                        BCT_lower[ep, el, til[j]:tfl[j], iband] = True

        # Combine the upper and lower bad channel-time matrices to get the final BCT
        # by taking the logical OR across all frequency bands
        BCT = np.logical_or(np.any(BCT_upper, axis=3), np.any(BCT_lower, axis=3))
        
        return BCT

    def reject_electrodes_based_on_power(self, raw, time_window, time_window_step, freq_band, thresh,
                                        use_relative_thresh=None):
        """
        Rejects segments of raw EEG data based on power spectrum thresholds in specified frequency bands.

        Parameters
        ----------
        raw : MNE Raw object
            The object containing the raw EEG data and related information such as sampling frequency.
        data_to_reject_upper_band : ndarray
            A boolean array with the same shape as the raw EEG data indicating which data points exceed
            the upper power threshold in their respective frequency band.
        data_to_reject_lower_band : ndarray
            A boolean array with the same shape as the raw EEG data indicating which data points fall below
            the lower power threshold in their respective frequency band.
        i_t : ndarray
            An array with indices representing the start and end samples of each time window within the raw EEG data.
        freq_band : list of tuples
            A list where each tuple contains the lower and upper frequencies of the band of interest, in Hz.

        Returns
        -------
        BCT : ndarray
            A boolean rejection matrix with the same number of epochs and electrodes as the raw EEG data, indicating
            whether a data point should be rejected (True) or not (False).

        Notes
        -----
        The function prints the percentage of data rejected in each frequency band due to surpassing the upper
        or lower power thresholds. The rejection is based on entire time windows, and the output matrix 'BCT' can
        be used to mask the raw EEG data array, effectively removing the segments of data considered to be artifacts.
        """
        
        # Ensure the use_relative_thresh variable is set; default to [True, True] if not provided
        use_relative_thresh = [True, True] if use_relative_thresh is None else use_relative_thresh

        # Define time windows
        bct_time_windows, i_t, n_time_window = define_time_window(raw, time_window, time_window_step)

        # Compute the power spectrum
        _, band_power = self.get_power_spectrum_per_time_window(raw, time_window, i_t, freq_band)

        # Determine which data segments to reject based on power thresholds
        data_to_reject_upper_band, data_to_reject_lower_band = self.get_data_to_reject_based_on_power(band_power, 
                                                                                                        bct_time_windows,
                                                                                                        thresh, 
                                                                                                        use_relative_thresh)
        
        # Execute the rejection of bad data segments from the EEG based on power criteria
        BCT = self.reject_bad_data_based_on_power(raw, 
                                                    data_to_reject_upper_band, 
                                                    data_to_reject_lower_band,
                                                    i_t, 
                                                    freq_band)
        return BCT


class ShortGoodSegments:
    """
    A class to identify and reject segments of raw EEG data that are shorter than a specified time limit.

    This class processes raw EEG data to find 'good' segments, i.e., segments that are not marked as artifacts,
    which are shorter than a defined time limit and flags them for rejection. This can be useful in scenarios
    where very short segments of good data are not considered reliable for analysis.

    Attributes:
    -----------
    params : dict
        Parameters for the EEG data processing including time limit, update flags and configurations.
    BCT : ndarray
        The Binary Contamination Threshold matrix after processing, indicating the rejected segments.
    n_rejected_data : int
        The number of data points that were flagged for rejection.

    Args:
    ----------
    raw : Raw
        The MNE Raw object containing the EEG data to be processed.
    time_limit : float, optional
        The time limit in seconds for the minimum duration of good segments, default is 2 seconds.
    update_BCT : bool, optional
        Flag indicating whether to update the binary correlation threshold matrix, default is True.
    update_summary : bool, optional
        Flag indicating whether to update and display a summary of the artifact rejection, default is False.
    update_algorithm : bool, optional
        Flag indicating whether to save the current algorithm state, default is False.
    config : bool, optional
        Flag indicating whether to use a pre-defined configuration for the analysis parameters, default is False.
    name : str, optional
        The name assigned to the instance of the ShortGoodSegments object, default is 'ShortGoodSegments'.
    loop_name : list, optional
        The list of names specifying loops in the configuration, default is an empty list.
            
    Methods:
    --------
    __init__(self, raw, time_limit=2., update_BCT=True, update_summary=False, update_algorithm=False, config=False,
            name='ShortGoodSegments', loop_name=[]):
        Constructor for the ShortGoodSegments class. Initializes data processing and segment rejection.

    reject_short_good_segments(raw, time_limit):
        Static method that processes the EEG data to reject short good segments based on a time limit.
    """
    
    def __init__(self, raw, time_limit=2., update_BCT=True, update_summary=False, update_algorithm=False, config=False,
                name='ShortGoodSegments', loop_name=[]):
        """
        Initializes the ShortGoodSegments class with parameters for EEG data processing.
        """

        # Initialize parameters for the ShortGoodSegments object.
        self.params = dict(time_limit=time_limit, update_BCT=update_BCT, update_summary=update_summary,
                            update_algorithm=update_algorithm)

        # Get configuration (user-input parameters)
        if config:
            from apice.artifacts_rejection import update_parameters_with_user_inputs
            self.params = update_parameters_with_user_inputs(self.params,
                                                            eval('apice.parameters.' + loop_name + '.' + name))

        print('\nRejecting short good segments...')

        # Initialize the artifact rejection matrix for the given EEG data.
        raw.artifacts = Artifacts(raw)

        # Call the method to reject short good segments and update the BCT matrix accordingly.
        BCT, BCT_input = self.reject_short_good_segments(raw, self.params['time_limit'])

        # Update the EEG's artifact rejection matrix with results from this class's processing.
        update_rejection_matrix(raw.artifacts, self.__class__.__name__, self.params, BCT)

        # Save a copy of the updated BCT matrix.
        self.BCT = BCT

        # Display the percentage of the total data rejected after processing.
        self.n_rejected_data = np.sum(BCT & np.logical_not(BCT_input))
        print(f'\nTotal rejected data: {np.round(self.n_rejected_data / np.size(BCT) * 100, 2)} %')

    @staticmethod
    def reject_short_good_segments(raw, time_limit):
        """
        Identifies and rejects short good segments from the EEG data.

        Parameters:
        ----------
        raw : Raw
            The MNE Raw object containing the EEG data to be processed.
        time_limit : float
            The time limit in seconds for the minimum duration of good segments.

        Returns:
        -------
        tuple
            A tuple containing the updated BCT matrix and the original BCT matrix before processing.
        """

        # Retrieve the dimensions of the data from the EEG object.
        n_electrodes, n_samples, n_epochs = Raw.get_data_size(raw)
        
        # Copy the current state of the BCT matrix and prepare a new one for processing.
        BCT_input = raw.artifacts.BCT.copy()
        BCT = np.full((n_epochs, n_electrodes, n_samples), False)
        
        # Convert the time limit from seconds to data samples.
        time_limit = np.round(time_limit * raw.info['sfreq'])
        
        # Ensure the time limit does not exceed the number of samples available.
        if n_samples <= time_limit:
            time_limit = n_samples - 1
        
        # Iterate over epochs and electrodes to reject short good segments.
        for ep in np.arange(n_epochs):
            bct_data = BCT_input[ep, :, :].copy()
            for el in np.arange(n_electrodes):
                good_data_time = np.asarray(~bct_data[el, :], dtype=int)
                # Get the start of the good segment
                temp1 = good_data_time[0]
                temp2 = np.asarray(good_data_time[1:] - good_data_time[:- 1] == 1, dtype=int)
                initial_good_data = np.zeros(len(good_data_time))
                initial_good_data[0] = temp1
                initial_good_data[1:] = temp2
                initial_good_data = np.where(initial_good_data)[0]
                # Get the end of the good segment
                temp3 = np.asarray(good_data_time[:- 1] - good_data_time[1:] == 1, dtype=int)
                temp4 = good_data_time[-1]
                final_good_data = np.zeros(len(good_data_time))
                final_good_data[:-1] = temp3
                final_good_data[-1] = temp4
                final_good_data = np.where(final_good_data)[0]
                # Get segment length
                if np.size(initial_good_data) > 0:
                    good_data_length = (final_good_data - initial_good_data + 1)
                    short_segment = np.where(good_data_length <= time_limit)[0]
                    if np.size(short_segment) > 0:
                        for i in np.arange(np.size(short_segment)):
                            ind = np.arange(initial_good_data[int(short_segment[i])], final_good_data[int(short_segment[i])] + 1)
                            bct_data[el, ind] = True
            BCT[ep, :, :] = np.logical_and(bct_data, np.logical_not(BCT_input[ep, :, :]))
            
        return BCT, BCT_input


class Amplitude:
    """
    A class to reject EEG data based on amplitude thresholds.

    The class handles the detection and rejection of artifacts in EEG data based on the amplitude exceeding
    specified thresholds. It supports absolute and relative thresholds, optionally on a per-electrode basis,
    and can perform the analysis on raw or z-scored data.

    Attributes:
    -----------
    params : dict
        Configuration parameters for artifact detection and rejection.
    BCT : ndarray
        The binary contamination matrix indicating which data points are rejected based on amplitude.
    n_rejected_data : int
        The number of data points that have been rejected.
    mu : ndarray, optional
        The mean value used for z-scoring, if z-scoring is applied.
    sd : ndarray, optional
        The standard deviation used for z-scoring, if z-scoring is applied.

    Args:
    -----------
    raw : instance of Raw
        The pre-loaded Raw EEG data to be processed.
    thresh : int or float, optional
        The amplitude threshold value above which data is considered bad.
    bad_data : str, optional
        A string indicating how to deal with bad data segments.
    mask : float, optional
        The duration (in seconds) to mask the EEG signal around artifact regions.
    do_reference_data : bool, optional
        If True, will reference the data before processing.
    do_zscore : bool, optional
        If True, will z-score the data before processing.
    use_relative_thresh : bool, optional
        If True, thresholds are relative to the distribution of the data.
    use_relative_thresh_per_electrode : bool, optional
        If True, apply relative thresholds per electrode.
    update_BCT : bool, optional
        If True, update the binary contamination matrix.
    update_summary : bool, optional
        If True, update the summary of rejected segments.
    update_algorithm : bool, optional
        If True, update the algorithm status.
    config : bool, optional
        If True, user configuration will be fetched and applied.
    name : str, optional
        The name identifier for this instance.
    loop_name : str, optional
        The name identifier for the processing loop, if applicable.
            
    Methods:
    --------
    __init__(self, raw, **kwargs):
        Constructs the Amplitude object, initializing the parameters for artifact detection, and
        processes the EEG data to reject artifacts based on amplitude.

    reject_data_based_on_signal_amplitude(raw, params):
        Static method that rejects data points in the EEG data where the signal amplitude exceeds
        the threshold(s) defined in params.
    """
    
    def __init__(self, raw, thresh=3, bad_data='none', mask=0.05, do_reference_data=False, do_zscore=False,
                use_relative_thresh=False, use_relative_thresh_per_electrode=False, update_BCT=True,
                update_summary=True, update_algorithm=True, config=False, name='Amplitude', loop_name=None):
        
        """
        Initializes the Amplitude object with EEG data and processing parameters.
        """

        # Initialize parameters dictionary with provided values and defaults.
        self.params = {
            'thresh': thresh,
            'bad_data': bad_data,
            'mask': mask,
            'do_reference_data': do_reference_data,
            'do_zscore': do_zscore,
            'use_relative_thresh': use_relative_thresh,
            'use_relative_thresh_per_electrode': use_relative_thresh_per_electrode,
            'update_BCT': update_BCT,
            'update_summary': update_summary,
            'update_algorithm': update_algorithm,
            'loop_name': loop_name,
            'name': name
        }

        # Get configuration (user-input parameters)
        if config:
            from apice.artifacts_rejection import update_parameters_with_user_inputs
            self.params = update_parameters_with_user_inputs(self.params, eval('apice.parameters.' + loop_name + '.' + name))

        # Configure threshold parameters
        self.params = configure_threshold_parameters(self.params, name='Amplitude')

        # Print rejection algorithm, parameters
        print(f"\nRejecting data based on the signal amplitude...")
        print(f"-- referenced data: {self.params['do_reference_data']}")
        print(f"-- z-score data: {self.params['do_zscore']}")
        print(f"-- relative threshold: {np.squeeze(self.params['use_relative_thresh'])}")
        print(f"-- relative threshold per electrode: {np.squeeze(self.params['use_relative_thresh_per_electrode'])}")

        # Initialize artifact rejection matrix
        raw.artifacts = Artifacts(raw)

        # Set reference to mean amplitude of channels
        if self.params['do_reference_data']:
            set_reference(raw, bad_data=self.params['bad_data'], save_reference=False)

        # Compute z-score for the artifacts
        if self.params['do_zscore']:
            raw._data, self.mu, self.sd = compute_z_score(raw)

        # Reject data based on the signal amplitude
        BCT = self.reject_data_based_on_signal_amplitude(raw, self.params)

        # Mask around artifacts
        if self.params['mask']:
            print(f"\n--Masking around artifacts: mask length {self.params['mask']} s")
            BCT = mask_around_artifacts_BCT(BCT, self.params['mask'], raw.info['sfreq'])

        # Update rejection matrix
        update_rejection_matrix(raw.artifacts, self.__class__.__name__, self.params, BCT)

        # Get data back
        if self.params['do_zscore']:
            return_data_after_zscore(raw, self.sd, self.mu)
        if self.params['do_reference_data']:
            return_data_after_referencing(raw)

        # Save a copy of BCT
        self.BCT = BCT

        # Display the total rejected data
        self.n_rejected_data = np.round(np.sum(BCT))
        print(f'\nTotal rejected data: {np.round(self.n_rejected_data / np.size(BCT) * 100, 2)} %')

    @staticmethod
    def reject_data_based_on_signal_amplitude(raw, params):
        """
        Reject epochs of EEG data that exceed certain amplitude thresholds.
        
        This function calculates the rejection threshold either globally or for each electrode
        based on the 75th percentile of the amplitude distribution, plus a multiple of the
        interquartile range (IQR). Epochs with amplitudes exceeding this threshold are marked for rejection.
        
        Parameters:
        ----------
        raw : instance of Raw
            The Raw EEG object containing the data to be processed.
        params : dict
            A dictionary containing parameters for the rejection process. Keys should include
            'use_relative_thresh' to determine if thresholds are relative, 'thresh' to specify
            the multiple of the IQR to be used as threshold, and 'use_relative_thresh_per_electrode'
            to specify if thresholds should be calculated for each electrode individually.
        
        Returns:
        -------
        numpy.ndarray
            A binary contamination matrix (BCT) indicating which epochs have been rejected.
        """
        
        # Retrieve the size of the data to determine how many electrodes, samples, and epochs we have.
        n_electrodes, n_samples, n_epochs = Raw.get_data_size(raw)
        
        # Initialize a matrix to store thresholds for each electrode and each condition.
        n_relative_thresh = np.size(params['use_relative_thresh'])
        threshold_matrix = np.empty((n_electrodes, n_relative_thresh))
        threshold_matrix[:] = np.nan
        
        # Create a boolean matrix to mark data that will be rejected.
        data_to_reject = np.full((n_epochs, n_electrodes, n_samples), False)

        # Copy and reshape the raw data for processing.
        raw_data = np.reshape(raw._data.copy(), (n_epochs, n_electrodes, n_samples))  # * 1e6
        BCT = raw.artifacts.BCT.copy()

        # Iterate over all conditions for relative thresholding.
        for i in np.arange(n_relative_thresh):
            # If using a relative threshold for this condition...
            if params['use_relative_thresh'][i]:
                # If thresholds are calculated per electrode...
                if params['use_relative_thresh_per_electrode'][i]:
                    # Initialize the sum of rejected data.
                    ru_sum = 0  # upper threshold
                    # Iterate over electrodes
                    for el in np.arange(n_electrodes):
                        data = np.abs(raw_data[:, el, :].copy())
                        data[BCT[:, el, :]] = np.nan
                        perc = np.nanpercentile(data, 75, interpolation='midpoint')
                        # Get the half distribution centered at zero
                        IQ = 2 * perc
                        t_u_el = perc + params['thresh'][i] * IQ
                        data_to_reject[:, el, :] = data_to_reject[:, el, :] | (np.abs(raw_data[:, el, :]) > t_u_el)
                        threshold_matrix[el, i] = t_u_el
                        ru_sum = ru_sum + np.sum(np.abs(raw_data[:, el, :]) > t_u_el)
                else:
                    # Global threshold for all electrodes.
                    data = np.abs(raw_data)
                    perc = np.percentile(data[np.logical_not(BCT)], 75, interpolation='midpoint')
                    IQ = 2 * perc
                    t_u = perc + params['thresh'][i] * IQ
                    data_to_reject = np.logical_or(data_to_reject, (np.abs(raw_data) > t_u))
                    threshold_matrix[:, i] = t_u
                    ru_sum = np.sum(np.abs(raw_data) > t_u)
            else:
                # Fixed threshold not based on data distribution.
                t_u = params['thresh'][i]
                data_to_reject = data_to_reject | (np.abs(raw_data) > t_u)
                threshold_matrix[:, i] = t_u
                ru_sum = np.sum(np.abs(raw_data) > t_u)

            # Calculate and display the percentage of rejected data.
            n = n_epochs * n_electrodes * n_samples
            print(f'\nData rejected based on signal amplitude: upper threshold {np.round(ru_sum / n * 100, 2)}%')

        # Update the BCT with the data to be rejected.
        BCT = data_to_reject
        
        return BCT


class TimeVariance:
    """
    A class for rejecting EEG data based on the time variance method.

    Attributes:
    -----------
        params (dict): A dictionary of parameters for the time variance method.
        BCT (np.ndarray): Binary Contamination Matrix indicating where artifacts are detected.
        n_rejected_data (int): Number of data points rejected.
        mu (float): Mean of the data, used for z-scoring.
        sd (float): Standard deviation of the data, used for z-scoring.

    Parameters:
    -----------
        raw (Raw object): The EEG data to process.
        time_window (float): Length of the sliding time window in seconds.
        time_window_step (float): Step size of the sliding time window in seconds.
        thresh (list): Threshold values for artifact detection.
        bad_data (str): Strategy for handling bad data.
        mask (int): The number of seconds to mask around detected artifacts.
        do_reference_data (bool): Whether to apply a reference to the EEG data.
        do_zscore (bool): Whether to apply z-score normalization to the EEG data.
        use_relative_thresh (bool): Whether to use relative thresholding for artifact detection.
        use_relative_thresh_per_electrode (bool): Whether to apply relative thresholding per electrode.
        update_BCT (bool): Whether to update the Binary Contamination Matrix.
        update_summary (bool): Whether to update the summary of rejected data.
        update_algorithm (bool): Whether to update the algorithm parameters.
        config (bool): Whether to use external configuration for parameters.
        name (str): Name identifier for the artifact rejection method.
        loop_name (str or None): Name of the loop if used in iterative processing.

    Methods:
    ---------
        __init__(self, raw, time_window=0.5, time_window_step=0.1, thresh=None, bad_data='none', mask=0,
                    do_reference_data=False, do_zscore=False, use_relative_thresh=True,
                    use_relative_thresh_per_electrode=True, update_BCT=True, update_summary=True,
                    update_algorithm=True, config=False, name='TimeVariance', loop_name=None)
            Initializes the TimeVariance object with the given EEG data and parameters.

        reject_electrodes_based_on_time_variance(raw): Main method to apply the time variance artifact rejection.
    """

    def __init__(self, raw, time_window=0.5, time_window_step=0.1, thresh=None, bad_data='none', mask=0,
                do_reference_data=False, do_zscore=False, use_relative_thresh=True,
                use_relative_thresh_per_electrode=True, update_BCT=True, update_summary=True,
                update_algorithm=True, config=False, name='TimeVariance', loop_name=None):
        """
        Initialize the TimeVariance object with EEG data and processing parameters.
        """
        
        # Simplified default parameter setting for 'loop_name' and 'thresh'
        # If 'loop_name' is not provided, default to an empty list
        loop_name = loop_name or []

        # If 'thresh' is not provided, default to a list with values [-3, 3]
        thresh = thresh or [-3, 3]

        # Initialize a dictionary to store all the parameters
        self.params = {
            'time_window': time_window,
            'time_window_step': time_window_step,  
            'thresh': thresh,  
            'bad_data': bad_data, 
            'mask': mask,  
            'do_reference_data': do_reference_data, 
            'do_zscore': do_zscore,  
            'use_relative_thresh': use_relative_thresh, 
            'use_relative_thresh_per_electrode': use_relative_thresh_per_electrode, 
            'update_BCT': update_BCT, 
            'update_summary': update_summary,  
            'update_algorithm': update_algorithm,  
            'loop_name': loop_name 
        }

        # Get configuration (user-input parameters)
        if config:
            from apice.artifacts_rejection import update_parameters_with_user_inputs
            self.params = update_parameters_with_user_inputs(self.params, eval('apice.parameters.' + loop_name + '.' + name))

        # Configure threshold parameters
        self.params = configure_threshold_parameters(self.params, name='TimeVariance')

        # Print out the rejection settings for the data all at once
        print(
            '\nRejecting data based on the time variance...\n'
            f"-- referenced data: {self.params['do_reference_data']}\n"
            f"-- z-score data: {self.params['do_zscore']}\n"
            f"-- relative threshold: {np.squeeze(self.params['use_relative_thresh'])}\n"
            f"-- relative threshold per electrode: {np.squeeze(self.params['use_relative_thresh_per_electrode'])}"
        )

        # Initialize artifact rejection matrix
        raw.artifacts = Artifacts(raw)

        # Set reference to mean amplitude of channels
        if self.params['do_reference_data']:
            set_reference(raw, bad_data=self.params['bad_data'], save_reference=False)

        # Compute z-score for the artifacts
        if self.params['do_zscore']:
            raw._data, self.mu, self.sd = compute_z_score(raw)

        # Reject data based on the time variance
        BCT, BCT_upper, BCT_lower = self.reject_electrodes_based_on_time_variance(raw)

        # Apply a mask to the artifact time points if specified in the parameters.
        if self.params['mask']:
            print(f"\n--Masking around artifacts: mask length {self.params['mask']} s")
            BCT = mask_around_artifacts_BCT(BCT, self.params['mask'], raw.info['sfreq'])

        # Update rejection matrix
        update_rejection_matrix(raw.artifacts, self.__class__.__name__, self.params, BCT)

        # Get data back
        if self.params['do_zscore']:
            return_data_after_zscore(raw, self.sd, self.mu)
        if self.params['do_reference_data']:
            return_data_after_referencing(raw)

        # Save a copy of the Bad Channel Times (BCT) array as an attribute of the class instance
        self.BCT = BCT

        # Calculate the percentage of data rejected by upper and lower thresholds
        rejected_upper_percentage = np.round(np.sum(BCT_upper) / np.size(BCT) * 100, 2)
        rejected_lower_percentage = np.round(np.sum(BCT_lower) / np.size(BCT) * 100, 2)
        total_rejected_percentage = np.round(np.sum(BCT) / np.size(BCT) * 100, 2)

        # Store the total number of rejected data points
        self.n_rejected_data = np.round(np.sum(BCT))

        # Print out the percentages of rejected data
        print(f'\nData rejected based on time variance:'
            f'\n--- upper threshold: {rejected_upper_percentage}%'
            f'\n--- lower threshold: {rejected_lower_percentage}%'
            f'\n\nTotal rejected data: {total_rejected_percentage}%'
            )

    @staticmethod
    def compute_time_variability(raw, n_time_window, i_t):
        """
        Computes the time variability within the defined time windows.

        Parameters:
            raw (Raw object): The EEG data to process.
            n_time_window (int): The number of time windows to consider.
            i_t (np.ndarray): Indices representing the time windows.
        
        Returns:
            np.ndarray: An array of standard deviation values representing the time variability.
        """
        # Get the dimensions of the EEG data.
        n_electrodes, n_samples, n_epochs = Raw.get_data_size(raw)

        # Reshape the raw data to facilitate processing.
        # It's formatted as epochs x electrodes x samples.
        raw_data = np.reshape(raw._data.copy(), (n_epochs, n_electrodes, n_samples))

        # Initialize a matrix to hold the standard deviation values, filled with NaNs.
        STD = np.full((n_epochs, n_electrodes, n_time_window), np.nan)

        # Loop through each epoch and electrode to compute the standard deviation within each time window.
        for ep in range(n_epochs):
            for el in range(n_electrodes):
                # Extract the data for the current electrode and epoch.
                data = raw_data[ep, el, :]
                
                # Select the relevant time window from the data.
                data_time_window = data[i_t]
                
                # Check if there is enough non-NaN data to compute the standard deviation.
                # If there are more than two non-NaN values, compute the standard deviation.
                if np.count_nonzero(~np.isnan(data_time_window)) > 2:
                    STD[ep, el, :] = np.nanstd(data_time_window, axis=0)

        return STD

    @staticmethod
    def get_data_to_reject_based_on_time_variance(STD, bct_time_windows, params):
        """
        Determines which data points to reject based on time variance thresholds.

        Parameters:
            STD (np.ndarray): An array of standard deviation values representing time variability.
            bct_time_windows (np.ndarray): Array representing the bad time windows.
            params (dict): Parameters for the time variance method.

        Returns:
            tuple: Arrays indicating data points to reject based on upper and lower thresholds, and the threshold matrix.
        """
        # Determine the shape of the standard deviation array.
        n_epoch, n_electrodes, n_time_window = np.shape(STD)

        # Get the number of relative thresholds specified in the parameters.
        n_relative_thresh = len(params['use_relative_thresh'])

        # Initialize a matrix to store thresholds for each electrode. It has a dimension for 
        # the number of electrodes, two for upper and lower thresholds, and the number of relative thresholds.
        threshold_matrix = np.zeros((n_electrodes, 2, n_relative_thresh))

        # Create boolean arrays to mark where data will be rejected based on upper and lower thresholds.
        # These arrays are initially filled with False, indicating no data is rejected yet.
        data_to_reject_upper_thresh = np.full(np.shape(STD), False)
        data_to_reject_lower_thresh = np.full(np.shape(STD), False)

        # Iterate over thresholds
        for i in np.arange(n_relative_thresh):
            
            # If relative threshold is to be used.
            if [params['use_relative_thresh']][i]:
                STD_ = np.log(np.divide(STD, np.median(STD)))
                
                # If the absolute threshold is per electrode
                if [params['use_relative_thresh_per_electrode']][i]:
                    # Loop over each electrode to calculate rejection thresholds and mark data for rejection.
                    for el in np.arange(n_electrodes):
                        # Extract the standard deviation for the current electrode and set artifact-contaminated windows to NaN.
                        STD_el = STD_[:, el, :].copy()
                        STD_el[bct_time_windows[:, el, :]] = np.nan
                        # Calculate the 25th, 50th, and 75th percentiles of the standard deviation, ignoring NaN values.
                        perc = np.nanpercentile(STD_el, [25, 50, 75], interpolation='midpoint')
                        # Calculate the interquartile range (IQR).
                        IQ = perc[2] - perc[0]
                        # Determine the lower and upper thresholds.
                        t_l_el = perc[0] + params['thresh'][0] * IQ
                        t_u_el = perc[2] + params['thresh'][1] * IQ
                        # Update the boolean arrays marking data for rejection based on these calculated thresholds.
                        data_to_reject_upper_thresh[:, el, :] = np.logical_or(data_to_reject_upper_thresh[:, el, :], (STD_[:, el, :] > t_u_el))
                        data_to_reject_lower_thresh[:, el, :] = np.logical_or(data_to_reject_lower_thresh[:, el, :], (STD_[:, el, :] < t_l_el))
                        # Store the calculated thresholds in the threshold_matrix for later use.
                        threshold_matrix[el, 0, i] = t_u_el
                        threshold_matrix[el, 1, i] = t_l_el
                else:
                    perc = np.nanpercentile(STD_[np.logical_not(bct_time_windows)], [25, 50, 75], interpolation='midpoint')
                    IQ = perc[2] - perc[0]
                    t_l = perc[0] + params['thresh'][0] * IQ
                    t_u = perc[2] + params['thresh'][1] * IQ
                    # Data to reject
                    data_to_reject_upper_thresh = np.logical_or(data_to_reject_upper_thresh, (STD_ > t_u))
                    data_to_reject_lower_thresh = np.logical_or(data_to_reject_lower_thresh, (STD_ < t_l))
                    # Thresholds
                    threshold_matrix[:, 0, i] = t_l
                    threshold_matrix[:, 1, i] = t_u
            # If absolute threshold is to be used.
            else:
                t_l = params['thresh'][0]
                t_u = params['thresh'][1]
                # Data to reject
                data_to_reject_upper_thresh = np.logical_or(data_to_reject_upper_thresh, (STD > t_u))
                data_to_reject_lower_thresh = np.logical_or(data_to_reject_lower_thresh, (STD < t_l))
                # Thresholds
                threshold_matrix[:, 0, i] = t_l
                threshold_matrix[:, 1, i] = t_u
                
        return data_to_reject_upper_thresh, data_to_reject_lower_thresh, threshold_matrix

    @staticmethod
    def reject_bad_data_based_on_time_variance(data_to_reject_upper_thresh, data_to_reject_lower_thresh, data_size, i_t):
        """
        Rejects the bad data based on the previously calculated thresholds.

        Parameters:
            data_to_reject_upper_thresh (np.ndarray): Data points to reject above the upper threshold.
            data_to_reject_lower_thresh (np.ndarray): Data points to reject below the lower threshold.
            data_size (tuple): Size of the EEG data (epochs, electrodes, samples).
            i_t (np.ndarray): Indices representing the time windows.

        Returns:
            tuple: Binary matrices indicating rejected data points for upper, lower, and combined thresholds.
        """
        
        # Define the size of the data.
        n_epochs, n_electrodes, n_samples = data_size

        # Initialize boolean arrays for tracking data to be rejected above and below the threshold.
        BCT_upper = np.full((n_epochs, n_electrodes, n_samples), False)
        BCT_lower = np.full((n_epochs, n_electrodes, n_samples), False)

        # Get the indices for the start and end of time windows.
        start_indices = i_t[0, :]
        end_indices = i_t[-1, :]

        # Loop over each epoch and electrode to map rejection from time windows back to sample indices.
        for ep in range(n_epochs):
            for el in range(n_electrodes):
                # Determine which time windows to reject for the current electrode and epoch (upper and lower thresholds).
                reject_upper = data_to_reject_upper_thresh[ep, el, :]
                reject_lower = data_to_reject_lower_thresh[ep, el, :]

                # Get the corresponding sample indices for these time windows.
                start_upper = start_indices[reject_upper]
                end_upper = end_indices[reject_upper]
                start_lower = start_indices[reject_lower]
                end_lower = end_indices[reject_lower]

                # Expand the ranges to include the initial and final samples if needed.
                if len(end_upper) > 1:
                    start_upper, end_upper = insert_initial_and_final_samples(start_upper, end_upper)
                if len(end_lower) > 1:
                    start_lower, end_lower = insert_initial_and_final_samples(start_lower, end_lower)

                # Update the BCT_upper and BCT_lower arrays with the rejected samples for this electrode and epoch.
                for j in range(len(start_upper)):
                    BCT_upper[ep, el, start_upper[j]:end_upper[j] + 1] = True
                for j in range(len(start_lower)):
                    BCT_lower[ep, el, start_lower[j]:end_lower[j] + 1] = True

        # Combine the upper and lower rejection arrays into a single array indicating all rejections.
        BCT = np.logical_or(BCT_upper, BCT_lower)

        return BCT, BCT_upper, BCT_lower

    def reject_electrodes_based_on_time_variance(self, EEG):
        """
        Orchestrates the rejection of electrodes based on time variance through various sub-methods.

        Parameters:
            EEG (EEG object): The EEG data to process.

        Returns:
            tuple: Binary Contamination Matrices for upper, lower, and combined thresholds.
        """
        # Define time windows
        bct_time_windows, i_t, n_time_window = define_time_window(EEG, self.params['time_window'],
                                                                    self.params['time_window_step'])

        # Compute time variability
        STD = self.compute_time_variability(EEG, n_time_window, i_t)

        # Determine the data to be rejected
        data_to_reject_upper_thresh, data_to_reject_lower_thresh, self.params['threshold_matrix'] = \
            self.get_data_to_reject_based_on_time_variance(STD, bct_time_windows, self.params)

        # Reject data based on time variance
        n_electrodes, n_samples, n_epochs = Raw.get_data_size(EEG)
        BCT, BCT_upper, BCT_lower = self.reject_bad_data_based_on_time_variance(data_to_reject_upper_thresh,
                                                                                data_to_reject_lower_thresh,
                                                                                [n_epochs, n_electrodes, n_samples],
                                                                                i_t)
        return BCT, BCT_upper, BCT_lower


class RunningAverage:
    """
    The RunningAverage class implements a running average algorithm to detect and reject artifacts in EEG data.
    
    The class calculates a fast and slow running average of the EEG signal and uses the difference between them,
    along with user-defined thresholds, to identify periods of the EEG that should be rejected as artifacts.
    
    Attributes:
    -----------
        - params (dict): A dictionary of parameters used for artifact detection and rejection.
        - BCT (ndarray): Binary matrix indicating time points that are to be rejected based on running average.
        - n_rejected_data (int): The number of data points rejected after running average analysis.
        - mu (float): Mean of the EEG data, used in z-score calculation.
        - sd (float): Standard deviation of the EEG data, used in z-score calculation.
    
    Args:
    -----
        - raw (Raw object): The EEG data to process.
        - thresh_fast (float): Threshold for fast running average to detect sharp peaks.
        - thresh_diff (float): Threshold for difference between fast and slow running averages.
        - bad_data (str): Specifies how to deal with bad data during reference data processing.
        - mask (float): Duration in seconds to mask before and after detected artifacts.
        - do_reference_data (bool): If True, perform referencing of the data.
        - do_zscore (bool): If True, normalize data using z-score.
        - use_relative_thresh (bool): If True, use relative threshold for detection.
        - use_relative_thresh_per_electrode (bool): If True, apply relative thresholding per electrode.
        - update_BCT (bool): If True, update the Bad Channel Time (BCT) matrix.
        - update_summary (bool): If True, update the summary of the artifact rejection process.
        - update_algorithm (bool): If True, update the artifact rejection algorithm.
        - config (bool or dict): Configuration parameters, if any.
        - name (str): Name of the running average instance.
        - loop_name (str or None): Name of the loop, if part of a looped process.
    
    Methods:
    --------
        __init__(self, raw, thresh_fast=3, thresh_diff=3, bad_data='none', mask=0.05,
                do_reference_data=False, do_zscore=False, use_relative_thresh=True,
                use_relative_thresh_per_electrode=True, update_BCT=True, update_summary=True,
                update_algorithm=True, config=False, name='RunningAverage', loop_name=None)
            Initializes the RunningAverage object with the given EEG data and parameters.
        
        reject_data_based_on_running_average (raw): Main method to apply the running average artifact rejection.
    """
    def __init__(self, raw, thresh_fast=3, thresh_diff=3, bad_data='none', mask=0.05,
                do_reference_data=False, do_zscore=False, use_relative_thresh=True,
                use_relative_thresh_per_electrode=True, update_BCT=True, update_summary=True,
                update_algorithm=True, config=False, name='RunningAverage', loop_name=None):

        """
        A class to perform running average artifact rejection on EEG data.
        """
    
        # Parameters
        self.params = {
            'thresh_fast': thresh_fast, 
            'thresh_diff': thresh_diff, 
            'bad_data': bad_data, 
            'mask': mask,
            'do_reference_data': do_reference_data, 
            'do_zscore': do_zscore,
            'use_relative_thresh': use_relative_thresh,
            'use_relative_thresh_per_electrode': use_relative_thresh_per_electrode, 
            'update_BCT': update_BCT,
            'update_summary': update_summary, 
            'update_algorithm': update_algorithm, 
            'loop_name': loop_name
        }

        # Get configuration (user-input parameters)
        if config:
            from apice.artifacts_rejection import update_parameters_with_user_inputs
            self.params = update_parameters_with_user_inputs(self.params, eval('apice.parameters.' + loop_name + '.' + name))

        # Configure threshold parameters
        self.params = configure_threshold_parameters(self.params, name='RunningAverage')

        # Generate the status messages for running average process and print them all at once.
        print(
            f"\nRejecting data based on the running average...\n",
            f"-- referenced data: {self.params['do_reference_data']}\n",
            f"-- z-score data: {self.params['do_zscore']}\n",
            f"-- relative threshold: {np.squeeze(self.params['use_relative_thresh'])}\n",
            f"-- relative threshold per electrode: {np.squeeze(self.params['use_relative_thresh_per_electrode'])}\n"
            )

        # Initialize artifacts
        raw.artifacts = Artifacts(raw)

        # Set reference to mean amplitude of channels
        if self.params['do_reference_data']:
            set_reference(raw, bad_data=self.params['bad_data'], save_reference=False)

        # Compute z-score for the artifacts
        if self.params['do_zscore']:
            raw._data, self.mu, self.sd = compute_z_score(raw)

        # Reject data based on the running average
        BCT, BCT_fast, BCT_diff = self.reject_data_based_on_running_average(raw)

        # Mask around artifacts
        if self.params['mask']:
            print(f"\n--Masking around artifacts: mask length {self.params['mask']} s")
            BCT = mask_around_artifacts_BCT(BCT, self.params['mask'], raw.info['sfreq'])

        # Update rejection matrix
        update_rejection_matrix(raw.artifacts, self.__class__.__name__, self.params, BCT)

        # Get data back
        if self.params['do_zscore']:
            return_data_after_zscore(raw, self.sd, self.mu)
        if self.params['do_reference_data']:
            return_data_after_referencing(raw)

        # Save a copy of BCT to the instance
        self.BCT = BCT

        # Display rejected data
        n = np.size(BCT)
        self.n_rejected_data = np.round(np.sum(BCT))
        print(
            f"\nData rejected based on the running average:\n",
            f"--- fast running average threshold {np.round(np.sum(BCT_fast) / n * 100, 2)} %\n",
            f"--- difference running average threshold {np.round(np.sum(BCT_diff) / n * 100, 2)} %",
            f"\nTotal rejected data: {np.round(np.sum(BCT) / n * 100, 2)} %\n"
            )
        
    @staticmethod
    def compute_running_average(raw):
        """
        Compute the running average for artifact rejection.

        This method calculates both fast and slow running averages from EEG data, which are used
        to detect artifacts based on the difference between these averages.

        Parameters:
        - raw (Raw object): The EEG data to process.

        Returns:
        - fast_average (ndarray): The fast running average of the EEG data.
        - diff_average (ndarray): The difference between the fast and slow running averages.
        """
        
        # Retrieve the dimensions of the raw data to define the shape of the arrays.
        n_electrodes, n_samples, n_epochs = Raw.get_data_size(raw)
        
        # Initialize the fast_average array with NaN values.
        fast_average = np.empty((n_epochs, n_electrodes, n_samples))
        fast_average[:] = np.nan

        # Initialize the slow_average array with NaN values.
        slow_average = np.empty((n_epochs, n_electrodes, n_samples))
        slow_average[:] = np.nan

        # Copy the raw EEG data to avoid modifying the original data.
        raw_data = raw._data.copy()  # * 1e6

        # Calculate the mean of the raw data along the time axis (axis=1) for each electrode,
        # resulting in a baseline value for each electrode. 
        baseline = np.nanmean(raw_data, axis=1)
        # Subtract the baseline from the raw data:
        raw_data = np.subtract(raw_data, np.tile(baseline, (np.shape(raw_data)[1], 1)).T)
        # Reshape the raw data to separate epochs, electrodes, and samples for further analysis.
        raw_data = np.reshape(raw_data, (n_epochs, n_electrodes, n_samples))

        # Initialize the running averages for the first sample based on the raw data.
        fast_average[:, :, 0] = 0.800 * np.zeros((n_epochs, n_electrodes)) + 0.200 * raw_data[:, :, 0]
        slow_average[:, :, 0] = 0.975 * np.zeros((n_epochs, n_electrodes)) + 0.025 * raw_data[:, :, 0]

        # Compute the running averages for each point in time after the first sample.
        for i in range(1, n_samples):
            # Update the fast average with a higher weight on the most recent sample.
            fast_average[:, :, i] = 0.800 * fast_average[:, :, i - 1] + 0.200 * raw_data[:, :, i]
            
            # Update the slow average with a higher weight on the past values.
            slow_average[:, :, i] = 0.975 * slow_average[:, :, i - 1] + 0.025 * raw_data[:, :, i]

        # Calculate the difference between the fast and slow averages to identify peaks
        # or shifts in the signal that may indicate artifacts.
        diff_average = fast_average - slow_average

        return fast_average, diff_average

    @staticmethod
    def get_data_to_reject_based_on_running_average(raw, fast_average, diff_average, params):
        """
        Identify data to reject based on running average.

        Using the computed running averages, this method applies thresholds to determine which data
        points should be rejected as artifacts.

        Parameters:
        - raw (Raw object): The EEG data to process.
        - fast_average (ndarray): The fast running average of the EEG data.
        - diff_average (ndarray): The difference between the fast and slow running averages.
        - params (dict): Parameters for thresholding and artifact detection.

        Returns:
        - BCT (ndarray): Boolean matrix indicating where artifacts were detected.
        - BCT_fast (ndarray): Boolean matrix for artifacts detected by the fast running average.
        - BCT_diff (ndarray): Boolean matrix for artifacts detected by the running average difference.
        """
        # Retrieve the dimensions of the raw data.
        n_electrodes, n_samples, n_epochs = Raw.get_data_size(raw)

        # Initialize a matrix to hold thresholds for artifact rejection.
        n_relative_thresh = np.size(params['use_relative_thresh'])
        threshold_matrix = np.full((n_electrodes, 2, n_relative_thresh), np.nan)

        # Create Boolean matrices to mark artifacts based on the fast running average (BCT_fast) 
        # and the difference between fast and slow running averages (BCT_diff).
        BCT_fast = np.zeros((n_epochs, n_electrodes, n_samples), dtype=bool)
        BCT_diff = np.zeros((n_epochs, n_electrodes, n_samples), dtype=bool)

        # Loop over each relative threshold level.
        for i in np.arange(n_relative_thresh):
            
            # Check if the relative threshold is used for this index.
            if params['use_relative_thresh'][i]:
                
                # Further check if the relative threshold is to be applied per electrode.
                if params['use_relative_thresh_per_electrode'][i]:
                    
                    # Calculate the absolute values of the fast average, ignore artifacts by setting them to NaN.
                    fast_average_el = np.abs(fast_average)
                    fast_average_el[raw.artifacts.BCT] = np.nan
                    
                    # Reshape for percentile calculation: combine all epochs and samples for each electrode.
                    fast_average_el = np.reshape(fast_average_el, (n_electrodes, n_samples * n_epochs))
                    
                    # Repeat the process for the difference between fast and slow averages.
                    diff_average_el = np.abs(diff_average)
                    diff_average_el[raw.artifacts.BCT] = np.nan
                    diff_average_el = np.reshape(diff_average_el, (n_electrodes, n_samples * n_epochs))
                    
                    # Calculate the 75th percentile as a base for threshold setting, with midpoint interpolation.
                    perc_fast = np.nanpercentile(fast_average_el, 75, axis=1, interpolation='midpoint')
                    
                    # Determine the inter-quartile range, assume a symmetric distribution around zero.
                    IQ = 2 * perc_fast 
                    
                    # Set the threshold for fast averages using the calculated percentile and the user-defined multiplier.
                    t_fast = perc_fast + params['thresh_fast'][i] * IQ
                    
                    # Repeat threshold calculation for the difference average.
                    perc_diff = np.nanpercentile(diff_average_el, 75, axis=1, interpolation='midpoint')
                    IQ = 2 * perc_diff  # half of the distribution centered at zero
                    t_diff = perc_diff + params['thresh_diff'][i] * IQ
                else:
                    # If not per electrode, the process is similar but we consider all data together.
                    fast_average_all = np.abs(fast_average)
                    fast_average_all[raw.artifacts.BCT] = np.nan
                    diff_average_all = np.abs(diff_average)
                    diff_average_all[raw.artifacts.BCT] = np.nan
                    
                    # Calculate thresholds for all electrodes together.
                    perc_fast = np.nanpercentile(fast_average_all, 75, interpolation='midpoint')
                    IQ = 2 * perc_fast
                    t_fast = perc_fast + params['thresh_fast'][i] * IQ
                    t_fast = np.repeat(t_fast, n_electrodes)
                    perc_diff = np.nanpercentile(diff_average_all, 75, interpolation='midpoint')
                    IQ = 2 * perc_diff
                    t_diff = perc_diff + params['thresh_diff'][i] * IQ
                    t_diff = np.repeat(t_diff, n_electrodes)

                # Apply the thresholds to reject artifacts.
                # Create a mask of the values exceeding the threshold.
                bct_fast = fast_average > np.tile(t_fast, (n_samples, 1)).T
                bct_diff = diff_average > np.tile(t_diff, (n_samples, 1)).T
                
                # Combine the new mask with the previous artifact masks using a logical OR.
                BCT_fast = np.logical_or(BCT_fast, bct_fast)
                BCT_diff = np.logical_or(BCT_diff, bct_diff)

                # Store the computed thresholds in the threshold matrix for reference.
                threshold_matrix[:, 0, i] = t_fast
                threshold_matrix[:, 1, i] = t_diff
                
            else:
                # If the relative threshold is not used, apply the user-defined absolute thresholds.
                # Again, create a mask of the values exceeding the threshold.
                bct_fast = fast_average > np.tile(params['thresh_fast'][i], (n_epochs, n_electrodes, n_samples))
                bct_diff = diff_average > np.tile(params['thresh_diff'][i], (n_epochs, n_electrodes, n_samples))
                
                # Update the artifact masks with the new values.
                BCT_fast = np.logical_or(BCT_fast, bct_fast)
                BCT_diff = np.logical_or(BCT_diff, bct_diff)
                
                # Record these thresholds in the threshold matrix.
                threshold_matrix[:, 0, i] = np.repeat(params['thresh_fast'][i], n_electrodes)
                threshold_matrix[:, 1, i] = np.repeat(params['thresh_diff'][i], n_electrodes)

        # Determine the bad data
        BCT = np.logical_or(BCT_fast, BCT_diff)
        
        return BCT, BCT_fast, BCT_diff

    def reject_data_based_on_running_average(self, raw):
        """
        Wrapper method to reject data based on the running average approach.

        It computes the running average, identifies data to reject, and applies the rejection
        process to the EEG data.

        Parameters:
        - raw (Raw object): The EEG data to process.

        Returns:
        - BCT (ndarray): Boolean matrix indicating where artifacts were detected.
        - BCT_fast (ndarray): Boolean matrix for artifacts detected by the fast running average.
        - BCT_diff (ndarray): Boolean matrix for artifacts detected by the running average difference.
        """
        # Compute running averages for the given raw data.
        fast_average, diff_average = self.compute_running_average(raw)

        # Determine which data points to reject based on the computed running averages.
        BCT, BCT_fast, BCT_diff = self.get_data_to_reject_based_on_running_average(raw, fast_average, diff_average, self.params)
        
        return BCT, BCT_fast, BCT_diff


class AmplitudeVariance:
    
    """
    A class to handle rejection of EEG data based on the amplitude variance across electrodes.

    Attributes:
    ----------
        - params (dict): Parameters for the amplitude variance rejection method.
        - BCT (numpy.ndarray): A boolean array indicating bad channels over time.
        - mu (numpy.ndarray): The mean used for z-scoring the data, if applicable.
        - sd (numpy.ndarray): The standard deviation used for z-scoring the data, if applicable.
        - n_rejected_data (int): The number of data points rejected based on amplitude variance.
    
    Args:
    ----
        - raw: The Raw EEG data structure.
        - thresh (float): The threshold multiplier for rejecting data.
        - bad_data (str): The criteria for what constitutes 'bad' data.
        - mask (float): The time (in seconds) to mask around artifacts.
        - do_reference_data (bool): Whether to set a reference to the mean amplitude of channels.
        - do_zscore (bool): Whether to perform z-scoring on the data.
        - update_BCT (bool): Whether to update the bad channel time matrix.
        - update_summary (bool): Whether to update the summary of rejected data.
        - update_algorithm (bool): Whether to update the rejection algorithm status.
        - config (bool): Whether to load custom configuration for parameters.
        - name (str): The name of the amplitude variance rejection method.
        - loop_name (str): The loop name for parameter configuration, if applicable.
        
    Methods:
    --------
        __init__(self, raw, thresh=3, bad_data='none', mask=0.05, do_reference_data=False, 
            do_zscore=False, update_BCT=True, update_summary=True, update_algorithm=True, 
            config=False, name='AmplitudeVariance', loop_name=None)
        Initializes the AmplitudeVariance object with EEG data and rejection parameters.
    
    """

    def __init__(self, raw, thresh=3, bad_data='none', mask=0.05,
                do_reference_data=False, do_zscore=False, update_BCT=True, update_summary=True,
                update_algorithm=True, config=False, name='AmplitudeVariance', loop_name=None):
        
        """
        Initializes the AmplitudeVariance object with EEG data and rejection parameters.
        """

        # Parameters
        self.params = {
            'thresh':thresh, 
            'bad_data':bad_data, 
            'mask': mask, 
            'do_reference_data':do_reference_data,
            'do_zscore':do_zscore, 
            'update_BCT':update_BCT, 
            'update_summary':update_summary,
            'update_algorithm':update_algorithm, 
            'loop_name':loop_name
        }
        
        # Get configuration (user-input parameters)
        if config:
            from apice.artifacts_rejection import update_parameters_with_user_inputs
            self.params = update_parameters_with_user_inputs(self.params, eval('apice.parameters.' + loop_name + '.' + name))

        # Print rejection parameters, all at once.
        print(
            f"\nRejecting data based on the amplitude variance across electrodes...\n",
            f"-- referenced data: {self.params['do_reference_data']}\n",
            f"-- z-score data: {self.params['do_zscore']}"
        )

        # Initialize artifact rejection matrix
        raw.artifacts = Artifacts(raw)

        # Set reference to mean amplitude of channels
        if self.params['do_reference_data']:
            set_reference(raw, bad_data=self.params['bad_data'], save_reference=False)

        # Compute z-score for the artifacts
        if self.params['do_zscore']:
            raw._data, self.mu, self.sd = compute_z_score(raw)

        # Reject data based on amplitude variance
        BCT, self.params = self.reject_data_based_on_amplitude_variance(raw, self.params)

        # Mask around artifacts
        if self.params['mask']:
            print(f"\n--Masking around artifacts: mask length {self.params['mask']} s")
            BCT = mask_around_artifacts_BCT(BCT, self.params['mask'], raw.info['sfreq'])

        # Update rejection matrix
        update_rejection_matrix(raw.artifacts, self.__class__.__name__, self.params, BCT)

        # Get data back
        if self.params['do_zscore']:
            return_data_after_zscore(raw, self.sd, self.mu)
        if self.params['do_reference_data']:
            return_data_after_referencing(raw)

        # Save a copy of BCT
        self.BCT = BCT

        # Display rejected data
        n = np.size(BCT)
        self.n_rejected_data = np.round(np.sum(BCT))
        print(f"\nData rejected based on the amplitude variance: {np.round(np.sum(BCT) / n * 100, 2)} %")

    @staticmethod
    def reject_data_based_on_amplitude_variance(raw, params):
        """
        Rejects raw EEG data based on amplitude variance across electrodes.

        Parameters:
            raw: The raw EEG data structure.
            params (dict): Parameters for the amplitude variance rejection method.

        Returns:
            numpy.ndarray: A boolean array indicating bad channels over time after rejection.
            dict: Updated parameters including the threshold matrix after data rejection.
        """
        
        # Retrieve the dimensions of the data:
        n_electrodes, n_samples, n_epochs = Raw.get_data_size(raw)
        # Reshape the raw EEG data to a 3D array
        raw_data = np.reshape(raw._data.copy(), (n_epochs, n_electrodes, n_samples))
        
        # Rule out bad data
        raw_data[raw.artifacts.BCT] = np.nan
        
        # Identify 'bad times' - periods where the data is unreliable, either due to pre-identified bad times (raw.artifacts.BT)
        # or when more than half of the electrodes are marked as bad for a given time point (i.e., sum of BCT across electrodes > 0.5 * n_electrodes).
        bad_times = np.logical_or(raw.artifacts.BT, (np.sum(raw.artifacts.BCT, axis=1) / n_electrodes > 0.5))
        
        # Create a tiled array of bad times to match the dimensions of raw_data for broadcasting.
        bad_times_ = np.tile(bad_times, (1, n_electrodes, 1))
        
        # Loop through each epoch and assign NaN to all electrodes at the bad times.
        # This effectively removes these time points from subsequent data analysis.
        for ep in range(n_epochs):
            raw_data[ep][bad_times_[ep]] = np.nan
            
        # Reshape the data array to consolidate the epochs and samples into a single dimension.
        raw_data = np.reshape(raw_data, (n_electrodes, n_samples * n_epochs))
        
        # Calculate the mean and standard deviation of the data across electrodes, ignoring NaNs.
        # Axis 0 refers to the electrode axis after reshaping, which is now the rows of the 2D array.
        raw_data_mean = np.nanmean(raw_data, axis=0)
        raw_data_std = np.nanstd(raw_data, axis=0)
        
        # Normalize the data: for each time point, subtract the mean and divide by the standard deviation.
        raw_data = np.divide(raw_data - raw_data_mean, raw_data_std)
        
        # Calculate the 25th, 50th, and 75th percentiles of the normalized data, which are
        # the first quartile (Q1), median, and third quartile (Q3) respectively. We use 'midpoint'
        # interpolation, which gives the midpoint of the range when there are even numbers of data points.
        perc = np.nanpercentile(raw_data, [25, 50, 75], interpolation='midpoint')
        
        # Calculate the inter-quartile range (IQR) which is the range between the first and third quartile.
        # This range captures the middle 50% of the data and is used to define the thresholds for extreme values.
        IQ = perc[2] - perc[0]
        
        # Define the upper threshold as Q3 plus 'thresh' times the IQR. Data points above this value
        # are considered unusually high and may be artifacts or outliers.
        thresh_upper = perc[2] + params['thresh'] * IQ
        
        # Similarly, define the lower threshold as Q1 minus 'thresh' times the IQR. Data points below
        # this value are considered unusually low.
        thresh_lower = perc[0] - params['thresh'] * IQ
        
        # Store the lower and upper thresholds in the parameters dictionary. This allows the
        # threshold values to be accessed later for comparison with the raw data to identify
        # artifacts or outliers.
        params['threshold_matrix'] = [thresh_lower, thresh_upper]

        # Reshape the original raw data array 
        raw_data = np.reshape(raw._data.copy(), (n_electrodes, n_samples * n_epochs))
        
        # Reshape the bad times array, matching the shape of the raw_data array.
        bad_times = np.reshape(bad_times, (1, n_samples * n_epochs))
        
        # Compute the mean and standard deviation of the raw data along the time axis, ignoring NaNs.
        # This will provide the baseline for normalizing the data.
        raw_data_mean_bad = np.nanmean(raw_data, axis=0)
        raw_data_std_bad = np.nanstd(raw_data, axis=0)
        
        # The bad times indicate where too many electrodes have artifacts at the same time.
        # For these times, we update the mean and standard deviation with the computed values from the bad times.
        # This step ensures that we have a complete set of mean and standard deviation values 
        # even for the times that were previously marked as bad.
        raw_data_mean[np.squeeze(bad_times)] = raw_data_mean_bad[np.squeeze(bad_times)]
        raw_data_std[np.squeeze(bad_times)] = raw_data_std_bad[np.squeeze(bad_times)]
        
        # Normalize the raw data by subtracting the mean and dividing by the standard deviation.
        raw_data = np.divide(raw_data - raw_data_mean, raw_data_std)
        
        # Reject data based on calculated thresholds:
        BCT = np.logical_or((raw_data > thresh_upper), (raw_data < thresh_lower))
        # Reshape the boolean matrix back to its original shape.
        BCT = np.reshape(BCT, (n_epochs, n_electrodes, n_samples))
        
        return BCT, params


class ShortBadSegments:
    
    """
    This class is designed to identify and handle very short segments that have been marked as bad
    in the EEG data. It allows for customization of the length of segments to be considered too short
    and provides options to update various processing flags.

    Attributes:
    -----------
        - params (dict): A dictionary of parameters used for artifact rejection.
        - BCT (ndarray): A copy of the bad channel time matrix after processing.
        - n_rejected_data (float): The total number of data points marked for rejection.
    
    Parameters:
    -----------
        - raw (Raw EEG object): The EEG data object to be processed.
        - time_limit (float): The duration limit below which segments are considered too short and re-included (default is 0.020 seconds).
        - update_BCT (bool): Flag indicating whether to update the bad channel time matrix (default True).
        - update_summary (bool): Flag indicating whether to update the summary of rejected data (default False).
        - update_algorithm (bool): Flag indicating whether to update the algorithm parameters (default False).
        - config (bool): Flag indicating whether to use an external configuration for parameters (default False).
        - name (str): The name identifier for this processing step (default 'ShortBadSegments').
        - loop_name (list): List of loop names if used in iterative processing (default empty list).
    
    Methods:
    --------
        __init__(self, raw, time_limit=0.020, update_BCT=True, update_summary=False, update_algorithm=False,
            config=False, name='ShortBadSegments', loop_name=[])
        Initialize the ShortBadSegments object with EEG data and processing parameters.
    """

    def __init__(self, raw, time_limit=0.020, update_BCT=True, update_summary=False, update_algorithm=False, 
                config=False, name='ShortBadSegments', loop_name=[]):
        """
            Initialize the ShortBadSegments object with EEG data and processing parameters.
        """
        
        # Parameters
        self.params = {
            'time_limit':time_limit, 
            'update_BCT':update_BCT, 
            'update_summary':update_summary,
            'update_algorithm':update_algorithm
        }

        # Get configuration (user-input parameters)
        if config:
            from apice.artifacts_rejection import update_parameters_with_user_inputs
            self.params = update_parameters_with_user_inputs(self.params, eval('apice.parameters.' + loop_name + '.' + name))

        print('\nKeeping too short segments...')

        # Initialize artifact rejection matrix
        raw.artifacts = Artifacts(raw)

        # Keep short bad segments
        include_segments, self.params = self.include_short_bad_segments(raw, self.params)

        # Update rejection matrix
        update_rejection_matrix(raw.artifacts, self.__class__.__name__, self.params, include_segments)

        # Save a copy of BCT
        self.BCT = raw.artifacts.BCT.copy()
        self.n_rejected_data = np.round(np.sum(include_segments))

    @staticmethod
    def include_short_bad_segments(raw, params):
        """
        Identifies and includes short bad segments from EEG data that are below a time threshold.

        Args:
            - raw (Raw EEG object): The EEG dataset containing artifacts information.
            - params (dict): A dictionary with parameters, including 'time_limit' key indicating the
                        threshold below which bad segments are reconsidered for inclusion.

        Returns:
            - include_segments (np.ndarray): A boolean array indicating which segments should be included after reconsideration.
            - params (dict): The parameters dictionary potentially updated during processing.
        """
        # Convert the time limit from seconds to number of samples based on the sampling frequency
        params['time_limit'] = np.round(params['time_limit'] * raw.info['sfreq'])
        
        # Initialize the count of samples to potentially restore after identifying short bad segments
        restore = 0
        
        # Obtain the dimensions of the data from the EEG Raw object
        n_electrodes, n_samples, n_epochs = Raw.get_data_size(raw)
        
        # Create a 3D boolean array initialized to False to track which samples to include
        include_segments = np.full((n_epochs, n_electrodes, n_samples), False)

        # Iterate over all epochs and electrodes to identify and include short bad segments
        for ep in range(n_epochs):
            
            for el in range(n_electrodes):
                
                # Extract the binary contamination array for the current electrode and epoch
                bad_times = raw.artifacts.BCT[ep, el, :].copy()
                bad_times = np.asarray(bad_times, dtype=int)
                
                # Find the start indices of bad segments
                temp1 = bad_times[0]
                temp2 = (bad_times[1:] - bad_times[0:-1] == 1)
                start_indices = np.insert(temp2, 0, temp1)
                start_indices = np.where(start_indices)[0]
                
                # Find the end indices of bad segments
                temp3 = (bad_times[0:-1] - bad_times[1:] == 1)
                temp4 = bad_times[-1]
                end_indices = np.insert(temp3, len(temp3), temp4)
                end_indices = np.where(end_indices)[0]
                
                # If there are any short segments
                if np.size(start_indices) > 0:
                    # Calculate the duration of each good segment
                    good_duration = end_indices - start_indices
                    short_segments = np.where(good_duration < params['time_limit'])[0]
                    
                    if np.size(short_segments) > 0:
                        # Loop through each short segment
                        for i in np.arange(np.size(short_segments)):
                            # Mark the short segment as included by setting the corresponding slice to True
                            include_segments[ep, el,
                            start_indices[int(short_segments[i])]:end_indices[int(short_segments[i])]] = True
                            # Update the restore count
                            restore += (start_indices[short_segments[i]] - end_indices[short_segments[i]] + 1)

        # Calculate the total number of data points across all epochs, electrodes, and samples
        total_data_points = n_epochs * n_electrodes * n_samples

        # Calculate the percentage of data that is being restored
        percentage_re_included = (restore / total_data_points) * 100

        # Print out the percentage of data re-included, rounded to two decimal places
        print(f'Total data re-included: {percentage_re_included:.2f}%')

        return include_segments, params


class FastChange:
    """
    A class to detect and reject EEG data based on rapid changes that exceed a certain threshold.

    Attributes:
    -----------
        - params (dict): Configuration parameters for detecting fast changes in EEG data.
        - BCT (numpy.ndarray): Binary matrix indicating the time points that are to be rejected.
        - n_rejected_data (int): The number of data points rejected due to fast changes.
        - mu (numpy.ndarray): Mean values used for z-scoring, retained for returning data to original state.
        - sd (numpy.ndarray): Standard deviation values used for z-scoring, retained for reverting z-score.
        
    Args:
    -----
        - raw (Raw): The EEG data to process.
        - time_window (float, optional): Time window size in seconds to consider for a rapid change. Defaults to 0.02.
        - thresh (float or list, optional): The threshold(s) to use for detecting rapid changes. Defaults to 3.
        - bad_data (str, optional): Strategy for handling bad data. Defaults to 'none'.
        - mask (int, optional): Number of samples to mask around detected artifacts. Defaults to 0.
        - do_reference_data (bool, optional): Whether to reference the data before processing. Defaults to False.
        - do_zscore (bool, optional): Whether to z-score the data before processing. Defaults to False.
        - use_relative_thresh (bool, optional): Whether to use a relative threshold for detection. Defaults to True.
        - use_relative_thresh_per_electrode (bool, optional): Use a relative threshold for each electrode. Defaults to True.
        - update_BCT (bool, optional): Whether to update the Binary Change Test matrix. Defaults to True.
        - update_summary (bool, optional): Whether to update the summary of changes. Defaults to True.
        - update_algorithm (bool, optional): Whether to update the algorithm state. Defaults to True.
        - config (bool, optional): Whether to use custom user-input configuration parameters. Defaults to False.
        - name (str, optional): The name of the algorithm instance. Defaults to 'FastChange'.
        - loop_name (str, optional): The name of the loop if used in iterative processing. Defaults to None.
    
    Methods:
    --------
        __init__(self, raw, time_window=0.02, thresh=3, bad_data='none', mask=0, do_reference_data=False, 
                do_zscore=False, use_relative_thresh=True, use_relative_thresh_per_electrode=True, 
                update_BCT=True, update_summary=True, update_algorithm=True, config=False, name='FastChange',
                loop_name=None)
            Initialize the FastChange object with EEG data and processing parameters.

        reject_data_due_to_fast_changes(raw): Processes the EEG data and rejects segments based on fast changes.

    """

    def __init__(self, raw, time_window=0.02, thresh=3, bad_data='none', mask=0, do_reference_data=False, 
                do_zscore=False, use_relative_thresh=True, use_relative_thresh_per_electrode=True,
                update_BCT=True, update_summary=True, update_algorithm=True, config=False, name='FastChange',
                loop_name=None):

        # Store parameters
        self.params = {
            'time_window':time_window, 
            'thresh':thresh, 
            'bad_data':bad_data, 
            'mask':mask,
            'do_reference_data':do_reference_data, 
            'do_zscore':do_zscore,
            'use_relative_thresh':use_relative_thresh,
            'use_relative_thresh_per_electrode':use_relative_thresh_per_electrode,
            'update_BCT':update_BCT,
            'update_summary':update_summary,
            'update_algorithm':update_algorithm, 
            'loop_name':loop_name
        }

        # Get configuration (user-input parameters)
        if config:
            from apice.artifacts_rejection import update_parameters_with_user_inputs
            self.params = update_parameters_with_user_inputs(self.params, eval('apice.parameters.' + loop_name + '.' + name))

        # Configure threshold parameters
        self.params = configure_threshold_parameters(self.params, name='FastChange')

        # Print rejection algorithm settings, all at once
        print(
            f"\nRejecting data based on fast changes...\n",
            f"-- referenced data: {self.params['do_reference_data']}\n",
            f"-- z-score data: {self.params['do_zscore']}\n",
            f"-- relative threshold: {np.squeeze(self.params['use_relative_thresh'])}\n",
            f"-- relative threshold per electrode: {np.squeeze(self.params['use_relative_thresh_per_electrode'])}\n"
        )

        # Initialize artifact rejection matrix
        raw.artifacts = Artifacts(raw)

        # Set reference to mean amplitude of channels
        if self.params['do_reference_data']:
            set_reference(raw, bad_data=self.params['bad_data'], save_reference=False)

        # Compute z-score for the artifacts
        if self.params['do_zscore']:
            raw._data, self.mu, self.sd = compute_z_score(raw)

        # Reject data due to fast changes
        BCT = self.reject_data_due_to_fast_changes(raw)

        # Mask around artifacts
        if self.params['mask']:
            print(f"\n--Masking around artifacts: mask length {self.params['mask']} s")
            BCT = mask_around_artifacts_BCT(BCT, self.params['mask'], raw.info['sfreq'])

        # Update rejection matrix
        update_rejection_matrix(raw.artifacts, self.__class__.__name__, self.params, BCT)

        # Get data back
        if self.params['do_zscore']:
            return_data_after_zscore(raw, self.sd, self.mu)
        if self.params['do_reference_data']:
            return_data_after_referencing(raw)

        # Save a copy of BCT
        self.BCT = BCT

        # Display rejected data
        n = np.size(BCT)
        self.n_rejected_data = np.round(np.sum(BCT))
        print(f"\nData rejected due to fast changes: {np.round(np.sum(BCT) / n * 100, 2)} %")

    @staticmethod
    def get_maximum_change_per_time_window(raw, params):
        """
        Calculates the maximum derivative within a specified time window for each electrode.

        Parameters:
            raw (Raw): The EEG data to analyze.
            params (dict): Parameters including the time window to use.

        Returns:
            numpy.ndarray: A matrix of the absolute maximum changes per time window for each electrode.
        """
        
        # Get the number of electrodes, samples, and epochs from the raw EEG data
        n_electrodes, n_samples, n_epochs = Raw.get_data_size(raw)
        
        # Calculate the number of samples that fit into the specified time window
        n_samples_time_window = np.round(params['time_window'] * raw.info['sfreq'])
        
        # Make the number of samples even if it's odd (for symmetry)
        if np.mod(n_samples_time_window, 2) == 1:
            n_samples_time_window = n_samples_time_window - 1

        # Copy the raw EEG data to prevent modifying the original data
        raw_data = raw._data.copy()
        
        # Calculate the first derivative (the change between each consecutive sample) of the EEG data
        raw_data_prime = np.diff(raw_data.copy())
        
        # Since np.diff reduces the number of samples by 1, add the last sample back to maintain the original shape
        raw_data_prime = np.insert(raw_data_prime.copy(), n_samples - 1, raw_data[:, -1], axis=1)
        
        # Reshape the array to match the original data structure with dimensions:
        raw_data_prime = np.reshape(raw_data_prime.copy(), (n_epochs, n_electrodes, n_samples))

        # Create an array of indices that span the time window around each sample
        id = np.arange(-n_samples_time_window / 2, n_samples_time_window / 2, dtype=int)
        id = np.tile(id, (n_samples, 1)) + np.tile((np.arange(n_samples)), (int(n_samples_time_window), 1)).T
        
        # Create a mask with the same shape as the id array
        id_mask = np.ones(np.shape(id), dtype=int)
        
        # Set the mask to 0 for indices that are out of bounds (less than 0 or greater than the max sample index)
        # This prevents the selection of non-existing data points
        id_mask[id < 0] = 0
        id_mask[id > (n_samples - 2)] = 0
        
        # Correct out-of-bounds indices in the id array to be within the valid range of data indices
        # This step ensures that when the id array is used to index the data, it won't cause an index error
        id[id < 0] = 1
        id[id > (n_samples - 2)] = n_samples - 1

        # Initialize an array to hold the maximum change values with NaNs
        change = np.empty((n_epochs, n_electrodes, n_samples))
        change[:] = np.nan
        
        # Iterate over all epochs and electrodes to calculate changes
        for ep in np.arange(n_epochs):
            for el in np.arange(n_electrodes):
                # Get the derivative of the raw data for the current electrode and epoch
                raw_data_prime_el = raw_data_prime[ep, el, :].copy().T
                # Apply the id indices and mask to select the relevant window around each sample point
                # and then calculate the sum of changes within each window
                raw_data_prime_el_ = raw_data_prime_el[id] * id_mask
                change_el = np.sum(raw_data_prime_el_, axis=1)
                # Store the total change per window in the corresponding position of the 'change' array
                change[ep, el, :] = change_el
        
        # Take the absolute value of the change array
        change = np.abs(change.copy())

        return change

    @staticmethod
    def get_data_to_reject(raw, params, change):
        """
        Identifies data points to reject based on the maximum change exceeding specified thresholds.

        Parameters:
            raw (Raw): The EEG data to process.
            params (dict): Detection parameters including thresholds.
            change (numpy.ndarray): The matrix of absolute maximum changes per time window for each electrode.

        Returns:
            tuple: A tuple containing:
                - numpy.ndarray: A boolean matrix indicating which data points to reject.
                - numpy.ndarray: The calculated thresholds for each electrode.
        """

        # Retrieve parameters for thresholds and data dimensions
        n_relative_thresh = np.size(params['use_relative_thresh'])
        n_electrodes, n_samples, n_epochs = Raw.get_data_size(raw)
        
        # Initialize a matrix to hold threshold values for each electrode
        threshold_matrix = np.empty((n_electrodes, n_relative_thresh))
        threshold_matrix[:] = np.nan
        
        # Initialize a 3D boolean array to flag data to reject based on the upper threshold
        data_to_reject_upper_thresh = np.full((n_epochs, n_electrodes, n_samples), False)

        # Determine data to reject based on thresholds
        for j in np.arange(n_relative_thresh):
            
            # Check if using relative threshold for the current index j
            if params['use_relative_thresh'][j]:
                
                # If using per electrode thresholding
                if params['use_relative_thresh_per_electrode'][j]:
                    # Sum of data points to be rejected per electrode
                    ru_sum = 0  
                    
                    for el in np.arange(n_electrodes):
                        # Copy the change data for the current electrode
                        data = change[:, el, :].copy()
                        data[raw.artifacts.BCT[:, el, :]] = np.nan
                        
                        # Calculate the 75th percentile and inter-quartile range (IQR)
                        perc = np.nanpercentile(data, 75, interpolation='midpoint')
                        
                        # IQR is double the 75th percentile since distribution is centered to zero
                        IQ = 2 * perc  
                        
                        # Calculate upper threshold
                        thresh_upper_el = perc + params['thresh'][j] * IQ
                        
                        # Flag data above the upper threshold
                        data_to_reject_upper_thresh[:, el, :] = np.logical_or(
                            data_to_reject_upper_thresh[:, el, :],
                            (change[:, el, :] > thresh_upper_el)
                            )
                        
                        # Update the threshold matrix and rejected data sum
                        threshold_matrix[el, j] = thresh_upper_el
                        ru_sum = ru_sum + np.sum(change[:, el, :] > thresh_upper_el)
                        
                else:
                    # Calculate thresholds and flag data for rejection across all electrodes
                    data = change[np.logical_not(raw.artifacts.BCT.copy())] # Use only good data
                    perc = np.nanpercentile(data, 75, interpolation='midpoint')
                    IQ = 2 * perc
                    thresh_upper = perc + params['thresh'][j] * IQ
                    data_to_reject_upper_thresh = np.logical_or(data_to_reject_upper_thresh, (change > thresh_upper))
                    threshold_matrix[:, j] = thresh_upper
                    ru_sum = np.sum(change > thresh_upper)
            else:
                # If not using relative threshold, use the absolute threshold from params
                data = change.copy()
                thresh_upper = params['thresh'][j]
                data_to_reject_upper_thresh = np.logical_or(
                    data_to_reject_upper_thresh,
                    (data > thresh_upper)
                    )
                threshold_matrix[:, j] = thresh_upper
                ru_sum = np.sum(change > thresh_upper)

            # Display the percentage of data rejected based on fast changes
            n = n_epochs * n_electrodes * n_samples
            print(
                f"\nData rejected based on fast changes:",
                f"\n--- upper threshold {np.round(ru_sum / n * 100, 2)} %"
                )

        # Copy data indices to reject
        data_to_reject = data_to_reject_upper_thresh.copy()

        return data_to_reject, threshold_matrix

    @staticmethod
    def reject_data_around_change_detected(raw, params, BCT):
        """
        Rejects data around the time points where a rapid change has been detected.

        Parameters:
            raw (Raw): The EEG data to process.
            params (dict): Parameters including the time window to use for masking.
            BCT (numpy.ndarray): A boolean matrix indicating the initial points of detected change.

        Returns:
            numpy.ndarray: The updated boolean matrix after masking around the detected changes.
        """
        
        # Retrieve the number of electrodes, samples, and epochs from the EEG data
        n_electrodes, n_samples, n_epochs = Raw.get_data_size(raw)
        
        # Create a copy of the BCT array to keep the original untouched
        BCT_input = BCT.copy()
        
        # Initialize the BCT matrix to False, indicating no artifacts initially detected
        BCT = np.full((n_epochs, n_electrodes, n_samples), False)

        # Calculate the number of samples to use as a buffer around the detected artifact
        n_samples_time_window = np.round(params['time_window'] * raw.info['sfreq'])
        buffer = n_samples_time_window / 2 
        
        # Loop over each epoch
        for ep in np.arange(n_epochs):
            # Get a copy of the artifact information for the current epoch
            bct = BCT_input[ep, :, :].copy()
            
            # Loop over each electrode
            for el in np.arange(n_electrodes):
                # Find indices where artifacts have been detected
                bad = bct[el, :]
                bad_idx = np.where(bad)[0]
                
                # If artifacts have been detected, calculate their surrounding buffer
                if np.size(bad_idx) > 0:
                    # Create a range around each artifact index
                    temp1 = np.tile(bad_idx, [2 * int(buffer) + 1, 1]).T
                    temp2 = np.tile(np.arange(-int(buffer), int(buffer) + 1), ([np.size(bad_idx), 1]))
                    bad_idx = temp1 + temp2
                    
                    # Ensure indices are within the bounds of the sample range
                    bad_idx = np.unique(bad_idx)
                    bad_idx = bad_idx[(bad_idx > 0) & (bad_idx <= n_samples - 1)]

                # Mark the indices around the artifact as bad in the artifact matrix
                bct[el, bad_idx] = True
                
            # Combine the new artifacts with the original, ensuring we don't mark previous artifacts
            BCT[ep, :, :] = np.logical_and(bct, np.logical_not(BCT_input[ep, :, :]))
            
        # Combine the new artifact information with the input artifact matrix
        BCT_new = np.logical_or(BCT, BCT_input)

        return BCT_new

    def reject_data_due_to_fast_changes(self, raw):
        """
        The primary method used to reject data based on fast changes detected in the EEG.

        Parameters:
            raw (Raw): The EEG data to process.

        Returns:
            numpy.ndarray: A binary matrix indicating the time points that are to be rejected.
        """
        # Assess changes within each time window and captures the maximum change
        change = self.get_maximum_change_per_time_window(raw, self.params)

        # Determine which data should be rejected based on the changes detected
        data_to_reject, threshold_matrix = self.get_data_to_reject(raw, self.params, change)

        # Reject data surrounding the points where fast changes were detected
        BCT = self.reject_data_around_change_detected(raw, self.params, data_to_reject)
        
        return BCT

