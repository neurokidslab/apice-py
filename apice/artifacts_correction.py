# %% LIBRARIES
import time
import numpy as np
import progressbar

import apice.parameters
from apice.io import Raw
import mne
from apice.electrode_positions import _check_origin

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

def prepare_data_for_splicing(data, bad_if, epoch_if, bct, bt, bc):
    """
    Prepare data and related information for splicing interpolated segments.

    Args:
        data (numpy.ndarray): EEG data to be prepared.
        bad_if (numpy.ndarray): Bad segment indices to interpolate.
        epoch_if (numpy.ndarray): Epoch indices to interpolate.
        bct (numpy.ndarray): Bad channel indices to interpolate.
        bt (numpy.ndarray): Bad time indices to interpolate.
        bc (numpy.ndarray): Bad channel indices for the data.
        
    Returns:
        tuple: A tuple containing the prepared data and related information.
    """
    
    # Check the shape of the input data and create a copy if needed.
    if data.ndim == 1:
        data_ = np.reshape(data, (1, np.size(data)))
    elif data.ndim == 2:
        data_ = data.copy()

    # Check and reshape bct, bt, and bc if provided.
    if bct is None or np.size(bct) == 0:
        bct = np.full(np.shape(data_), False)
    else:
        bct = np.reshape(bct, np.shape(data_))

    if bt is None or np.size(bt) == 0:
        bt = np.full((1, np.shape(data_)[1]), False)
    else:
        bt = np.reshape(bt, (1, np.shape(data_)[1]))

    if bc is None or np.size(bc) == 0:
        bc = np.full((np.shape(data_)[0], 1), False)
    else:
        bc = np.reshape(bc, (np.shape(data_)[0], 1))

    # Extract indices for bad and epoch segments.
    if np.size(np.shape(bad_if)) > 1:
        I_all = bad_if[:, 0].T
        F_all = bad_if[:, 1].T
    else:
        I_all = np.reshape(bad_if[0], (1, 1))
        F_all = np.reshape(bad_if[1], (1, 1))

    if np.size(np.shape(epoch_if)) > 1:
        Epoch_I = epoch_if[:, 0].T
        Epoch_F = epoch_if[:, 1].T
    else:
        Epoch_I = epoch_if[0]
        Epoch_F = epoch_if[1]
        
    return data_, bad_if, epoch_if, bct, bt, bc, I_all, F_all, Epoch_I, Epoch_F

def do_splice_segments(data, bad_intervals, epoch_intervals, bct=None, bt=None, bc=None):
    """
    Splices segments in data based on various criteria.

    This function processes an input dataset to align and correct segments based on the status of electrodes
    and other specified criteria. It handles different cases of bad and good electrodes, aligning each segment
    accordingly.

    Parameters:
    data : ndarray
        The original dataset that needs to be processed.
    bad_intervals : ndarray
        Array indicating bad intervals in the data.
    epoch_intervals : ndarray
        Epoch intervals in the data.
    bct : ndarray, optional
        Binary array indicating bad electrodes for each time point. Default is None.
    bt : ndarray, optional
        Binary array indicating bad time segments. Default is None.
    bc : ndarray, optional
        Binary array for additional criteria. Default is None.

    Returns:
    dN : ndarray
        The processed dataset with spliced segments.
    """
    
    # Preparation of data for splicing
    prepared_data, bad_intervals, epoch_intervals, bct, bt, bc, initial_intervals, final_intervals, epoch_start, epoch_end = prepare_data_for_splicing(data, bad_intervals,
                                                                                                                                                        epoch_intervals, bct, bt, bc)
    
    # Initialize the output array with a copy of the input data
    processed_data = prepared_data.copy()

    # Check if there are intervals to process
    if np.size(initial_intervals) > 0:
        # Selecting relevant intervals within the epochs
        sample_I = initial_intervals[(initial_intervals >= epoch_start) & (initial_intervals <= epoch_end)]
        sample_F = final_intervals[(final_intervals >= epoch_start) & (final_intervals <= epoch_end)]
        unique_intervals = np.unique(np.hstack([epoch_start, sample_I, sample_F + 1, epoch_end + 1]))
        unique_intervals = np.asarray(unique_intervals, dtype=int)

        if np.size(unique_intervals) > 2:
            print('Splicing segments...')
            
            # Progress bar initialization
            bar = progressbar.ProgressBar(maxval=int(np.size(unique_intervals) - 1))
            bar.start()

            # Iterate through each segment for splicing
            for segment_index in np.arange(1, np.size(unique_intervals) - 1):
                # Identify which electrodes are fine and which ones are bad
                if bt[0, unique_intervals[segment_index] - 1]:  # Check if previous segment is bad
                    id_good_electrode = np.full((np.shape(processed_data)[0], 1), False)
                    id_bad_electrode = np.full((np.shape(processed_data)[0], 1), True)
                else:
                    # Identify good and bad electrodes
                    id_good_electrode = ~bct[:, unique_intervals[segment_index] - 1] # Good electrodes before the segment
                    id_bad_electrode = bct[:, unique_intervals[segment_index] - 1] # Bad electrodes before the segment
                # Reshape for compatibility
                id_good_electrode = np.reshape(id_good_electrode,
                                                (np.size(id_good_electrode), 1))  # Electrodes with bad values before
                id_bad_electrode = np.reshape(id_bad_electrode, (np.size(id_bad_electrode), 1))

                # Align the good electrodes with the previous segment
                if (id_good_electrode & ~bc).any():
                    # Identify electrodes that are good in this segment and not part of 'bc'
                    el = id_good_electrode[:, 0] & ~bc[:, 0]
                    # Calculate the difference between the last value of the previous segment
                    # and the first value of the current segment for these electrodes
                    alignment_shift = processed_data[el, unique_intervals[segment_index] - 1] - prepared_data[el, unique_intervals[segment_index]]
                    # Apply this difference to align the good electrodes in the current segment
                    # with the end of the previous segment.
                    processed_data[el, unique_intervals[segment_index]:unique_intervals[segment_index + 1] - 1 + 1] = prepared_data[el, unique_intervals[segment_index]:unique_intervals[segment_index + 1] - 1 +1] + \
                                                            np.tile(alignment_shift, (unique_intervals[segment_index + 1] - unique_intervals[segment_index], 1)).T

                # If it is a bad electrode, align it to the good segment before and after
                if np.any(id_bad_electrode[:, 0] & ~bc[:, 0]):
                    # Identify electrodes that are bad in this segment and not part of 'bc'
                    el = id_bad_electrode[:, 0] & ~bc[:, 0]
                    el = np.asarray(np.where(el)[0])
                    # Define the range for the current segment
                    segment_range = np.arange(unique_intervals[segment_index - 1], unique_intervals[segment_index])
                    
                    # Case 1: Segment starts at the beginning of the data array
                    if segment_range[0] == 0 and segment_range[-1] != np.shape(processed_data)[1] - 1:
                        for i in el:
                            alignment_target = processed_data[i, segment_range[-1] + 1]
                            processed_data[i, segment_range] = processed_data[i, segment_range].copy() + (alignment_target - processed_data[i, segment_range[-1]])  #  np.tile((y_f - dN[i, x[-1]]), (x[-1] + 1, 1)).T
                    
                    # Case 2: Segment ends at the end of the data array
                    elif segment_range[-1] == (np.shape(processed_data)[1] - 1) and segment_range[0] != 0:
                        for i in el:
                            alignment_target = processed_data[i, segment_range[0] - 1]
                            processed_data[i, segment_range] = processed_data[i, segment_range].copy() + (alignment_target - processed_data[i, segment_range[0]])  # np.tile(y_f - dN[i, x[0]], (x[-1] + 1, 1)).T
                    
                    # Case 3: Segment is in the middle of the data array
                    elif segment_range[-1] != (np.shape(processed_data)[1] - 1) and segment_range[0] != 0:
                        for i in el:
                            alignment_target = np.mean(np.vstack([processed_data[i, segment_range[0] - 1], processed_data[i, segment_range[-1] + 1]]))
                            processed_data[i, np.arange(segment_range[-1]+1)] = processed_data[i, np.arange(segment_range[-1]+1)] + alignment_target - processed_data[i, segment_range[0] - 1]
                            temp1 = processed_data[i, (segment_range[-1]+1):]
                            temp2 = alignment_target - processed_data[i, segment_range[-1] + 1]
                            processed_data[i, (segment_range[-1]+1):] = temp1 + temp2
                            progression_rate = (processed_data[i, segment_range[-1] + 1] - processed_data[i, segment_range[-1]]) / (np.size(segment_range) + 1)
                            incremental_shift = progression_rate * (segment_range - segment_range[0] + 1)
                            processed_data[i, segment_range] = processed_data[i, segment_range] + incremental_shift
                
                # Update progress bar
                bar.update(segment_index)
            
            # End of splicing method
            bar.finish()
            
    return processed_data

from joblib import Parallel, delayed
from multiprocessing import Process, Manager

def process_bad_channel(bad_idx, ch_names, ch_names_montage, positions, bad_channel_indices, new_exclude_index, data, distances_matrix):
    
    #Processes a single bad channel and performs spherical spline interpolation.    
    bad_ch_name = ch_names[bad_idx]
    bad_pos = positions[ch_names_montage.index(ch_names[bad_idx])]
    bad_pos = np.reshape(bad_pos, (1, 3))

    distances = distances_matrix[bad_idx,:]
    distances[bad_channel_indices] = np.inf
    distances[new_exclude_index] = np.inf
    
    neighbors_idx = np.where(distances < np.inf)[0]
   
    #neighbor_data = data[good_electrodes, :]
    #neighbor_positions = positions[good_electrodes]

    neighbor_data = data[neighbors_idx, :]
    neighbor_positions = positions[neighbors_idx]
    
    try:
        interpolated_row = spherical_spline_inter(neighbor_positions, bad_pos, neighbor_data)
        return bad_idx, interpolated_row # Return the bad_idx and the interpolated row.
    except Exception as e:
        print(f"Spherical Spline interpolation failed for channel {bad_ch_name}: {e}")
        return bad_idx, None #return the bad_idx and None if there is an error.

def parallel_interpolate(bad_channel_indices, ch_names, ch_names_montage, positions, new_exclude_index, data, distances_matrix, n_jobs):
    #Parallelizes the spherical spline interpolation for bad channels.
    result_interpolation = np.copy(data) # Create a copy to store the interpolated data

    results = Parallel(n_jobs=n_jobs)(delayed(process_bad_channel)(bad_idx, ch_names, ch_names_montage, positions, bad_channel_indices, new_exclude_index, data, distances_matrix) for bad_idx in bad_channel_indices)

    for bad_idx, interpolated_row in results:
        if interpolated_row is not None:
            result_interpolation[bad_idx, :] = interpolated_row # Place the row into the copied array.
    
    return result_interpolation

def do_spherical_spline_interpolation(raw, distances_matrix, positions, adjacency_matrix, bad_neighbor_proportion, bad_channels_to_interpolate, all_bad_channels=None, interpolation_channels=False, n_jobs=-1):
    """
    Perform spherical spline interpolation on EEG data to correct for bad channels.

    Parameters:
    raw: MNE raw object
        The raw EEG data to be interpolated.
    adjacency_matrix: numpy.ndarray
        A matrix that defines the which electrodes are neighbors.
    bad_neighbor_proportion: float
        The proportion of bad neighboring channels over neighboring channels.
    bad_channels_to_interpolate: array_like
        Indices of bad channels that need to be interpolated.
    all_bad_channels: array_like, optional
        Indices of all bad channels in the EEG data. Default is None.
    n_jobs: integer number, optional
        Number of core used for the parallel computation. Default is all available.

    Returns:
    tuple
        A tuple containing the interpolated EEG data and a boolean array indicating the channels
        that were interpolated.
    """

    # Mark the bad channels in the raw data information
    bads_list = list(np.asarray(raw.ch_names, dtype=str)[all_bad_channels])

    # Get electrode positions
    spec_ch_pos = raw.info.get_montage().get_positions()['ch_pos']
    ch_names_montage = list(spec_ch_pos.keys()) # Channel names from the montage

    # Get data and channel names
    data = raw.get_data()
    ch_names = raw.ch_names

    # Get common channels names
    common_channels = list(set(ch_names) & set(ch_names_montage))
    montage_indices = [ch_names_montage.index(ch) for ch in common_channels]

    # Find indices of bad channels
    bad_channel_indices = [ch_names.index(ch) for ch in bads_list]

    # Determine which channels to exclude from the interpolation
    exclude_channels = np.logical_xor(all_bad_channels, bad_channels_to_interpolate)
    
    if interpolation_channels:
        exclude_channels = all_bad_channels

    exclude = list(np.asarray(raw.ch_names, dtype=str)[exclude_channels])
    
    # Get the indices of the electrodes to exclude
    exclude_indices = []
    for electrode in exclude:
        exclude_indices.append(raw.ch_names.index(electrode)) 
    
    # Determine which channels doesn't have sufficient good neighbors    
    new_exclude = []
    for el in exclude_indices:
        # Find the number of bad neighbors per electrode
        el_neighbors = list(np.where(adjacency_matrix[el])[0])
        el_neighbors.remove(el)
        # Determine the bad channel neighbors
        bad_el_neighbors = [i for i in exclude_indices if i in el_neighbors]
        # Get the ratio of bad neighbors
        bad_neighbors_proportion = len(bad_el_neighbors) / len(el_neighbors)
        # If the ratio is greater than bad_neighbor_proportion, remove the electrode from exclude list
        if bad_neighbors_proportion <= bad_neighbor_proportion:
            new_exclude.append(raw.ch_names[el])
    
    new_exclude_index = [ch_names.index(ch) for ch in new_exclude]
    
    # Removing the all bad channels
    if not interpolation_channels:
        bad_channel_indices = list(set(bad_channel_indices).symmetric_difference(set(exclude_indices)))
        exclude_channels[bad_channel_indices] = True
    

    # Interpolate bad channels in the EEG data
    new_interpolated_data = parallel_interpolate(bad_channel_indices, ch_names, ch_names_montage, positions, new_exclude_index, data, distances_matrix, n_jobs=n_jobs)
    
    # Identify which channels were successfully interpolated
    interpolated_bad_channels = np.full(len(raw.ch_names), False)
    for el in range(len(raw.ch_names)):
        #interpolated_bad_channels[el] = (raw.ch_names[el] in bads_list)
        interpolated_bad_channels[el] = (raw.ch_names[el] in [ch_names[i] for i in bad_channel_indices])
        raw.info["bads"] = [ch for ch in raw.info["bads"] if ch in [ch_names[i] for i in bad_channel_indices]]

    return new_interpolated_data, interpolated_bad_channels

from numpy.polynomial.legendre import legval
from scipy.linalg import pinv

def calc_g(cosang, stiffness=4, n_legendre_terms=7):
    factors = [
        (2 * n + 1) / (n**stiffness * (n + 1) ** stiffness * 4 * np.pi)
        for n in range(1, n_legendre_terms + 1)
    ]
    return legval(cosang, [0] + factors)

def normalize_vectors(rr):
    """Normalize surface vertices."""
    size = np.linalg.norm(rr, axis=1)
    mask = size > 0
    rr[mask] /= size[mask, np.newaxis]  # operate in-place
    return size

def spherical_spline_inter(good_pos, bad_pos, good_data):

    normalize_vectors(good_pos)
    normalize_vectors(bad_pos)
    
    Gelec = calc_g(good_pos.dot(good_pos.T)) # from
    Gsph = calc_g(bad_pos.dot(good_pos.T)) # to

    Gelec.flat[:: len(Gelec) + 1] += 1e-5

    n_from = Gelec.shape[0]
    n_to = Gsph.shape[0]

    C = np.vstack(
        [
            np.hstack([Gelec, np.ones((n_from, 1))]),
            np.hstack([np.ones((1, n_from)), [[0]]]),
        ]
    )
    C_inv = pinv(C)

    interpolation = np.hstack([Gsph, np.ones((n_to, 1))]) @ C_inv[:, :-1]

    interpdata = np.matmul(interpolation, good_data)

    return interpdata.copy() 

# %% CLASSES

class TargetPCA:
    """
    Class for performing Target Principal Component Analysis (PCA) on EEG data to correct for artifacts.
    
    Attributes:
    -----------
        - params (dict): Dictionary containing the parameters used for the PCA.
        - intertime (numpy.ndarray): Array for time intervals.
        - interchannel (numpy.ndarray): Array for channel intervals.
        
    Args:
    -----
        - raw (mne.io.Raw): The MNE Raw object containing EEG data.
        - max_time (float): Maximum time for artifact removal, in seconds. Defaults to 0.100.
        - components_to_remove (list): List of components to remove. Defaults to an empty list.
        - variance_to_remove (float): Variance to remove. Defaults to 0.9.
        - mask_time (float): Mask time for artifact removal, in seconds. Defaults to 0.05.
        - all_time (str): Time selection ('all', 'no_bad_time', 'bad_time'). Defaults to 'no_bad_time'.
        - all_channel (str): Channel selection ('all', 'no_bad_channel', 'bad_channel'). Defaults to 'no_bad_channel'.
        - config (bool): Use configuration parameters. Defaults to False.

    Methods:
    --------
        __init__(self, raw, max_time=0.100, components_to_remove=[], variance_to_remove=0.9, mask_time=0.05, 
                all_time='no_bad_time', all_channel='no_bad_channel', config=False): 
            Initialize the TargetPCA object and perform artifact correction.
        target_PCA_per_electrode: Perform Target PCA for each electrode.
        target_PCA: Perform Target PCA on EEG data.
        find_bad_segments: Find bad data segments for interpolation.
    """

    def __init__(self, raw, max_time=0.100, components_to_remove=[], variance_to_remove=0.9, mask_time=0.05,
                all_time='no_bad_time', all_channel='no_bad_channel', config=False):
        """
        Initialize the TargetPCA object and perform artifact correction on EEG data.
        """
        
        # Initialize parameters
        self.params = {
            'max_time': max_time, 
            'components_to_remove': components_to_remove, 
            'variance_to_remove': variance_to_remove,  
            'mask_time': mask_time, 
            'order': 3, 
            'wsize': 4,  
            'all_time': all_time, 
            'all_channel': all_channel,  
            'save_corrected': True, 
            'index_to_remove': [], 
            'silent': False, 
            'config': config  
        }

        # Get configuration (user-input parameters)
        if config:
            for keys in list(self.params.keys()):
                if hasattr(apice.parameters.PCA, keys):
                    self.params[keys] = apice.parameters.PCA.__dict__.get(keys)

        # Initialize self.intertime based on parameters
        self.intertime = None
        if self.params['all_time'] == 'no_bad_time':
            self.intertime = ~raw.artifacts.BT.copy()
        elif self.params['all_time'] == 'bad_time':
            self.intertime = raw.artifacts.BT.copy()


        # Initialize self.interchannel based on parameters
        self.interchannel = None
        if self.params['all_channel'] == 'no_bad_channel':
            self.interchannel = ~raw.artifacts.BC.copy()
        elif self.params['all_channel'] == 'bad_channel':
            self.interchannel = raw.artifacts.BC.copy()

        # Print headers
        print_header('CORRECTING ARTIFACTS')
        print_header('Performing PCA per Electrode', separator='-')

        # Check if there is EEG data
        if np.size(raw._data) > 0:
            start_time = time.time()
            # Perform Target PCA
            raw._data, interpolation_matrix = self.target_PCA_per_electrode(raw)

            # Check if saving the corrected data is enabled
            if self.params['save_corrected']:
                # Initialize the 'CCT' attribute if it doesn't exist
                if not hasattr(raw.artifacts, 'CCT'):
                    raw.artifacts.CCT = np.full(np.shape(raw._data), False)
                
                # Mark interpolated data by applying 'interpolation_matrix'
                idx = interpolation_matrix & raw.artifacts.BCT
                raw.artifacts.BCT[idx] = False
                
                # Combine 'interpolation_matrix' with 'CCT' to update marked interpolated data
                raw.artifacts.CCT = raw.artifacts.CCT | interpolation_matrix

            print(
                f"--- Elapsed time during PCA: {time.time() - start_time} seconds\n",
            )

            # Print summary of artifacts
            raw.artifacts.print_summary()
        else:
            print('No data, nothing will be done.')

        print('\n')


    def target_PCA_per_electrode(self, raw):
        """
        Perform target PCA for each electrode on the raw data.

        Parameters:
        - raw: Raw data object containing EEG data.

        Returns:
        - good_data: Corrected EEG data.
        - interpolation_matrix: Matrix indicating interpolated data.
        """
        
        # Extract the raw data dimensions
        n_electrodes, n_samples, n_epochs = Raw.get_data_size(raw)

        # Handle missing parameter values for target PCA
        if len(self.intertime) == 0:
            self.intertime = np.full((1, n_samples, n_epochs), True)
        if len(self.interchannel) == 0:
            self.interchannel = np.full((1, n_samples, n_epochs), True)
        
        # Handle missing component removal and variance removal values
        if self.params['components_to_remove'] == [] and self.params['variance_to_remove'] == []:
            self.params['components_to_remove'] = 0.95
            self.params['variance_to_remove'] = 1
        if self.params['components_to_remove'] == []:
            self.params['components_to_remove'] = 1
        if self.params['variance_to_remove'] == []:
            self.params['variance_to_remove'] = 0.001

        # Apply target PCA
        # Reshape the EEG data, bad component matrix, and inter-time matrix for processing
        DATA = np.reshape(raw._data.copy(), (n_electrodes, n_samples * n_epochs))
        BCT = np.reshape(raw.artifacts.BCT.copy(), (n_electrodes, n_samples * n_epochs))
        IT = np.reshape(self.intertime, (1, n_samples * n_epochs))
        
        # Determine if we need inter-channel interpolation based on the number of epochs
        if n_epochs > 1:
            IC = np.full((n_electrodes, 1), True)
        else:
            IC = np.reshape(self.interchannel, (n_electrodes))
        
        # Initialize the interpolation matrix to False
        interpolation_matrix = np.full((n_electrodes, n_samples * n_epochs), False)

        # Normalize so all electrodes have equal variance
        # Create an empty array to store normalized variances for each electrode
        norm_electrodes = np.full((n_electrodes, 1), np.nan)
        
        # Iterate through each electrode to calculate normalized variance
        for el in range(n_electrodes):
            non_bad_samples = DATA[el, ~BCT[el, :]]
            if len(non_bad_samples) > 2:
                norm_electrodes[el, 0] = np.nanstd(non_bad_samples)

        # Identify electrodes with NaN or very small amplitude
        idx = (np.isnan(norm_electrodes) | (norm_electrodes < 1e-9)) 
        
        # Replace problematic values with the mean of non-problematic values
        norm_electrodes[idx] = np.nanmean(norm_electrodes)
        
        # Normalize the data
        DATA_normalized = np.divide(DATA, np.tile(norm_electrodes, (1, n_samples * n_epochs)))
        
        # Set NaN values in the normalized data to 0
        DATA_normalized[np.isnan(DATA_normalized)] = 0

        # Correct artifacts
        good_data = DATA_normalized.copy()
        for el in np.arange(n_electrodes):
            # Check if the electrode is marked as bad
            if IC[el]:
                # Calculate the number of good electrodes before the current one
                good_electrode = np.sum(IC[0:el])

                # Print the electrode number
                if not self.params['silent']:
                    print('Electrode ', el + 1, ': ', end='')

                # Find the segments to interpolate
                bad_if = self.find_bad_segments(BCT[el, :], IT, n_epochs, n_samples,
                                                self.params['mask_time'] * raw.info['sfreq'],
                                                self.params['max_time'] * raw.info['sfreq']
                                                )
                # Check if there are bad segments to interpolate
                if np.size(bad_if) > 0:
                    
                    # Check if 'silent' parameter is not set to True
                    if not self.params['silent']:

                        total_bad_data = np.sum(BCT[el, :])
                        percent_bad_data = np.round(total_bad_data / (n_samples * n_epochs) * 100, 2)
                        data_to_apply_pca = int(np.sum(bad_if[:, 1] - bad_if[:, 0]))
                        percent_data_to_apply_pca = np.round(data_to_apply_pca / (n_samples * n_epochs) * 100, 2)

                        print(f'Total bad data {total_bad_data} ({percent_bad_data}%). '
                            f'Data to apply PCA {data_to_apply_pca} ({percent_data_to_apply_pca}%).')

                    # Target PCA
                    d, tC = self.target_PCA(DATA_normalized[IC, :], bad_if,
                                            self.params['components_to_remove'],
                                            self.params['variance_to_remove'],
                                            good_electrode)
                    
                    # Extract data for the current good electrode
                    d = d[good_electrode, :]
                    
                    # Update interpolation matrix for the current electrode
                    interpolation_matrix[el, tC[0, :]] = True

                    # Splice the segments of data together
                    epoch_if = np.asarray([0, n_samples - 1]).T
                    d = do_splice_segments(d, bad_if, epoch_if)

                    # Store the corrected data for the current electrode
                    good_data[el, :] = d
                else:
                    print('No segments to apply target PCA were found.')

            else:
                print(f"Electrode {el + 1}: Bad channel.")

        # Rescale the corrected data back using the norm_electrodes
        good_data *= np.tile(norm_electrodes, (1, n_samples * n_epochs))
        
        # Also rescale the original DATA with norm_electrodes (for consistency)
        DATA = DATA_normalized * np.tile(norm_electrodes, (1, n_samples * n_epochs))

        # Reshape the data
        if n_epochs > 1:
            good_data = good_data.reshape((n_epochs, n_electrodes, n_samples))
            interpolation_matrix = interpolation_matrix.reshape((n_epochs, n_electrodes, n_samples)).astype(bool)

        return good_data, interpolation_matrix

    @staticmethod
    def target_PCA(data, bad_segment, num_singular_values, variance_threshold, electrode_selection):
        """
        Apply Target PCA (Principal Component Analysis) to remove artifacts from EEG data.

        Parameters:
            data (numpy.ndarray): EEG data matrix.
            bad_segments (numpy.ndarray): Array indicating bad data segments.
            num_singular_values (int): Number of singular values (principal components) to retain.
            variance_threshold (float): Variance threshold to determine the number of singular values.
            electrode_selection (numpy.ndarray): Array indicating which electrodes to process.

        Returns:
            numpy.ndarray: Processed EEG data after applying Target PCA.
            numpy.ndarray: Array indicating components retained after PCA.
        """
        
        # If no electrode selection is provided, select all electrodes
        if electrode_selection is None or (isinstance(electrode_selection, list) and not electrode_selection):
            electrode_selection = np.full((np.shape(data)[0], 1), True)

        # Initialize the limit for segment processing and a flag array for retained components
        n_limit = np.inf
        retained_components = np.full((1, np.shape(data)[1]), False)

        # Process each bad segment until none are left
        while np.size(bad_segment) > 0:
            
            # Create a flag array for the current segment
            retained_components_i = np.full((1, np.shape(data)[1]), False)

            # Calculate the cumulative duration of segments
            cumulative_duration = np.cumsum(bad_segment[:, 1] - bad_segment[:, 0])

            # Find segments that meet the time limit criteria
            idx = cumulative_duration <= n_limit

            # Ensure at least one segment is retained
            if not np.any(idx):
                idx[0] = True

            # Get a copy of the segments that meet the criteria and update the list of bad segments
            bad_segment_i = bad_segment[idx, :].copy()
            bad_segment = bad_segment[~idx, :].copy()

            # Try to extract the data within the selected segment range and handle any exceptions
            try:
                y = data[:, np.arange(int(bad_segment_i[0, 0]), int(bad_segment_i[0, 1] + 1))].T
            except:
                # If the end index is missing, extract data up to the available data
                y = data[:, np.arange(int(bad_segment_i[0, 0]), int(bad_segment_i[0, 1]))].T
            
            # Remove the mean from the extracted data along the axis of time
            y -= np.nanmean(y, axis=0)
            
            # Try to mark the corresponding elements in 'retained_components_i' as 'True'
            try:
                retained_components_i[0, np.arange(int(bad_segment_i[0, 0]), int(bad_segment_i[0, 1] + 1))] = True
            except:
                # If the end index is missing, mark elements up to the available data as 'True'
                retained_components_i[0, np.arange(int(bad_segment_i[0, 0]), int(bad_segment_i[0, 1]))] = True
            
            # Loop through additional segments in 'bad_segment_i'
            for iseg in np.arange(1, np.shape(bad_segment_i)[0]):
                # Extract data for the current segment and remove the mean
                yi = data[:, int(bad_segment_i[iseg, 0]):int(bad_segment_i[iseg, 1] + 1)].T
                yi -= np.nanmean(yi, axis=0)
                
                # Concatenate the current segment's data with the previous segments
                y = np.concatenate((y, yi))
                
                # Try to mark the corresponding elements in 'retained_components_i' as 'True'
                try:
                    retained_components_i[0, np.arange(int(bad_segment_i[iseg, 0]), int(bad_segment_i[iseg, 1] + 1))] = True
                except:
                    # If the end index is missing, mark elements up to the available data as 'True'
                    retained_components_i[0, np.arange(int(bad_segment_i[iseg, 0]), int(bad_segment_i[iseg, 1]))] = True
            
            # Remove the mean from the concatenated data
            y -= np.nanmean(y, axis=0)

            '''Perform Principal Component Analysis (PCA) on the concatenated data y.'''
            
            # Calculate the covariance matrix of 'y' (ignoring row variations)
            cov_matrix = np.cov(y, rowvar=False)
            
            # Compute eigenvalues and eigenvectors of the covariance matrix
            [eigenvalues, V] = np.linalg.eigh(cov_matrix)
            
            # Rearrange eigenvectors in descending order of eigenvalues
            eigenvectors = V[:, np.arange(np.shape(cov_matrix)[0] - 1, -1, -1)]
            
            # Calculate the scores by projecting 'y' onto the eigenvectors
            score = np.dot(y, eigenvectors)
            
            # Compute the explained variance for each principal component
            exp_var = np.var(score, axis=0).T
            
            # Normalize the explained variance to sum up to 1
            exp_var = exp_var / np.sum(exp_var)

            # Determine the number of singular values (principal components) to retain based on the variance threshold
            ev_exp = np.cumsum(exp_var) >= variance_threshold
            variance_threshold = np.where(ev_exp)[0]

            # Extract the first index of the variance threshold
            variance_threshold = variance_threshold[0]

            # Ensure that the number of singular values to retain is at least as large as the computed threshold
            num_singular_values = np.max((num_singular_values, variance_threshold))

            # Create a diagonal matrix to retain selected singular values (principal components)
            ev = np.zeros(np.shape(eigenvectors)[0])
            ev[0:num_singular_values + 1] = 1
            ev = np.diag(ev)

            # Perform dimensionality reduction by projecting 'y' onto selected principal components
            yc = y - np.matmul(np.matmul(y, eigenvectors), np.matmul(ev, eigenvectors.T))

            # Store the corrected data back into the original data array for selected electrodes and components
            data[electrode_selection, retained_components_i[0, :]] = yc[:, electrode_selection]
            
            # Update the retained components mask to mark the retained components as True
            retained_components [retained_components_i] = True
        else:
            # If there are no more bad segments to process, set the 'bad_segment' list to an empty list
            bad_segment = []

            # Print the PCA time

        return data, retained_components 

    @staticmethod
    def find_bad_segments(bad_data, intertime, n_epochs, n_samples, mask, maxtime):
        """
        Find bad data segments within epochs based on criteria.

        Parameters:
            bad_data (array): Binary array indicating bad data points.
            intertime (array): Binary array indicating intervals of interest.
            n_epochs (int): Number of epochs.
            n_samples (int): Number of samples per epoch.
            mask (int): Time mask to expand segments (in samples).
            maxtime (int): Maximum allowable segment duration (in samples).

        Returns:
            bad_if (array): Array containing the beginning and end indices of bad segments.

        This function processes `bad_data` and `intertime` arrays within epochs to identify bad data segments.
        It combines segments that overlap and removes segments that exceed the `maxtime` duration.
        """
        
        # Initialize the list to store bad data segment indices
        segment_indices = []
        
        # Loop through each epoch
        for ep in np.arange(n_epochs):
            # Extract binary arrays for bad data and intervals of interest for the current epoch
            temp1 = np.asarray(bad_data[ep * n_samples: (ep + 1) * n_samples], dtype=int)
            temp2 = np.asarray(intertime[0, ep * n_samples: (ep + 1) * n_samples], dtype=int)
            
            # Combine the two binary arrays to identify segments that need interpolation
            to_interpolate = temp1 & temp2

            # Calculate the differences in 'to_interpolate'
            diff_to_interpolate = np.diff(to_interpolate)
            
            # Beginning and end of each segment
            segment_start_flags = np.concatenate(([to_interpolate[0]], diff_to_interpolate == 1))
            segment_end_flags = np.concatenate((diff_to_interpolate == -1, [to_interpolate[-1]]))

            # Find the indices of segment start and end flags
            segment_start_indices = np.where(segment_start_flags)[0]
            segment_end_indices = np.where(segment_end_flags)[0]

            # Calculate the beginning and end of each segment with the mask
            segment_ranges = np.column_stack((segment_start_indices - mask, segment_end_indices + mask))

            # Ensure that indices are within bounds
            segment_ranges[:, 0] = np.maximum(segment_ranges[:, 0], 0)
            segment_ranges[:, 1] = np.minimum(segment_ranges[:, 1], n_samples - 1)

            # Initialize the segment indices
            segment_indices = [] 
            
            if len(segment_ranges) > 1:
                # Find indices of overlapping segments
                overlap_indices = np.where(segment_ranges[:-1, 1] >= segment_ranges[1:, 0])[0]

                # Merge overlapping segments
                for j in overlap_indices:
                    segment_ranges[j, 1] = segment_ranges[j + 1, 1]

                # Remove the merged segments
                segment_ranges = np.delete(segment_ranges, overlap_indices + 1, axis=0)

                # Calculate the duration of each segment
                segment_duration = segment_ranges[:, 1] - segment_ranges[:, 0] + 1

                # Subtract the mask duration from each segment
                segment_duration -= 2 * mask

                # Find indices of segments that exceed the maximum allowed duration
                idx_to_remove = segment_duration > maxtime

                # Delete segments that exceed the maximum duration
                segment_ranges = np.delete(segment_ranges, idx_to_remove, axis=0)

                # Adjust segment indices to be relative to the total number of samples
                segment_ranges = segment_ranges + ep * n_samples

                # Initialize or concatenate segment_indices based on the epoch
                if ep == 0:
                    segment_indices = segment_ranges
                else:
                    segment_indices = np.concatenate(segment_indices, segment_ranges)

        return segment_indices

class SegmentSphericalSplineInterpolation:
    """
    Class for performing interpolation using Spherical Spline per Segment on EEG data.
    
    Attributes:
    -----------
        - params (dict): A dictionary containing the input parameters.
    
    Args:
    -----
        - raw (mne.Raw): The EEG data.
        - n_jobs (int) : Number of core used for the parallel computation.
        - p (float): Percentage of bad data channels to consider for interpolation.
        - p_neighbors (int): Number of neighboring channels to use for interpolation.
        - min_good_time (float): Minimum duration of good data segments in seconds.
        - min_intertime (int): Minimum inter-segment time in samples.
        - max_loop (int): Maximum number of loops for the interpolation process.
        - mask_time (float): Time in seconds to use as a mask for bad segments.
        - save_corrected (bool): Whether to save the corrected data.
        - silent (bool): Whether to suppress printing progress messages.
        - config (bool): Whether to use user-input parameters from a configuration file.

    Methods:
    --------
        __init__(self, raw, p=0.5, p_neighbors=1, min_good_time=2.00, min_intertime=1, max_loop=10,
                mask_time=0, save_corrected=True, silent=False, config=False):
            Initialize class for performing Spherical Spline Interpolation per Segment on EEG data.
        nearest_neigbor_interpolation: Perform Spherical Spline Interpolation on EEG data.
        nearest_neigbor_interpolation_of_segments: Perform Spherical Spline Interpolation on EEG data.
        find_segments_to_interpolate: Find segments in EEG data to interpolate.
    """

    def __init__(self, raw, n_jobs, p=0.5, p_neighbors=1, min_good_time=2.00, min_intertime=1, max_loop=10, 
                mask_time=0, save_corrected=True, silent=False, config=False):
        """
        Initialize class for performing Spherical Spline Interpolation per Segment on EEG data.
        """

        # Create a dictionary 'params' to store the input parameters.
        self.params = {
            'p': p,
            'p_neighbors': p_neighbors,
            'min_good_time': min_good_time,
            'min_intertime': min_intertime,
            'max_loop': max_loop,
            'mask_time': mask_time,
            'save_corrected': save_corrected,
            'silent': silent,
            'config': config
        }
        
        print_header('Performing Spherical Spline Interpolation per Segment', separator="-")

        # Get configuration (user-input parameters)
        if config:
            for keys in list(self.params.keys()):
                if hasattr(apice.parameters.Spherical_Spline_Interpolation, keys):
                    self.params[keys] = apice.parameters.Spherical_Spline_Interpolation.__dict__.get(keys)

        # Check if there is data to perform interpolation on.
        if np.size(raw._data) > 0:

            # If 'min_good_time' is greater than 0, mark too short bad segments as 'bad'.
            if self.params['min_good_time'] > 0:
                from apice.artifacts_detection import ShortGoodSegments
                ShortGoodSegments(raw, time_limit=self.params['min_good_time'])

            # Perform Spherical Spline interpolation of spatial segments.
            raw._data, interpolation_matrix = self.spherical_spline_interpolation(raw, n_jobs)

            # If 'save_corrected' is True, update 'CCT' and 'BCT' attributes in artifacts.
            if self.params['save_corrected']:
                if not hasattr(raw.artifacts, 'CCT'):
                    raw.artifacts.CCT = np.full((np.shape(raw._data)), False)
                idx = np.logical_and(interpolation_matrix, raw.artifacts.BCT)
                raw.artifacts.BCT[idx] = False
                raw.artifacts.CCT = np.logical_or(raw.artifacts.CCT, interpolation_matrix)
            
            # Print a summary of artifacts.
            raw.artifacts.print_summary()
            
        else:
            print('No data, nothing will be done.')

    def spherical_spline_interpolation(self, raw, n_jobs):
        """
        Perform spherical spline interpolation on bad segments of EEG data.

        This function iteratively interpolates bad segments in the EEG data using nearest neighbor interpolation.
        It continues until there are no significant changes in the interpolated data or a maximum number of iterations is reached.

        Parameters:
        raw: Raw EEG data object
            The EEG data object containing the data to be interpolated.
        n_jobs: integer number
            Number of core used for the parallel computation.

        Returns:
        interpolatedData: ndarray
            The EEG data after interpolation of bad segments.
        interpolation_matrix: ndarray
            A matrix indicating the segments of the data that were interpolated.
        """

        print('\nSpherical Spline Interpolation of Bad Segments')

        # Get data size for electrodes, samples, and epochs
        n_electrodes, n_samples, n_epochs = Raw.get_data_size(raw)
        
        # Initialize a matrix to track interpolated data
        interpolation_matrix = np.full((n_epochs, n_electrodes, n_samples), False)
        numInterpolatedPoints = np.sum(interpolation_matrix)
        
        # Copy the original data for processing
        interpolatedData = raw._data.copy()
        
        # Copy the bad channel tracking (BCT) and good channel information
        BCT = raw.artifacts.BCT.copy()
        good_channels = ~raw.artifacts.BC.copy()

        # Initialize variables for the interpolation loop
        isInterpolationComplete = False
        iterationCounter = 0

        # Computing the distances only once
        from scipy.spatial.distance import cdist
        spec_ch_pos = raw.info.get_montage().get_positions()['ch_pos']
        positions = np.array(list(spec_ch_pos.values())) - _check_origin("auto", raw.info)
        distance_matrix = cdist(positions, positions, metric='euclidean')

        # Get adjacency matrix
        print('\nExtracting electrode adjacency matrix.')
        adjacency_matrix = mne.channels.find_ch_adjacency(raw.info, 'eeg')[0].toarray()

        # Interpolation loop
        while not isInterpolationComplete and (iterationCounter < self.params['max_loop']):
            iterationCounter += 1

            # Identify segments that need to be interpolated
            badChannelsForInterpolation = self.find_segments_to_interpolate(raw, BCT)

            # Perform nearest neighbor interpolation on identified segments
            interpolatedData, interpolation_matrix, BCT = self.spherical_spline_interpolation_of_segments(raw, BCT,
                                                                                                    distance_matrix,
                                                                                                    positions,
                                                                                                    adjacency_matrix,
                                                                                                    good_channels,
                                                                                                    badChannelsForInterpolation,
                                                                                                    interpolation_matrix,
                                                                                                    interpolatedData, n_jobs)

            # Update the BCT based on the interpolation results
            BCT[interpolation_matrix & BCT] = False
            newInterpolatedDataCount = np.sum(interpolation_matrix)
            
            # Check if interpolation is complete
            if (newInterpolatedDataCount - numInterpolatedPoints) / np.size(interpolation_matrix) * 100 == 0:
                isInterpolationComplete = True
            else:
                numInterpolatedPoints = newInterpolatedDataCount

        return interpolatedData, interpolation_matrix

    def spherical_spline_interpolation_of_segments(self, raw, BCT, distances_matrix, positions, adjacency_matrix, goodChannelsPerEpoch, bct_to_interpolate,
                                                    interpolation_matrix, good_data, n_jobs):
        """
        Performs spherical spline interpolation on EEG data segments.

        This function interpolates bad segments in EEG data using spherical spline interpolation. 
        It identifies segments with bad channels and interpolates them if there are enough good channels available.

        Parameters:
        raw : Raw EEG object
            EEG data structure containing raw data, metadata, and artifacts.
        BCT : ndarray
            Binary channel-time matrix indicating bad channels.
        goodChannelsPerEpoch : ndarray
            Binary matrix indicating good channels for each epoch.
        bct_to_interpolate : ndarray
            Binary channel-time matrix indicating channels to interpolate.
        interpolation_matrix : ndarray
            Matrix to store interpolation flags for each channel and sample.
        good_data : ndarray
            Matrix to store the interpolated data.
            
        n_jobs : integer number
            Number of core used for the parallel computation.

        Returns:
        good_data : ndarray
            The EEG data with interpolated segments.
        interpolation_matrix : ndarray
            Updated matrix with interpolation flags.
        BCT : ndarray
            Updated binary channel-time matrix after interpolation.
        """

        # Extract data dimensions
        n_electrodes, n_samples, n_epochs = Raw.get_data_size(raw)

        # Process each epoch separately
        for ep in np.arange(n_epochs):
            start_time = time.time()

            print('\nInterpolating Epoch ', ep + 1, '...')

            # Initialize arrays for storing segment indices
            initialSegmentIndices = np.empty((1, n_samples))
            initialSegmentIndices[:] = np.nan
            finalSegmentIndices = np.empty((1, n_samples))
            finalSegmentIndices[:] = np.nan

            currentSegmentIndex = 0

            # Identify the changes in the number of bad channels
            change_channel = np.where(np.any(np.diff(bct_to_interpolate[ep, :, :], axis=1), axis=0))[0]
            segmentStartIndices = np.unique(np.hstack([change_channel + 1, 0]))
            segmentEndIndices = np.unique(np.hstack([change_channel, n_samples - 1]))
            badIntervals = np.asarray([segmentStartIndices.T, segmentEndIndices.T]).T
            
            # Initialize progress bar
            bar = progressbar.ProgressBar(maxval=int(np.shape(badIntervals)[0] - 1))
            bar.start()

            # Copy EEG data for processing
            raw_data = raw._data.copy()
            t = raw.times.copy()
            croppedEEGData = raw.copy()
            croppedEEGData._data = []
            croppedEEGData._times = []

            # Iterate through each segment for interpolation
            for segmentIndex in np.arange(np.shape(badIntervals)[0]):
                # Check if the segment contains any bad channels to interpolate
                if np.any(bct_to_interpolate[ep, :, np.arange(badIntervals[segmentIndex, 0], badIntervals[segmentIndex, 1] + 1)]):

                    currentSegmentIndex += 1

                    # Define segment boundaries
                    segmentStart = badIntervals[segmentIndex, 0]
                    segmentEnd = badIntervals[segmentIndex, 1]
                    initialSegmentIndices[ep, currentSegmentIndex - 1] = segmentStart
                    finalSegmentIndices[ep, currentSegmentIndex - 1] = segmentEnd

                    # Determine bad channels in the segment
                    all_bad_channels = np.any(bct_to_interpolate[ep, :, np.arange(segmentStart, segmentEnd + 1)], axis=0) | \
                                        ~goodChannelsPerEpoch[ep, :, 0]
                    bad_channels_to_interpolate = np.any(bct_to_interpolate[ep, :, np.arange(segmentStart, segmentEnd + 1)], axis=0) & \
                                                    goodChannelsPerEpoch[ep, :, 0]

                    # Interpolate if there are enough good channels
                    if self.params['p']:
                        if np.sum(all_bad_channels) / np.size(all_bad_channels) <= self.params['p']:

                            # Get a copy of the segments to interpolate
                            croppedEEGData._data = raw_data[:, np.arange(segmentStart, segmentEnd+1)]
                            croppedEEGData._times = t[np.arange(segmentStart, segmentEnd + 1)]

                            # Perform interpolation
                            interpolated_data, interpolated_bad_channels  = do_spherical_spline_interpolation(croppedEEGData, 
                                                                                                                distances_matrix,
                                                                                                                positions,
                                                                                                                adjacency_matrix, 
                                                                                                                self.params['p_neighbors'], 
                                                                                                                bad_channels_to_interpolate, 
                                                                                                                all_bad_channels, False, n_jobs)
    
                            
                            # Store the interpolated data
                            bad_ch = np.where(interpolated_bad_channels)[0]
                            for i in bad_ch:
                                interpolation_matrix[ep][i, np.arange(segmentStart, segmentEnd + 1)] = True
                                good_data[i, np.arange(segmentStart, segmentEnd + 1)] = interpolated_data[i, :]
                                #good_data[i, np.arange(segmentStart, segmentEnd + 1)] = interpolated_data[np.where(bad_ch == i)[0][0], :]
                                BCT[ep][i, np.arange(segmentStart, segmentEnd + 1)] = False

                # Update progress bar
                bar.update(segmentIndex)

            # End of interpolation
            bar.finish()
            
            # Finalize the indices for interpolated segments
            initialSegmentIndices = initialSegmentIndices[~np.isnan(initialSegmentIndices)]
            finalSegmentIndices = finalSegmentIndices[~np.isnan(finalSegmentIndices)]

            # Splice the interpolated segments together
            badIntervals = np.asarray([initialSegmentIndices, finalSegmentIndices], dtype=int).T
            epoch_if = np.asarray([0, n_samples - 1])
            bct = BCT[ep, :, :]
            bt = raw.artifacts.BT[ep, :, :].copy()
            bc = ~goodChannelsPerEpoch
            good_data = do_splice_segments(good_data, badIntervals, epoch_if, bct, bt, bc)

            # Print summary
            print(
                f"--- Elapsed time during interpolation: {time.time() - start_time} seconds\n",
                f"--- Percentage of interpolated data: {np.round(np.sum(interpolation_matrix[ep, :, :]) / (n_electrodes * n_samples) * 100, 2)} %\n"
            )

        return good_data, interpolation_matrix, BCT

    def find_segments_to_interpolate(self, raw, BCT):
        """
        Identifies segments in EEG data that require interpolation.

        This function goes through EEG data to find segments with bad channels that need interpolation.
        It considers various parameters like minimum inter-time, masking time, and percentage of bad channels
        to decide which segments should be interpolated.

        Parameters:
        raw : Raw EEG object
            EEG data structure containing raw data, metadata, and artifacts.
        BCT : ndarray
            Binary channel-time matrix indicating bad channels.

        Returns:
        segmentsToInterpolateMatrix : ndarray
            A matrix indicating the segments in the EEG data that require interpolation.
        """

        # Extract data dimensions and initialize interpolation matrix
        n_electrodes, n_samples, n_epochs = Raw.get_data_size(raw)
        segmentsToInterpolateMatrix = np.full((n_epochs, n_electrodes, n_samples), False)

        # Calculate minimum inter-time and mask time based on sampling frequency
        min_intertime = self.params['min_intertime'] * raw.info['sfreq']
        mask_time = self.params['mask_time'] * raw.info['sfreq']

        # Identify good channels and inter-time periods
        goodChannelsMatrix = ~raw.artifacts.BC.copy()
        validTimePeriods = ~raw.artifacts.BT.copy()

        # Optionally print total bad time information
        if not self.params['silent']:
            print('Total bad time : ', np.sum(~validTimePeriods), '(',
                  np.round(np.sum(~validTimePeriods) / (n_samples * n_epochs) * 100, 2), '%).')

        # Process each electrode and epoch to find segments to interpolate
        for electrodeIndex in np.arange(n_electrodes):
            for epochIndex in np.arange(n_epochs):
                
                # Combine various conditions to identify bad segments
                v1 = BCT[epochIndex, electrodeIndex, :]
                v2 = validTimePeriods[epochIndex, :]
                v3 = np.tile(goodChannelsMatrix[epochIndex, electrodeIndex, :], (1, n_samples))
                v4 = ((np.sum(BCT[epochIndex, :, :], axis=0) / n_electrodes) <= self.params['p'])
                to_interpolate = v1 & v2 & v3 & v4

                # Identify the start and end points of each bad segment
                to_interpolate_ = np.asarray(to_interpolate, dtype=int)
                segmentStartIndices = np.full(np.shape(to_interpolate_), False)
                segmentStartIndices[0, 0] = to_interpolate_[0, 0]
                segmentStartIndices[0, 1:] = (np.diff(to_interpolate_) == 1)
                segmentStartIndices = np.where(segmentStartIndices)[1]
                segmentEndIndices = np.full(np.shape(to_interpolate_), False)
                segmentEndIndices[0, :-1] = (np.diff(to_interpolate_) == -1)
                segmentEndIndices[0, -1] = to_interpolate_[0, -1]
                segmentEndIndices = np.where(segmentEndIndices)[1]

                # Remove segments that are too short
                bad_duration_per_epoch = segmentEndIndices - segmentStartIndices + 1
                indicesToRemove = (bad_duration_per_epoch < min_intertime)
                segmentStartIndices = np.delete(segmentStartIndices, indicesToRemove)
                segmentEndIndices = np.delete(segmentEndIndices, indicesToRemove)

                # Apply mask time to segments
                interpolatedSegmentIndices = np.asarray([segmentStartIndices - mask_time, segmentEndIndices + mask_time], dtype=int).T
                interpolatedSegmentIndices[interpolatedSegmentIndices[:, 0] < 0, 0] = 0
                interpolatedSegmentIndices[interpolatedSegmentIndices[:, 1] > n_samples - 1, 1] = n_samples - 1
                
                # Check and merge overlapping segments                
                if np.size(interpolatedSegmentIndices) > 0:
                    # Put together the overlapping segments
                    if np.size(interpolatedSegmentIndices[:, 0]) > 1:
                        overlappingSegmentIndices = np.where(interpolatedSegmentIndices[:-1, 1] >= interpolatedSegmentIndices[1: ,0])[0]
                        for j in np.arange(np.size(overlappingSegmentIndices)):
                            interpolatedSegmentIndices[overlappingSegmentIndices[j], 1] = interpolatedSegmentIndices[overlappingSegmentIndices[j] + 1, 1]
                        interpolatedSegmentIndices = np.delete(interpolatedSegmentIndices, overlappingSegmentIndices + 1, axis=0)

                    # Mark segments for interpolation in the matrix
                    for i in np.arange(np.shape(interpolatedSegmentIndices)[0]):
                        segmentsToInterpolateMatrix[epochIndex, electrodeIndex, np.arange(interpolatedSegmentIndices[i, 0], interpolatedSegmentIndices[i, 1] + 1)] = True

                # Optionally print information about each electrode's bad data and data to interpolate
                if not self.params['silent']:
                    print(
                        f"Electrode {electrodeIndex + 1}: Bad data {np.sum(BCT[:, electrodeIndex, :])} ({np.round(np.sum(BCT[:, electrodeIndex, :]) / (n_epochs * n_samples) * 100, 2)} %).",
                        f"Data to interpolate {np.sum(segmentsToInterpolateMatrix[:, electrodeIndex, :])} ({np.round(np.sum(segmentsToInterpolateMatrix[:, electrodeIndex, :]) / (n_samples * n_epochs) * 100, 2)} %)."
                        )

        return segmentsToInterpolateMatrix
  

class ChannelsSphericalSplineInterpolation:
    """
    A class to perform spherical spline interpolation on EEG data.

    This class is designed to handle the interpolation of non-working electrodes in EEG data using
    a spherical spline method. It initializes with parameters for interpolation and processes the EEG
    data accordingly.

    Attributes:
    -----------
        - params (dict): A dictionary containing parameters for the interpolation process.
    
    Parameters:
        - raw (Raw EEG object): The EEG data structure to be interpolated.
        - n_jobs (int) : Number of core used for the parallel computation.
        - p (float): The percentage threshold for determining bad channels.
        - p_neighbors (int): The number of neighboring electrodes to consider in the interpolation.
        - min_good_time (float): The minimum duration for considering a segment as good.
        - min_intertime (float): The minimum duration between interpolations.
        - max_loop (int): The maximum number of iterations for the interpolation process.
        - mask_time (float): The time duration to mask around bad segments.
        - save_corrected (bool): Flag to determine if corrected data should be saved.
        - silent (bool): Flag to suppress print statements.
        - config (bool): Flag to use custom configuration parameters.
    
    Methods:
        __init__(self, raw, p=0.5, p_neighbors=1, min_good_time=2.00, min_intertime=1, max_loop=10,
                mask_time=0, save_corrected=True, silent=False, config=False): 
            Initializes the ChannelsSphericalSplineInterpolation class with EEG data and interpolation parameters.
    """
    
    def __init__(self, raw, n_jobs, p=0.5, p_neighbors=1, min_good_time=2.00, min_intertime=1, max_loop=10, 
                mask_time=0, save_corrected=True, silent=False, config=False):
        """
        Initializes the ChannelsSphericalSplineInterpolation class with EEG data and interpolation parameters.
        """

        print_header('Performing Spherical Spline Interpolation (Nonworking electrodes)', separator='-')

        # Store parameters in a dictionary
        self.params = {
            'p':p, 
            'p_neighbors':p_neighbors, 
            'min_good_time':min_good_time,
            'min_intertime':min_intertime, 
            'max_loop':max_loop,
            'mask_time':mask_time, 
            'save_corrected':save_corrected,
            'silent':silent,
            'config':config
        }

        # Update parameters with user-defined configuration if config is set
        if config:
            for keys in list(self.params.keys()):
                if hasattr(apice.parameters.Spherical_Spline_Interpolation, keys):
                    self.params[keys] = apice.parameters.Spherical_Spline_Interpolation.__dict__.get(keys)

        # Determine the good channels in the EEG data
        good_channels = ~raw.artifacts.BC.copy()
        
        # Get adjacency matrix
        print('\nExtracting electrode adjacency matrix.')
        adjacency_matrix = mne.channels.find_ch_adjacency(raw.info, 'eeg')[0].toarray()
        
        # Computing the distances only once
        from scipy.spatial.distance import cdist
        spec_ch_pos = raw.info.get_montage().get_positions()['ch_pos']
        positions = np.array(list(spec_ch_pos.values())) - _check_origin("auto", raw.info)
        distance_matrix = cdist(positions, positions, metric='euclidean')
        #distance_matrix = np.linalg.norm(positions, axis=1)

        # Interpolation process starts here
        if np.size(raw._data) > 0:

            # Interpolate entire channels and process EEG data
            n_electrodes, n_samples, n_epochs = Raw.get_data_size(raw)
            interpolation_matrix = np.full((n_epochs, n_electrodes, n_samples), False)

            print('\nSpatial interpolation of electrodes not working during the whole epoch...')

            # Copy and reshape the raw EEG data for processing
            eeg_data = raw._data.copy()
            eeg_data = np.reshape(eeg_data, (n_epochs, n_electrodes, n_samples))
            
            # Copy the time points associated with the raw EEG data
            t = raw.times.copy()
            
            # If there is only one epoch, reshape the time array to match the dimensions of the EEG data
            if n_epochs == 1:
                t = np.reshape(t, (1, 1, n_samples))

            # Create a copy of the raw EEG object for isolated manipulation
            eeg_copy = raw.copy()
            
            # Reset the data and times in the copied EEG object to empty
            eeg_copy._data = []
            eeg_copy._times = []

            # Loop through each epoch to handle bad channels
            for epochIndex in np.arange(n_epochs):
                # Identify bad channels and interpolate them
                channelsToInterpolate = ~good_channels[epochIndex][:, :][:,0]
                print('\nEpoch :', epochIndex + 1)

                if np.any(channelsToInterpolate):
                    start_time = time.time()
                    print('Bad electrodes: ', np.where(channelsToInterpolate)[0] + 1)

                    # Get a copy of the EEG data and info
                    eeg_copy._data = eeg_data[epochIndex][:, :]
                    
                    # Set time values
                    if n_epochs == 1:
                        eeg_copy._times = t[epochIndex][:, :]
                    else:
                        eeg_copy._times = t

                    # Perform spline interpolation
                    if np.sum(channelsToInterpolate) / np.size(channelsToInterpolate) <= self.params['p']:
                        interpolated_data, interpolated_channels = do_spherical_spline_interpolation(eeg_copy, 
                                                                                                distance_matrix,
                                                                                                positions,
                                                                                                adjacency_matrix, 
                                                                                                self.params['p_neighbors'], 
                                                                                                channelsToInterpolate, 
                                                                                                channelsToInterpolate,
                                                                                                True, n_jobs) 
                                                                                              
                        # Store data
                        for badElectrodeIndex in np.where(channelsToInterpolate)[0]:
                            if n_epochs == 1:
                                raw._data[badElectrodeIndex, :] = interpolated_data[badElectrodeIndex, :]
                                #raw._data[badElectrodeIndex, :] = interpolated_data[np.where(np.where(channelsToInterpolate)[0] == badElectrodeIndex)[0][0], :]
                            else:
                                raw._data[epochIndex][badElectrodeIndex, :] = interpolated_data[badElectrodeIndex, :]
                                #raw._data[epochIndex][badElectrodeIndex, :] = interpolated_data[np.where(np.where(channelsToInterpolate)[0] == badElectrodeIndex)[0][0], :]
                            interpolation_matrix[epochIndex][badElectrodeIndex, :] = True

                        if np.all(interpolated_channels == channelsToInterpolate):
                            print('--- All bad channels were interpolated.')
                            print(
                                  f"--- Elapsed time during whole channel interpolation: {time.time() - start_time} seconds\n"
                            )
                        elif np.any(np.where(interpolated_channels)[0] == np.where(channelsToInterpolate)[0]) and not np.all(
                                interpolated_channels == channelsToInterpolate):
                            print('--- Some channels were interpolated.')
                            print(
                                  f"--- Elapsed time during whole channel interpolation: {time.time() - start_time} seconds\n"
                            )
                    else:
                        print('--- No bad channels could be interpolated.')

                else:
                    print('No bad channels to interpolate.')

            # Update EEG artifacts based on interpolation results
            if self.params['save_corrected']:
                if not hasattr(raw.artifacts, 'CCT'):
                    raw.artifacts.CCT = np.full((n_epochs, n_electrodes, n_samples), False)
                interpolationIndices = np.logical_and(interpolation_matrix, raw.artifacts.BCT)
                raw.artifacts.BCT[interpolationIndices] = False
                raw.artifacts.CCT = np.logical_or(raw.artifacts.CCT, interpolation_matrix)

                # Mark samples for the interpolated channels as bad during BT
                for epochIndex in np.arange(n_epochs):
                    interpolationIndices = interpolation_matrix[epochIndex][:, :] & np.tile(raw.artifacts.BT[epochIndex][:, :], (n_electrodes, 1))
                    raw.artifacts.BCT[epochIndex][interpolationIndices] = True

                # Mark samples for the interpolated channels as bad if bad data was used for the interpolation
                bct = raw.artifacts.BCT.copy()
                bct[np.tile(np.logical_not(raw.artifacts.BC), (1, 1, n_samples))] = False
                badChannelAggregate = np.tile(np.any(bct, axis=1), (n_electrodes, 1, 1))
                badChannelAggregate = np.moveaxis(badChannelAggregate, [0, 1], [1, 0])
                interpolationIndices = interpolation_matrix & np.tile(~raw.artifacts.BT, (1, n_electrodes, 1)) & badChannelAggregate
                raw.artifacts.BCT[interpolationIndices] = True

                interpolationIndices = np.all(interpolation_matrix, axis=2)
                raw.artifacts.BC[interpolationIndices, :] = False

                raw.artifacts.print_summary()
        
        else:
            print('No data, nothing will be done.')

        print('\n')