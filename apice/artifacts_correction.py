"""Artifact correction algorithms for APICE EEG preprocessing.

This module provides correction methods applied after artifact detection,
including target PCA-based correction and spherical spline interpolation at
channel and segment levels.
"""

# %% LIBRARIES
import os
import time
import numpy as np
import progressbar
from progressbar import (ProgressBar, Percentage, Bar)


import mne
from apice.artifacts_detection import (ShortGoodSegments, ShortBadSegments, Mask)

from apice.electrode_positions import _check_origin
from apice.utils import (find_true_segments_edges, mask_bad_segments, include_short_bad_segments, reject_short_good_segments)



def _print_header(header, separator="="):
    """Print a title wrapped by repeated separator lines.

    Parameters
    ----------
    header : str
        Header text to display.
    separator : str, default='='
        Single character (or string) repeated to form the border line.

    Returns
    -------
    None
        Prints formatted text to stdout.
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


# %% FUNCTIONS FOR SPLICING

def _splice_segments(data, bad_if, epoch_if, bct=None, bt=None, bc=None, method=1):
    """Splice corrected segments to reduce discontinuities at segment borders.

    Parameters
    ----------
    data : numpy.ndarray
        Input data array with shape (channels, samples) or (samples,).
    bad_if : numpy.ndarray
        Array of segment start/end sample indices to splice.
    epoch_if : numpy.ndarray
        Epoch boundaries as start/end sample indices.
    bct : numpy.ndarray | None, default=None
        Bad-channel-time mask.
    bt : numpy.ndarray | None, default=None
        Bad-time mask.
    bc : numpy.ndarray | None, default=None
        Bad-channel mask.
    method : int | None, default=1
        Splicing strategy. ``1`` applies linear alignment, ``None`` disables
        splicing.

    Returns
    -------
    dN : numpy.ndarray
        Spliced data with reduced boundary jumps.
    """
    if not method:
        dN = data
        print('Not splicing for alingment was performed')
    elif method == 1:
        dN = _splice_segments1(data, bad_if, epoch_if, bct=bct, bt=bt, bc=bc)
    else:
        raise Exception("the only possible method is 1 (or None) ")
    return dN


def _prepare_data_for_splicing(data, bad_if, epoch_if, bct, bt, bc):
    """Normalize splicing inputs to canonical shapes and masks.

    Parameters
    ----------
    data : numpy.ndarray
        Data as 1D or 2D array.
    bad_if : numpy.ndarray
        Bad segment boundaries.
    epoch_if : numpy.ndarray
        Epoch boundaries.
    bct : numpy.ndarray | None
        Bad-channel-time mask.
    bt : numpy.ndarray | None
        Bad-time mask.
    bc : numpy.ndarray | None
        Bad-channel mask.

    Returns
    -------
    tuple
        Normalized data/masks and segment boundary vectors used by
        splicing functions.
    """
    if np.size(np.shape(data)) == 1:
        data_ = np.reshape(data, (1, np.size(data)))
    elif np.size(np.shape(data)) == 2:
        data_ = data.copy()

    if bct is None or np.size(bct) == 0:
        bct = np.full(np.shape(data_), False)
    else:
        bct = np.reshape(bct, np.shape(data_))
        bct = bct==1

    if bt is None or np.size(bt) == 0:
        bt = np.full(np.shape(data_)[1], False)
    else:
        bt = np.reshape(bt, np.shape(data_)[1])
        bt = bt==1

    if bc is None or np.size(bc) == 0:
        bc = np.full(np.shape(data_)[0], False)
    else:
        bc = np.reshape(bc, np.shape(data_)[0])
        bc = bc==1

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
        Epoch_I = [epoch_if[0]]
        Epoch_F = [epoch_if[1]]
        
    return data_, bad_if, epoch_if, bct, bt, bc, I_all, F_all, Epoch_I, Epoch_F


def _splice_segments1(data, bad_if, epoch_if, bct=None, bt=None, bc=None):
    """Splice by aligning each segment to the previous segment endpoint.

    Parameters
    ----------
    data : numpy.ndarray
        Data to splice.
    bad_if : numpy.ndarray
        Segment start/end boundaries.
    epoch_if : numpy.ndarray
        Epoch start/end boundaries.
    bct : numpy.ndarray | None, default=None
        Bad-channel-time mask.
    bt : numpy.ndarray | None, default=None
        Bad-time mask.
    bc : numpy.ndarray | None, default=None
        Bad-channel mask.

    Returns
    -------
    dN : numpy.ndarray
        Data after segment alignment.
    """
    data, bad_if, epoch_if, bct, bt, bc, I_all, F_all, Epoch_I, Epoch_F = _prepare_data_for_splicing(data, bad_if, epoch_if, bct, bt, bc)
    
    # Splice Segments
    dN = data.copy()
    
    n_epochs = np.size(Epoch_I)
    if np.size(I_all) > 0:
        
        for iep in range(n_epochs):
            
            sample_I = I_all[(I_all >= Epoch_I[iep]) & (I_all <= Epoch_F[iep])]
            sample_F = F_all[(F_all >= Epoch_I[iep]) & (F_all <= Epoch_F[iep])]
            the_I = np.unique(np.hstack([Epoch_I, sample_I, sample_F, Epoch_F]))
            the_I = np.asarray(the_I, dtype=int)
    
            if np.size(the_I) > 2:
                
                for kk in np.arange(1, np.size(the_I) - 1):
    
                    # Identify which electrodes are fine and which ones are bad in the previous segment
                    if bt[the_I[kk] - 1]:  # If the previous segment is bad, all electrodes are bad
                        id_good_electrode = np.full(np.shape(dN)[0], False)
                        id_bad_electrode = np.full(np.shape(dN)[0], True)
                    else:
                        id_good_electrode = bct[:, the_I[kk] - 1]==False 
                        id_bad_electrode = bct[:, the_I[kk] - 1]
                    
                    # Align the segments of the electrodes with good data in 
                    # previous segment with the previous segment 
                    if np.any(np.logical_and(id_good_electrode,  bc==False)):
                        el = np.logical_and(id_good_electrode,  bc==False)
                        yadd = dN[el, the_I[kk] - 1] - data[el, the_I[kk]]
                        dN[el, the_I[kk]:the_I[kk+1]] = data[el, the_I[kk]:the_I[kk+1]] + np.tile(yadd, (the_I[kk+1]-the_I[kk], 1)).T
    
                    # If the previous segment is bad, align the previous  
                    # segment (the one with bad data, kk-1) to the surronfing  
                    # ones (kk-2 and kk) by fitting a linear trend
                    if np.any(np.logical_and(id_bad_electrode,  bc==False)):
                        el = np.logical_and(id_bad_electrode,  bc==False)
                        el = np.asarray(np.where(el)[0])
                        x = np.arange(the_I[kk - 1], the_I[kk])
                        deltaX = x[-1] - x[0]
                        if deltaX!=0:
                            for i in el:
                                deltaN = dN[i, x[-1]] - dN[i, x[-1]+1]
                                p = deltaN / deltaX
                                y_substract = p * (x - x[0])
                                dN[i, x] = dN[i, x] - y_substract
                        else:
                            print('too short segment')
                                
         
    return dN

# %% FUNCTIONS FOR PCA CORRECTION
def _find_bad_segments_pca(bad_data, intertime, n_epochs, n_samples, mask, maxtime):
    """Find artifact segments eligible for target PCA correction.

    Parameters
    ----------
    bad_data : numpy.ndarray
        Flattened bad-data mask.
    intertime : numpy.ndarray
        Boolean mask indicating time points eligible for interpolation.
    n_epochs : int
        Number of epochs.
    n_samples : int
        Number of samples per epoch.
    mask : int | float
        Mask radius in samples.
    maxtime : int | float
        Maximum allowed segment length in samples.

    Returns
    -------
    bad_if : numpy.ndarray
        Segment boundaries (start/end) in flattened sample indexing.
    """

    bad_if = []
    for ep in np.arange(n_epochs):
        
        temp1 = bad_data[ep * n_samples: (ep + 1) * n_samples]==1
        temp2 = intertime[ep * n_samples: (ep + 1) * n_samples]
        to_interpolate = np.logical_and(temp1==1, temp2==1)

        # mask
        to_interpolate, _ = mask_bad_segments(to_interpolate, mask)
        
        # Beginning and end of each segment
        badi, badf = find_true_segments_edges(to_interpolate)
        badif_ep = np.concatenate( (badi[:,np.newaxis], badf[:,np.newaxis]), axis=1)
                
        if np.shape(badif_ep)[0]>0:

            # Remove too long segments
            bad_duration_ep = badif_ep[:, 1] - badif_ep[:, 0]
            bad_duration_ep = bad_duration_ep - 2 * mask
            idx_to_remove = bad_duration_ep > maxtime
            badif_ep = np.delete(badif_ep, idx_to_remove, axis=0)
                
            if np.shape(badif_ep)[0]>0:
                # Go to total samples
                badif_ep = badif_ep + ep * n_samples
                # append
                bad_if.append(badif_ep)
    
    if bad_if:            
        bad_if = np.vstack(bad_if)
            
    return bad_if


def _target_PCA(data, bad_segment, nSV, vSV, el):
    """Apply target PCA correction on selected bad segments.

    Parameters
    ----------
    data : numpy.ndarray
        Input data matrix with electrodes x samples.
    bad_segment : numpy.ndarray
        Segment boundaries where correction should be applied.
    nSV : int | None
        Number of singular vectors/components to remove.
    vSV : float | None
        Cumulative explained variance threshold for component removal.
    el : int | None
        Electrode index to map corrected output back.

    Returns
    -------
    data : numpy.ndarray
        Corrected data matrix.
    tC : numpy.ndarray
        Boolean mask marking corrected time points.
    """

    if el is None or np.size(el) == 0:
        el = np.full(np.shape(data)[0], True)

    tC = np.full(np.shape(data)[1], False)
        
    # Get the data to correct and remove the mean
    y = data[:,int(bad_segment[0, 0]):int(bad_segment[0, 1])].T
    y = y - np.nanmean(y, axis=0)
    tC[int(bad_segment[0, 0]):int(bad_segment[0, 1])] = True
    for iseg in np.arange(1, np.shape(bad_segment)[0]):
        yi = data[:, int(bad_segment[iseg, 0]):int(bad_segment[iseg, 1])].T
        yi = yi - np.nanmean(yi, axis=0)
        y = np.concatenate((y, yi))
        tC[int(bad_segment[iseg, 0]):int(bad_segment[iseg, 1])] = True
    y = y - np.nanmean(y, axis=0)

    # PCA
    cov_matrix = np.cov(y, rowvar=False)
    [eigenvalues, V] = np.linalg.eigh(cov_matrix)
    eigenvectors = V[:, np.arange(np.shape(cov_matrix)[0] - 1, -1, -1)]
    score = np.dot(y, eigenvectors)
    exp_var = np.var(score, axis=0).T
    exp_var = exp_var / np.sum(exp_var)
    
    n_remove = 0
    if vSV:
        n_remove_var = np.where(np.cumsum(exp_var) >= vSV)[0][0]+1
        n_remove = np.max((n_remove_var, n_remove))
    if nSV:
        n_remove = np.max((nSV, n_remove))

    ev = np.zeros(np.shape(eigenvectors)[0])
    ev[0:n_remove ] = 1
    ev = np.diag(ev)

    yc = y - np.matmul(np.matmul(y, eigenvectors), np.matmul(ev, eigenvectors.T))

    # Store the corrected data
    data[el, tC] = yc[:, el]
        
    return data, tC



# %% FUNCTIONS FOR SPHERICAL SPLINE INTERPOLATION
from joblib import Parallel, delayed
from multiprocessing import Process, Manager

def _process_bad_channel(bad_idx, ch_names, ch_names_montage, positions, bad_channel_indices, new_exclude_index, data, distances_matrix):
    """Interpolate one bad channel using spherical spline neighbors.

    Parameters
    ----------
    bad_idx : int
        Index of the bad channel to interpolate.
    ch_names : list[str]
        Channel names in data order.
    ch_names_montage : list[str]
        Channel names in montage order.
    positions : numpy.ndarray
        Electrode positions.
    bad_channel_indices : list[int]
        Indices of all bad channels.
    new_exclude_index : list[int]
        Indices to exclude from neighborhood interpolation.
    data : numpy.ndarray
        EEG data matrix (channels x samples).
    distances_matrix : numpy.ndarray
        Pairwise electrode distance matrix.

    Returns
    -------
    tuple[int, numpy.ndarray | None]
        Channel index and interpolated signal, or ``None`` if interpolation
        failed.
    """
    
    #Processes a single bad channel and performs spherical spline interpolation.    
    bad_ch_name = ch_names[bad_idx]
    bad_pos = positions[ch_names_montage.index(ch_names[bad_idx])]
    bad_pos = np.reshape(bad_pos, (1, 3))

    distances = distances_matrix[bad_idx,:].copy()
    distances[bad_channel_indices] = np.inf
    distances[new_exclude_index] = np.inf
    
    neighbors_idx = np.where(distances < np.inf)[0]
   
    #neighbor_data = data[good_electrodes, :]
    #neighbor_positions = positions[good_electrodes]

    neighbor_data = data[neighbors_idx, :]
    neighbor_positions = positions[neighbors_idx]
    
    try:
        interpolated_row = _spherical_spline_inter(neighbor_positions, bad_pos, neighbor_data)
        return bad_idx, interpolated_row # Return the bad_idx and the interpolated row.
    except Exception as e:
        print(f"Spherical Spline interpolation failed for channel {bad_ch_name}: {e}")
        return bad_idx, None #return the bad_idx and None if there is an error.

def _parallel_interpolate(bad_channel_indices, ch_names, ch_names_montage, positions, new_exclude_index, data, distances_matrix, n_jobs):
    """Run channel-wise spline interpolation in parallel.

    Parameters
    ----------
    bad_channel_indices : list[int]
        Bad channels to interpolate.
    ch_names : list[str]
        Channel names in data order.
    ch_names_montage : list[str]
        Channel names in montage order.
    positions : numpy.ndarray
        Electrode positions.
    new_exclude_index : list[int]
        Indices to exclude from interpolation neighborhoods.
    data : numpy.ndarray
        EEG data matrix (channels x samples).
    distances_matrix : numpy.ndarray
        Pairwise distance matrix.
    n_jobs : int
        Number of parallel workers.

    Returns
    -------
    result_interpolation : numpy.ndarray
        Data matrix with interpolated bad channels replaced when successful.
    """
    #Parallelizes the spherical spline interpolation for bad channels.
    result_interpolation = np.copy(data) # Create a copy to store the interpolated data

    results = Parallel(n_jobs=n_jobs)(delayed(_process_bad_channel)(bad_idx, ch_names, ch_names_montage, positions, bad_channel_indices, new_exclude_index, data, distances_matrix) for bad_idx in bad_channel_indices)

    for bad_idx, interpolated_row in results:
        if interpolated_row is not None:
            result_interpolation[bad_idx, :] = interpolated_row # Place the row into the copied array.
    
    return result_interpolation

def _do_spherical_spline_interpolation(raw, distances_matrix, positions, adjacency_matrix, bad_neighbor_proportion, bad_channels_to_interpolate, all_bad_channels=None, interpolation_channels=False, n_jobs=-1):
    """Interpolate selected bad channels using spherical spline interpolation.

    Parameters
    ----------
    raw : RawAPICE
        Input recording object containing channel info and data.
    distances_matrix : numpy.ndarray
        Pairwise distance matrix between electrodes.
    positions : numpy.ndarray
        Electrode Cartesian coordinates.
    adjacency_matrix : numpy.ndarray
        Binary adjacency matrix defining neighboring electrodes.
    bad_neighbor_proportion : float
        Maximum allowed proportion of bad neighbors for a channel to be
        considered interpolable.
    bad_channels_to_interpolate : array_like
        Boolean/index mask for channels targeted for interpolation.
    all_bad_channels : array_like | None, default=None
        Boolean/index mask of all currently bad channels.
    interpolation_channels : bool, default=False
        If True, use channel-level interpolation mode.
    n_jobs : int, default=-1
        Number of parallel workers.

    Returns
    -------
    new_interpolated_data : numpy.ndarray
        Data matrix with interpolated channels replaced when successful.
    interpolated_bad_channels : numpy.ndarray
        Boolean vector indicating channels that were interpolated.
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
    new_interpolated_data = _parallel_interpolate(bad_channel_indices, ch_names, ch_names_montage, positions, new_exclude_index, data, distances_matrix, n_jobs=n_jobs)
    
    # Identify which channels were successfully interpolated
    interpolated_bad_channels = np.full(len(raw.ch_names), False)
    for el in range(len(raw.ch_names)):
        #interpolated_bad_channels[el] = (raw.ch_names[el] in bads_list)
        interpolated_bad_channels[el] = (raw.ch_names[el] in [ch_names[i] for i in bad_channel_indices])
        raw.info["bads"] = [ch for ch in raw.info["bads"] if ch in [ch_names[i] for i in bad_channel_indices]]

    return new_interpolated_data, interpolated_bad_channels

from numpy.polynomial.legendre import legval
from scipy.linalg import pinv

def _calc_g(cosang, stiffness=4, n_legendre_terms=7):
    """Compute Perrin spherical spline kernel values.

    Parameters
    ----------
    cosang : numpy.ndarray
        Cosine of angular distances between electrode positions.
    stiffness : int, default=4
        Spline stiffness parameter.
    n_legendre_terms : int, default=7
        Number of Legendre polynomial terms.

    Returns
    -------
    numpy.ndarray
        Kernel matrix values used for spherical spline interpolation.
    """
    factors = [
        (2 * n + 1) / (n**stiffness * (n + 1) ** stiffness * 4 * np.pi)
        for n in range(1, n_legendre_terms + 1)
    ]
    return legval(cosang, [0] + factors)

def _normalize_vectors(rr):
    """Normalize surface vertices."""
    size = np.linalg.norm(rr, axis=1)
    mask = size > 0
    rr[mask] /= size[mask, np.newaxis]  # operate in-place
    return size

def _spherical_spline_inter(good_pos, bad_pos, good_data):
    """Interpolate bad positions from good channels using spherical splines.

    Parameters
    ----------
    good_pos : numpy.ndarray
        Cartesian coordinates of good electrodes (n_good, 3).
    bad_pos : numpy.ndarray
        Cartesian coordinates of target electrodes (n_bad, 3).
    good_data : numpy.ndarray
        Data from good electrodes (n_good, n_samples).

    Returns
    -------
    interpdata : numpy.ndarray
        Interpolated data at bad electrode positions.
    """

    _normalize_vectors(good_pos)
    _normalize_vectors(bad_pos)
    
    Gelec = _calc_g(good_pos.dot(good_pos.T)) # from
    Gsph = _calc_g(bad_pos.dot(good_pos.T)) # to

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


def _build_interpolation_matrix_spline(raw, min_good_time, min_intertime, mask_time, min_segment_time):
    """Build a cleaned interpolation mask for segment-wise spline correction.

    Parameters
    ----------
    raw : RawAPICE
        Input recording with artifact matrices.
    min_good_time : float
        Minimum good segment duration (seconds) to preserve.
    min_intertime : float
        Minimum bad segment duration (seconds) required for interpolation.
    mask_time : float
        Temporal mask extension around bad segments (seconds).
    min_segment_time : float
        Minimum segment duration (seconds) used for segment consolidation.

    Returns
    -------
    numpy.ndarray
        Boolean/integer interpolation matrix aligned with ``raw.artifacts.BCT``.
    """

    n_electrodes, n_samples, n_epochs = raw.get_data_size()
    raw_int=raw.copy()

    # remove segments to interpolate that are too short
    shortbad = ShortBadSegments(time_limit=min_intertime, verbose=False)
    shortbad.reject(raw_int)
    
    # mask interpolation segmets
    maskrej = Mask(mask_length=mask_time, verbose=False)
    maskrej.reject(raw_int)

    # mark as interpolation segmets too short no interpolation segmets
    shortgood = ShortGoodSegments(time_limit=min_good_time, verbose=False)
    shortgood.reject(raw_int)

    # remove bad times and bad channels from the interpolation matrix
    if len(raw_int._data.shape)==2:
        # remove bad channels
        raw_int.artifacts.BCT[np.tile(raw.artifacts.BC,(1, n_samples))] = 0
        # remove bad times
        raw_int.artifacts.BCT[np.tile(raw.artifacts.BT,(n_electrodes, 1))] = 0
    elif len(raw_int._data.shape)==3:
        # remove bad channels
        raw_int.artifacts.BCT[np.tile(raw.artifacts.BC,(1, 1, n_samples))] = 0
        # remove bad times
        raw_int.artifacts.BCT[np.tile(raw.artifacts.BT,(1, n_electrodes, 1))] = 0

    # remove segments to interpolate that are too short
    shortbad = ShortBadSegments(time_limit=min_intertime, verbose=False)
    shortbad.reject(raw_int)

    # modify the rejection to avoid having too short separate segments to 
    # interpolate due to different channels to interpolate
    segment_samples = int(np.round(min_segment_time*raw.info['sfreq']))
    max_samples = int(np.round(mask_time/2*raw.info['sfreq']))
    if len(raw_int._data.shape)==2:
        raw_int.artifacts.BCT = _modify_rej(raw_int.artifacts.BCT, segment_samples, max_samples)
    elif len(raw_int._data.shape)==3:
        for iep in range(n_epochs):
            raw_int.artifacts.BCT[iep,:,:] = _modify_rej(raw_int.artifacts.BCT[iep,:,:], segment_samples, max_samples)

    return raw_int.artifacts.BCT
    

def _modify_rej(rej, segment_samples, max_samples):
    """Regularize rejection mask within fixed-length segments.

    Parameters
    ----------
    rej : numpy.ndarray
        Rejection matrix (channels x samples).
    segment_samples : int
        Number of samples per segment.
    max_samples : int
        Maximum rejected samples per segment before marking the full segment
        as rejected for that channel.

    Returns
    -------
    rej_out : numpy.ndarray
        Regularized rejection matrix.
    """
    
    n_samples = np.shape(rej)[1]
    
    # initialize new rejection matrix
    rej_out = np.full(np.shape(rej), 0, dtype=int)
    
    # define segments
    seg = np.arange(0, n_samples-segment_samples, segment_samples)
    
    # iterate over segments
    for segi in seg:
        m = rej[:,segi:segi+segment_samples]
        nreji = np.sum(m,axis=1)
        idx_ch_rej = nreji>max_samples
        rej_out[idx_ch_rej, segi:segi+segment_samples] = 1
        
    # last segment
    segi = seg[-1]+segment_samples
    m = rej[:,segi:]
    nreji = np.sum(m,axis=1)
    idx_ch_rej = nreji>max_samples
    rej_out[idx_ch_rej, segi:] = 1
    
    return rej_out

def _find_segments_interpolation_spline(rej, bad_channels):
    """Find contiguous interpolation segments and involved channels.

    Parameters
    ----------
    rej : numpy.ndarray
        Rejection matrix (channels x samples).
    bad_channels : numpy.ndarray
        Boolean mask of channels excluded from interpolation.

    Returns
    -------
    bad_if : numpy.ndarray
        Segment boundaries (start/end indices).
    cha_interpolate : list[numpy.ndarray]
        Channel masks indicating channels to interpolate per segment.
    """
    
    # remove bad channesl from rejection
    rej[bad_channels,:] = 0
    
    # find segments where changes happend
    bad_if = _find_segments_change(rej)
    n_segm = np.shape(bad_if)[0]
    
    # keep the segments with something to interpolate
    segm_rmv = np.full((n_segm), False)
    for i in range(n_segm):
        if not np.any(np.any(rej[:,bad_if[i,0]:bad_if[i,1]]==1, axis=1)):
            segm_rmv[i] = True
    bad_if = bad_if[segm_rmv==False,:]
    n_segm = np.shape(bad_if)[0]
    
    # get the channels to interpolate in each segment
    cha_interpolate = [np.any(rej[:,segmi[0]:segmi[1]]==1, axis=1) for segmi in bad_if]
    # cha_interpolate = [list(np.where(np.any(rej[:,segmi[0]:segmi[1]]==1, axis=1))[0]) for segmi in bad_if]
     
    return bad_if, cha_interpolate

def _find_segments_change(rej):
    """Compute segment boundaries where rejection pattern changes.

    Parameters
    ----------
    rej : numpy.ndarray
        Rejection matrix (channels x samples).

    Returns
    -------
    bad_if : numpy.ndarray
        Segment boundaries (start/end indices).
    """
    
    # Identify the changes in the number of bad channels
    n_samples = np.shape(rej)[1]
    change_channel = np.where(np.any(np.diff(rej, axis=1), axis=0))[0]
    seg_i = np.unique(np.hstack([change_channel + 1, 0]))
    seg_f = np.unique(np.hstack([change_channel + 1, n_samples]))
    bad_if = np.asarray([seg_i.T, seg_f.T]).T
    
    return bad_if


def _interpolate_spline_segment_task(raw, raw_data_2d, segment_start, segment_end,
                                     bad_channels_to_interpolate, all_bad_channels,
                                     distance_matrix, positions, adjacency_matrix,
                                     p_neighbors, n_jobs):
    """Interpolate one segment and return only the updated channels for safe merging."""
    cropped_eeg = raw.copy()
    cropped_eeg._data = raw_data_2d[:, segment_start:segment_end]
    seg_len = int(segment_end - segment_start)
    cropped_eeg._times = np.arange(seg_len, dtype=float) / raw.info['sfreq']

    interpolated_data, interpolated_bad_channels = _do_spherical_spline_interpolation(
        cropped_eeg,
        distance_matrix,
        positions,
        adjacency_matrix,
        p_neighbors,
        bad_channels_to_interpolate,
        all_bad_channels,
        False,
        n_jobs,
    )

    bad_ch = np.where(interpolated_bad_channels)[0]
    if np.size(bad_ch) == 0:
        return segment_start, segment_end, bad_ch, np.empty((0, seg_len))

    return segment_start, segment_end, bad_ch, interpolated_data[bad_ch, :]

# %% CLASSES FOR ARTIFACTS CORRECTION

class ArtCorrection:
    """Base class for artifact correction algorithms.

    Provides shared parameter handling and bookkeeping utilities used by
    concrete correction methods.

    Parameters
    ----------
    verbose : bool, default=True
        Whether to print correction progress details.
    """

    def __init__(self, verbose=True):

        # Arreange parameters
        self.params = dict()
        self.verbose = verbose

    def steps_pre_interpolation(self):
        # print info
        if self.verbose:
            print('\nInterpolation parameters')
            for k in self.params.keys():
                print('-- {}: '.format(k), self.params[k])
        
    def steps_post_interpolation(self, raw, interpolation_matrix):
        
        n_electrodes, n_samples, n_epochs = raw.get_data_size()
        
        # Mark the interpolated data
        if self.params['save_corrected']:
            BCT = np.reshape(raw.artifacts.BCT.copy(), (n_epochs, n_electrodes, n_samples))
            INT = np.reshape(interpolation_matrix.copy(), (n_epochs, n_electrodes, n_samples))
            BCT[np.logical_and(INT, BCT==1)] = 0
            
            # reshape if necessary
            if len(raw._data.shape)==2:
                BCT = np.squeeze(np.transpose(BCT,(1,2,0)), axis=2) 
                INT = np.squeeze(np.transpose(INT,(1,2,0)), axis=2)
                
            # update the rejection
            raw.artifacts.BCT = BCT
            raw.artifacts.CCT[INT] = True
            
        # save interpolated data
        self.interpolation_matrix = interpolation_matrix
        self.interpolated = np.sum(interpolation_matrix.flatten())/np.size(interpolation_matrix)
        

class TargetPCA(ArtCorrection):
    """Artifact correction based on target PCA per electrode.

    Parameters
    ----------
    max_time : float, default=0.100
        Maximum segment duration in seconds eligible for correction.
    components_to_remove : int | None, default=None
        Fixed number of components to remove.
    variance_to_remove : float, default=0.98
        Cumulative variance threshold for component removal.
    mask_time : float, default=0.05
        Temporal masking around bad segments in seconds.
    all_time : str, default='all'
        Time selection strategy for PCA.
    all_channel : str, default='no_bad_channel'
        Channel selection strategy for PCA.
    all_epochs : str, default='all'
        Epoch selection strategy for PCA.
    splice_method : int, default=1
        Splicing strategy applied after correction.
    save_corrected : bool, default=True
        Whether to update correction masks in ``raw.artifacts``.
    """

    def __init__(self, max_time=0.100, components_to_remove=None, variance_to_remove=0.98, mask_time=0.05,
                 all_time='all', all_channel='no_bad_channel', all_epochs='all', splice_method=1, save_corrected=True):
        
        super().__init__()
        
        # Arreange parameters for rejection
        self.params = dict(max_time=max_time, components_to_remove=components_to_remove,
                           variance_to_remove=variance_to_remove, mask_time=mask_time,
                           all_time=all_time, all_channel=all_channel, all_epochs=all_epochs,
                           save_corrected=save_corrected, index_to_remove=[], splice_method=splice_method, verbose=True)
        

    def correct(self, raw):         
        """Run target PCA correction and update artifact bookkeeping.

        Parameters
        ----------
        raw : RawAPICE
            Input object to correct in place.

        Returns
        -------
        None
            Data are modified in place.
        """
            
        _print_header('Performing Target PCA per Electrode', separator="-")
        
        # Steps pre interpolation
        self.steps_pre_interpolation()
        
        # Target PCA
        raw._data, interpolation_matrix = self.apply_target_PCA(raw, 
                                                                max_time=self.params['max_time'],
                                                                components_to_remove=self.params['components_to_remove'],
                                                                variance_to_remove=self.params['variance_to_remove'],
                                                                mask_time=self.params['mask_time'],
                                                                all_time=self.params['all_time'],
                                                                all_channel=self.params['all_channel'],
                                                                all_epochs=self.params['all_epochs'],
                                                                splice_method=self.params['splice_method'],
                                                                )
        
        # Steps post interpolation
        self.steps_post_interpolation(raw, interpolation_matrix)
            
        print('\n')
     
    @staticmethod
    def apply_target_PCA(raw, max_time=0.100, components_to_remove=None, variance_to_remove=0.98, mask_time=0.05,
                 all_time='all', all_channel='no_bad_channel', all_epochs='all', splice_method=1):
        """Apply target PCA correction and return corrected data plus mask.

        Parameters
        ----------
        raw : RawAPICE
            Input raw/epochs object.
        max_time : float, default=0.100
            Maximum segment duration in seconds for PCA correction.
        components_to_remove : int | None, default=None
            Number of components to remove.
        variance_to_remove : float, default=0.98
            Cumulative explained variance threshold for component removal.
        mask_time : float, default=0.05
            Temporal mask in seconds around bad segments.
        all_time : str, default='all'
            Time selection mode.
        all_channel : str, default='no_bad_channel'
            Channel selection mode.
        all_epochs : str, default='all'
            Epoch selection mode.
        splice_method : int, default=1
            Splicing method for corrected segments.

        Returns
        -------
        data_pca : numpy.ndarray
            Corrected data.
        interpolation_matrix : numpy.ndarray
            Boolean mask of corrected samples.
        """

        # get data size
        n_electrodes, n_samples, n_epochs_all = raw.get_data_size()
        
        # get the times that are considered for PCA
        if all_time == 'all':
            if len(np.shape(raw._data))==2:
                IT = np.full((1, n_samples), True)
            else:
                IT = np.full((n_epochs_all, 1, n_samples), True)    
        elif all_time == 'no_bad_time':
            IT = raw.artifacts.BT.copy()==False
        elif all_time == 'bad_time':
            IT = raw.artifacts.BT.copy()==True

        # get the channels that are considered for PCA
        if all_channel == 'all':
            if len(np.shape(raw._data))==2:
                IC = np.full(n_electrodes, True)
            else:
                IC = np.full((n_epochs_all, n_electrodes), True)
        elif all_channel == 'no_bad_channel':
            IC = raw.artifacts.BC.copy()==False
            IC = np.reshape(IC, (n_epochs_all, n_electrodes))
        elif all_channel == 'bad_channel':
            IC = raw.artifacts.BC.copy()==True
            IC = np.reshape(IC, (n_epochs_all, n_electrodes))
        
        # get the epochs considered for PCA
        if len(np.shape(raw._data))==2:
            IE = np.full(1, True)
        else:
            if all_epochs == 'all':
                IE = np.full((n_epochs_all, 1, 1), True)
            elif all_epochs == 'no_bad_epoch':
                IE = raw.artifacts.BE.copy()==False
            elif all_epochs == 'bad_epoch':
                IE = raw.artifacts.BE.copy()==True
        get_some_epochs = np.any(IE==0)
        
        # get the data and rejection matrix
        DATA = raw._data.copy()
        BCT = raw.artifacts.BCT.copy()
        BT = raw.artifacts.BT.copy()
        
        # mark bad times as bad for all channnels
        if len(np.shape(raw._data))==2:
            BCT[np.tile(BT,(n_electrodes,1))] = 1
        else:
            BCT[np.tile(BT,(1, n_electrodes, 1))] = 1
        
        # get epochs to interpolate
        if get_some_epochs and len(np.shape(raw._data))==3:
            DATA = DATA[IE,:,:]
            BCT = BCT[IE,:,:]
            BT = BT[IE,:,:]
            IC = IC[IE,:]
            IT = IT[IE,:,:]
            n_epochs = np.sum(IE)
        else:
            n_epochs = n_epochs_all
        
        # reshape
        if len(np.shape(raw._data))==3:
            DATA = np.reshape(np.transpose(DATA, (1,2,0)), (n_electrodes, n_samples*n_epochs), order='F')
            BCT = np.reshape(np.transpose(BCT, (1,2,0)), (n_electrodes, n_samples*n_epochs), order='F')
            IT = np.reshape(np.transpose(IT, (1,2,0)), (1, n_samples*n_epochs), order='F')
            IC = np.any(IC,axis=0)
            
        IT = np.reshape(IT, n_samples*n_epochs)
        IC = np.reshape(IC, n_electrodes)
        
        # initialize rejection matrix
        interpolation_matrix = np.full((n_electrodes, n_samples*n_epochs), False)

        # normalize the data so all electrodes have equal variance
        norm_electrodes = np.full((n_electrodes, 1), np.nan)
        for el in np.arange(n_electrodes):
            bt = BCT[el, :]==1
            d = DATA[el, :]
            if np.sum(bt==False) > 2:
                norm_electrodes[el] = np.nanstd(d[bt==False])
        idx = (np.isnan(norm_electrodes) | (norm_electrodes < 1e-9)) # 1e-3
        norm_electrodes[idx] = np.nanmean(norm_electrodes[idx==False])
        DATA = np.divide(DATA, np.tile(norm_electrodes, (1, n_samples*n_epochs)))
        DATA[np.isnan(DATA)] = 0

        # Apply PCA
        data_pca = DATA.copy()
        print('\n')
        for el in np.arange(n_electrodes):
    
            if IC[el]:
                
                # print info
                print('Electrode ', el + 1, ': ', end='')
    
                # index for the channel within the included ones
                good_electrode = np.sum(IC[0:el])
    
                # Find the segments to interpolate
                bad_if = _find_bad_segments_pca(BCT[el, :], IT, n_epochs, n_samples,
                                                mask_time * raw.info['sfreq'],
                                                max_time * raw.info['sfreq']
                                                )
    
                if np.size(bad_if) > 0:
    
                    print('Total bad data ', np.sum(BCT[el, :]), '(',
                          np.round(np.sum(BCT[el, :]) / (n_samples * n_epochs) * 100, 2), '%).',
                          'Data to apply PCA ', int(np.sum(bad_if[:, 1] - bad_if[:, 0])),
                          '(', np.round(np.sum(bad_if[:, 1] - bad_if[:, 0]) / (n_samples * n_epochs) * 100, 2),
                          '%).')
    
                    # Target PCA
                    d, tC = _target_PCA(DATA[IC, :], bad_if, components_to_remove, variance_to_remove, good_electrode)
                    d = d[good_electrode, :]
                    interpolation_matrix[el, tC] = True
    
                    # Splice the segments of data together
                    if splice_method:
                        epoch_i = np.arange(0, n_samples*n_epochs, n_samples)
                        epoch_f = epoch_i+n_samples
                        epoch_if = np.concatenate((epoch_i[:,np.newaxis], epoch_f[:,np.newaxis]), axis=1)
                        d = _splice_segments(d, bad_if, epoch_if, method=splice_method)
    
                    # Store the data
                    data_pca[el, :] = d
                else:
                    print('No segments to apply target PCA were found.')
    
            else:
                print('Electrode ', el + 1, ': Bad channel.')

        # Rescale the data back
        data_pca = data_pca * np.tile(norm_electrodes, (1, n_samples * n_epochs))
        DATA = DATA * np.tile(norm_electrodes, (1, n_samples * n_epochs))

        # Reshape the data
        if len(np.shape(raw._data))==3:
            data_pca = np.reshape(data_pca, (n_electrodes, n_samples, n_epochs), order='F')
            data_pca = data_pca.transpose(2,0,1)
            interpolation_matrix = np.reshape(interpolation_matrix, (n_electrodes, n_samples, n_epochs), order='F')
            interpolation_matrix = interpolation_matrix.transpose(2,0,1)
            
        # If only some epochs were corrected put the corrected data in arrays the size of the original data
        if get_some_epochs and len(np.shape(raw._data))==3:
            data_pca_ = data_pca.copy()
            interpolation_matrix_ = interpolation_matrix.copy()
            data_pca = raw._data.copy()
            interpolation_matrix = np.full(np.shape(raw._data), False, dtype=bool)
            data_pca[IE,:,:] = data_pca_
            interpolation_matrix[IE,:,:] = interpolation_matrix_

        return data_pca, interpolation_matrix


class ChannelsSphericalSplineInterpolation(ArtCorrection):
    """Spherical spline interpolation for globally bad channels.

    Parameters
    ----------
    p : float, default=0.3
        Maximum allowed proportion of bad channels to attempt interpolation.
    p_neighbors : float, default=1
        Maximum proportion of bad neighbors accepted for interpolation.
    save_corrected : bool, default=True
        Whether to update correction masks in ``raw.artifacts``.
    verbose : bool, default=True
        Whether to print progress information.
    n_jobs : int, default=-1
        Number of parallel workers.
    """

    def __init__(self, p=0.3, p_neighbors=1, save_corrected=True, verbose=True, n_jobs=-1):
        
        super().__init__(verbose=verbose)
        
        # Arrange parameters for rejection
        self.params = dict(p=p, p_neighbors=p_neighbors, save_corrected=save_corrected, verbose=verbose, n_jobs=n_jobs)

    def correct(self, raw):         
        """Run channel-wise spline interpolation and update bookkeeping.

        Parameters
        ----------
        raw : RawAPICE
            Input object to correct in place.

        Returns
        -------
        None
            Data are modified in place.
        """
            
        _print_header('Performing Spherical Spline Interpolation (Bad channels)', separator="-")
        
        # Steps pre interpolation
        self.steps_pre_interpolation()
        
        # Interpolation channels
        raw._data, interpolation_matrix = self.spherical_spline_interpolation(raw, self.params['p'], self.params['p_neighbors'], n_jobs=self.params['n_jobs'])

        # Steps post interpolation
        self.steps_post_interpolation(raw, interpolation_matrix)
            
        print('\n')

    @staticmethod
    def spherical_spline_interpolation(raw, p, p_neighbors, n_jobs=-1):
        """Interpolate bad channels using spherical splines.

        Parameters
        ----------
        raw : RawAPICE
            Input raw/epochs object.
        p : float
            Maximum proportion of bad channels to allow interpolation.
        p_neighbors : float
            Maximum proportion of bad neighbors for eligible interpolation.
        n_jobs : int, default=-1
            Number of parallel workers.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            Corrected data and interpolation mask.
        """

        n_electrodes, n_samples, n_epochs = raw.get_data_size()
        good_channels = np.reshape(raw.artifacts.BC,(n_epochs, n_electrodes, 1), order='F')==False
        if np.size(raw._data) > 0:

            # Get adjacency matrix
            print('\nExtracting electrode adjacency matrix.')
            adjacency_matrix = mne.channels.find_ch_adjacency(raw.info, 'eeg')[0].toarray()
            
            # Computing the distances only once
            from scipy.spatial.distance import cdist
            spec_ch_pos = raw.info.get_montage().get_positions()['ch_pos']
            positions = np.array(list(spec_ch_pos.values())) - _check_origin("auto", raw.info)
            distance_matrix = cdist(positions, positions, metric='euclidean')

            # Initialize the interpolation matrix to False
            interpolation_matrix = np.full((n_epochs, n_electrodes, n_samples), False)

            # Copy and reshape the raw EEG data for processing
            eeg_data = raw._data.copy()
            eeg_data = np.reshape(eeg_data, (n_epochs, n_electrodes, n_samples), order='F')
            eeg_copy = raw.copy()
            eeg_copy._data = []
            eeg_copy._times = []
            
            # Loop through each epoch to handle bad channels
            for epochIndex in np.arange(n_epochs):
                # Identify bad channels and interpolate them
                channelsToInterpolate = ~good_channels[epochIndex][:, :][:,0]
                print('\nEpoch :', epochIndex + 1)

                if np.any(channelsToInterpolate):
                    start_time = time.time()
                    
                    print('Bad electrodes: ', [raw.info['ch_names'][i] for i in np.where(channelsToInterpolate)[0]])

                    # Get a copy of the EEG data and info
                    eeg_copy._data = eeg_data[epochIndex][:, :]                    
                    eeg_copy._times = raw.times.copy()

                    # Perform spline interpolation
                    if np.sum(channelsToInterpolate) / n_electrodes <= p:
                        interpolated_data, interpolated_channels = _do_spherical_spline_interpolation(eeg_copy, 
                                                                                                distance_matrix,
                                                                                                positions,
                                                                                                adjacency_matrix, 
                                                                                                p_neighbors, 
                                                                                                channelsToInterpolate, 
                                                                                                channelsToInterpolate,
                                                                                                True, n_jobs) 
                                                                                              
                        # Store data
                        for badElectrodeIndex in np.where(channelsToInterpolate)[0]:
                            if len(raw._data.shape) == 2:
                                raw._data[badElectrodeIndex, :] = interpolated_data[badElectrodeIndex, :]
                            elif len(raw._data.shape) == 3:
                                raw._data[epochIndex][badElectrodeIndex, :] = interpolated_data[badElectrodeIndex, :]
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

        # reshape if necessary         
        if len(raw._data.shape) == 2:
            interpolation_matrix = np.squeeze(np.transpose(interpolation_matrix,(1,2,0)), axis=2)
            
        return raw._data, interpolation_matrix



class SegmentSphericalSplineInterpolation(ArtCorrection):
    """Spherical spline interpolation applied on bad temporal segments.

    Parameters
    ----------
    n_jobs : int, default=-1
        Number of parallel workers.
    p : float, default=0.5
        Maximum proportion of bad channels allowed for interpolation per
        segment.
    p_neighbors : float, default=1
        Maximum proportion of bad neighbors for interpolation eligibility.
    min_good_time : float, default=1.00
        Minimum good segment duration in seconds.
    min_intertime : float, default=0.100
        Minimum bad segment duration in seconds for interpolation.
    mask_time : float, default=0.100
        Mask duration in seconds around bad segments.
    min_segment_time : float, default=0.200
        Minimum segment length in seconds for rejection regularization.
    splice_method : int, default=1
        Splicing method after interpolation.
    save_corrected : bool, default=True
        Whether to update correction masks in ``raw.artifacts``.
    verbose : bool, default=True
        Whether to print progress information.
    parallelize_mode : {'auto', 'channels', 'segments'}, default='auto'
        Parallelization strategy.
    """

    def __init__(self, n_jobs=-1, p=0.5, p_neighbors=1, min_good_time=1.00, min_intertime=0.100, mask_time=0.100, 
                 min_segment_time=0.200, splice_method=1, save_corrected=True, verbose=True,
                 parallelize_mode='auto'):
        
        super().__init__(verbose=verbose)

        valid_parallelize_mode = {'auto', 'channels', 'segments'}
        if parallelize_mode not in valid_parallelize_mode:
            raise ValueError(
                "parallelize_mode must be one of {'auto', 'channels', 'segments'}, "
                f"got {parallelize_mode!r}"
            )
        
        # Arreange parameters for rejection
        self.params = dict(n_jobs=n_jobs, p=p, p_neighbors=p_neighbors, min_good_time=min_good_time,
                           min_intertime=min_intertime, mask_time=mask_time, 
                           min_segment_time=min_segment_time,
                           splice_method=splice_method,
                           save_corrected=save_corrected, verbose=verbose,
                           parallelize_mode=parallelize_mode)
    
    def correct(self, raw):
        """Run segment-wise spline interpolation and update bookkeeping.

        Parameters
        ----------
        raw : RawAPICE
            Input object to correct in place.

        Returns
        -------
        None
            Data are modified in place.
        """
        
        _print_header('Performing Spherical Spline Interpolation per Segment', separator="-")

        # Steps pre interpolation
        self.steps_pre_interpolation()
        
        # Interpolation of Spatial Segments
        raw._data, interpolation_matrix = self._spherical_spline_interpolation(raw)
        
        # Steps post interpolation
        self.steps_post_interpolation(raw, interpolation_matrix)
            
        print('\n')

    def _spherical_spline_interpolation(self, raw):
        """Interpolate bad channel-time segments using spherical splines.

        Parameters
        ----------
        raw : RawAPICE
            Input raw/epochs object.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            Interpolated data and interpolation mask.
        """

        n_channels, n_samples, n_epochs = raw.get_data_size()

        # Find the segments to interpolate
        print('\nDefining data to interpolate...')
        bct_to_interpolate = _build_interpolation_matrix_spline(raw, self.params['min_good_time'], self.params['min_intertime'], self.params['mask_time'], self.params['min_segment_time'])

        # Computing the distances only once
        from scipy.spatial.distance import cdist
        spec_ch_pos = raw.info.get_montage().get_positions()['ch_pos']
        positions = np.array(list(spec_ch_pos.values())) - _check_origin("auto", raw.info)
        distance_matrix = cdist(positions, positions, metric='euclidean')

        # Get adjacency matrix
        print('\nExtracting electrode adjacency matrix.')
        adjacency_matrix = mne.channels.find_ch_adjacency(raw.info, 'eeg')[0].toarray()


        badIntervals = []
        chaInterpolate = []
        allBadChannels = []
        for ep in np.arange(n_epochs):
            
            # bad channels in that epoch are not interpolated
            if len(np.shape(raw._data))==2:
                badChannelsPerEpoch = np.squeeze(raw.artifacts.BC[:,0]) | raw.artifacts.BCmanual
            elif len(np.shape(raw._data))==3:
                badChannelsPerEpoch = np.squeeze(raw.artifacts.BC[ep,:,0]) | raw.artifacts.BCmanual
            
            # data to interpolate
            if len(np.shape(raw._data))==2:
                bct_to_interpolate_ep = bct_to_interpolate
            elif len(np.shape(raw._data))==3:
                bct_to_interpolate_ep = np.squeeze(bct_to_interpolate[ep,:,:])
                
            # Identify the segments and channels to interpolate
            badIntervals_ep, chaInterpolate_ep = _find_segments_interpolation_spline(bct_to_interpolate_ep, badChannelsPerEpoch)
            for segmentIndex in np.arange(np.shape(badIntervals_ep)[0]):
                allBadChannels_ep = chaInterpolate_ep[segmentIndex] | badChannelsPerEpoch
                # Transform to indexes across epochs
                badIntervals.append((badIntervals_ep[segmentIndex] + ep * n_samples).astype(int))
                chaInterpolate.append(chaInterpolate_ep[segmentIndex])
                allBadChannels.append(allBadChannels_ep)

        if len(badIntervals) == 0:
            interpolation_matrix = np.zeros_like(raw._data, dtype=bool)
            return raw._data.copy(), interpolation_matrix

        badIntervals = np.asarray(badIntervals, dtype=int)
        
        # Initialize matrices
        raw_data = raw._data.copy()
        if len(raw._data.shape) == 3:
            raw_data_2d = np.transpose(raw_data, (1, 2, 0))
            raw_data_2d = np.reshape(raw_data_2d, (n_channels, n_samples * n_epochs), order='F')
        else:
            raw_data_2d = raw_data.copy()

        interpolatedData = raw_data_2d.copy()
        interpolationMatrix = np.full((n_channels, n_samples * n_epochs), False, dtype=bool)

        # Initialize arrays for storing segment indices
        initialSegmentIndices = []
        finalSegmentIndices = []
        
        requested_mode = self.params['parallelize_mode']
        selected_mode = requested_mode
        if requested_mode == 'auto':
            n_segments = int(np.shape(badIntervals)[0])
            n_jobs = self.params['n_jobs']
            n_jobs = os.cpu_count() or 1 if n_jobs in (-1, None) else max(1, int(n_jobs))
            avg_channels_to_interpolate = np.mean([np.sum(x) for x in chaInterpolate]) if n_segments else 0
            max_channels_allowed = self.params['p'] * n_channels
            if n_segments >= max(8, 2 * n_jobs) and avg_channels_to_interpolate <= max_channels_allowed:
                selected_mode = 'segments'
            else:
                selected_mode = 'channels'
            print(f"Auto parallelization selected mode: {selected_mode}")

        # Initialize progress bar
        start_time = time.time()
        if selected_mode == 'channels':
            widgets = ['Interpolating...', Percentage(), Bar()]
            bar = ProgressBar(maxval=int(np.shape(badIntervals)[0] - 1), widgets=widgets)
            bar.start()

            for segmentIndex in np.arange(np.shape(badIntervals)[0]):
                segmentStart = int(badIntervals[segmentIndex, 0])
                segmentEnd = int(badIntervals[segmentIndex, 1])
                bad_channels_to_interpolate = chaInterpolate[segmentIndex]
                all_bad_channels = allBadChannels[segmentIndex]

                if (np.sum(all_bad_channels) / n_channels) <= self.params['p']:
                    seg_start, seg_end, bad_ch, bad_rows = _interpolate_spline_segment_task(
                        raw,
                        raw_data_2d,
                        segmentStart,
                        segmentEnd,
                        bad_channels_to_interpolate,
                        all_bad_channels,
                        distance_matrix,
                        positions,
                        adjacency_matrix,
                        self.params['p_neighbors'],
                        self.params['n_jobs'],
                    )
                    if np.size(bad_ch) > 0:
                        initialSegmentIndices.append(seg_start)
                        finalSegmentIndices.append(seg_end)
                        interpolationMatrix[bad_ch, seg_start:seg_end] = True
                        interpolatedData[bad_ch, seg_start:seg_end] = bad_rows

                bar.update(segmentIndex)

            bar.finish()
        else:
            eligible_segments = [
                idx for idx in np.arange(np.shape(badIntervals)[0])
                if (np.sum(allBadChannels[idx]) / n_channels) <= self.params['p']
            ]

            print(f"Interpolating segments in parallel: {len(eligible_segments)} eligible / {np.shape(badIntervals)[0]} total")
            if len(eligible_segments) > 0:
                results = Parallel(n_jobs=self.params['n_jobs'], prefer='threads')(
                    delayed(_interpolate_spline_segment_task)(
                        raw,
                        raw_data_2d,
                        int(badIntervals[idx, 0]),
                        int(badIntervals[idx, 1]),
                        chaInterpolate[idx],
                        allBadChannels[idx],
                        distance_matrix,
                        positions,
                        adjacency_matrix,
                        self.params['p_neighbors'],
                        1,
                    )
                    for idx in eligible_segments
                )

                for seg_start, seg_end, bad_ch, bad_rows in results:
                    if np.size(bad_ch) == 0:
                        continue
                    initialSegmentIndices.append(seg_start)
                    finalSegmentIndices.append(seg_end)
                    interpolationMatrix[bad_ch, seg_start:seg_end] = True
                    interpolatedData[bad_ch, seg_start:seg_end] = bad_rows

        print('--- Elapsed time during interpolation: ', time.time() - start_time, 'seconds')
        print('--- Percentage of interpolated data: ', np.round(np.sum(interpolationMatrix[:]) / np.size(interpolationMatrix) * 100, 2), '%')
        
        # reshape
        if len(np.shape(raw._data))==3:
            interpolatedData = np.reshape(interpolatedData, (n_channels, n_samples, n_epochs), order='F')
            interpolatedData = np.transpose(interpolatedData, (2, 0, 1))
            interpolationMatrix = np.reshape(interpolationMatrix, (n_channels, n_samples, n_epochs), order='F')
            interpolationMatrix = np.transpose(interpolationMatrix, (2, 0, 1))
            
        if len(np.shape(raw._data))==2:
            interpolatedData = np.reshape(interpolatedData, (n_epochs, n_channels, n_samples), order='F')
            interpolationMatrix = np.reshape(interpolationMatrix, (n_epochs, n_channels, n_samples), order='F')
 

        # Splice the interpolated segments together
        initialSegmentIndices = np.asarray(initialSegmentIndices, dtype=int)
        finalSegmentIndices = np.asarray(finalSegmentIndices, dtype=int)
        print('Splicing segments...')
        for ep in np.arange(n_epochs):
            int_i = interpolationMatrix[ep, :, :]
            # get the segments to splice in that epoch
            segments_i = initialSegmentIndices[(initialSegmentIndices < n_samples*(ep+1)) & (initialSegmentIndices >= n_samples*ep)]
            segments_i = segments_i - ep * n_samples
            segments_f = finalSegmentIndices[(finalSegmentIndices <= n_samples*(ep+1)) & (finalSegmentIndices > n_samples*ep)]
            segments_f = segments_f - ep * n_samples
            bad_if = np.asarray([segments_i, segments_f], dtype=int).T
            if np.size(bad_if)>0:
                d_i = np.squeeze(interpolatedData[ep,:,:])
                epoch_if = np.asarray([0, n_samples])
                if len(np.shape(raw._data))==2:
                    bct_i = np.squeeze(raw.artifacts.BCT)
                    bc_i = raw.artifacts.BC
                    bt_i = raw.artifacts.BT           
                else:
                    bct_i = np.squeeze(raw.artifacts.BCT[ep,:,:])
                    bc_i = np.squeeze(raw.artifacts.BC[ep,:,0])
                    bt_i = np.squeeze(raw.artifacts.BT[ep,0,:])
                bct_i[int_i] = 0  
                dspliced = _splice_segments(d_i, bad_if, epoch_if, bct=bct_i, bt=bt_i, bc=bc_i, method=1)
                interpolatedData[ep,:,:] = dspliced
           
        # reshape if necessary         
        if len(np.shape(raw._data))==2:
            interpolationMatrix = np.squeeze(np.transpose(interpolationMatrix,(1,2,0)), axis=2)
            interpolatedData = np.squeeze(np.transpose(interpolatedData,(1,2,0)), axis=2)
        
        return interpolatedData, interpolationMatrix
    







        # # Initialize a matrix to track interpolated data
        # interpolationMatrix = np.full((n_epochs, n_channels, n_samples), False)

        # # Copy the original data for processing
        # interpolatedData = raw._data.copy()


        # # Process each epoch separately
        # for ep in np.arange(n_epochs):
        #     start_time = time.time()

        #     print('\nInterpolating Epoch ', ep + 1, '...')

        #     # data to interpolate
        #     bct_to_interpolate_ep = bct_to_interpolate[ep, :, :]
            
        #     # bad channels in that epoch are not interpolated
        #     if len(raw._data.shape) == 2:
        #         badChannelsPerEpoch = np.squeeze(raw.artifacts.BC[:,0]) | raw.artifacts.BCmanual
        #     else:
        #         badChannelsPerEpoch = np.squeeze(raw.artifacts.BC[ep,:,0]) | raw.artifacts.BCmanual
               

        #     # Initialize arrays for storing segment indices
        #     initialSegmentIndices = np.empty((1, n_samples))
        #     initialSegmentIndices[:] = np.nan
        #     finalSegmentIndices = np.empty((1, n_samples))
        #     finalSegmentIndices[:] = np.nan

        #     currentSegmentIndex = 0

        #     # Identify the changes in the number of bad channels
        #     # change_channel = np.where(np.any(np.diff(bct_to_interpolate_ep, axis=1), axis=0))[0]
        #     # segmentStartIndices = np.unique(np.hstack([change_channel + 1, 0]))
        #     # segmentEndIndices = np.unique(np.hstack([change_channel, n_samples - 1])) + 1
        #     # badIntervals = np.asarray([segmentStartIndices.T, segmentEndIndices.T]).T

        #     badIntervals, chaInterpolate = _find_segments_interpolation_spline(bct_to_interpolate_ep, badChannelsPerEpoch)
        #     # bad = np.where(bad_channels_ep)[0]
        #     # if np.size(bad)==0:
        #     #     bad = []
        #     # else:
        #     #     bad = list(bad)
        #     # cha_bad_ep = [bad for i in range(np.shape(bad_if_ep)[0])]
            
        #     # Initialize progress bar
        #     start_time = time.time()
        #     widgets = ['Interpolating...', Percentage(), Bar()]
        #     bar = ProgressBar(maxval=int(np.shape(badIntervals)[0] - 1), widgets=widgets)
        #     bar.start()

        #     # Copy EEG data for processing
        #     raw_data = raw._data.copy()
        #     t = raw.times.copy()
        #     croppedEEGData = raw.copy()
        #     croppedEEGData._data = []
        #     croppedEEGData._times = []

        #     # Iterate through each segment for interpolation
        #     for segmentIndex in np.arange(np.shape(badIntervals)[0]):
        #         # Check if the segment contains any bad channels to interpolate
        #         if np.any(bct_to_interpolate[ep, :, np.arange(badIntervals[segmentIndex, 0], badIntervals[segmentIndex, 1])]):

        #             currentSegmentIndex += 1

        #             # Define segment boundaries
        #             segmentStart = badIntervals[segmentIndex, 0]
        #             segmentEnd = badIntervals[segmentIndex, 1]
        #             initialSegmentIndices[ep, currentSegmentIndex - 1] = segmentStart
        #             finalSegmentIndices[ep, currentSegmentIndex - 1] = segmentEnd

        #             # Determine bad channels in the segment
        #             bad_channels_to_interpolate = chaInterpolate[segmentIndex]
        #             all_bad_channels = np.any(bct_to_interpolate[ep, :, np.arange(segmentStart, segmentEnd)], axis=0) | badChannelsPerEpoch
        #             # bad_channels_to_interpolate = np.any(bct_to_interpolate[ep, :, np.arange(segmentStart, segmentEnd)], axis=0) & ~badChannelsPerEpoch

        #             # Interpolate if there are enough good channels
        #             if self.params['p']:
        #                 if np.sum(all_bad_channels) / np.size(all_bad_channels) <= self.params['p']:

        #                     # Get a copy of the segments to interpolate
        #                     croppedEEGData._data = raw_data[ep, :, np.arange(segmentStart, segmentEnd)] if len(raw._data.shape) == 3 else raw_data[:, np.arange(segmentStart, segmentEnd)]
        #                     croppedEEGData._times = t[np.arange(segmentStart, segmentEnd)]

        #                     # Perform interpolation
        #                     interpolated_data, interpolated_bad_channels  = _do_spherical_spline_interpolation(croppedEEGData, 
        #                                                                                                         distance_matrix,
        #                                                                                                         positions,
        #                                                                                                         adjacency_matrix, 
        #                                                                                                         self.params['p_neighbors'], 
        #                                                                                                         bad_channels_to_interpolate, 
        #                                                                                                         all_bad_channels, False, self.params['n_jobs'])
    
                            
        #                     # Store the interpolated data
        #                     bad_ch = np.where(interpolated_bad_channels)[0]
        #                     for i in bad_ch:
        #                         interpolationMatrix[ep][i, np.arange(segmentStart, segmentEnd)] = True
        #                         interpolatedData[i, np.arange(segmentStart, segmentEnd)] = interpolated_data[i, :]

        #         # Update progress bar
        #         bar.update(segmentIndex)

        #     # End of interpolation
        #     bar.finish()
            
        #     # Finalize the indices for interpolated segments
        #     initialSegmentIndices = initialSegmentIndices[~np.isnan(initialSegmentIndices)]
        #     finalSegmentIndices = finalSegmentIndices[~np.isnan(finalSegmentIndices)]

        #     # Splice the interpolated segments together
        #     badIntervals = np.asarray([initialSegmentIndices, finalSegmentIndices], dtype=int).T
        #     epoch_if = np.asarray([0, n_samples - 1])
        #     if len(raw._data.shape) == 3:
        #         bct = raw.artifacts.BCT[ep, :, :].copy()
        #         bt = raw.artifacts.BT[ep, :, :].copy()
        #         bc = badChannelsPerEpoch
        #     elif len(raw._data.shape) == 2:
        #         bct = raw.artifacts.BCT.copy()
        #         bt = raw.artifacts.BT.copy()
        #         bc = badChannelsPerEpoch
        #     interpolatedData = _splice_segments(interpolatedData, badIntervals, epoch_if, bct=bct, bt=bt, bc=bc, method=1)

        #     # Print summary
        #     print(
        #         f"--- Elapsed time during interpolation: {time.time() - start_time} seconds\n",
        #         f"--- Percentage of interpolated data: {np.round(np.sum(interpolationMatrix[ep, :, :]) / (n_channels * n_samples) * 100, 2)} %\n"
        #     )

        # return interpolatedData, interpolationMatrix
