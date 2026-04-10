import glob
import os
import warnings
import numpy as np
import json
from importlib import resources
from pathlib import Path

def find_true_segments_edges(m):
    """
    This function finds the indexs for the starting and ending points of segments with True values.
    :return idx_i: samples for the starting of the segment
            idx_f: samples for the ending of the segment
    """
    
    m = np.asarray(m, dtype=int)
    m = np.reshape(m, np.size(m))

    if m.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    
    temp1 = m[0]
    temp2 = (m[1:] - m[0:-1]) == 1
    idx_i = np.insert(temp2, 0, temp1)
    idx_i = np.where(idx_i)[0]
    
    temp3 = (m[0:-1] - m[1:]) == 1
    temp4 = m[-1]
    idx_f = np.insert(temp3, len(temp3), temp4)
    idx_f = np.where(idx_f)[0]+1
    
    return idx_i, idx_f

def get_onset_and_duration(event_array, time_vector):
    
    isbad_i, isbad_f = find_true_segments_edges(event_array)
    onset = time_vector[isbad_i]
    # duration = time_vector[isbad_f-1]-time_vector[isbad_i] 
    sfreq = 1 / np.mean(np.diff(time_vector))
    duration = (isbad_f - isbad_i) / sfreq   
        
    return onset, duration

def reshape_axis_first(m, axis):
    shape_org = np.shape(m)
    perm_vals = np.empty((len(shape_org)), dtype=int)
    perm_vals[0] = axis
    perm_vals[np.arange(1,len(shape_org))] = np.delete(np.arange(0,len(shape_org)),axis)
    m = np.transpose(m, perm_vals)
    m = np.reshape(m, (np.shape(m)[0],-1))
    return m, shape_org, perm_vals


def back_to_original_shape(m, axis, shape_org, perm_vals):
    m = np.reshape(m, np.array(shape_org)[perm_vals])
    perm_vals = np.empty((len(shape_org)), dtype=int)
    perm_vals[axis] = 0
    perm_vals[np.delete(np.arange(0,len(shape_org)),axis)] = np.arange(1,len(shape_org))
    m = np.transpose(m,perm_vals)
    return m


def reject_short_good_segments_1d(bt, samples_limit):

    bt_out = bt.copy()==1
    
    # good segments
    good_data_time = bt==False 
    
    # get the start and end of the segments
    isgood_i, isgood_f = find_true_segments_edges(good_data_time) 
    
    # get segment length and if too short reject
    if np.size(isgood_i) > 0:
        good_data_length = (isgood_f - isgood_i)
        short_segment = np.where(good_data_length <= samples_limit)[0]
        if np.size(short_segment) > 0:
            for i in np.arange(np.size(short_segment)):
                ind = np.arange(isgood_i[int(short_segment[i])], isgood_f[int(short_segment[i])])
                bt_out[ind] = True
    
    change = bt!=bt_out   
        
    return bt_out, change
        
   
def include_short_bad_segments_1d(bt, samples_limit):
    
    bt_out = bt.copy()==1
    
    # get the start and end of the segments
    isbad_i, isbad_f = find_true_segments_edges(bt)
    
    # get segment length and if too short include        
    if np.size(isbad_i) > 0:
        good_duration = isbad_f - isbad_i
        short_segments = np.where(good_duration < samples_limit)[0]
        if np.size(short_segments) > 0:
            for i in np.arange(np.size(short_segments)):
                ind = np.arange(isbad_i[int(short_segments[i])], isbad_f[int(short_segments[i])])
                bt_out[ind] = False
    
    change = bt!=bt_out   
        
    return bt_out, change

def mask_bad_segments_1d(bt, mask_samples):
 
    bt_out = bt.copy()==1

    n_samples = len(bt)
    if mask_samples > 0:
        buffer = int(np.round(mask_samples))
        
        ZA = np.concatenate(([0], bt, [0]))
        indices = np.flatnonzero(ZA[1:] != ZA[:-1])
        counts = indices[1:] - indices[:-1]
        bad_i = indices[::2]
        duration = counts[::2]
        bad_f = bad_i + duration - 1
    
        if np.size(bad_i) > 0:
            for i in np.arange(np.size(bad_i)):
                bad_idx_i = np.asarray(np.arange(bad_i[i] - buffer, bad_i[i]), dtype=int)# + 1
                bad_idx_i = np.delete(bad_idx_i, bad_idx_i < 0)
                bt_out[bad_idx_i] = True
            for i in np.arange(np.size(bad_f)):
                bad_idx_f = np.asarray(np.arange(bad_f[i] + 1, bad_f[i] + buffer + 1), dtype=int)
                bad_idx_f = np.delete(bad_idx_f, bad_idx_f >= n_samples)
                bt_out[bad_idx_f] = True
            
    change = bt!=bt_out   
        
    return bt_out, change


def reject_short_good_segments(m, samples_limit, axis=None):
    
    if len(np.shape(m))==1:
        m_out, change = reject_short_good_segments_1d(m, samples_limit)
    
    else:
        
        if axis is None:
            axis = np.argmax(np.shape(m))
        
        # reshape to apply in the desired dimension
        m, m_shape, perm_vals = reshape_axis_first(m, axis)
        
        # reject
        m_out = m.copy()
        change = np.full(np.shape(m),False)
        for i in range(np.shape(m)[1]):
            m_out[:,i], change[:,i] = reject_short_good_segments_1d(m[:,i], samples_limit)
            
        # back to intial shape
        m_out = back_to_original_shape(m_out, axis, m_shape, perm_vals)
        change = back_to_original_shape(change, axis, m_shape, perm_vals)
        
    return m_out, change


def include_short_bad_segments(m, samples_limit, axis=None):
    
    if len(np.shape(m))==1:
        m_out, change = include_short_bad_segments_1d(m, samples_limit)
    
    else:
        
        if axis is None:
            axis = np.argmax(np.shape(m))
        
        # reshape to apply in the desired dimension
        m, m_shape, perm_vals = reshape_axis_first(m, axis)
        
        # reject
        m_out = m.copy()
        change = np.full(np.shape(m),False)
        for i in range(np.shape(m)[1]):
            m_out[:,i], change[:,i] = include_short_bad_segments_1d(m[:,i], samples_limit)
            
        # back to intial shape
        m_out = back_to_original_shape(m_out, axis, m_shape, perm_vals)
        change = back_to_original_shape(change, axis, m_shape, perm_vals)
        
    return m_out, change


def mask_bad_segments(m, mask_samples, axis=None):
    
    if len(np.shape(m))==1:
        m_out, change = mask_bad_segments_1d(m, mask_samples)
    
    else:
        
        if axis is None:
            axis = np.argmax(np.shape(m))
        
        # reshape to apply in the desired dimension
        m, m_shape, perm_vals = reshape_axis_first(m, axis)
        
        # reject
        m_out = m.copy()
        change = np.full(np.shape(m),False)
        for i in range(np.shape(m)[1]):
            m_out[:,i], change[:,i] = mask_bad_segments_1d(m[:,i], mask_samples)
            
        # back to intial shape
        m_out = back_to_original_shape(m_out, axis, m_shape, perm_vals)
        change = back_to_original_shape(change, axis, m_shape, perm_vals)
        
    return m_out, change


def update_parameters_with_user_inputs(params, new_params):
    """
    Update the default parameters with user-provided parameters.

    Args:
        params (dict): The default parameters to be updated.
        new_params (dict): The user-provided parameters to update the defaults.

    Returns:
        dict: The updated parameters after merging defaults with user inputs.
    """
    updated_params = params.copy()

    for key, value in new_params.items():
        if key in updated_params:
            updated_params[key] = value
        else:
            warnings.warn(
                f"'{key}' is not a recognized parameter and will be ignored.",
                UserWarning,
                stacklevel=2,
            )
    return updated_params


def print_header(header, separator="="):
    """
    Print a header with separator lines of the same length.

    Args:
        header (str): The header text to be printed.
        separator (str, optional): The character used to create separator lines. Defaults to "-".
    """
    separator = separator * len(header)
    print("\n" + separator)
    print(header)
    print(separator + "\n")


def get_files_in_folder(inputDir, pattern):
    filePattern = os.path.join(inputDir,  pattern)
    matchingFiles = glob.glob(filePattern)
    fileNames = [os.path.basename(file) for file in matchingFiles]
    return fileNames

def get_data_size(obj):
    """
    Get the shape of the EEG continuous signal.

    Args:
        - obj (mne.io.Raw or mne.io.Epochs): Object containing the EEG data.

    Returns:
        - n_channels (int): Number of channels.
        - n_samples (int): Number of data points per epoch.
        - n_epochs (int): Number of continuous segments.
    """
    import mne

    # Accept custom wrappers/proxies that already expose a get_data_size method.
    if hasattr(obj, 'get_data_size') and callable(obj.get_data_size):
        return obj.get_data_size()

    # Prefer metadata attributes to avoid loading full data arrays when possible.
    n_channels = len(obj.ch_names)
    n_samples = int(obj.n_times)

    if isinstance(obj, mne.io.BaseRaw):
        n_epochs = 1
    elif isinstance(obj, mne.Epochs):
        n_epochs = len(obj)
    else:
        raise TypeError(f"Unsupported object type for get_data_size: {type(obj)}")
    
    return n_channels, n_samples, n_epochs


def get_cfg(cfg, default_cfg):
    if cfg is None:
        resource = resources.files(__package__).joinpath("default_cfg", default_cfg)
        with resource.open('r', encoding='utf-8') as f:
            cfg = json.load(f)
    elif isinstance(cfg, str) or isinstance(cfg, Path):
        with open(cfg, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
    elif isinstance(cfg, dict):
        pass
    else:        
        raise ValueError(f"cfg must be either None, a path to a json file, or a dictionary. Got {type(cfg)}")
    return cfg
