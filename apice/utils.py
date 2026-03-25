import mne
import numpy as np


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
    import os
    import glob
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
        - n_electrodes (int): Number of electrodes.
        - n_samples (int): Number of data points per epoch.
        - n_epochs (int): Number of continuous segments.
    """
    
    data_shape = obj.get_data().shape

    if len(data_shape) == 2:  # Continuous data (Raw)
        n_epochs = 1  # Continuous data is considered as one epoch
        n_electrodes = data_shape[0]  # Number of electrodes is the first dimension
        n_samples = data_shape[1]  # Number of samples is the second dimension
    else:
        n_epochs = data_shape[0]  # Number of epochs in the Epochs object
        n_electrodes = data_shape[1]  # Number of electrodes is the second dimension
        n_samples = data_shape[2]  # Number of samples is the third dimension
    
    return n_electrodes, n_samples, n_epochs



