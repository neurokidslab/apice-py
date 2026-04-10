
# %% LIBRARIES
from pathlib import Path
import numpy as np
import mne.io
import pandas as pd 

from apice.artifacts_structure import Artifacts
from apice.utils import print_header, get_data_size
from apice.data_structures import RawAPICE, EpochsAPICE
        

# %% FUNCTIONS


def load_rawapice(fname, bt_labels='badtime', bct_labels='artifact', cct_labels='corrected'):
    # load the data
    raw = mne.io.read_raw(fname, preload=False, verbose=None)
    # Initialize RawAPICE structure
    raw = RawAPICE(raw, bt_label=bt_labels, bct_label=bct_labels, cct_label=cct_labels)
    return raw

def load_epochapice(fname):
    # load the epochs
    epochs = mne.read_epochs(fname)
    # Initialize EpochsAPICE structure
    epochs = EpochsAPICE(epochs)
    # get the path to the parent folder
    folder_path = Path(fname).parent
    art_file = folder_path / (Path(fname).stem + '-artifacts.csv')
    # Read the artifacts information from the csv file and add it to the epochs object
    if art_file.is_file():
        artifacts_df = pd.read_csv(art_file)
        epochs = epochs.dataframe_to_rejection_matrix(artifacts_df)
    else:
        print(f"No artifacts file found at {art_file}. Returning epochs without artifacts information.")
    return epochs

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
                         processed_file_pattern='*-preproc.fif',
                         ):
    """
    Get a list of files to process based on the specified data selection method.

    Parameters:
    - input_dir (str): The input directory containing raw data files.
    - output_dir (str, optional): The output directory where preprocessed files will be saved.
    - data_selection_method (str | list[str], optional): The method to select files:
        'all' - All valid files in the input directory.
        'new' - Files not already processed in the input directory and output directory.
        list[str] - File patterns used to select files from input_dir.
    - processed_file_pattern (str, optional): The pattern to identify preprocessed files in the output directory (default: '*-preproc.fif').

    Returns:
    - list: A list of file paths to process.
    """

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

    elif data_selection_method == 'new':
        # Method 'new': Include files not already processed in the output directory.
        if output_dir is None:
            raise ValueError("output_dir must be provided when data_selection_method='new'.")

        if not output_dir.exists():
            files_to_process = valid_files
        else:
            output_folder_files = {
                f.stem for f in output_dir.glob(processed_file_pattern) if f.is_file()
            }
            files_to_process = []
            for file in valid_files:
                expected_output_stem = Path(
                    processed_file_pattern.replace("*", file.stem)
                ).stem
                if expected_output_stem not in output_folder_files:
                    files_to_process.append(file)

    elif isinstance(data_selection_method, list) and all(
        isinstance(pattern, str) for pattern in data_selection_method
    ):
        # Method list[str]: Include files matching any provided pattern.
        matched_files = set()
        for pattern in data_selection_method:
            for file in input_dir.glob(pattern):
                if file.is_file():
                    matched_files.add(file)
        files_to_process = sorted(matched_files, key=lambda p: p.name.lower())

    else:
        raise ValueError(
            "Invalid data_selection_method. Use 'all', 'new', or a list of string file patterns."
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
                              processed_file_pattern='*-preproc.fif'):
    """
    Get a list of BIDS files to process based on the specified data selection method.

    Parameters:
    - bids_root (str): The root directory containing BIDS data.
    - output_dir (str, optional): The output directory where preprocessed files will be saved.
    - data_selection_method (str, optional): The method to select files:
        'all' - All valid BIDS files in the input directory.
        'new' - BIDS files not already processed in the input directory and output directory.
        'manual' - Manually input BIDS file paths.
    - processed_file_pattern (str, optional): The pattern to identify preprocessed files in the output directory (default: '*-preproc.fif').
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
                f.stem for f in output_dir.glob(processed_file_pattern) if f.is_file()
            }
            files_to_process = []
            for file in list_files:
                expected_output_stem = Path(
                    processed_file_pattern.replace("*", file.basename)
                ).stem
                if expected_output_stem not in output_folder_files:
                    files_to_process.append(file)


    return files_to_process


