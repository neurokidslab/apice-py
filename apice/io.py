"""Input/output helpers for APICE raw and epoched workflows.

This module provides loading wrappers that initialize APICE data structures and
file-discovery utilities for regular folders and BIDS datasets.
"""

# %% LIBRARIES
from pathlib import Path
import numpy as np
import mne.io
import pandas as pd 

from apice.artifacts_structure import Artifacts
from apice.utils import print_header, get_data_size
from apice.data_structures import RawAPICE, EpochsAPICE
        

# %% FUNCTIONS


def load_rawapice(fname, bt_labels='badtime', bct_labels='artifact', cct_labels='corrected',
                  verbose=None, **kwargs):
    """Load raw data and wrap it as ``RawAPICE``.

    Parameters
    ----------
    fname : str | pathlib.Path
        Path to an EEG file readable by MNE.
    bt_labels : str, default='badtime'
        Annotation label used for bad-time segments.
    bct_labels : str, default='artifact'
        Annotation label used for bad channel-time segments.
    cct_labels : str, default='corrected'
        Annotation label used for corrected segments.
    verbose : bool | str | int | None, default=None
        MNE verbosity setting passed to ``RawAPICE``.
    **kwargs
        Additional keyword arguments forwarded to ``ArtifactsRaw`` via
        ``RawAPICE`` (e.g. custom threshold parameters).

    Returns
    -------
    raw : RawAPICE
        Loaded and wrapped raw object.
    """
    # load the data
    raw = mne.io.read_raw(fname, preload=False, verbose=verbose)
    # Initialize RawAPICE structure
    raw = RawAPICE(raw, bt_label=bt_labels, bct_label=bct_labels, cct_label=cct_labels,
                   verbose=verbose, **kwargs)
    return raw

def load_epochapice(fname, verbose=None, **kwargs):
    """Load epochs data and wrap it as ``EpochsAPICE``.

    Parameters
    ----------
    fname : str | pathlib.Path
        Path to an epochs FIF file.
    verbose : bool | str | int | None, default=None
        MNE verbosity setting passed to ``EpochsAPICE``.
    **kwargs
        Additional keyword arguments forwarded to ``ArtifactsEpochs`` via
        ``EpochsAPICE``.

    Returns
    -------
    epochs : EpochsAPICE
        Loaded and wrapped epochs object.
    """
    # load the epochs
    epochs = mne.read_epochs(fname, verbose=verbose)
    # Initialize EpochsAPICE structure
    epochs = EpochsAPICE(epochs, verbose=verbose, **kwargs)
    # get the path to the parent folder
    folder_path = Path(fname).parent
    art_file = folder_path / (Path(fname).stem + '-artifacts.csv')
    # Read the artifacts information from the csv file and add it to the epochs object
    if art_file.is_file():
        artifacts_df = pd.read_csv(art_file)
        epochs.dataframe_to_rejection_matrix(artifacts_df)
    else:
        print(f"No artifacts file found at {art_file}. Returning epochs without artifacts information.")
    return epochs

def is_valid_extension(filename, valid_extensions):
    """Check whether a filename extension belongs to an allowed set.

    Parameters
    ----------
    filename : str | pathlib.Path
        Filename to validate.
    valid_extensions : collection of str
        Allowed extensions without leading dots.

    Returns
    -------
    is_valid : bool
        True when extension is accepted.
    """
    return Path(filename).suffix.lstrip(".").lower() in valid_extensions


def get_files_to_process(input_dir, 
                         output_dir=None, 
                         data_selection_method: str ='all',
                         processed_file_pattern='*-preproc.fif',
                         ):
    """Return input files to process based on selection strategy.

    Parameters
    ----------
    input_dir : str | pathlib.Path
        Directory containing candidate raw files.
    output_dir : str | pathlib.Path | None, default=None
        Directory containing processed outputs (required for ``'new'`` mode).
    data_selection_method : str | list[str], default='all'
        Selection mode:
        - ``'all'``: include all valid files.
        - ``'new'``: include only files without matching processed outputs.
        - ``list[str]``: include files matching any provided glob pattern.
    processed_file_pattern : str, default='*-preproc.fif'
        Output filename pattern used to detect already processed files.

    Returns
    -------
    files_to_process : list of pathlib.Path
        Sorted list of files selected for processing.

    Raises
    ------
    ValueError
        If directories or selection mode are invalid.
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
    """Return BIDS files to process based on selection strategy.

    Parameters
    ----------
    bids_root : str | pathlib.Path
        Root of the BIDS dataset.
    session : str | None, default=None
        Optional BIDS session selector.
    task : str | None, default=None
        Optional BIDS task selector.
    run : str | None, default=None
        Optional BIDS run selector.
    subject : str | None, default=None
        Optional BIDS subject selector.
    datatype : str, default='eeg'
        BIDS datatype.
    suffix : str, default='eeg'
        BIDS suffix.
    extension : str, default='.vhdr'
        File extension to match.
    output_dir : str | pathlib.Path | None, default=None
        Directory containing processed outputs (required for ``'new'`` mode).
    data_selection_method : {'all', 'new'}, default='all'
        Selection mode.
    processed_file_pattern : str, default='*-preproc.fif'
        Output filename pattern used to detect already processed files.

    Returns
    -------
    files_to_process : list
        List of matching ``mne_bids.BIDSPath`` entries.

    Raises
    ------
    ValueError
        If selection mode is invalid or ``output_dir`` is missing in ``'new'`` mode.
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


