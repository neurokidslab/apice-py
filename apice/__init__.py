"""Python library for automated EEG artifact detection and correction.

APICE (Artifact Processing In Continuous EEG) provides a modular, configurable
pipeline for detecting and correcting physiological and instrumental artifacts
in EEG recordings. The library extends MNE-Python's RawArray and EpochsArray
classes with artifact management structures and integrates multiple detection
algorithms (amplitude, spectral, cross-electrode correlation) with correction
methods (PCA, spherical spline interpolation).

Key Features:
    - Configurable multi-algorithm artifact detection
    - Spatial and temporal artifact correction
    - Optional BIDS format support
    - Detailed logging and HTML reports

Main Classes:
    RawAPICE : Wrapper around mne.io.RawArray with artifact tracking
    EpochsAPICE : Wrapper around mne.EpochsArray with epoch-wise rejection
    
Main Functions:
    run_preprocessing : High-level preprocessing pipeline
    run_segmentation : Epoching and bad-epoch detection pipeline

See Also:
    mne : MEG/EEG signal processing library
    mne_bids : BIDS format support for MNE

Examples:
    Load and preprocess EEG data:
    >>> from apice import run_preprocessing
    >>> run_preprocessing(input_dir='raw/', output_dir='preproc/')

    Work directly with data objects:
    >>> from apice import RawAPICE
    >>> raw = RawAPICE(mne_raw_object)
    >>> raw.plot_artifact_structure()
"""

import mne
from mne.io import read_raw

from apice.data_structures import RawAPICE, EpochsAPICE
from apice.pipeline import run_preprocessing, run_segmentation
from apice.erp_statistics import compute_sme
