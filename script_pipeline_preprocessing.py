from apice.pipeline import run_preprocessing

def main():

    # Input Output parameters
    # ============================================================================================

    # Directory for input data
    INPUT_DIR = r"test_data/raw"

    # Directory for output data
    OUTPUT_DIR = r"test_data/preprocessed"

    # BIDS parameters if the input data is organized in BIDS format, otherwise these parameters will be ignored and can be set to None
    INPUT_DIR_BIDS = False
    BIDS_SESSION = '01' 
    BIDS_TASK = 'SLCatLearn'
    BIDS_RUN = '01'
    BIDS_SUBJECT = None
    BIDS_EXTENSION = '.vhdr'
    BIDS_DATATYPE = 'eeg'
    BIDS_SUFFIX = 'eeg'

    # Data selection parameters: 
    # "all" to process all files in the input directory, 
    # "new" to process only files that have not been processed yet (i.e., files that do not have a corresponding preprocessed file in the output directory), 
    # list of strings with file patterns use to select specific files (only valid if input_dir_bids=False, otherwise the BIDS file selection will be used)
    DATA_SELECTION_METHOD = "all"
    PREPROCESSED_FILE_PATTERN = '*-preproc-raw.fif'

    # Whether to create and save an HTML report with the preprocessing steps and results.
    SAVE_REPORT = True
    # Whether to print the preprocessing log to the console and save it to a text file in the output directory.
    SAVE_LOG = False
    # Whether to save a summary of the preprocessing steps and results.
    SAVE_SUMMARY = True


    # Preprocessing parameters
    # ============================================================================================
    # The order of the steps peformed in the preprocessing pipeline is the following:
    # 1. Initial steps: croping, resampling, filtering, picking electrodes, setting the montage (if specified)
    # 2. ICA (if specified)
    # 3. APICE preprocessing steps: bad channels detection, artifacts detection, local correction of artifacts using PCA and spline methods, etc. with the specified configurations for each step (or the default configurations if not specified).

    # List of electrodes to drop (e.g., dead channels, etc.) 
    DROP_ELECTRODES = ['E125','E126','E127','E128']

    # If the reference channel(s) are included in the data, indicate them here to avoid them being detected as bad channels and interpolated. 
    # If the reference channel(s) are not included in the data, set this parameter to None or an empty list.
    REFERENCE_CHANNELS = ['VREF']   
    
    # Pick only eeg channels for preprocessing (e.g., for infant EEG data, it is common to have many non-EEG channels such as EOG, EMG, etc. that can be excluded from preprocessing). 
    PICKS = 'eeg'

    # Time window to crop the data (e.g., to remove long periods of recording before and after the actual experiment). Set to None to not crop the data.
    CROP_TIMES = None

    # Time to crop from the beginning of the recording (e.g., to remove long periods of recording before the actual experiment). Set to None to not crop from the beginning.
    CROP_FROM_BEGINNING = None

    # Time to crop from the end of the recording (e.g., to remove long periods of recording after the actual experiment). Set to None to not crop from the end.
    CROP_FROM_END = None

    # Resampling frequency (Hz). Set to None to not resample the data.
    RESAMPLE_FREQ = None

    # Whether to convert stim channels to annotations
    STIM_CHANNELS_TO_ANNOTATIONS = False

    # Montage to set to the data. It can be mne.channels.DigMontage object, a path to a montage file (e.g., .sfp, .elc, .xyz, etc.) or a string corresponding to a standard montage available in MNE (e.g., 'standard_1020', 'GSN-HydroCel-128', etc.).
    # Set to None to not set a montage if the data already has a montage. 
    # Note that if not montage is provided and the data does not have a montage an error will be raised
    MONTAGE = None
    
    # Filtering parameters for the initial preprocessing steps (set to None to not apply band-pass filtering)
    L_FREQ_INITIAL = 0.10
    H_FREQ_INITIAL = 40
    LINE_NOISE_FREQ_INITIAL = None # Frequency for line noise removal (set to None to not apply line noise removal). The harmonics of the line noise frequency will also be removed up to the Nyquist frequency.

    # Parameter for ICA if applied
    APPLY_ICA = True
    ICA_PARAMETERS = {
        'cfg_artifacts_detection': None, # Set to None to apply the default configuration for artifact detection in the ICA pipeline. If a dictionary with specific parameters is provided or a path to a configuration file is provided, these parameters will be used for artifact detection in the ICA pipeline.
        'cfg_bcbt': None, # Set to None to apply the default configuration for bad channels detection based on the BCBT method in the ICA pipeline. If a dictionary with specific parameters is provided or a path to a configuration file is provided, these parameters will be used for bad channels detection based on the BCBT method in the ICA pipeline.
        'l_freq_ica': 1.0,  # Low-pass frequency for ICA (set to None to not apply low-pass filtering for ICA)
        'h_freq_ica': None,  # High-pass frequency for ICA
        'l_freq_artifacts':None,  # Low-pass frequency for artifact detection (set to None to not apply specific low-pass filtering for artifact detection)
        'h_freq_artifacts':None,  # High-pass frequency for artifact detection (set to None to not apply specific high-pass filtering for artifact detection)
        'exclude_ica': None, # List of channels to exclude from ICA (e.g., reference channels, etc.). Set to None or an empty list to not exclude any channels from ICA.
        'n_components': 'auto',  # Number of ICA components to compute. If float between 0 and 1, select the number of components to explain the specified variance. If int > 1, select the specified number of components. If None, select all components. If 'auto', the number of components will be automatically determined based on the number of channels and amount good data available to fit the ICA model (n_samples ≥ 30 x n_channel**2).
        'method': 'picard',  # ICA method to use (e.g., 'fastica', 'infomax', 'picard', etc.). See MNE documentation for more details.
        'fit_params': dict(ortho=False, extended=True),  # Additional parameters to pass to the ICA fit method. See MNE documentation for more details.
        'random_state': 42,  # Random state for ICA reproducibility.
        'label_components_method': 'iclabel',  # Method to use for labeling ICA components (e.g., 'iclabel', 'correlation', etc.). If 'iclabel', the ICLabel algorithm will be used to automatically label components. If 'correlation', the correlation of the ICA components with EOG and ECG channels will be used to label components. 
        'iclabel_lim_probability':0.9, # Probability threshold for labeling ICA components as artifacts using the ICLabel algorithm. Components with a probability of being an artifact above this threshold will be labeled as artifacts. 
        'iclabel_labels_to_exclude': ['eye blink', 'muscle artifact', 'heart beat', 'line noise', 'channel noise'], # List of ICLabel labels to consider as artifacts and exclude. The possible labels are: 'eye blink', 'eye movement', 'muscle artifact', 'heart beat', 'line noise', 'channel noise', 'other'. 
    }
   
    # Filtering parameters for the APICE preprocessing steps (set to None to not apply band-pass filtering). L_FREQ_APICE should be set to a value >0 to apply the high-pass filtering in the APICE default pipeline, which is required for filtering after local artifacts correction steps.
    L_FREQ_APICE = 0.10
    H_FREQ_APICE = None
    LINE_NOISE_FREQ_APICE = None

    # Filter parameters to apply for artifact detection steps. 
    # A copy of the data will be filtered with these parameters and used only for artifact detection steps (e.g., bad channels detection, artifacts detection, etc.) to improve the detection of artifacts. 
    # Set to None to not apply specific filtering for artifact detection and use the same filtered data as for the rest of preprocessing steps (i.e., l_freq, h_freq, etc.). 
    L_FREQ_ARTIFACTS = None
    H_FREQ_ARTIFACTS = None

    # Set to None to apply the default configuration.
    CFG_BAD_CHANNELS_DETECTION = None
    CFG_GLITCHES_DETECTION = None
    CFG_TARGET_PCA = None
    CFG_ARTIFACTS_DETECTION = None
    CFG_SPLINE_SEGMENTS = None
    CFG_SPLINE_CHANNELS = None

    # Number of parallel jobs to run for computationally intensive steps (e.g., bad channels detection, artifacts detection, etc.). 
    # Set to -1 to use all available cores.
    N_JOBS = -1

    # Whether to show figures during preprocessing (e.g., plots of detected bad channels, artifacts, etc.). Set to False to not show figures.
    SHOW_FIGURES = False


    # Run the preprocessing pipeline with the specified parameters
    # ============================================================================================

    run_preprocessing(
        INPUT_DIR, 
        OUTPUT_DIR, 
        input_dir_bids=INPUT_DIR_BIDS,
        bids_session=BIDS_SESSION,
        bids_task=BIDS_TASK,
        bids_run=BIDS_RUN,
        bids_subject=BIDS_SUBJECT,
        bids_extension=BIDS_EXTENSION,
        bids_datatype=BIDS_DATATYPE,
        bids_suffix=BIDS_SUFFIX,
        processed_file_pattern=PREPROCESSED_FILE_PATTERN,
        data_selection_method=DATA_SELECTION_METHOD,
        drop_electrodes=DROP_ELECTRODES,
        reference_channels=REFERENCE_CHANNELS,
        picks=PICKS,
        crop_times=CROP_TIMES,
        crop_from_beginning=CROP_FROM_BEGINNING,
        crop_from_end=CROP_FROM_END,
        resample_freq=RESAMPLE_FREQ,
        stim_channels_to_annotations=STIM_CHANNELS_TO_ANNOTATIONS,
        montage=MONTAGE,
        l_freq_initial=L_FREQ_INITIAL,
        h_freq_initial=H_FREQ_INITIAL,
        line_noise_freq_initial=LINE_NOISE_FREQ_INITIAL,
        l_freq_apice=L_FREQ_APICE,
        h_freq_apice=H_FREQ_APICE,
        line_noise_freq_apice=LINE_NOISE_FREQ_APICE,
        l_freq_artifacts=L_FREQ_ARTIFACTS,
        h_freq_artifacts=H_FREQ_ARTIFACTS,
        apply_ica=APPLY_ICA,
        ica_parameters=ICA_PARAMETERS,
        cfg_bad_channels_detection=CFG_BAD_CHANNELS_DETECTION,
        cfg_glitches_detection=CFG_GLITCHES_DETECTION,
        cfg_target_pca=CFG_TARGET_PCA,
        cfg_artifacts_detection=CFG_ARTIFACTS_DETECTION,
        cfg_spline_segments=CFG_SPLINE_SEGMENTS,
        cfg_spline_channels=CFG_SPLINE_CHANNELS,
        n_jobs=N_JOBS,
        save_log=SAVE_LOG,
        save_report=SAVE_REPORT,
        save_summary=SAVE_SUMMARY,
        show_figures=SHOW_FIGURES,
        )

if __name__ == "__main__":
    print("\nRunning APICE (Automated Pipeline for Infants Continuous EEG)...\n")
    main()

  