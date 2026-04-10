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
    PREPROCESSED_FILE_PATTERN = '*-preproc.fif'

    # Whether to create and save an HTML report with the preprocessing steps and results.
    SAVE_REPORT = True
    # Whether to print the preprocessing log to the console and save it to a text file in the output directory.
    SAVE_LOG = False
    # Whether to save a summary of the preprocessing steps and results.
    SAVE_SUMMARY = True


    # Preprocessing parameters
    # ============================================================================================

    # List of electrodes to drop (e.g., dead channels, etc.) 
    DROP_ELECTRODES = ['E125','E126','E127','E128']

    # If the reference channel(s) are included in the data, indicate them here to avoid them being detected as bad channels and interpolated. 
    # If the reference channel(s) are not included in the data, set this parameter to None or an empty list.
    REFERENCE_CHANNELS = ['VREF']   
    
    # Pick only eeg channels for preprocessing (e.g., for infant EEG data, it is common to have many non-EEG channels such as EOG, EMG, etc. that can be excluded from preprocessing). 
    PICKS = 'eeg'

    # Time window to crop the data (e.g., to remove long periods of recording before and after the actual experiment). Set to None to not crop the data.
    CROP_TIMES = None

    # Time to crop from the beggining of the recording (e.g., to remove long periods of recording before the actual experiment). Set to None to not crop from the beggining.
    CROP_FROM_BEGGINNING = None

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
    
    # Frequency parameters for band-pass filtering (set to None to not apply band-pass filtering)
    L_FREQ = 0.1
    H_FREQ = 40
    L_TRANS_BANDWIDTH = 0.1
    H_TRANS_BANDWIDTH = 10

    # Configuration parameters for specific preprocessing steps (e.g., bad channels detection, artifacts detection, etc.). 
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
        crop_from_beginnning=CROP_FROM_BEGGINNING,
        crop_from_end=CROP_FROM_END,
        resample_freq=RESAMPLE_FREQ,
        stim_channels_to_annotations=STIM_CHANNELS_TO_ANNOTATIONS,
        montage=MONTAGE,
        save_log=SAVE_LOG,
        save_report=SAVE_REPORT,
        save_summary=SAVE_SUMMARY,
        l_freq=L_FREQ,
        h_freq=H_FREQ,
        l_trans_bandwidth=L_TRANS_BANDWIDTH,
        h_trans_bandwidth=H_TRANS_BANDWIDTH,
        n_jobs=N_JOBS,
        cfg_bad_channels_detection=CFG_BAD_CHANNELS_DETECTION,
        cfg_glitches_detection=CFG_GLITCHES_DETECTION,
        cfg_target_pca=CFG_TARGET_PCA,
        cfg_artifacts_detection=CFG_ARTIFACTS_DETECTION,
        cfg_spline_segments=CFG_SPLINE_SEGMENTS,
        cfg_spline_channels=CFG_SPLINE_CHANNELS,
        )

if __name__ == "__main__":
    print("\nRunning APICE (Automated Pipeline for Infants Continuous EEG)...\n")
    main()

  