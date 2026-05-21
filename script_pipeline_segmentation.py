from apice.pipeline import run_segmentation  

def main():

    # Input Output parameters
    # ============================================================================================

    # Directory for input data
    INPUT_DIR = r"test_data/preprocessed"

    # Directory for output data
    OUTPUT_DIR = r"test_data/segmented"

    # Data selection parameters: 
    # "all" to process all files in the input directory, 
    # "new" to process only files that have not been processed yet (i.e., files that do not have a corresponding preprocessed file in the output directory), 
    # list of strings with file patterns use to select specific files (only valid if input_dir_bids=False, otherwise the BIDS file selection will be used)
    PROCESSED_FILE_PATTERN = '*-epo.fif'
    DATA_SELECTION_METHOD = "all"

    # Whether to save the segmented epochs in the output directory.
    SAVE_EPOCHS = True
    # Whether to save only the good epochs or all epochs in the output directory.
    SAVE_ONLY_GOOD_EPOCHS = False
    # Whether to save the segmented evoked data in the output directory.
    SAVE_EVOKED = True
    # Whether to create and save an HTML report with the segmentation steps and results.
    SAVE_REPORT = True
    # Whether to save a summary of the segmentation steps and results.
    SAVE_SUMMARY = True
    # Whether to print the segmentation log to the console and save it to a text file in the output directory.
    SAVE_LOG = False
    # Whether to save the configuration parameters used for artifacts detection and correction.
    SAVE_CFG = True


    # Segmentation parameters
    # ============================================================================================

    # Frequency parameters for band-pass filtering (set to None to not apply band-pass filtering)
    L_FREQ = 0.2
    H_FREQ = 20
    
    # Time window for epoching the data around the events of interest (e.g., from -2 to 2 seconds around the event).
    EVENT_TIME_WINDOW = (-0.2, 2.0)

    # Dictionary of keyword arguments to pass to the function mne.events_from_annotations that extracts events from annotations for segmentation.
    KWARGS_EVENTS_FROM_ANNOTATIONS_FOR_SEGMENTATION = dict(regexp='animal*', verbose='WARNING')

    # Dictionary of keyword arguments to pass to the function mne.events_from_annotations that extracts events from annotations for metadata creation.
    # Set it to do not create metadata from the annotations for the epochs
    KWARGS_EVENTS_FROM_ANNOTATIONS_FOR_METADATA = dict(regexp='shape*|animal*|Tini*|Sound|eye*|k*', verbose='WARNING')
    # Dictionary of keyword arguments setting the parameters for creating metadata from the events.
    # - tmin and tmax indicate the time window around the events to consider for creating the metadata. If not set, the entire epoch will be considered.
    # - columns_events_to_keep indicate the columns of the events to keep in the metadata. If not set, all columns will be kept.
    KWARGS_MAKE_METADATA = dict(tmin=-0.2, tmax=2.0, columns_events_to_keep=['event_name', 'Tini', 'Sound', 'eyeO1', 'eyeC', 'eyeO2', 'animal', 'shape','eyeOend', 'katt','kesc','krsm'])
    
    # Baseline correction parameters for the epochs. Set to None to not apply baseline correction.
    BASELINE = (None, 0)

    # Parameters for setting the reference of the data. 
    # Set to None to not set the reference. 
    # If set to a dictionary, the dictionary will be passed as keyword arguments to the function mne.Epochs.set_eeg_reference.
    SET_REFERENCE = {'ref_channels':'average'}

    # Whether to create evoked objects for each event type by averaging the epochs of each event type. 
    # If None, only the epochs will be saved.
    # If "all", evoked responses will be computed for all event types in the data and saved in a single file.
    # If a list of strings with event types, evoked responses will be computed for each event type in the list and saved in separate files. 
    EVOKED_BY = ["animal"]

    # Configuration parameters for specific segmentation steps (e.g., defining BCBT epochs, spline interpolation, etc.). Set to None to apply the default configuration.
    CFG_DEFINE_BCBT_EPOCHS = None
    CFG_SPLINE_CHANNELS = None
    CFG_BAD_EPOCHS = None

    # Number of parallel jobs to run for computationally intensive steps (e.g., bad channels detection, artifacts detection, etc.). Set to -1 to use all available cores.
    N_JOBS = -1

    # Whether to show figures during preprocessing (e.g., plots of detected bad channels, artifacts, etc.). Set to False to not show figures.
    SHOW_FIGURES = False


    # Run the segmentation pipeline with the specified parameters
    # ============================================================================================

    run_segmentation(
        INPUT_DIR, 
        OUTPUT_DIR,
        KWARGS_EVENTS_FROM_ANNOTATIONS_FOR_SEGMENTATION, 
        EVENT_TIME_WINDOW,
        processed_file_pattern=PROCESSED_FILE_PATTERN,
        data_selection_method=DATA_SELECTION_METHOD,
        l_freq=L_FREQ,
        h_freq=H_FREQ,
        kwargs_events_from_annotations_for_metadata=KWARGS_EVENTS_FROM_ANNOTATIONS_FOR_METADATA,
        kwargs_make_metadata=KWARGS_MAKE_METADATA,                             
        evoked_by=EVOKED_BY,
        baseline=BASELINE, 
        set_reference=SET_REFERENCE,
        save_log=SAVE_LOG,
        save_epochs=SAVE_EPOCHS,
        save_only_good_epochs=SAVE_ONLY_GOOD_EPOCHS,
        save_evoked=SAVE_EVOKED,
        save_report=SAVE_REPORT,
        save_summary=SAVE_SUMMARY,
        save_cfg=SAVE_CFG,
        cfg_define_bcbt_epochs=CFG_DEFINE_BCBT_EPOCHS,
        cfg_spline_channels=CFG_SPLINE_CHANNELS,  
        cfg_bad_epochs=CFG_BAD_EPOCHS,              
        n_jobs=N_JOBS,
        show_figures=SHOW_FIGURES,
        )


    
if __name__ == "__main__":
    print("\nRunning APICE default segmentation pipeline on test data...\n")
    main()