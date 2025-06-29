# Libraries
import matplotlib

# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from apice.pipeline import run
from apice.artifacts_rejection import * 

import os
import time

# %%

def main():

    # Directory for input data
    INPUT_DIR = r"input"

    # Directory for output data
    OUTPUT_DIR = r"output"

    # Selection method ( 1 - Run it for all the files found and overwrite previous output files (default)
    # 2 - Run it only for the new files
    # 3 - Run specific files. Space key + enter key to stop the input prompt.)
    SELECTION_METHOD = 1

    # Montage (information regarding the sensor locations, - built_in mne montage or - electrode layout file)
    MONTAGE = r"electrode_layout/GSN-HydroCel-128.sfp"
    
    # Event keys for segmentation (array of event types relative to the epochs)
    EVENT_KEYS_FOR_SEGMENTATION = ['Icue', 'Ieye', 'Iout'] 

    # Event time window for segmentation (start and end time of the epochs in seconds)
    EVENT_TIME_WINDOW = [-1.600, 2.200] 
    
    # Baseline time window for segmentation (time interval to consider as baseline when applying baseline correction of epochs, in seconds)
    BASELINE_TIME_WINDOW = [-1.600, 0]
    
    # Flag to indicate whether to process data by event type
    BY_EVENT_TYPE = True

    # Flag to save preprocessed raw data
    SAVE_PREPROCESSED_RAW = True

    # Flag to save segmented data
    SAVE_SEGMENTED_DATA = True  

    # Flag to save evoked response data
    SAVE_EVOKED_RESPONSE = True

    # Flag to save log files
    SAVE_LOG = True 

    # Number of core used in the computation (-1 to use all the available, faster computation)
    N_JOBS = -1
    
    """ PROCESSING PIPELINE """
    
    run(
        INPUT_DIR,
        OUTPUT_DIR,
        SELECTION_METHOD,
        EVENT_KEYS_FOR_SEGMENTATION,
        EVENT_TIME_WINDOW,
        BASELINE_TIME_WINDOW,
        MONTAGE,
        by_event_type=BY_EVENT_TYPE,
        save_preprocessed_raw=SAVE_PREPROCESSED_RAW,
        save_segmented_data=SAVE_SEGMENTED_DATA,
        save_evoked_response=SAVE_EVOKED_RESPONSE,
        save_log=SAVE_LOG,
        n_jobs=N_JOBS,
    )

# %% MAIN

if __name__ == "__main__":
    
    print("\nRunning APICE (Automated Pipeline for Infants Continuous EEG)...\n")

    main()
