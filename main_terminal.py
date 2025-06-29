# Libraries
import matplotlib

# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import apice.pipeline
import apice
from apice.artifacts_rejection import *
import os

from apice.argparser_main import get_parsed_arguments


# %%

def main():
    """
    APICE: Automated Pipeline for Infants Continuous EEG

    This is the python implementation of APICE.
    Taken from Flo et al. (2022)

    Date: June 27, 2023
    Laboratory: UNICOG, NeuroSpin
    """
    args = get_parsed_arguments()

    """
    ********************************************* FILE DIRECTORIES **********************************************
    Set the paths of the following:
    input_dir = input directory containing the raw data to be processed
    output_dir = output directory where output data will be saved
    """
    input_dir = args.input_dir
    output_dir = args.output_dir

    '''
    ********************************************* DATA SELECTION ************************************************
    OPTIONS: Chose the data to process
    1. Run it for all the files found and overwrite previous output files (1)
    2. Run it only for the new files (2)
    3. Run specific files (3). Space key + enter key to stop the input prompt.
    '''
    data_selection_method = args.selection_method

    '''
    ********************************************* FILE DESCRIPTION **********************************************
    montage = information regarding the sensor locations,
                run mne.channels.get_builtin_montages(descriptions=True) for built-in montages
    '''
    montage = args.montage

    '''
    ************************************************ SEGMENTATION *************************************************
    Set the parameters for segmentation.
    event_keys_for_segmentation = events relative to the epochs
    event_time_window = start and end time of the epochs, in seconds
    baseline_time_window = time interval to consider as 'baseline' when applying baseline correction of epochs, 
                            in seconds
    by_event_type = evoked response per stimulus type, default to True
    '''
    event_keys_for_segmentation = args.event_keys_for_segmentation
    event_time_window = args.event_time_window
    baseline_time_window = args.baseline_time_window
    by_event_type = args.by_event_type

    '''
    ********************************************* PARALLEL PROCESSING *********************************************
    n_jobs = Number of core used in the computation (-1 to use all the available, faster computation)
    '''
    n_jobs = args.n_jobs 

    '''
    *********************************************** EXPORTING DATA ************************************************
    save_preprocessed_raw = export the preprocessed continuous data
    save_segmeneted_data = export the segmented data, returns the epoched data
    save_evoked_response = export the ERPs relative to the chosen stimulus, 
                                returns and array of channels x time per stimulus
    save_log = create and save the logs for each preprocessed data
    '''
    save_preprocessed_raw = args.save_preprocessed_raw
    save_segmented_data = args.save_segmented_data
    save_evoked_response = args.save_evoked_response
    save_log = args.save_log

    '''=============================================================================================================
                                            PRE-PROCESSING PIPELINE
    ============================================================================================================='''

    apice.pipeline.run(input_dir, output_dir, data_selection_method, 
                        event_keys_for_segmentation, event_time_window,
                        baseline_time_window, montage,
                        by_event_type=by_event_type,
                        save_preprocessed_raw=save_preprocessed_raw,
                        save_segmented_data=save_segmented_data,
                        save_evoked_response=save_evoked_response,
                        save_log=save_log, n_jobs=n_jobs)


# %% MAIN

if __name__ == '__main__':
    print('\nRunning APICE (Automated Pipeline for Infants Continuous EEG)...\n')
    main()
