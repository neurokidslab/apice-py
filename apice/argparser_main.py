""" Parse input arguments
"""
import argparse
import pathlib
import numpy

def get_parsed_arguments():
     parser = argparse.ArgumentParser()

     # File Directories
     parser.add_argument('--input_dir', dest='input_dir', type=pathlib.Path, required=True,
                         help='input directory containing the raw data to be processed')
     parser.add_argument('--output_dir', dest='output_dir', type=str, required=True,
                         help='output directory where output data will be saved')
     parser.add_argument('--selection_method', dest='selection_method', type=int, required=False, default=1,
                         help='OPTIONS: Chose the data to process. '
                              '\t 1 - Run it for all the files found and overwrite previous output files (default)'
                              '\t 2 - Run it only for the new files'
                              '\t 3 - Run specific files. Space key + enter key to stop the input prompt.')

     # File Description
     parser.add_argument('--montage', dest='montage', type=str, required=False, default=None,
                         help='information regarding the sensor locations, '
                              '\t - built_in mne montage or '
                              '\t - electrode layout file')

     # Segmentation
     parser.add_argument('--event_keys_for_segmentation', dest='event_keys_for_segmentation', type=str, nargs='+',
                         required=False, default=None,
                         help='array of event types relative to the epochs')
     parser.add_argument('--event_time_window', dest='event_time_window', default=None, type=float, nargs='+',
                         required=False, help='start and end time of the epochs in seconds')
     parser.add_argument('--baseline_time_window', dest='baseline_time_window', default=None, type=float, nargs='+',
                         required=False, help=' time interval to consider as baseline when applying baseline correction '
                                                  'of epochs, in seconds')
     parser.add_argument('--by_event_type', dest='by_event_type', default=True, type=bool,
                         required=False, help='evoked response per stimulus type, default to True')
     
     # Parallel processing
     parser.add_argument('--n_jobs', dest='n_jobs', type=int, required=False, 
                         help='number of cores to be used for parallel processing', default=-1)

     # Saving
     parser.add_argument('--save_preprocessed_raw', dest='save_preprocessed_raw', type=bool,
                         required=False, default=True, help='export the preprocessed continuous data')
     parser.add_argument('--save_segmented_data', dest='save_segmented_data', type=bool,
                         required=False, default=True, help='export the segmented data')
     parser.add_argument('--save_evoked_response', dest='save_evoked_response', type=bool,
                         required=False, default=True, help='export the ERPs relative to the chosen stimulus, '
                                                            'returns and array of channels x time per stimulus')
     parser.add_argument('--save_log', dest='save_log', type=bool, required=False, default=False,
                         help='create and save the logs for each preprocessed data')
     

     args = parser.parse_args()

     return args


# Check
if __name__ == '__main__':
     print("Get parsed arguments: including base path and data path \n")
     args_ = get_parsed_arguments()
     print(args_)
