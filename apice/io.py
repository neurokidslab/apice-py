
# %% LIBRARIES
import os
import numpy as np
import mne.io
import apice.artifacts_structure
import pandas as pd 

# %% FUNCTIONS

def is_valid_extension(filename, valid_extensions):
    """
    Check if a filename has a valid extension.

    Parameters:
    - filename (str): The filename to check.
    - valid_extensions (list): A list of valid file extensions.

    Returns:
    - bool: True if the filename has a valid extension, False otherwise.
    """
    ext = os.path.splitext(filename)[-1][1:]
    return ext in valid_extensions


def get_files_to_process(input_dir, output_dir=None, data_selection_method=3):
    """
    Get a list of files to process based on the specified data selection method.

    Parameters:
    - input_dir (str): The input directory containing raw data files.
    - output_dir (str, optional): The output directory where preprocessed files will be saved.
    - data_selection_method (int, optional): The method to select files:
        1 - All valid files in the input directory.
        2 - Files not already processed in the input directory and output directory.
        3 - Manually input filenames.

    Returns:
    - list: A list of file paths to process.
    """
    # List of valid extensions
    valid_extensions = [
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
    ]

    # Get all files in the input directory
    filenames = os.listdir(input_dir)

    # Filter filenames based on valid extensions
    valid_filenames = [
        filename
        for filename in filenames
        if is_valid_extension(filename, valid_extensions)
    ]

    # Initialize subjects list
    subjects = []

    j = 0  # Count subjects

    if data_selection_method == 1:
        # Method 1: Include all valid files in the input directory
        subjects = [os.path.join(input_dir, filename) for filename in valid_filenames]
        j += len(valid_filenames)

    if data_selection_method == 2 and output_dir:
        # Method 2: Include files not already processed in input and output directories
        output_folder_files = os.listdir(output_dir)
        # Remove extensions
        for i, file_name in enumerate(output_folder_files):
            base_name, ext = os.path.splitext(file_name)
            output_folder_files[i] = base_name.replace('_prp', '')

        for i, file_name in enumerate(valid_filenames):
            base_name, ext = os.path.splitext(file_name)
            base_name = base_name.replace('_raw', '')
            if ext[1:] in valid_extensions:
                matching_output_file = any(
                    base_name in output_file for output_file in output_folder_files
                )
                if not matching_output_file:
                    subjects.append(os.path.join(input_dir, file_name))
                    j += 1

    if data_selection_method == 3:
        # Method 3: Manually input filenames
        while True:
            input_name = input("File name (press Enter to finish): ")
            if not input_name.strip():
                break
            if is_valid_extension(input_name, valid_extensions):
                subjects.append(os.path.join(input_dir, input_name))
                j += 1
            else:
                print(
                    f"Invalid file extension. Valid extensions are: {', '.join(valid_extensions)}"
                )

    if len(subjects) == 0:
        print("No raw data to preprocess.")

    return subjects


def print_header(header):
    """
    Print a header with a separator line above and below it.

    Args:
        - header (str): The text to be displayed as the header.

    Returns:
        None
    """
    separator = "=" * len(header)  # Create a separator line of '=' characters
    print("\n" + separator)  # Print the separator line
    print(header)  # Print the header text
    print(separator + "\n")  # Print the separator line below the header


# %% CLASSES


class Raw:
    @staticmethod
    def import_raw(data_name, baseline_correction="mean", montage=None):
        """
        Import raw EEG data and perform preprocessing steps.

        Args:
            - data_name (str): The full path to the raw data file.
            - baseline_correction (str): Baseline correction method (default: 'mean').
            - montage (str or None): The montage to use for electrode positions (default: None).

        Returns:
            - mne.io.Raw: The raw EEG data.
        """
        from mne.io import read_raw, read_raw_egi
        
        print_header("IMPORTING RAW DATA")

        print("Subject:", os.path.basename(data_name), "\n")

        # Import raw
        try:
            ext = os.path.splitext(data_name)[-1].lower()
            if ext == '.raw':
                raw = read_raw_egi(data_name, preload=False, verbose=False)
            else:
                raw = read_raw(data_name, preload=False, verbose=False)
        
            # Process raw structure
            if "stim" in np.unique(raw.get_channel_types()):
                Raw.stim_channels_to_annotations(raw)
                
                # Remove STIM channels
                raw.pick(meg=False, eeg=True, stim=False)
                
            # Drop VREF
            if "VREF" in raw.ch_names:
                raw.drop_channels(["VREF"]) 

        except Exception as e:
            print(f"\nError reading raw data: {e}")

        # Set montage
        if montage is not None:
            raw = Raw.set_eeg_montage(raw, montage)

        # Process and set annotations, events, and event ids
        raw = Raw.process_raw_annotations(raw)

        # Baseline Correction
        if baseline_correction is not None:
            raw._data = mne.baseline.rescale(
                raw.get_data(),
                raw.times,
                baseline=(None, None),
                mode=baseline_correction,
                copy=False,
            )

        return raw

    @staticmethod
    def set_eeg_montage(raw, montage):
        """
        Sets the EEG montage for a raw EEG object.

        This function attempts to read a custom montage first and, if that fails,
        it uses a standard montage with automatic head size determination.

        Args:
            - raw (mne.io.Raw): The MNE raw object containing EEG data.
            - montage (str): The name or file path of the EEG montage to be set.

        Returns:
            - mne.io.Raw: The raw EEG object with the new montage applied.
        """
        try:
            # Try to read a custom montage
            montage = mne.channels.read_custom_montage(montage)
            #montage = mne.channels.read_dig_fif(montage)
        except:
            # If reading fails, use a standard montage with automatic head size determination
            montage = mne.channels.make_standard_montage(montage)

        # Set the EEG montage for the raw object
        raw.set_montage(montage)

        return raw

    @staticmethod
    def process_raw_annotations(raw):
        """
        Process annotations in a raw MNE object.

        This function extracts annotations, separates them based on descriptions ('artifact', 'corrected', 'badtime'),
        and sets the event annotations accordingly.

        Args:
            - raw (mne.io.Raw): The MNE raw object containing EEG data with annotations.

        Returns:
            - raw (mne.io.Raw): The processed MNE raw object.
        """

        # Extract annotations and create event_ids
        annotations = apice.artifacts_structure.extract_annotations(raw)

        # Separate the annotations for artifacts and get the event annotations
        annotations_temp = annotations[
            ~annotations["Description"].isin(["artifact", "corrected", "badtime"])
        ].copy()

        # Create new annotations without the artifact annotations
        annotations_events = mne.Annotations(
            onset=list(annotations_temp["Onset"]),
            duration=list(annotations_temp["Duration"]),
            description=list(annotations_temp["Description"]),
            ch_names=list(annotations_temp["Channel"]),
        )

        # Set raw with new annotations
        raw.set_annotations(annotations_events)

        # Create events and event_ids attributes
        raw.events, raw.event_ids = mne.events_from_annotations(raw)

        # Put all original annotations back
        annotations = mne.Annotations(
            onset=list(annotations["Onset"]),
            duration=list(annotations["Duration"]),
            description=list(annotations["Description"]),
            ch_names=list(annotations["Channel"]),
        )

        # Set raw with original annotations
        raw.set_annotations(annotations)

        return raw

    @staticmethod
    def get_data_size(raw):
        """
        Get the shape of the EEG continuous signal.

        Args:
            - raw (Raw): Object containing the EEG data.

        Returns:
            - n_electrodes (int): Number of electrodes.
            - n_samples (int): Number of data points per epoch.
            - n_epochs (int): Number of continuous segments.
        """
        
        data_shape = np.shape(raw._data)  # Get the shape of raw data

        n_epochs = (
            1 if len(data_shape) == 2 else data_shape[0]
        )  # Determine the number of epochs
        n_electrodes = (
            data_shape[0] if len(data_shape) == 2 else data_shape[1]
        )  # Number of electrodes
        n_samples = (
            data_shape[1] if len(data_shape) == 2 else data_shape[2]
        )  # Number of data points per epoch

        return n_electrodes, n_samples, n_epochs

    @staticmethod
    def export_raw(raw, file_name, output_path):
        
        print_header('EXPORTING DATA')
        print('Subject:', file_name.replace('_prp.fif', ''), '\n')

        full_path = os.path.join(output_path, file_name)

        if hasattr(raw, 'events') and raw.events.size != 0:
            # Create a copy of annotations
            from apice.artifacts_structure import extract_annotations
            annotations_copy = extract_annotations(raw)

            # Create annotations from events
            events = np.squeeze(np.asarray(raw.events))
            mapping = dict()

            for key in raw.event_ids.keys():
                mapping[int(raw.event_ids[key])] = key

            annot_from_events = mne.annotations_from_events(events=events,
                                                            event_desc=mapping,
                                                            sfreq=raw.info['sfreq'],
                                                            orig_time=raw.info['meas_date'])

            # Combine annotations
            annotations = mne.Annotations(onset=annot_from_events.onset,
                                            ch_names=annot_from_events.ch_names,
                                            duration=annot_from_events.duration,
                                            description=annot_from_events.description)
            annotations.append(onset=list(annotations_copy['Onset']),
                                ch_names=list(annotations_copy['Channel']),
                                duration=list(annotations_copy['Duration']),
                                description=list(annotations_copy['Description']))

            # Annotate
            raw.set_annotations(annotations)

        full_path = os.path.join(output_path, file_name)
        print('\nSaving preprocessed raw...')
        mne.io.Raw.save(raw, full_path, overwrite=True)
    
    def stim_channels_to_annotations(raw):
        # Get a copy of the original annotations
        df_annotations = Raw.extract_annotations(raw=raw)
        
        # Convert stim channels to annotations
        print("\nConverting STIMs to annotations...")
        
        stims = raw.copy().pick(meg=False, eeg=False, stim=True)
        
        if stims:

            # Detect all events
            from mne import find_events
            events = find_events(raw, stim_channel=stims.ch_names, verbose=False)
            
            # Assuming event IDs are directly the data values in the stim channel
            event_ids = np.unique(events[:, 2])  # Unique event identifiers
            
            onsets = events[:, 0] / raw.info['sfreq']  # Convert sample indices to times
            
            ch_names = [()] * len(events)
            
            durations = [Raw.get_stim_duration(raw)] * len(events) 
            
            event_map = {event_id: f"{stims.ch_names[event_id - 1]}" for event_id in event_ids if event_id < len(raw.ch_names)}
            
            descriptions = [event_map[event_id] for event_id in events[:, 2]]

            # Create Annotations object
            df_annotations_from_stims = pd.DataFrame(dict(Onset=onsets, Duration=durations, Description=descriptions, Channel=ch_names))
            
            # List of DataFrames to combine
            dfs = [df_annotations, df_annotations_from_stims]
            
            # Filter out empty DataFrames
            dfs = [df for df in dfs if not df.empty]
            
            if dfs:
                # Concatenating DataFrames
                df_combined = pd.concat(dfs, ignore_index=True)

                # Dropping duplicates and sorting by 'Onset'
                df_final = df_combined.drop_duplicates().sort_values(by="Onset").reset_index(drop=True)

                from mne import Annotations 
                annotations = Annotations(onset=list(df_final["Onset"]),
                                            duration=list(df_final["Duration"]),
                                            description=list(df_final["Description"]),
                                            ch_names=list(df_final["Channel"]))
                
                raw.set_annotations(annotations)
    
    @staticmethod
    def extract_annotations(raw):
        """
        Extracts annotations from an MNE-Python EEG object and returns them as a pandas DataFrame.
        
        Parameters:
        - raw: Raw object
            The EEG object from MNE-Python containing annotations.
        
        Returns:
        - DataFrame
            A pandas DataFrame with the annotations data, including channels, descriptions, onsets, and durations.
            
        Notes:
        - The function assumes that the annotations in the EEG object are structured with ch_names, description, onset, and duration attributes.
        - The function will fail if the EEG object does not contain annotations or if the annotations do not have the expected attributes.
        """
        
        # Check if the EEG object has the attribute 'annotations'
        if hasattr(raw, 'annotations') and raw.annotations:
            # Create a DataFrame to hold the annotations
            annotations = pd.DataFrame(data=[], columns=['Channel', 'Description', 'Onset', 'Duration'])
            
            # Assign each column in the DataFrame by extracting corresponding data from the annotations
            annotations['Channel'] = raw.annotations.ch_names if hasattr(raw.annotations, 'ch_names') else ['N/A'] * len(raw.annotations.onset)
            annotations['Description'] = raw.annotations.description
            annotations['Onset'] = raw.annotations.onset
            annotations['Duration'] = raw.annotations.duration
            
            return annotations
        else:
            # If there are no annotations, return an empty DataFrame with the same structure
            return pd.DataFrame(columns=['Channel', 'Description', 'Onset', 'Duration'])
    
    @staticmethod
    def get_stim_duration(raw):

        # Load data from the specified stimulus channel
        stim_data = raw.copy().pick(meg=False, eeg=False, stim=True).get_data()
        
        above_baseline = np.where(stim_data > 0.01, 1, 0)
        diff = np.diff(above_baseline, prepend=0)

        # Event onsets (where diff == 1) and offsets (where diff == -1)
        onsets = np.where(diff == 1)[1]
        offsets = np.where(diff == -1)[1]

        # Check for the case where the last event doesn't have an offset
        if len(onsets) > len(offsets):
            if len(offsets) == 0 or onsets[-1] > offsets[-1]:
                offsets = np.append(offsets, len(stim_data[0])) 

        # Calculate durations in seconds
        durations = (offsets - onsets) / raw.info['sfreq']

        return np.max(durations)

def export_epoch(epochs, file_name, output_dir):

        # Remove the artifact annotation because it contains the raw annotations
        annotations = mne.Annotations(ch_names=[], description=[], duration=[], onset=[])
        epochs.set_annotations(annotations)

        import pandas as pd
        from apice.artifacts_structure import calculate_event_onsets_and_durations
        artifacts_df = pd.DataFrame(columns=['Epoch', 'Channel', 'Artifact', 'Onset', 'Duration'])
        
        # Save artifacts structure as dataframe
        
        # BCT
        for ep in np.arange(np.shape(epochs.artifacts.BCT)[0]):
            for el in np.arange(np.shape(epochs.artifacts.BCT)[1]):
                    onset, duration = calculate_event_onsets_and_durations(epochs.artifacts.BCT[ep, el, :], epochs.times, epochs.info['sfreq'])
                    if len(onset) > 0:
                        for i in range(len(onset)):
                            artifacts_df.loc[len(artifacts_df)] = [ep, epochs.ch_names[el], 'bad data', onset[i], duration[i]]
        # BC
        for ep in np.arange(np.shape(epochs.artifacts.BC)[0]):
            for el in np.arange(np.shape(epochs.artifacts.BCT)[1]):
                    if epochs.artifacts.BC[ep, el, 0]:
                        artifacts_df.loc[len(artifacts_df)] = [ep, epochs.ch_names[el], 'bad channel', '()', '()']

        # BE
        for ep in np.arange(np.shape(epochs.artifacts.BE)[0]):
            if epochs.artifacts.BE[ep, 0, 0]:
                artifacts_df.loc[len(artifacts_df)] = [ep, '()', 'bad epoch', '()', '()']
        
        # BT
        for ep in np.arange(np.shape(epochs.artifacts.BT)[0]):
            onset, duration = calculate_event_onsets_and_durations(epochs.artifacts.BT[ep, 0, :], epochs.times, epochs.info['sfreq'])
            if len(onset) > 0:
                for i in range(len(onset)):
                    artifacts_df.loc[len(artifacts_df)] = [ep, 'all channels', 'bad time', onset[i], duration[i]]
        
        # BCT
        for ep in np.arange(np.shape(epochs.artifacts.CCT)[0]):
            for el in np.arange(np.shape(epochs.artifacts.CCT)[1]):
                    onset, duration = calculate_event_onsets_and_durations(epochs.artifacts.CCT[ep, el, :], epochs.times, epochs.info['sfreq'])
                    if len(onset) > 0:
                        for i in range(len(onset)):
                            artifacts_df.loc[len(artifacts_df)] = [ep, epochs.ch_names[el], 'corrected data', onset[i], duration[i]]
        
        import os
        # Define fine name
        folder_path = os.path.join(output_dir, 'artifacts')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        full_path = os.path.join(folder_path, file_name.split(sep='.')[0]+'_art.csv')
        
        # Save data frame
        artifacts_df.to_csv(full_path, index=False)
        print(f"\nEpochs artifacts information saved at {full_path}.")
        
        # Remove bad channel in info
        epochs.info['bads'] = []
        
        # Save epochs
        print('\nExporting epochs...')
        full_path = os.path.join(output_dir, 'epochs', file_name)
        print(f"Writing {full_path}")
        epochs.save(full_path, overwrite=True)
        print(f"Closing {full_path}")
        print('[done]')