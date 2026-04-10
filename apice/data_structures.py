# Import necessary modules
import json

import mne  
from mne.io import BaseRaw
from mne import BaseEpochs
import numpy as np 
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone


# Import specific modules from your project's modules
from apice.utils import (print_header, get_onset_and_duration, get_cfg)
from apice.artifacts_structure import ArtifactsEpochs, ArtifactsRaw
from apice.artifacts_rejection import run_algorithms
from apice.artifacts_correction import (TargetPCA, ChannelsSphericalSplineInterpolation, SegmentSphericalSplineInterpolation)




# %% CLASSES TO MANIPULATE THE RAW AND EPOCH DATA WITH THE ARTIFACTS REJECTION MATRICES

class RawAPICE(mne.io.RawArray):

    def __init__(self, raw: BaseRaw, verbose=None,
                 bt_label='badtime', bct_label='artifact', cct_label='corrected',
                 **kwargs):
        if not isinstance(raw, BaseRaw):
            raise TypeError(f"Expected a BaseRaw instance, got {type(raw)}")

        # Ensure data is loaded before copying internal state
        if not raw.preload:
            raw.load_data()

        # Copy the internal MNE infrastructure from the source object
        super().__init__(raw._data.copy(), raw.info.copy(), raw.first_samp, verbose=verbose)

        # Set annotations 
        self.set_annotations(raw.annotations.copy())

        if self.info['meas_date'] is None:
            self.set_meas_date(datetime.now(tz=timezone.utc))

        # Set projectors if any
        self._projector = raw._projector

        # Initialize artifacts structure
        self.artifacts = ArtifactsRaw(self, **kwargs)
        self.annotations_to_rejection_matrix(bt_label=bt_label, bct_label=bct_label, cct_label=cct_label)
        self.remove_artifacts_annotations(bt_label=bt_label, bct_label=bct_label, cct_label=cct_label)
        self.define_bcbt()

    def update_artifacts_params(self, **kwargs):
        self.artifacts.update_params(**kwargs)

    def compute_psd(self, *args, **kwargs):
        """Compute PSD through a native MNE RawArray for Spectrum compatibility."""
        tmp_raw = mne.io.RawArray(
            self._data.copy(),
            self.info.copy(),
            first_samp=self.first_samp,
            verbose=False,
        )
        tmp_raw.set_annotations(self.annotations.copy())
        tmp_raw.info['bads'] = self.info['bads'].copy()
        return tmp_raw.compute_psd(*args, **kwargs)

    def get_data_size(self):
        n_channels = len(self.ch_names)
        n_samples = len(self.times)
        n_epochs = 1  # For Raw data, we consider it as one continuous segment
        return n_channels, n_samples, n_epochs

    def export(self, file_name, output_dir, data_suffix='-preproc'):
        # rejection matrix to annotations
        self.annotate_bads(channels=True, times=True, data=True, corrected=True)
        # save preprocessed raw
        full_path = Path(output_dir) / (file_name + data_suffix + '.fif')
        self.save(full_path, overwrite=True)

    def bc_to_bads(self):        
        bad_channels_idx = np.where(self.artifacts.BC[:, 0])[0].astype(int)
        bad_channels = [self.ch_names[i] for i in bad_channels_idx]
        bad_channels_idx_manual = np.where(self.artifacts.BCmanual)[0].astype(int)
        bad_channels_manual = [self.ch_names[i] for i in bad_channels_idx_manual]
        bad_channels = self.info['bads'].copy() + bad_channels + bad_channels_manual
        self.info['bads'] = list(set(bad_channels))
        
    def annotate_bads(self, channels=True, times=True, data=True, corrected=True, bt_labels='badtime', bct_labels='artifact', cct_labels='corrected'):
        """
        Annotates bad channels, times, and artifacts in an EEG raw data structure.
        
        Parameters:
        - raw: The raw EEG data structure (usually an instance of mne.io.Raw or similar).
        - channels (bool): If True, annotate bad channels based on the 'BC' (bad channels) artifact flag.
        - times (bool): If True, annotate bad times based on the 'BT' (bad times) artifact flag.
        - data (bool): If True, annotate bad data based on the 'BCT' (bad channel times) artifact flag.
        - corrected (bool): If True, annotate data that has been corrected based on the 'CCT' (corrected channel times) artifact flag.
        
        Modifies the raw data structure by adding annotations for any identified bad data.
        """

        # Extract raw data dimensions
        n_channels, n_samples, n_epochs = self.get_data_size()

        # Annotate bad channels if specified and the artifact attribute exists
        if channels:
            self.bc_to_bads()

        # Keep annotation timing consistent even when first_samp != 0.
        # MNE expects onsets in Raw-time coordinates, i.e. relative to first_time.
        onset_offset = float(self.first_time)

        # Start from existing non-artifact annotations and append fresh artifact ones.
        self.remove_artifacts_annotations(
            bt_label=bt_labels,
            bct_label=bct_labels,
            cct_label=cct_labels,
        )
        annotations = self.annotations.copy()

        # Ensure orig_time is aligned with this Raw object.
        if annotations.orig_time is None:
            annotations = mne.Annotations(
                onset=list(annotations.onset),
                duration=list(annotations.duration),
                description=list(annotations.description),
                ch_names=list(annotations.ch_names),
                orig_time=self.info['meas_date'],
            )

        # Annotate bad times if specified
        if times and np.sum(self.artifacts.BT):
            onset, duration = get_onset_and_duration(self.artifacts.BT[0, :], self.times)
            onset = onset + onset_offset
            annotations.append(onset=onset, duration=duration, description=bt_labels)

        # Annotate bad data if specified
        if data and np.sum(self.artifacts.BCT):
            for el in np.arange(n_channels):
                bt_cha = np.asarray(self.artifacts.BCT[el, :], dtype=int)
                onset, duration = get_onset_and_duration(bt_cha, self.times)
                onset = onset + onset_offset
                description = [bct_labels] * len(onset)
                ch_names = [(self.ch_names[el],)] * len(onset)
                annotations.append(onset=onset, duration=duration, description=description, ch_names=ch_names)

        # Annotate corrected artifacts if specified 
        if corrected and np.sum(self.artifacts.CCT):
            for el in np.arange(n_channels):
                CCT = np.asarray(self.artifacts.CCT[el, :], dtype=int)
                onset, duration = get_onset_and_duration(CCT, self.times)
                onset = onset + onset_offset
                description = [cct_labels] * len(onset)
                ch_names = [(self.ch_names[el],)] * len(onset)
                annotations.append(onset=onset, duration=duration, description=description, ch_names=ch_names)

        self.set_annotations(annotations)


    def annotations_to_rejection_matrix(self, bt_label='badtime', bct_label='artifact', cct_label='corrected') -> None:
        """
        Converts annotations in an EEG raw data structure to a rejection matrix format.


        """

        print("Converting annotations to artifacts matrix")
        
        # Convert annotations to a DataFrame for easier manipulation
        annotations_df = self.annotations.to_data_frame(time_format=None)
        if 'ch_names' not in annotations_df.columns:
            annotations_df['ch_names'] = [[] for _ in range(len(annotations_df))]
            # annotations_df['ch_names'] = None 

        # Get time vector and channel list from the raw data structure
        t = self.times
        ch_names = np.asarray(self.info['ch_names'])

        # Get data size information from the custom Raw object
        n_channels, n_samples, n_epochs = self.get_data_size()

        # Create a rejection matrix for bad channels (BC)
        # Get indices of bad channels and ensure they are integers
        bad_channel_indices = np.array([np.where(ch_names == el)[0] for el in self.info['bads']], dtype=int).flatten()

        # Apply the bad channel mask efficiently
        self.artifacts.BC[bad_channel_indices, 0] = True

        # Create a rejection matrix for bad times (BT)
        bad_time = annotations_df[annotations_df['description'] == bt_label].reset_index(drop=True)

        # Vectorize search for nearest indices
        onset_indices = np.searchsorted(t, bad_time['onset'])
        end_indices = np.searchsorted(t, bad_time['onset'] + bad_time['duration'])

        # Efficiently apply the artifacts mask
        for start, end in zip(onset_indices, end_indices):
            self.artifacts.BT[0, start:end] = True

        # Create a rejection matrix for bad data (BCT)
        bad_data = annotations_df[annotations_df['description'] == bct_label].reset_index(drop=True)

        # Vectorized search for nearest indices
        onset_indices = np.searchsorted(t, bad_data['onset'])
        end_indices = np.searchsorted(t, bad_data['onset'] + bad_data['duration'])

        # Precompute channel indices
        channel_indices = np.array([np.where(ch_names == ch)[0][0] for ch in bad_data['ch_names']])

        # Efficiently apply the artifacts mask
        for el, start, end in zip(channel_indices, onset_indices, end_indices):
            self.artifacts.BCT[el, start:end] = True  

        # Create a rejection matrix for corrected data (CCT)
        corrected_data = annotations_df[annotations_df['description'] == cct_label].reset_index(drop=True)

        # Vectorized search for nearest indices
        onset_indices = np.searchsorted(t, corrected_data['onset'])
        end_indices = np.searchsorted(t, corrected_data['onset'] + corrected_data['duration'])

        # Precompute channel indices
        channel_indices = np.array([np.where(ch_names == ch)[0][0] for ch in corrected_data['ch_names']])

        # Efficiently apply the corrected artifacts mask
        for el, start, end in zip(channel_indices, onset_indices, end_indices):
            self.artifacts.CCT[el, start:end] = True 

    def remove_artifacts_annotations(self, bt_label='badtime', bct_label='artifact', cct_label='corrected') -> None:

        print("Removing artifact-related annotations")

        # get the annotations description as a list        
        annotations_description = self.annotations.description.copy()

        indexes_to_remove = [i for i, desc in enumerate(annotations_description) if desc in [bt_label, bct_label, cct_label]]
        indexes_to_remove = np.array(indexes_to_remove)

        # remove the annotations with the specified descriptions
        if len(indexes_to_remove) > 0:
            self.annotations.delete(indexes_to_remove)
        
    
    def segment_continuous_data(self, 
                                events,
                                event_id,
                                epoching_kwargs={}
                                ):
        """
        Segments continuous EEG data into epochs based on specified events.

        Parameters:
        raw : Raw EEG object
            Continuous EEG data to be segmented.
        epoching_kwargs : dict, optional
            Additional arguments to pass to the `mne.Epochs` constructor.
        
        Returns:
        epochs : mne.Epochs object
            The segmented epochs.
        """
        
        # Print a header for the segmentation process
        print_header('SEGMENTING CONTINUOUS DATA', separator='=')

        # Create epochs from the continuous raw data using the extracted events
        epochs = mne.Epochs(self, 
                            events=events, 
                            event_id=event_id, 
                            **epoching_kwargs,
                            )
                
        # Additional code to handle the rejection matrix and update artifacts in the epochs
        # Calculate left and right limits for the time window
        samples_start = ((-epochs.times[0]) * self.info['sfreq']).astype(int)
        samples_epoch = len(epochs.times)
        
        # Extract the stimulus times from the events
        stimulus_times = events[:, 0]

        # Identify the events to be dropped based on the drop log of epochs
        stimulus_events_to_drop = []
        for ep in np.arange(len(epochs.drop_log)):
            if len(epochs.drop_log[ep]) > 0:
                stimulus_events_to_drop.append(ep)

        # Update stimulus times by removing the dropped events
        stimulus_times = np.delete(stimulus_times, stimulus_events_to_drop).astype(int)

        # Initialize artifact structures in the epochs
        n_epochs = len(epochs)
        epochs = EpochsAPICE(epochs)

        # Update the artifact structures with information from the raw data
        epochs.artifacts.BCmanual = self.artifacts.BCmanual.copy()
        for ep in np.arange(n_epochs):
            try:
                # Convert absolute sample indices to indices relative to RawAPICE by accounting for first_samp
                epoch_start_time = (stimulus_times[ep] - self.first_samp - samples_start).astype(int)
                epoch_end_time = (epoch_start_time + samples_epoch - 1).astype(int)
                            
                epochs.artifacts.BCT[ep] = self.artifacts.BCT[:, epoch_start_time:epoch_end_time+1]
                epochs.artifacts.BT[ep] = self.artifacts.BT[:, epoch_start_time:epoch_end_time+1]
                epochs.artifacts.BC[ep] = self.artifacts.BC
                epochs.artifacts.CCT[ep] = self.artifacts.CCT[:, epoch_start_time:epoch_end_time+1]
            except IndexError:
                print(f"IndexError: Skipping epoch {ep} due to out-of-bounds indexing.")

        return epochs    
    

    def plot_percentage_of_bad_data_across_sensors(self):

        from matplotlib import pyplot as plt

        # Get the percentage of bad data per electrodes
        data = []
        for i, ch in enumerate(self.ch_names):
            n_bads = np.sum(self.artifacts.BCT[i, :])
            n_per = (n_bads / np.shape(self.artifacts.BCT[i, :])[0]) * 100
            data.append(n_per)
        
        # Create a figure explicitly
        fig, ax = plt.subplots()
        
        # Plot the topomap
        im, _ = mne.viz.plot_topomap(data, self.info, 
                            ch_type='eeg', 
                            names=self.ch_names, 
                            size=4, 
                            cmap='viridis',
                            axes=ax,
                            show=False)

        # Add a colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Percentage of bad data (%)')  # More descriptive label

        # Return the figure instead of the image
        return fig
    
    def plot_artifact_structure(self, artifact='all',time_step=50, color_scheme='gnuplot'):
        return self.artifacts.plot_artifact_structure(artifact=artifact, time_step=time_step, color_scheme=color_scheme)
    
    def run_algorithms(self, cfg_algorithms):
        run_algorithms(self, cfg_algorithms)

    def define_bcbt(self, keep_rejected_previous=None, plot_rejection_matrix=False):
        self.artifacts.define_bcbt(keep_rejected_previous=keep_rejected_previous, plot_rejection_matrix=plot_rejection_matrix)   

    def detect_bad_channels(self, cfg_bad_channels_detection=None):
        
        # if the cfg_bad_channels_detection is None load the default configuration for bad channels detection
        cfg_bad_channels_detection = get_cfg(cfg_bad_channels_detection, 'detect_bad_channels_config.json')
        
        # run the bad channels detection algorithm
        self.run_algorithms(cfg_bad_channels_detection)

    def detect_glitches(self, cfg_glitches_detection=None):
        
        # if the cfg_glitches_detection is None load the default configuration for glitches detection
        cfg_glitches_detection = get_cfg(cfg_glitches_detection, 'detect_artifacts_glitches_config.json')
        
        # run the glitches detection algorithm
        self.run_algorithms(cfg_glitches_detection)
        
    def detect_artifacts(self, cfg_artifacts_detection=None):
        
        # if the cfg_artifacts_detection is None load the default configuration for artifacts detection
        cfg_artifacts_detection = get_cfg(cfg_artifacts_detection, 'detect_artifacts_all_config.json')
        
        # run the artifacts detection algorithm
        self.run_algorithms(cfg_artifacts_detection)
        
    def correct_target_pca(self, cfg_target_pca=None):
        
        # correct using target PCA
        cfg_target_pca = get_cfg(cfg_target_pca, 'correction_target_pca_config.json')
        targetPCA = TargetPCA(**cfg_target_pca)
        targetPCA.correct(self)

        self.define_bcbt()

    def correct_spline_segments(self, cfg_spline_segments=None):
        
        # if the cfg_spline_segments is None load the default configuration for spline segments correction
        cfg_spline_segments = get_cfg(cfg_spline_segments, 'correction_spline_segments_config.json')
        
        # correct using spherical spline interpolation
        spline_segm = SegmentSphericalSplineInterpolation(**cfg_spline_segments)
        spline_segm.correct(self)

        self.define_bcbt()

    def correct_spline_channels(self, cfg_spline_channels=None):
        
        # if the cfg_spline_channels is None load the default configuration for bad channels correction
        cfg_spline_channels = get_cfg(cfg_spline_channels, 'correction_spline_channels_config.json')
        
        # correct using spherical spline interpolation
        spline_chan = ChannelsSphericalSplineInterpolation(**cfg_spline_channels)
        spline_chan.correct(self)

        self.define_bcbt()

    # write a methods that returns the mne.io.BaseRaw object without the artifacts structure (e.g., for compatibility with mne functions that require a BaseRaw object as input)
    def to_mne_raw(self, annotate_channels=True, annotate_times=True, annotate_data=True, annotate_corrected=True):
        # Convert rejection matrices to annotations before creating the new Raw object
        self.annotate_bads(channels=annotate_channels, times=annotate_times, data=annotate_data, corrected=annotate_corrected)
        # Create a new mne.io.Raw object with the same data and info as the current RawAPICE object
        raw_noart = mne.io.RawArray(self._data.copy(), self.info.copy(), self.first_samp, verbose="WARNING")
        raw_noart.set_annotations(self.annotations.copy())
        raw_noart._projector = self._projector  # Copy projectors if any
        return raw_noart

    def deal_with_reference_channels(self, reference_channels):
        if reference_channels is not None:
            idx_reference_channels = [self.ch_names.index(ch) for ch in reference_channels if ch in self.ch_names]
            if len(idx_reference_channels) > 0:
                self.artifacts.BC[idx_reference_channels, 0] = False  # Ensure reference channels are not marked as bad channels in BC
                self.artifacts.BCT[idx_reference_channels, :] = False  # Ensure reference channels are not marked as bad channels in BCT
                self.artifacts.BCT[idx_reference_channels, self.artifacts.BT[0,:]] = True


class EpochsAPICE(mne.EpochsArray):
    """
    A class for managing and processing EEG epoch data.

    This class includes methods for segmenting continuous EEG data into epochs, defining bad epochs based on various criteria, and removing bad epochs from the dataset.
    """
    
    def __init__(self, epochs: BaseEpochs, verbose=None, **kwargs):
        if not isinstance(epochs, BaseEpochs):
            raise TypeError(f"Expected a BaseEpochs instance, got {type(epochs)}")

        # Ensure data is loaded
        epochs.load_data()
        data = epochs.get_data().copy()  # (n_epochs, n_channels, n_times)

        super().__init__(
            data=data,
            info=epochs.info.copy(),
            events=epochs.events.copy(),
            event_id=epochs.event_id.copy(),
            tmin=epochs.tmin,
            baseline=epochs.baseline,
            reject=None,
            flat=None,
            proj=False,                         # projectors copied manually below
            on_missing='ignore',
            drop_log=epochs.drop_log,
            metadata=epochs.metadata.copy() if epochs.metadata is not None else None,
            selection=epochs.selection.copy(),
            verbose=verbose,
        )


        # Copy projectors if any
        self._projector = epochs._projector

        # Store source type for traceability
        self._source_type = type(epochs).__name__

        # Initialize artifacts structure
        self.artifacts = ArtifactsEpochs(self, **kwargs)
        self.define_bcbt()

    def update_artifacts_params(self, **kwargs):
        self.artifacts.update_params(**kwargs)

    def get_data_size(self):
        n_channels = len(self.ch_names)
        n_samples = len(self.times)
        n_epochs = len(self.events)
        return n_channels, n_samples, n_epochs

    def rejection_matrix_to_data_frame(self):

        artifacts_df = pd.DataFrame(columns=['epoch', 'ch_names', 'description', 'onset', 'duration'])  
        
        # BCT
        for ep in np.arange(np.shape(self.artifacts.BCT)[0]):
            for el in np.arange(np.shape(self.artifacts.BCT)[1]):
                    onset, duration = get_onset_and_duration(self.artifacts.BCT[ep, el, :], self.times)
                    if len(onset) > 0:
                        for i in range(len(onset)):
                            artifacts_df.loc[len(artifacts_df)] = [ep, self.ch_names[el], 'artifact', onset[i], duration[i]]
        # BC
        for ep in np.arange(np.shape(self.artifacts.BC)[0]):
            for el in np.arange(np.shape(self.artifacts.BC)[1]):
                    if self.artifacts.BC[ep, el, 0]:
                        artifacts_df.loc[len(artifacts_df)] = [ep, self.ch_names[el], 'badchannel', None, None]

        # BE
        for ep in np.arange(np.shape(self.artifacts.BE)[0]):
            if self.artifacts.BE[ep]:
                artifacts_df.loc[len(artifacts_df)] = [ep, None, 'badepoch', None, None]
        
        # BT
        for ep in np.arange(np.shape(self.artifacts.BT)[0]):
            onset, duration = get_onset_and_duration(self.artifacts.BT[ep, 0, :], self.times)
            if len(onset) > 0:
                for i in range(len(onset)):
                    artifacts_df.loc[len(artifacts_df)] = [ep, None, 'badtime', onset[i], duration[i]]
        
        # BCT
        for ep in np.arange(np.shape(self.artifacts.CCT)[0]):
            for el in np.arange(np.shape(self.artifacts.CCT)[1]):
                    onset, duration = get_onset_and_duration(self.artifacts.CCT[ep, el, :], self.times)
                    if len(onset) > 0:
                        for i in range(len(onset)):
                            artifacts_df.loc[len(artifacts_df)] = [ep, self.ch_names[el], 'corrected', onset[i], duration[i]]
        
        return artifacts_df

    def dataframe_to_rejection_matrix(self, artifacts_df):
        
        if 'ch_names' not in artifacts_df.columns:
            artifacts_df['ch_names'] = None 
        if 'epoch' not in artifacts_df.columns:
            artifacts_df['epoch'] = None

        # Get time vector and channel list from the raw data structure
        t = self.times
        ch_names = np.asarray(self.ch_names)

        # Get data size information from the custom Raw object
        n_channels, n_samples, n_epochs = self.get_data_size()

        # Set in the rejection matrix the bad channels
        for ep in np.arange(n_epochs):
            bad_channels = artifacts_df[(artifacts_df['description'] == 'badchannel') & (artifacts_df['epoch'] == ep)]['ch_names'].values
            bad_channel_indices = np.array([np.where(ch_names == el)[0] for el in bad_channels], dtype=int).flatten()
            self.artifacts.BC[ep, bad_channel_indices, :] = True

        # Set in the rejection matrix the bad times
        for ep in np.arange(n_epochs):
            bad_time = artifacts_df[(artifacts_df['description'] == 'badtime') & (artifacts_df['epoch'] == ep)].reset_index(drop=True)
            onset_indices = np.searchsorted(t, bad_time['onset'])
            end_indices = np.searchsorted(t, bad_time['onset'] + bad_time['duration'])
            for start, end in zip(onset_indices, end_indices):
                self.artifacts.BT[ep, :, start:end] = True

        # Set in the rejection matrix the bad data
        for ep in np.arange(n_epochs):
            bad_data = artifacts_df[(artifacts_df['description'] == 'artifact') & (artifacts_df['epoch'] == ep)].reset_index(drop=True)
            onset_indices = np.searchsorted(t, bad_data['onset'])
            end_indices = np.searchsorted(t, bad_data['onset'] + bad_data['duration'])
            channel_indices = np.array([np.where(ch_names == ch)[0][0] for ch in bad_data['ch_names']])
            for el, start, end in zip(channel_indices, onset_indices, end_indices):
                self.artifacts.BCT[ep, el, start:end] = True

        # Set in the rejection matrix the corrected data
        for ep in np.arange(n_epochs):  
            corrected_data = artifacts_df[(artifacts_df['description'] == 'corrected') & (artifacts_df['epoch'] == ep)].reset_index(drop=True)
            onset_indices = np.searchsorted(t, corrected_data['onset'])
            end_indices = np.searchsorted(t, corrected_data['onset'] + corrected_data['duration'])
            channel_indices = np.array([np.where(ch_names == ch)[0][0] for ch in corrected_data['ch_names']])
            for el, start, end in zip(channel_indices, onset_indices, end_indices):
                self.artifacts.CCT[ep, el, start:end] = True
        
        # Set in the rejection matrix the bad epochs
        for ep in np.arange(n_epochs):
            if len(artifacts_df[(artifacts_df['description'] == 'badepoch') & (artifacts_df['epoch'] == ep)]) > 0:
                self.artifacts.BE[ep, 0, 0] = True

    
    def run_algorithms(self, cfg_algorithms):
        run_algorithms(self, cfg_algorithms)

    def define_bcbt(self, keep_rejected_previous=None, plot_rejection_matrix=False):
        self.artifacts.define_bcbt(keep_rejected_previous=keep_rejected_previous, plot_rejection_matrix=plot_rejection_matrix)   

    def detect_bad_channels(self, cfg_bad_channels_detection=None):
        
        # if the cfg_bad_channels_detection is None load the default configuration for bad channels detection
        cfg_bad_channels_detection = get_cfg(cfg_bad_channels_detection, 'detect_bad_channels_config.json')
        
        # run the bad channels detection algorithm
        self.run_algorithms(cfg_bad_channels_detection)

    def detect_glitches(self, cfg_glitches_detection=None):
        
        # if the cfg_glitches_detection is None load the default configuration for glitches detection
        cfg_glitches_detection = get_cfg(cfg_glitches_detection, 'detect_artifacts_glitches_config.json')
        
        # run the glitches detection algorithm
        self.run_algorithms(cfg_glitches_detection)
        
    def detect_artifacts(self, cfg_artifacts_detection=None):
        
        # if the cfg_artifacts_detection is None load the default configuration for artifacts detection
        cfg_artifacts_detection = get_cfg(cfg_artifacts_detection, 'detect_artifacts_all_config.json')
        
        # run the artifacts detection algorithm
        self.run_algorithms(cfg_artifacts_detection)
        
    def correct_target_pca(self, cfg_target_pca=None):
        
        # correct using target PCA
        cfg_target_pca = get_cfg(cfg_target_pca, 'correction_target_pca_config.json')
        targetPCA = TargetPCA(**cfg_target_pca)
        targetPCA.correct(self)

        self.define_bcbt()

    def correct_spline_segments(self, cfg_spline_segments=None):
        
        # if the cfg_spline_segments is None load the default configuration for spline segments correction
        cfg_spline_segments = get_cfg(cfg_spline_segments, 'correction_spline_segments_config.json')
        
        # correct using spherical spline interpolation
        spline_segm = SegmentSphericalSplineInterpolation(**cfg_spline_segments)
        spline_segm.correct(self)

        self.define_bcbt()

    def correct_spline_channels(self, cfg_spline_channels=None):
        
        # if the cfg_spline_channels is None load the default configuration for bad channels correction
        cfg_spline_channels = get_cfg(cfg_spline_channels, 'correction_spline_channels_config.json')
        
        # correct using spherical spline interpolation
        spline_chan = ChannelsSphericalSplineInterpolation(**cfg_spline_channels)
        spline_chan.correct(self)

        self.define_bcbt()

    def define_bad_epochs(self, bad_data = 1, bad_time = 0, bad_channel = 0.3, lim_dist=2, lim_gfp=2):
        self.define_bad_epochs_artifacts(bad_data=bad_data, bad_time=bad_time, bad_channel=bad_channel, keeppre=False)
        if lim_dist:
            self.define_bad_epochs_dist(lim_dist=lim_dist, keeppre=True)
        if lim_gfp:
            self.define_bad_epochs_gfp(lim_gfp=lim_gfp, keeppre=True)

    def define_bad_epochs_artifacts(self, bad_data = 1, bad_time = 0, bad_channel = 0.3,
                        tmin=[], tmax=[], keeppre=True):

        print('\nIdentifying bad epochs based on the amount of bad data...')

        # initialize some stuff
        n_electrodes, n_samples, n_epochs = self.get_data_size()

        # Find the times to consider
        if not tmin:
            tmin = self.times[0]
        if not tmax:
            tmax = self.times[-1]
        idx_t = (self.times >= tmin) & (self.times <= tmax)
        n_samples = np.sum(idx_t)

        # get rejection
        bct = self.artifacts.BCT
        bt = self.artifacts.BT
        bc = self.artifacts.BC
        
        # get rejected data
        TOT = np.empty((n_epochs, 3))
        TOT[:] = np.nan
        TOT[:, 0] = np.sum(np.sum(bct[:, :, idx_t], axis=1), axis=1) / (n_samples * n_electrodes)
        TOT[:, 1] = np.squeeze(np.sum(bt[:, :, idx_t], axis=2) / n_samples)
        TOT[:, 2] = np.squeeze(np.sum(bc, axis=1) / n_electrodes)

        # get bad epochs
        thresh = [bad_data, bad_time, bad_channel]
        R = TOT > np.tile(thresh, [n_epochs, 1])
        bad_epochs = np.any(R, axis=1)
        
        # set the artifacts
        be_old = self.artifacts.BE.copy()
        if keeppre:
            self.artifacts.update_be(bad_epochs)
        else:
            self.artifacts.set_be(bad_epochs)
            
        # Display rejected data
        be_new = np.logical_and(bad_epochs.ravel(), be_old.ravel()==False)
        print('--> Rejected epochs by this algorithm: {:n} out of {:n} ({:.1%})'.format(np.sum(bad_epochs), n_epochs, np.sum(bad_epochs)/n_epochs))
        print('    --> BCT threshold {:.2f}: {:n} ({:.1%})'.format(thresh[0], np.sum(R[:, 0]), np.sum(R[:, 0]) / n_epochs))
        print('    --> BT threshold {:.2f}: {:n} ({:.1%})'.format(thresh[1], np.sum(R[:, 1]), np.sum(R[:, 1]) / n_epochs))
        print('    --> BC threshold {:.2f}: {:n} ({:.1%})'.format(thresh[2], np.sum(R[:, 2]), np.sum(R[:, 2]) / n_epochs))
        print('--> Total rejected epochs:             {:n} out of {:n} ({:.1%})'.format(np.sum(self.artifacts.BE[:]), n_epochs, np.sum(self.artifacts.BE[:])/n_epochs))
        print('--> New rejected epochs:               {:n} out of {:n} ({:.1%})'.format(np.sum(be_new), n_epochs, np.sum(be_new)/n_epochs))    
            
        return bad_epochs

    def define_bad_epochs_dist(self, 
                            lim_dist = 2, lim_bad_time_dist = None, lim_mean_dist = None, lim_max_dist=None, 
                            relative=True, maxloops=1, where=[], rmvmean=False, normdist=True,
                            l_freq_filter=None, h_freq_filter=None, keeppre=True):
        
        print('\nIdentifying bad epochs based on the distance to the average ERP...')

        if not lim_bad_time_dist:
            lim_bad_time_dist = 0.10/(self.times[-1]-self.times[0])  # 100 ms
        
        if not where:
            where = [self.times[0], self.times[-1]]
            
        n_electrodes, n_samples, n_epochs = self.get_data_size()
        if n_epochs > 1:
            
            epochs_ = self.copy()
            
            if keeppre:
                be_old = self.artifacts.BE.copy()
                be_old = np.reshape(be_old, n_epochs)
            else:
                be_old = np.full(n_epochs, False)
                    
            # high pass filter if necessary
            if l_freq_filter or h_freq_filter:
                epochs_.filter(l_freq_filter, h_freq_filter)
            
            # reference to the mean
            epochs_.set_eeg_reference(ref_channels='average')
            
            # take the data
            data = epochs_._data.copy()
            data_avg = epochs_._data.copy()
            
            # set as nan datapoints rejected
            data_avg[self.artifacts.BCT==True] = np.nan
            
            # do not considered bad electrodes 
            el_bad = np.reshape(np.all(self.artifacts.BC, axis=0), n_electrodes)
            data = data[:,~el_bad,:]
            data_avg = data_avg[:,~el_bad,:]
            
            # keep the times of interes
            idxtime = np.logical_and(self.times>=where[0], self.times<=where[1])
            n_smpl = np.sum(idxtime)
            data = data[:,:,idxtime]
            data_avg = data_avg[:,:,idxtime]
            
            # remove the mean
            if rmvmean:
                data_avg = data_avg - np.tile(np.mean(data_avg,axis=2), (1, 1, n_smpl))
                
            # reject epochs having samples that are too far from the average    
            be = be_old.copy()
            ok = False;
            ci = 1;
            be_dist = np.full(n_epochs, False)
            while (~ok and ci<=maxloops and np.any(be==0)):
                
                M = np.nanmean(data_avg[~be,:,:], axis=0)
                
                # scale the mean based on the standard deviations (otherwise the mean and the single trial data have differnt amplitudes)
                sdM = np.nanstd(M,axis=1)
                d = np.transpose(data, (1,2,0))
                d = np.reshape(d, (np.shape(d)[0],np.shape(d)[1]*np.shape(d)[2]))
                sdD = np.nanstd(d,axis=1)
                M = np.multiply(M,  np.tile(sdD[:,np.newaxis], (1, np.shape(M)[1]))) 
                M = np.divide(M , np.tile(sdM[:,np.newaxis], (1, np.shape(M)[1])))
                
                # compute the distance
                D = data - np.tile(M[np.newaxis,:,:],(n_epochs, 1, 1))
                D = np.squeeze(np.sqrt(np.sum(np.multiply(D,D), axis=1)))
                
                # log transfomation to have a normal distribution
                D = np.log(D)    
                
                # normalize the distance such that the variance and mean are equal across samples
                if normdist:
                    Dmu = np.mean(D[~be,:],axis=0)
                    Dvar = np.std(D[~be,:],axis=0)
                    D = D - np.tile(Dmu[np.newaxis,:], (np.shape(D)[0], 1))
                    D = np.divide(D, np.tile(Dvar[np.newaxis,:], (np.shape(D)[0], 1)) )
                    
                # threshold for the distance
                if relative:
                    d = D[~be,:]
                    perc = np.nanpercentile(d[:], [25, 75], method='midpoint')
                    threshD = perc[1] + lim_dist*(perc[1]-perc[0])
                else:
                    threshD = lim_dist
                RR = D > threshD
                
                # threshold for the amount of data too far away
                if lim_bad_time_dist:
                    Rt = (np.sum(RR,axis=1)/n_smpl) > lim_bad_time_dist
                else:
                    Rt = np.full(n_epochs, False)
                    
                # threshold for the mean distance
                if lim_mean_dist:
                    if relative:
                        d = np.mean(D[~be,:],axis=1)
                        perc = np.nanpercentile(d, [25, 75], method='midpoint')
                        threshMeanD = perc[1] + lim_mean_dist*(perc[1]-perc[0])
                    else:
                        threshMeanD = lim_mean_dist
                    Rmean = np.mean(D,axis=1) > threshMeanD
                else:
                    Rmean = np.full(n_epochs, False)
                    
                # threshold for the max distance
                if lim_max_dist:
                    if relative:
                        d = np.max(D[~be,:],axis=1)
                        perc = np.nanpercentile(d, [25, 75], method='midpoint')
                        threshMaxD = perc[1] + lim_max_dist*(perc[1]-perc[0])
                    else:
                        threshMaxD = lim_max_dist
                    Rmax = np.mean(D,axis=1) > threshMaxD
                else:
                    Rmax = np.full(n_epochs, False)
                    
                # Rejection vector
                R = np.logical_or(Rmean, Rmax)
                R = np.logical_or(R, Rt)  
                
                # check if new data was rejected
                if np.all(np.logical_or(R, be)==be):
                    ok = True
                    
                # update
                be_dist = np.logical_or(be_dist, R)
                be = np.logical_or(be, R)
                ci+=1
            
            # Display rejected data
            be_new = np.logical_and(be==True, be_old==False)
            print('--> Rejected epochs by this algorithm: {:n} out of {:n} ({:.1%})'.format(np.sum(be_dist), n_epochs, np.sum(be_dist)/n_epochs))
            print('--> Total rejected epochs:             {:n} out of {:n} ({:.1%})'.format(np.sum(be), n_epochs, np.sum(be)/n_epochs))
            print('--> New rejected epochs:               {:n} out of {:n} ({:.1%})'.format(np.sum(be_new), n_epochs, np.sum(be_new)/n_epochs))
            
            # set the artifacts
            if keeppre:
                self.artifacts.update_be(be)
            else:
                self.artifacts.set_be(be)
                
        else:
            be_dist = np.full(n_epochs, False)
            
        return be_dist


    def define_bad_epochs_gfp(self, 
                            lim_gfp = 2, lim_bad_time_gfp = None, lim_mean_gfp = None, lim_max_gfp=None, 
                            relative=True, maxloops=1, where=[],
                            l_freq_filter=None, h_freq_filter=None, keeppre=True):
        
        print('\nIdentifying bad epochs based on the GFP...')

        if not lim_bad_time_gfp:
            lim_bad_time_gfp = 0.10/(self.times[-1]-self.times[0])  # 100 ms
        
        if not where:
            where = [self.times[0], self.times[-1]]
            
        n_electrodes, n_samples, n_epochs = self.get_data_size()
        if n_epochs > 1:
            
            epochs_ = self.copy()
            
            if keeppre:
                be_old = self.artifacts.BE.copy()
                be_old = np.reshape(be_old, n_epochs)
            else:
                be_old = np.full(n_epochs, False)
                    
            # high pass filter if necessary
            if l_freq_filter or h_freq_filter:
                epochs_.filter(l_freq_filter, h_freq_filter)
            
            # reference to the mean
            epochs_.set_eeg_reference(ref_channels='average')
            
            # take the data
            data = epochs_._data.copy()
            
            # do not considered bad electrodes 
            el_bad = np.reshape(np.all(self.artifacts.BC, axis=0), n_electrodes)
            data = data[:,~el_bad,:]
            
            # keep the times of interest
            idxtime = np.logical_and(self.times>=where[0], self.times<=where[1])
            n_smpl = np.sum(idxtime)
            data = data[:,:,idxtime]
            
            # compute GFP
            GFP = np.std(data.copy(),axis=1)
            
            # log transformation to have a normal distribution
            GFP = np.log(GFP)    
            
            # reject epochs having a too big GFP 
            # ---------------------------------------------------------------------S
            be = be_old.copy()
            be_gfp = np.full(n_epochs, False)
                
            # threshold for the GFP
            if relative:
                d = GFP[~be,:]
                perc = np.nanpercentile(d[:], [25, 75], method='midpoint')
                threshGFP = perc[1] + lim_gfp*(perc[1]-perc[0])
            else:
                threshGFP = lim_gfp
            RR = GFP > threshGFP
            
            # threshold for the amount of data too far away
            if lim_bad_time_gfp:
                Rt = (np.sum(RR,axis=1)/n_smpl) > lim_bad_time_gfp
            else:
                Rt = np.full(n_epochs, False)
                
            # threshold for the mean GFP
            if lim_mean_gfp:
                if relative:
                    d = np.mean(GFP[~be,:],axis=1)
                    perc = np.nanpercentile(d, [25, 75], method='midpoint')
                    threshMeanGFP = perc[1] + lim_mean_gfp*(perc[1]-perc[0])
                else:
                    threshMeanGFP = lim_mean_gfp
                Rmean = np.mean(GFP,axis=1) > threshMeanGFP
            else:
                Rmean = np.full(n_epochs, False)
                
            # threshold for the max distance
            if lim_max_gfp:
                if relative:
                    d = np.max(GFP[~be,:],axis=1)
                    perc = np.nanpercentile(d, [25, 75], method='midpoint')
                    threshMaxGFP = perc[1] + lim_max_gfp*(perc[1]-perc[0])
                else:
                    threshMaxGFP = lim_max_gfp
                Rmax = np.mean(GFP,axis=1) > threshMaxGFP
            else:
                Rmax = np.full(n_epochs, False)
                
            # Rejection vector
            R = np.logical_or(Rmean, Rmax)
            R = np.logical_or(R, Rt)  
                
            # update
            be_gfp = np.logical_or(be_gfp, R)
            be = np.logical_or(be, R)
            
            # Display rejected data
            be_new = np.logical_and(be==True, be_old==False)
            print('--> Rejected epochs by this algorithm: {:n} out of {:n} ({:.1%})'.format(np.sum(be_gfp), n_epochs, np.sum(be_gfp)/n_epochs))
            print('--> Total rejected epochs:             {:n} out of {:n} ({:.1%})'.format(np.sum(be), n_epochs, np.sum(be)/n_epochs))
            print('--> New rejected epochs:               {:n} out of {:n} ({:.1%})'.format(np.sum(be_new), n_epochs, np.sum(be_new)/n_epochs))
            
            # set the artifacts
            if keeppre:
                self.artifacts.update_be(be)
            else:
                self.artifacts.set_be(be)
                
        else:
            be_gfp = np.full(n_epochs, False)
            
        return be_gfp    


    def remove_bad_epochs(self):
        """
        Removes bad epochs from the EEG data.

        Parameters:
        epochs : mne.Epochs object
            The epochs from which bad epochs will be removed.

        Returns:
        None
        """

        # Drop the bad epochs from the epochs dat
        self.drop(self.artifacts.BE, reason='bad epoch')

        # Update the artifacts matrices to reflect the removal of bad epochs
        good_epochs = ~self.artifacts.BE
        self.artifacts.n_epochs = np.sum(good_epochs)
        self.artifacts.BE = self.artifacts.BE[good_epochs]
        self.artifacts.BCT = self.artifacts.BCT[good_epochs, :, :]
        self.artifacts.BT = self.artifacts.BT[good_epochs, :, :]
        self.artifacts.BC = self.artifacts.BC[good_epochs, :, :]
        self.artifacts.CCT = self.artifacts.CCT[good_epochs, :, :]
        
        

    def export(self, file_name, output_dir, data_suffix='-epo'):

        # get the artifacts in a dataframe
        artifacts_df = self.rejection_matrix_to_data_frame()
        delattr(self, 'artifacts')
                
        # Save the epochs and the artifacts information in a csv file in the output directory
        print('\nExporting epochs...')
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        epochs_fullpath = output_dir / (file_name + data_suffix + '.fif')
        self.save(epochs_fullpath, overwrite=True)
        print(f"Epochs saved at {epochs_fullpath}.")
        
        art_fullpath = output_dir / (file_name + data_suffix + '-artifacts.csv')
        artifacts_df.to_csv(art_fullpath, index=False)
        print(f"\nEpochs artifacts information saved at {art_fullpath}.")
        
    def deal_with_reference_channels(self, reference_channels):
        if reference_channels is not None:
            idx_reference_channels = [self.ch_names.index(ch) for ch in reference_channels if ch in self.ch_names]
            self.artifacts.BC[:, idx_reference_channels, 0] = False  # Ensure reference channels are not marked as bad channels in BC
            self.artifacts.BCT[:, idx_reference_channels, :] = False  # Ensure reference channels are not marked as bad channels in BCT
            for i in range(self._data.shape[0]):
                self.artifacts.BCT[i, idx_reference_channels, self.artifacts.BT[i,0,:]] = True
       

    def plot_percentage_of_bad_data_across_sensors(self):

        from matplotlib import pyplot as plt

        # Get the percentage of bad data per electrodes
        data = []
        for i, ch in enumerate(self.ch_names):
            n_bads = np.sum(self.artifacts.BCT[:, i, :])
            n_per = (n_bads / np.shape(self.artifacts.BCT[:, i, :])[1]) * 100
            data.append(n_per)
        
        # Create a figure explicitly
        fig, ax = plt.subplots()
        
        # Plot the topomap
        im, _ = mne.viz.plot_topomap(data, self.info, 
                            ch_type='eeg', 
                            names=self.ch_names, 
                            size=4, 
                            cmap='viridis',
                            axes=ax,
                            show=False)

        # Add a colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Percentage of bad data (%)')  # More descriptive label

        # Return the figure instead of the image
        return fig
    
    def plot_artifact_structure(self, artifact='all',time_step=50, color_scheme='gnuplot'):
        return self.artifacts.plot_artifact_structure(artifact=artifact, time_step=time_step, color_scheme=color_scheme)
 