"""Data containers extending MNE objects with APICE artifact structures.

This module defines ``RawAPICE`` and ``EpochsAPICE`` wrappers that attach
artifact masks, detection/correction utilities, and export helpers.
"""

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
    """Raw EEG container with APICE artifact matrices and utilities.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Source raw recording copied into this wrapper.
    verbose : bool | str | int | None, default=None
        MNE verbosity setting.
    bt_label : str, default='badtime'
        Annotation label interpreted as bad-time segments.
    bct_label : str, default='artifact'
        Annotation label interpreted as bad-channel-time segments.
    cct_label : str, default='corrected'
        Annotation label interpreted as corrected segments.
    **kwargs
        Additional parameters passed to ``ArtifactsRaw``.
    """

    def __init__(self, raw: BaseRaw, verbose=None,
                 bt_label='badtime', bct_label='artifact', cct_label='corrected',
                 **kwargs):
        """Initialize a ``RawAPICE`` object from an MNE raw object.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            Source raw object.
        verbose : bool | str | int | None, default=None
            MNE verbosity setting.
        bt_label : str, default='badtime'
            Annotation description used for bad times.
        bct_label : str, default='artifact'
            Annotation description used for bad channel-time samples.
        cct_label : str, default='corrected'
            Annotation description used for corrected samples.
        **kwargs
            Additional arguments forwarded to ``ArtifactsRaw``.

        Returns
        -------
        None
        """
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
        """Update artifact parameter values in-place.

        Parameters
        ----------
        **kwargs
            Parameter names and values accepted by ``self.artifacts``.

        Returns
        -------
        None
        """
        self.artifacts.update_params(**kwargs)

    def compute_psd(self, *args, **kwargs):
        """Compute power spectral density using a temporary MNE raw object.

        Parameters
        ----------
        *args
            Positional arguments forwarded to ``mne.io.Raw.compute_psd``.
        **kwargs
            Keyword arguments forwarded to ``mne.io.Raw.compute_psd``.

        Returns
        -------
        spectrum : mne.time_frequency.Spectrum
            PSD result returned by MNE.
        """
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
        """Return dimensions of the wrapped raw recording.

        Returns
        -------
        n_channels : int
            Number of channels.
        n_samples : int
            Number of samples.
        n_epochs : int
            Always ``1`` for continuous raw data.
        """
        n_channels = len(self.ch_names)
        n_samples = len(self.times)
        n_epochs = 1  # For Raw data, we consider it as one continuous segment
        return n_channels, n_samples, n_epochs

    def export(self, full_path, overwrite=False):
        """Export raw data to FIF after writing artifact annotations.

        Parameters
        ----------
        full_path : str | pathlib.Path
            Full path to the output file, including filename and extension.
        overwrite : bool, default=True
            If True, overwrite existing files.

        Returns
        -------
        None
        """
        # rejection matrix to annotations
        self.annotate_bads(channels=True, times=True, data=True, corrected=True)
        # save preprocessed raw
        full_path = Path(full_path)
        self.save(full_path, overwrite=overwrite)

    def bc_to_bads(self):
        """Copy bad-channel flags from artifact masks into ``info['bads']``.

        Returns
        -------
        None
        """
        bad_channels_idx = np.where(self.artifacts.BC[:, 0])[0].astype(int)
        bad_channels = [self.ch_names[i] for i in bad_channels_idx]
        bad_channels_idx_manual = np.where(self.artifacts.BCmanual)[0].astype(int)
        bad_channels_manual = [self.ch_names[i] for i in bad_channels_idx_manual]
        bad_channels = self.info['bads'].copy() + bad_channels + bad_channels_manual
        self.info['bads'] = list(set(bad_channels))
        
    def annotate_bads(self, channels=True, times=True, data=True, corrected=True, bt_labels='badtime', bct_labels='artifact', cct_labels='corrected'):
        """Write artifact masks into MNE annotations.

        Parameters
        ----------
        channels : bool, default=True
            If True, copy bad channels to ``info['bads']``.
        times : bool, default=True
            If True, annotate bad-time segments from ``BT``.
        data : bool, default=True
            If True, annotate bad channel-time segments from ``BCT``.
        corrected : bool, default=True
            If True, annotate corrected channel-time segments from ``CCT``.
        bt_labels : str, default='badtime'
            Description label for bad-time annotations.
        bct_labels : str, default='artifact'
            Description label for bad channel-time annotations.
        cct_labels : str, default='corrected'
            Description label for corrected annotations.

        Returns
        -------
        None
            Updates ``self.annotations`` in place.
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
        """Populate artifact masks from existing annotations.

        Parameters
        ----------
        bt_label : str, default='badtime'
            Annotation description interpreted as bad-time intervals.
        bct_label : str, default='artifact'
            Annotation description interpreted as bad channel-time intervals.
        cct_label : str, default='corrected'
            Annotation description interpreted as corrected intervals.

        Returns
        -------
        None
            Updates ``self.artifacts`` masks in place.
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
        """Remove artifact-related annotations from the raw object.

        Parameters
        ----------
        bt_label : str, default='badtime'
            Label for bad-time annotations.
        bct_label : str, default='artifact'
            Label for bad channel-time annotations.
        cct_label : str, default='corrected'
            Label for corrected annotations.

        Returns
        -------
        None
        """

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
        """Create ``EpochsAPICE`` from continuous data and transfer masks.

        Parameters
        ----------
        events : numpy.ndarray
            MNE events array.
        event_id : dict | int | list
            Event selection passed to ``mne.Epochs``.
        epoching_kwargs : dict, default={}
            Additional keyword arguments for ``mne.Epochs``.

        Returns
        -------
        epochs : EpochsAPICE
            Epoched data with artifact masks derived from raw masks.
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
        """Plot topographic percentage of bad data per channel.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Generated topomap figure.
        """
        from matplotlib import pyplot as plt

        # Get the percentage of bad data per electrodes
        data = []
        for i, ch in enumerate(self.ch_names):
            idx_t = self.artifacts.BT[0, :]==False
            n_bads = np.sum(self.artifacts.BCT[i, idx_t])
            n_per = (n_bads / np.sum(idx_t)) * 100
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
    
    def plot_artifact_structure(self, artifact='all',time_step=50, color_scheme='turbo'):
        """Plot raw artifact masks.

        Parameters
        ----------
        artifact : {'all', 'BCT', 'BT', 'BC', 'BE'}, default='all'
            Artifact layer to display.
        time_step : int, default=50
            Tick spacing for x-axis labels.
        color_scheme : str, default='turbo'
            Matplotlib colormap.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Artifact heatmap figure.
        """
        return self.artifacts.plot_artifact_structure(artifact=artifact, time_step=time_step, color_scheme=color_scheme)

    def plot_bad_channels_bar(self):
        """Bar plot of bad-data percentage per channel, excluding bad-time samples.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Bar chart figure.
        """
        return self.artifacts.plot_bad_channels_bar()

    def plot_bad_times_line(self):
        """Line plot of bad-channel percentage at each time sample, excluding bad channels.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Line plot figure.
        """
        return self.artifacts.plot_bad_times_line()

    def run_algorithms(self, cfg_algorithms, force_cfg=False, l_freq=None, h_freq=None):
        """Run configured detection/rejection algorithms on this raw object.

        Parameters
        ----------
        cfg_algorithms : dict
            Algorithm configuration dictionary.
        force_cfg : bool, default=False
            If True, bypass safety checks for ``update_artifacts`` flags.
        l_freq : float | None, default=None
            Optional high-pass cutoff used before running algorithms.
        h_freq : float | None, default=None
            Optional low-pass cutoff used before running algorithms.


        Returns
        -------
        None
        """
        run_algorithms(self, cfg_algorithms, force_cfg=force_cfg, l_freq=l_freq, h_freq=h_freq)

    def define_bcbt(self, keep_rejected_previous=None, plot_rejection_matrix=False):
        """Recompute ``BC`` and ``BT`` masks from current ``BCT``.

        Parameters
        ----------
        keep_rejected_previous : {'bt', 'bc'} | None, default=None
            Preserve previous bad-time or bad-channel flags.
        plot_rejection_matrix : bool, default=False
            If True, display the artifact matrix plot.

        Returns
        -------
        None
        """
        self.artifacts.define_bcbt(keep_rejected_previous=keep_rejected_previous, plot_rejection_matrix=plot_rejection_matrix)   

    def detect_bad_channels(self, cfg=None, l_freq=None, h_freq=None):
        """Detect bad channels using the configured or default pipeline.

        Parameters
        ----------
        cfg : None | str | pathlib.Path | dict, default=None
            Custom configuration source. ``None`` loads package defaults.
        l_freq : float | None, default=None
            Optional high-pass cutoff used before running algorithms.
        h_freq : float | None, default=None
            Optional low-pass cutoff used before running algorithms.

        Returns
        -------
        None
        """
        cfg_bad_channels_detection = get_cfg(cfg, 'detect_bad_channels_config.json')
        self.run_algorithms(cfg_bad_channels_detection, l_freq=l_freq, h_freq=h_freq)

    def detect_glitches(self, cfg=None, l_freq=None, h_freq=None):
        """Detect glitches using the configured or default pipeline.

        Parameters
        ----------
        cfg : None | str | pathlib.Path | dict, default=None
            Custom configuration source. ``None`` loads package defaults.
        l_freq : float | None, default=None
            Optional high-pass cutoff used before running algorithms.
        h_freq : float | None, default=None
            Optional low-pass cutoff used before running algorithms.

        Returns
        -------
        None
        """
        cfg_glitches_detection = get_cfg(cfg, 'detect_artifacts_glitches_config.json')
        self.run_algorithms(cfg_glitches_detection, l_freq=l_freq, h_freq=h_freq)
        
    def detect_artifacts(self, cfg=None, l_freq=None, h_freq=None):
        """Detect artifacts using the configured or default pipeline.

        Parameters
        ----------
        cfg : None | str | pathlib.Path | dict, default=None
            Custom configuration source. ``None`` loads package defaults.
        l_freq : float | None, default=None
            Optional high-pass cutoff used before running algorithms.
        h_freq : float | None, default=None
            Optional low-pass cutoff used before running algorithms.

        Returns
        -------
        None
        """
        cfg_artifacts_detection = get_cfg(cfg, 'detect_artifacts_all_config.json')
        self.run_algorithms(cfg_artifacts_detection, l_freq=l_freq, h_freq=h_freq)
        
    def correct_target_pca(self, cfg=None):
        """Apply target PCA artifact correction.

        Parameters
        ----------
        cfg : None | str | pathlib.Path | dict, default=None
            Correction configuration. ``None`` loads package defaults.

        Returns
        -------
        None
        """
        cfg_target_pca = get_cfg(cfg, 'correction_target_pca_config.json')
        targetPCA = TargetPCA(**cfg_target_pca)
        targetPCA.correct(self)
        self.define_bcbt()

    def correct_spline_segments(self, cfg=None):
        """Apply segment-wise spherical spline interpolation correction.

        Parameters
        ----------
        cfg : None | str | pathlib.Path | dict, default=None
            Correction configuration. ``None`` loads package defaults.

        Returns
        -------
        None
        """
        cfg_spline_segments = get_cfg(cfg, 'correction_spline_segments_config.json')
        spline_segm = SegmentSphericalSplineInterpolation(**cfg_spline_segments)
        spline_segm.correct(self)
        self.define_bcbt()

    def correct_spline_channels(self, cfg=None):
        """Apply channel-wise spherical spline interpolation correction.

        Parameters
        ----------
        cfg : None | str | pathlib.Path | dict, default=None
            Correction configuration. ``None`` loads package defaults.

        Returns
        -------
        None
        """
        cfg_spline_channels = get_cfg(cfg, 'correction_spline_channels_config.json')
        spline_chan = ChannelsSphericalSplineInterpolation(**cfg_spline_channels)
        spline_chan.correct(self)
        self.define_bcbt()

    def to_mne_raw(self, annotate_channels=True, annotate_times=True, annotate_data=True, annotate_corrected=True):
        """Return a plain MNE ``RawArray`` copy without APICE wrapper state.

        Parameters
        ----------
        annotate_channels : bool, default=True
            If True, propagate bad channels to ``info['bads']`` before export.
        annotate_times : bool, default=True
            If True, write bad-time annotations.
        annotate_data : bool, default=True
            If True, write bad channel-time annotations.
        annotate_corrected : bool, default=True
            If True, write corrected-data annotations.

        Returns
        -------
        raw_noart : mne.io.RawArray
            MNE raw object containing data and annotations only.
        """
        self.annotate_bads(channels=annotate_channels, times=annotate_times, data=annotate_data, corrected=annotate_corrected)
        raw_noart = mne.io.RawArray(self._data.copy(), self.info.copy(), self.first_samp, verbose="WARNING")
        raw_noart.set_annotations(self.annotations.copy())
        raw_noart._projector = self._projector  # Copy projectors if any
        return raw_noart

    def deal_with_reference_channels(self, reference_channels):
        """Ensure reference channels are handled consistently in masks.

        Parameters
        ----------
        reference_channels : list of str | None
            Channel names that should not be marked as globally bad channels.

        Returns
        -------
        None
        """
        if reference_channels is not None:
            idx_reference_channels = [self.ch_names.index(ch) for ch in reference_channels if ch in self.ch_names]
            if len(idx_reference_channels) > 0:
                self.artifacts.BC[idx_reference_channels, 0] = False
                self.artifacts.BCT[idx_reference_channels, :] = False
                self.artifacts.BCT[idx_reference_channels, self.artifacts.BT[0,:]] = True


# %% UTILITY FUNCTIONS FOR EPOCHS

def normalize_epochs(epochs, by_epochs='single', by_channels='all', where=None,
                     mean_value=0, std_value=None, rescale=None):
    """Normalize epoch data in place.

    Statistics are computed over the specified time window (or the full epoch
    if *where* is ``None``) and applied to the entire epoch duration.

    Parameters
    ----------
    epochs : mne.BaseEpochs
        Epochs object to normalize.  Modified in place.
    by_epochs : {'single', 'all'}, default='single'
        If ``'single'``, statistics are computed independently per epoch.
        If ``'all'``, data from all epochs are pooled before computing
        statistics.
    by_channels : {'single', 'all'}, default='all'
        If ``'single'``, statistics are computed independently per channel.
        If ``'all'``, data from all channels are pooled.
    where : list of float | None, default=None
        ``[tmin, tmax]`` window in seconds used to compute statistics.
        If ``None``, the full epoch duration is used.
    mean_value : float | numpy.ndarray | None, default=0
        Value to subtract.  If ``None``, the mean is computed from the data.
        Default ``0`` leaves the mean unchanged.
    std_value : float | numpy.ndarray | None, default=None
        Divisor.  If ``None``, the standard deviation is computed from the
        data.
    rescale : bool | None, default=None
        If ``True``, rescale after normalization so the global standard
        deviation matches the original data.

    Returns
    -------
    epochs : mne.BaseEpochs
        The same object with normalized ``_data``.
    """
    print('Epoch normalization...')
    print(f' - by_epochs={by_epochs!r}, by_channels={by_channels!r}')

    n_epochs = len(epochs.events)
    n_channels = len(epochs.ch_names)
    n_samples = len(epochs.times)

    if mean_value is None or std_value is None:

        if where is None:
            where = [epochs.times[0], epochs.times[-1]]
        print(f' - Statistics computed over [{where[0]:.3f}, {where[1]:.3f}] s')

        idxtime = np.logical_and(epochs.times >= where[0], epochs.times <= where[1])
        ref_data = epochs._data[:, :, idxtime].copy()  # (n_epochs, n_channels, n_window)

        if by_epochs == 'all':
            # Pool across all epochs: reshape to (n_channels, n_epochs * n_window)
            ref_data_t = np.transpose(ref_data, (1, 0, 2))
            ref_data_t = np.reshape(ref_data_t, (n_channels, -1))
            if by_channels == 'single':
                if mean_value is None:
                    mv = np.nanmean(ref_data_t, axis=1)       # (n_channels,)
                    mean_value = np.tile(mv[np.newaxis, :, np.newaxis], (n_epochs, 1, n_samples))
                if std_value is None:
                    sv = np.nanstd(ref_data_t, axis=1)        # (n_channels,)
                    std_value = np.tile(sv[np.newaxis, :, np.newaxis], (n_epochs, 1, n_samples))
            else:  # by_channels == 'all'
                if mean_value is None:
                    mean_value = float(np.nanmean(ref_data_t))
                if std_value is None:
                    std_value = float(np.nanstd(ref_data_t))

        else:  # by_epochs == 'single'
            if by_channels == 'single':
                if mean_value is None:
                    mv = np.nanmean(ref_data, axis=2)          # (n_epochs, n_channels)
                    mean_value = np.tile(mv[:, :, np.newaxis], (1, 1, n_samples))
                if std_value is None:
                    sv = np.nanstd(ref_data, axis=2)           # (n_epochs, n_channels)
                    std_value = np.tile(sv[:, :, np.newaxis], (1, 1, n_samples))
            else:  # by_channels == 'all'
                ref_flat = np.reshape(ref_data, (n_epochs, -1))  # (n_epochs, n_channels * n_window)
                if mean_value is None:
                    mv = np.nanmean(ref_flat, axis=1)          # (n_epochs,)
                    mean_value = np.tile(mv[:, np.newaxis, np.newaxis], (1, n_channels, n_samples))
                if std_value is None:
                    sv = np.nanstd(ref_flat, axis=1)           # (n_epochs,)
                    std_value = np.tile(sv[:, np.newaxis, np.newaxis], (1, n_channels, n_samples))

    if rescale:
        rescale_value = float(np.std(epochs._data))

    epochs._data = epochs._data - mean_value
    epochs._data = np.divide(epochs._data, std_value)

    if rescale:
        epochs._data = epochs._data * rescale_value

    return epochs


class EpochsAPICE(mne.EpochsArray):
    """Epoched EEG container with APICE artifact matrices and utilities.

    Parameters
    ----------
    epochs : mne.BaseEpochs
        Source epoched recording copied into this wrapper.
    verbose : bool | str | int | None, default=None
        MNE verbosity setting.
    **kwargs
        Additional parameters passed to ``ArtifactsEpochs``.
    """
    
    def __init__(self, epochs: BaseEpochs, verbose=None, **kwargs):
        """Initialize an ``EpochsAPICE`` object from an MNE epochs object.

        Parameters
        ----------
        epochs : mne.BaseEpochs
            Source epochs object.
        verbose : bool | str | int | None, default=None
            MNE verbosity setting.
        **kwargs
            Additional arguments forwarded to ``ArtifactsEpochs``.

        Returns
        -------
        None
        """
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
        """Update artifact parameter values in-place.

        Parameters
        ----------
        **kwargs
            Parameter names and values accepted by ``self.artifacts``.

        Returns
        -------
        None
        """
        self.artifacts.update_params(**kwargs)

    def get_data_size(self):
        """Return dimensions of the wrapped epochs object.

        Returns
        -------
        n_channels : int
            Number of channels.
        n_samples : int
            Number of samples per epoch.
        n_epochs : int
            Number of epochs.
        """
        n_channels = len(self.ch_names)
        n_samples = len(self.times)
        n_epochs = len(self.events)
        return n_channels, n_samples, n_epochs

    def rejection_matrix_to_data_frame(self):
        """Convert artifact masks to a long-form dataframe.

        Returns
        -------
        artifacts_df : pandas.DataFrame
            DataFrame with columns ``epoch``, ``ch_names``, ``description``,
            ``onset``, ``duration``, and ``reason``.  The ``reason`` column
            is populated only for ``badepoch`` rows and contains a
            semicolon-separated string of the criterion labels that flagged
            the epoch (e.g. ``'artifacts;distance'``).
        """

        artifacts_df = pd.DataFrame(columns=['epoch', 'ch_names', 'description', 'onset', 'duration', 'reason'])

        # BCT
        for ep in np.arange(np.shape(self.artifacts.BCT)[0]):
            for el in np.arange(np.shape(self.artifacts.BCT)[1]):
                    onset, duration = get_onset_and_duration(self.artifacts.BCT[ep, el, :], self.times)
                    if len(onset) > 0:
                        for i in range(len(onset)):
                            artifacts_df.loc[len(artifacts_df)] = [ep, self.ch_names[el], 'artifact', onset[i], duration[i], None]
        # BC
        for ep in np.arange(np.shape(self.artifacts.BC)[0]):
            for el in np.arange(np.shape(self.artifacts.BC)[1]):
                    if self.artifacts.BC[ep, el, 0]:
                        artifacts_df.loc[len(artifacts_df)] = [ep, self.ch_names[el], 'badchannel', None, None, None]

        # BE
        for ep in np.arange(np.shape(self.artifacts.BE)[0]):
            if self.artifacts.BE[ep]:
                reason_str = ';'.join(sorted(self.artifacts.rejection_reasons[ep])) if self.artifacts.rejection_reasons[ep] else ''
                artifacts_df.loc[len(artifacts_df)] = [ep, None, 'badepoch', None, None, reason_str]

        # BT
        for ep in np.arange(np.shape(self.artifacts.BT)[0]):
            onset, duration = get_onset_and_duration(self.artifacts.BT[ep, 0, :], self.times)
            if len(onset) > 0:
                for i in range(len(onset)):
                    artifacts_df.loc[len(artifacts_df)] = [ep, None, 'badtime', onset[i], duration[i], None]

        # CCT
        for ep in np.arange(np.shape(self.artifacts.CCT)[0]):
            for el in np.arange(np.shape(self.artifacts.CCT)[1]):
                    onset, duration = get_onset_and_duration(self.artifacts.CCT[ep, el, :], self.times)
                    if len(onset) > 0:
                        for i in range(len(onset)):
                            artifacts_df.loc[len(artifacts_df)] = [ep, self.ch_names[el], 'corrected', onset[i], duration[i], None]

        return artifacts_df

    def dataframe_to_rejection_matrix(self, artifacts_df):
        """Populate artifact masks from a dataframe representation.

        Parameters
        ----------
        artifacts_df : pandas.DataFrame
            DataFrame with artifact annotations per epoch/channel/time.

        Returns
        -------
        None
            Updates artifact matrices in place.
        """
        
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
        
        # Set in the rejection matrix the bad epochs and restore rejection reasons
        has_reason_col = 'reason' in artifacts_df.columns
        for ep in np.arange(n_epochs):
            bad_ep_rows = artifacts_df[(artifacts_df['description'] == 'badepoch') & (artifacts_df['epoch'] == ep)]
            if len(bad_ep_rows) > 0:
                self.artifacts.BE[ep] = True
                if has_reason_col:
                    reason_str = bad_ep_rows['reason'].iloc[0]
                    if pd.notna(reason_str) and reason_str != '':
                        self.artifacts.rejection_reasons[ep] = set(reason_str.split(';'))
                    else:
                        self.artifacts.rejection_reasons[ep] = {'unknown'}
                else:
                    self.artifacts.rejection_reasons[ep] = {'unknown'}


    def run_algorithms(self, cfg_algorithms):
        """Run configured detection/rejection algorithms on this epochs object.

        Parameters
        ----------
        cfg_algorithms : dict
            Algorithm configuration dictionary.

        Returns
        -------
        None
        """
        run_algorithms(self, cfg_algorithms)

    def define_bcbt(self, keep_rejected_previous=None, plot_rejection_matrix=False):
        """Recompute ``BC`` and ``BT`` masks from current ``BCT``.

        Parameters
        ----------
        keep_rejected_previous : {'bt', 'bc'} | None, default=None
            Preserve previous bad-time or bad-channel flags.
        plot_rejection_matrix : bool, default=False
            If True, display the artifact matrix plot.

        Returns
        -------
        None
        """
        self.artifacts.define_bcbt(keep_rejected_previous=keep_rejected_previous, plot_rejection_matrix=plot_rejection_matrix)   

    def detect_bad_channels(self, cfg=None):
        """Detect bad channels using the configured or default pipeline.

        Parameters
        ----------
        cfg : None | str | pathlib.Path | dict, default=None
            Custom configuration source. ``None`` loads package defaults.

        Returns
        -------
        None
        """
        cfg_bad_channels_detection = get_cfg(cfg, 'detect_bad_channels_config.json')
        self.run_algorithms(cfg_bad_channels_detection)

    def detect_glitches(self, cfg=None):
        """Detect glitches using the configured or default pipeline.

        Parameters
        ----------
        cfg : None | str | pathlib.Path | dict, default=None
            Custom configuration source. ``None`` loads package defaults.

        Returns
        -------
        None
        """
        cfg_glitches_detection = get_cfg(cfg, 'detect_artifacts_glitches_config.json')
        self.run_algorithms(cfg_glitches_detection)
        
    def detect_artifacts(self, cfg=None):
        """Detect artifacts using the configured or default pipeline.

        Parameters
        ----------
        cfg : None | str | pathlib.Path | dict, default=None
            Custom configuration source. ``None`` loads package defaults.

        Returns
        -------
        None
        """
        cfg_artifacts_detection = get_cfg(cfg, 'detect_artifacts_all_config.json')
        self.run_algorithms(cfg_artifacts_detection)
        
    def correct_target_pca(self, cfg=None):
        """Apply target PCA artifact correction.

        Parameters
        ----------
        cfg : None | str | pathlib.Path | dict, default=None
            Correction configuration. ``None`` loads package defaults.

        Returns
        -------
        None
        """
        cfg_target_pca = get_cfg(cfg, 'correction_target_pca_config.json')
        targetPCA = TargetPCA(**cfg_target_pca)
        targetPCA.correct(self)
        self.define_bcbt()

    def correct_spline_segments(self, cfg=None):
        """Apply segment-wise spherical spline interpolation correction.

        Parameters
        ----------
        cfg : None | str | pathlib.Path | dict, default=None
            Correction configuration. ``None`` loads package defaults.

        Returns
        -------
        None
        """
        cfg_spline_segments = get_cfg(cfg, 'correction_spline_segments_config.json')
        spline_segm = SegmentSphericalSplineInterpolation(**cfg_spline_segments)
        spline_segm.correct(self)
        self.define_bcbt()

    def correct_spline_channels(self, cfg=None):
        """Apply channel-wise spherical spline interpolation correction.

        Parameters
        ----------
        cfg : None | str | pathlib.Path | dict, default=None
            Correction configuration. ``None`` loads package defaults.

        Returns
        -------
        None
        """
        cfg_spline_channels = get_cfg(cfg, 'correction_spline_channels_config.json')
        spline_chan = ChannelsSphericalSplineInterpolation(**cfg_spline_channels)
        spline_chan.correct(self)
        self.define_bcbt()

    def normalize(self, by_epochs='single', by_channels='all', where=None,
                  mean_value=0, std_value=None, rescale=None):
        """Normalize epoch data in place.

        Parameters
        ----------
        by_epochs : {'single', 'all'}, default='single'
            If ``'single'``, statistics are computed independently per epoch.
            If ``'all'``, data from all epochs are pooled.
        by_channels : {'single', 'all'}, default='all'
            If ``'single'``, statistics are computed independently per channel.
            If ``'all'``, data from all channels are pooled.
        where : list of float | None, default=None
            ``[tmin, tmax]`` window in seconds used to compute statistics.
            If ``None``, the full epoch duration is used.
        mean_value : float | numpy.ndarray | None, default=0
            Value to subtract.  If ``None``, the mean is computed from the
            data.  Default ``0`` leaves the mean unchanged.
        std_value : float | numpy.ndarray | None, default=None
            Divisor.  If ``None``, the standard deviation is computed from
            the data.
        rescale : bool | None, default=None
            If ``True``, rescale after normalization so the global standard
            deviation matches the original data.

        Returns
        -------
        None
        """
        normalize_epochs(self, by_epochs=by_epochs, by_channels=by_channels,
                         where=where, mean_value=mean_value,
                         std_value=std_value, rescale=rescale)

    def define_bad_epochs(self, bad_data = 1, bad_time = 0, bad_channel = 0.3, lim_dist=2, lim_gfp=2):
        """Run all bad-epoch criteria and update ``BE``.

        Parameters
        ----------
        bad_data : float, default=1
            Threshold on bad channel-time proportion per epoch.
        bad_time : float, default=0
            Threshold on bad-time proportion per epoch.
        bad_channel : float, default=0.3
            Threshold on bad-channel proportion per epoch.
        lim_dist : float | None, default=2
            Distance-to-average-ERP threshold.
        lim_gfp : float | None, default=2
            Global field power threshold.

        Returns
        -------
        None
            Updates ``self.artifacts.BE`` in place.
        """
        self.define_bad_epochs_artifacts(bad_data=bad_data, bad_time=bad_time, bad_channel=bad_channel, keeppre=False)
        if lim_dist:
            self.define_bad_epochs_dist(lim_dist=lim_dist, keeppre=True)
        if lim_gfp:
            self.define_bad_epochs_gfp(lim_gfp=lim_gfp, keeppre=True)

    def _update_rejection_reasons(self, be_vector, reason, keeppre):
        """Update per-epoch rejection reasons.

        Parameters
        ----------
        be_vector : numpy.ndarray
            Boolean vector of length ``n_epochs``; True = epoch rejected by
            this criterion.
        reason : str
            Label identifying the criterion (e.g. ``'artifacts'``,
            ``'distance'``, ``'gfp'``).
        keeppre : bool
            When False, reset all existing reasons before recording the new
            ones (consistent with the ``set_be`` overwrite semantics).
        """
        n_epochs = len(self.artifacts.rejection_reasons)
        be_vector = np.reshape(be_vector, n_epochs)
        if not keeppre:
            self.artifacts.rejection_reasons = [set() for _ in range(n_epochs)]
        for ep, flagged in enumerate(be_vector):
            if flagged:
                self.artifacts.rejection_reasons[ep].add(reason)

    def define_bad_epochs_artifacts(self, bad_data = 1, bad_time = 0, bad_channel = 0.3,
                        tmin=[], tmax=[], keeppre=True):
        """Flag bad epochs using artifact-mask proportions.

        Parameters
        ----------
        bad_data : float, default=1
            Maximum allowed ``BCT`` proportion per epoch.
        bad_time : float, default=0
            Maximum allowed ``BT`` proportion per epoch.
        bad_channel : float, default=0.3
            Maximum allowed ``BC`` proportion per epoch.
        tmin : float | list, default=[]
            Start time (seconds) of evaluation window.
        tmax : float | list, default=[]
            End time (seconds) of evaluation window.
        keeppre : bool, default=True
            If True, keep previously flagged bad epochs.

        Returns
        -------
        bad_epochs : numpy.ndarray
            Boolean vector indicating epochs flagged by this criterion.
        """

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

        self._update_rejection_reasons(bad_epochs, 'artifacts', keeppre)
        return bad_epochs

    def define_bad_epochs_dist(self, 
                            lim_dist = 2, lim_bad_time_dist = None, lim_mean_dist = None, lim_max_dist=None, 
                            relative=True, maxloops=1, where=[], rmvmean=False, normdist=True,
                            l_freq_filter=None, h_freq_filter=None, keeppre=True):
        """Flag bad epochs using distance to the average ERP.

        Parameters
        ----------
        lim_dist : float, default=2
            Main distance threshold.
        lim_bad_time_dist : float | None, default=None
            Threshold on proportion of samples above distance threshold.
        lim_mean_dist : float | None, default=None
            Threshold on mean distance per epoch.
        lim_max_dist : float | None, default=None
            Threshold on maximum distance per epoch.
        relative : bool, default=True
            If True, derive thresholds from data percentiles.
        maxloops : int, default=1
            Maximum iterative rejection loops.
        where : list, default=[]
            Time window ``[tmin, tmax]`` in seconds.
        rmvmean : bool, default=False
            If True, remove per-channel temporal mean before distance.
        normdist : bool, default=True
            If True, z-normalize distances across samples.
        l_freq_filter : float | None, default=None
            Optional low cutoff for pre-filtering.
        h_freq_filter : float | None, default=None
            Optional high cutoff for pre-filtering.
        keeppre : bool, default=True
            If True, keep previously rejected epochs.

        Returns
        -------
        be_dist : numpy.ndarray
            Boolean vector indicating epochs rejected by distance criteria.
        """
        
        print('\nIdentifying bad epochs based on the distance to the average ERP...')

        if not lim_bad_time_dist:
            lim_bad_time_dist = 0.10/(self.times[-1]-self.times[0])  # 100 ms
        
        if not where:
            where = [self.times[0], self.times[-1]]
            
        n_electrodes, n_samples, n_epochs = self.get_data_size()
        be_dist = np.full(n_epochs, False)
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

        self._update_rejection_reasons(be_dist, 'distance', keeppre)
        return be_dist


    def define_bad_epochs_gfp(self, 
                            lim_gfp = 2, lim_bad_time_gfp = None, lim_mean_gfp = None, lim_max_gfp=None, 
                            relative=True, maxloops=1, where=[],
                            l_freq_filter=None, h_freq_filter=None, keeppre=True):
        """Flag bad epochs using global field power (GFP) criteria.

        Parameters
        ----------
        lim_gfp : float, default=2
            Main GFP threshold.
        lim_bad_time_gfp : float | None, default=None
            Threshold on proportion of samples above GFP threshold.
        lim_mean_gfp : float | None, default=None
            Threshold on mean GFP per epoch.
        lim_max_gfp : float | None, default=None
            Threshold on maximum GFP per epoch.
        relative : bool, default=True
            If True, derive thresholds from data percentiles.
        maxloops : int, default=1
            Reserved for compatibility with distance-based interface.
        where : list, default=[]
            Time window ``[tmin, tmax]`` in seconds.
        l_freq_filter : float | None, default=None
            Optional low cutoff for pre-filtering.
        h_freq_filter : float | None, default=None
            Optional high cutoff for pre-filtering.
        keeppre : bool, default=True
            If True, keep previously rejected epochs.

        Returns
        -------
        be_gfp : numpy.ndarray
            Boolean vector indicating epochs rejected by GFP criteria.
        """
        
        print('\nIdentifying bad epochs based on the GFP...')

        if not lim_bad_time_gfp:
            lim_bad_time_gfp = 0.10/(self.times[-1]-self.times[0])  # 100 ms
        
        if not where:
            where = [self.times[0], self.times[-1]]
            
        n_electrodes, n_samples, n_epochs = self.get_data_size()
        be_gfp = np.full(n_epochs, False)
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

        self._update_rejection_reasons(be_gfp, 'gfp', keeppre)
        return be_gfp


    def define_bad_epochs_manual(self, bad_epochs, rejection_reason='manual', keeppre=True):
        """Flag bad epochs using an arbitrary boolean vector.

        Parameters
        ----------
        bad_epochs : array-like of bool
            Boolean vector of length ``n_epochs``.  ``True`` marks an epoch
            as bad.
        rejection_reason : str, default='manual'
            Label to record in ``rejection_reasons`` for each flagged epoch.
            If an epoch was already flagged by another criterion, this label
            is added to its existing reasons.
        keeppre : bool, default=True
            If True, keep previously rejected epochs.  If False, reset ``BE``
            to only reflect this vector.

        Returns
        -------
        bad_epochs : numpy.ndarray
            Boolean vector (same as input, cast to ndarray).
        """
        bad_epochs = np.asarray(bad_epochs, dtype=bool)
        n_epochs = len(self.artifacts.BE)
        if bad_epochs.shape[0] != n_epochs:
            raise ValueError(
                f"bad_epochs length ({bad_epochs.shape[0]}) does not match "
                f"number of epochs ({n_epochs})."
            )

        if keeppre:
            self.artifacts.update_be(bad_epochs)
        else:
            self.artifacts.set_be(bad_epochs)

        self._update_rejection_reasons(bad_epochs, rejection_reason, keeppre)

        n_flagged = int(np.sum(bad_epochs))
        print(f'\nManual epoch rejection ({rejection_reason!r})...')
        print(f'--> Rejected epochs by this criterion: {n_flagged} out of {n_epochs} ({n_flagged / n_epochs:.1%})')
        print(f'--> Total rejected epochs:             {int(np.sum(self.artifacts.BE))} out of {n_epochs} ({np.sum(self.artifacts.BE) / n_epochs:.1%})')

        return bad_epochs


    def remove_bad_epochs(self, reasons=None):
        """Drop bad epochs and synchronize artifact matrices.

        Parameters
        ----------
        reasons : str | list of str | None, default=None
            If ``None``, remove all epochs flagged as bad (``BE == True``).
            Otherwise, remove only epochs whose rejection reasons include at
            least one of the specified labels (e.g. ``'artifacts'``,
            ``'distance'``, ``'gfp'``).  Epochs not matching the filter are
            left untouched even if ``BE`` is True.

        Returns
        -------
        None
            Removes rows corresponding to rejected epochs from data and masks.
        """

        if reasons is None:
            # Remove all bad epochs
            to_drop = self.artifacts.BE.copy()
        else:
            if isinstance(reasons, str):
                reasons = [reasons]
            reasons = set(reasons)
            to_drop = np.array([
                bool(r & reasons)
                for r in self.artifacts.rejection_reasons
            ])

        # Drop the selected epochs from the epochs data
        self.drop(to_drop, reason='bad epoch')

        # Update the artifacts matrices to reflect the removal of the dropped epochs
        good_epochs = ~to_drop
        self.artifacts.n_epochs = np.sum(good_epochs)
        self.artifacts.BE = self.artifacts.BE[good_epochs]
        self.artifacts.BCT = self.artifacts.BCT[good_epochs, :, :]
        self.artifacts.BT = self.artifacts.BT[good_epochs, :, :]
        self.artifacts.BC = self.artifacts.BC[good_epochs, :, :]
        self.artifacts.CCT = self.artifacts.CCT[good_epochs, :, :]
        self.artifacts.rejection_reasons = [
            r for ep, r in enumerate(self.artifacts.rejection_reasons)
            if good_epochs[ep]
        ]
        
        

    def export(self, full_path, overwrite=False):
        """Export epochs to FIF and artifact annotations to CSV.

        Parameters
        ----------
        full_path : str | pathlib.Path
            Full path to the output file, including filename.

        Returns
        -------
        None
        """
        # the extension must be .fif or no extension (in which case .fif will be added)
        if not str(full_path).endswith('.fif'):
            full_path = str(full_path) + '.fif'
        
        # if the extension is not .fif, raise an error        
        if not str(full_path).endswith('.fif'):
            raise ValueError('The output file must have a .fif extension or no extension.')

        # get the artifacts in a dataframe
        artifacts_df = self.rejection_matrix_to_data_frame()
        delattr(self, 'artifacts')
                
        # Save the epochs and the artifacts information in a csv file in the output directory
        print('\nExporting epochs...')
        full_path = Path(full_path)
        parent_dir = full_path.parent
        file_name = full_path.stem
        parent_dir.mkdir(parents=True, exist_ok=True)
        
        epochs_fullpath = parent_dir / (file_name + '.fif')
        self.save(epochs_fullpath, overwrite=overwrite)
        print(f"Epochs saved at {epochs_fullpath}.")
        
        # check if the artifacts .cs file already exists and if overwrite is False, raise an error
        art_fullpath = parent_dir / (file_name + '-artifacts.csv')
        if art_fullpath.exists() and not overwrite:
            raise FileExistsError(f"The artifact file {art_fullpath} already exists. Set overwrite=True to overwrite it.")
        artifacts_df.to_csv(art_fullpath, index=False)
        print(f"\nEpochs artifacts information saved at {art_fullpath}.")
        
    def deal_with_reference_channels(self, reference_channels):
        """Ensure reference channels are handled consistently in epoch masks.

        Parameters
        ----------
        reference_channels : list of str | None
            Channel names that should not be marked as globally bad channels.

        Returns
        -------
        None
        """
        if reference_channels is not None:
            idx_reference_channels = [self.ch_names.index(ch) for ch in reference_channels if ch in self.ch_names]
            self.artifacts.BC[:, idx_reference_channels, 0] = False  # Ensure reference channels are not marked as bad channels in BC
            self.artifacts.BCT[:, idx_reference_channels, :] = False  # Ensure reference channels are not marked as bad channels in BCT
            for i in range(self._data.shape[0]):
                self.artifacts.BCT[i, idx_reference_channels, self.artifacts.BT[i,0,:]] = True
       

    def plot_percentage_of_bad_data_across_sensors(self):
        """Plot topographic percentage of bad data per channel across epochs.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Generated topomap figure.
        """

        from matplotlib import pyplot as plt

        # Get the percentage of bad data per electrodes
        data = []
        for i, ch in enumerate(self.ch_names):
            idx_t = self.artifacts.BT[:, 0, :]==False
            bct_i = self.artifacts.BCT[:, i, :]
            n_bads = np.sum(bct_i[idx_t])
            n_per = (n_bads / np.sum(idx_t)) * 100
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
    
    def plot_artifact_structure(self, artifact='all',time_step=50, color_scheme='turbo'):
        """Plot epoch artifact masks.

        Parameters
        ----------
        artifact : {'all', 'BCT', 'BT', 'BC', 'BE'}, default='all'
            Artifact layer to display.
        time_step : int, default=50
            Tick spacing for x-axis labels.
        color_scheme : str, default='turbo'
            Matplotlib colormap.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Artifact heatmap figure.
        """
        return self.artifacts.plot_artifact_structure(artifact=artifact, time_step=time_step, color_scheme=color_scheme)

    def plot_bad_channels_bar(self):
        """Bar plot of bad-data percentage per channel, excluding bad-time samples.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Bar chart figure.
        """
        return self.artifacts.plot_bad_channels_bar()

    def plot_bad_times_line(self):
        """Line plot of bad-channel percentage per sample, concatenated across epochs.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Line plot figure.
        """
        return self.artifacts.plot_bad_times_line()

    def plot_rejection_summary(self):
        """Visualize epoch rejection with per-epoch rejection reasons.

        The figure has two panels:

        - **Top**: bar chart with one bar per epoch, coloured red (rejected)
          or green (good).  The title shows the total count.
        - **Bottom**: heatmap matrix where rows are rejection criteria and
          columns are epochs.  A coloured cell means that criterion flagged
          the epoch.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Rejection summary figure.
        """
        from matplotlib import pyplot as plt
        from matplotlib.patches import Patch

        reasons_list = self.artifacts.rejection_reasons
        n_epochs = len(reasons_list)

        all_reasons = sorted(set().union(*reasons_list)) if any(reasons_list) else []
        is_bad = np.array([bool(r) for r in reasons_list])
        n_bad = int(np.sum(is_bad))

        n_rows = 2 if all_reasons else 1
        fig_width = max(10, n_epochs * 0.12)
        fig, axes = plt.subplots(
            n_rows, 1,
            figsize=(fig_width, 3 * n_rows),
            squeeze=False,
        )

        # --- top panel: good / bad bar per epoch ---
        ax0 = axes[0, 0]
        colors = ['#d62728' if b else '#2ca02c' for b in is_bad]
        ax0.bar(np.arange(n_epochs), is_bad.astype(int), color=colors, width=1.0, linewidth=0)
        ax0.set_xlim(-0.5, n_epochs - 0.5)
        ax0.set_ylim(0, 1.3)
        ax0.set_yticks([0, 1])
        ax0.set_yticklabels(['good', 'bad'])
        ax0.set_xlabel('Epoch index')
        ax0.set_title(
            f'Epoch rejection summary  —  {n_bad} / {n_epochs} rejected '
            f'({n_bad / n_epochs:.1%})'
        )
        legend_elements = [
            Patch(facecolor='#d62728', label=f'Rejected ({n_bad})'),
            Patch(facecolor='#2ca02c', label=f'Good ({n_epochs - n_bad})'),
        ]
        ax0.legend(handles=legend_elements, loc='upper right', fontsize=8)

        # --- bottom panel: reason matrix ---
        if all_reasons:
            ax1 = axes[1, 0]
            reason_colors = plt.cm.tab10(np.linspace(0, 0.9, len(all_reasons)))
            matrix = np.zeros((len(all_reasons), n_epochs))
            for ep, ep_reasons in enumerate(reasons_list):
                for row_idx, r in enumerate(all_reasons):
                    if r in ep_reasons:
                        matrix[row_idx, ep] = row_idx + 1

            # grey background for all epochs, colored cells for flagged ones
            ax1.imshow(
                is_bad[np.newaxis, :].repeat(len(all_reasons), axis=0),
                aspect='auto', cmap='Greys', vmin=0, vmax=4,
                interpolation='none',
            )
            for row_idx, r in enumerate(all_reasons):
                flagged = matrix[row_idx, :] > 0
                if flagged.any():
                    ax1.scatter(
                        np.where(flagged)[0],
                        np.full(flagged.sum(), row_idx),
                        marker='s',
                        s=max(4, 400 / n_epochs),
                        color=reason_colors[row_idx],
                        zorder=3,
                        label=r,
                    )

            ax1.set_xlim(-0.5, n_epochs - 0.5)
            ax1.set_ylim(-0.5, len(all_reasons) - 0.5)
            ax1.set_yticks(range(len(all_reasons)))
            ax1.set_yticklabels(all_reasons)
            ax1.set_xlabel('Epoch index')
            ax1.set_ylabel('Rejection reason')
            ax1.legend(loc='upper right', fontsize=8, title='Reason')

        plt.tight_layout()
        return fig
