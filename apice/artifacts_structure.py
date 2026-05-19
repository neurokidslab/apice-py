"""Artifact mask structures and visualization helpers.

This module defines utility functions and container classes used to represent
bad-channel, bad-time, bad-epoch, and corrected-data masks for raw and epoched
EEG data.
"""

# %% LIBRARIES

# Import necessary modules
import copy
import mne  
import numpy as np 
from prettytable import PrettyTable 

# Import specific modules from your project's modules
from apice.utils import (get_data_size, include_short_bad_segments, reject_short_good_segments, mask_bad_segments, get_cfg)


# %% FUNCTIONS

def define_bcbt(bct, thresh_bad_times, thresh_bad_channels, bc=None, bt=None, bc_manual=None):
    """Infer bad-channel and bad-time masks from bad-channel-time data.

    Parameters
    ----------
    bct : numpy.ndarray
        Bad-channel-time mask with shape ``(n_channels, n_samples)`` for raw
        data or ``(n_epochs, n_channels, n_samples)`` for epochs.
    thresh_bad_times : list of float
        Per-cycle threshold for rejecting channels based on bad-time
        proportion.
    thresh_bad_channels : list of float
        Per-cycle threshold for rejecting time samples based on bad-channel
        proportion.
    bc : numpy.ndarray | None, default=None
        Initial bad-channel mask.
    bt : numpy.ndarray | None, default=None
        Initial bad-time mask.
    bc_manual : numpy.ndarray | None, default=None
        Manual bad-channel flags to force in the final mask.

    Returns
    -------
    bc : numpy.ndarray
        Updated bad-channel mask.
    bt : numpy.ndarray
        Updated bad-time mask.
    bcbt : numpy.ndarray
        Combined bad data mask generated from ``bc`` and ``bt``.
    """
    
    if len(bct.shape) == 2:
        n_epochs = 1
        n_channels = np.shape(bct)[0]
        n_samples = np.shape(bct)[1]
    if len(bct.shape) == 3:
        n_epochs = np.shape(bct)[0]
        n_channels = np.shape(bct)[1]
        n_samples = np.shape(bct)[2]
       
    if len(bct.shape) == 2:
        bct = np.transpose(np.expand_dims(bct, axis=2), (2, 0, 1))
        dimension_added_bct = True
    else: 
        dimension_added_bct = False
 
    thresh_bad_channels_all = thresh_bad_channels.copy()
    
    if len(np.unique(np.asarray([len(thresh_bad_times), len(thresh_bad_channels), len(thresh_bad_channels_all)])))>1:
        raise Exception("thresh_bad_times and thresh_bad_channels must have the same size")
    n_cycle = len(thresh_bad_times)

    if bc is None:
        bc = np.full((n_epochs, n_channels, 1), False)
    if bt is None:
        bt = np.full((n_epochs, 1, n_samples), False)
    if bc_manual is None:
        bc_manual = np.full((n_channels), False)
    bcbt = np.full((n_epochs, n_channels, n_samples), False)
    bcbt[np.tile(bc, [1, 1, n_samples])] = True
    bcbt[np.tile(bt, [1, n_channels, 1])] = True

    for i in np.arange(n_cycle):
        for ep in np.arange(n_epochs):
            
            bct_ep = bct[ep, :, :].copy()
            
            # Number of bad channels per sample
            thresh_bad_channels_i = thresh_bad_times[i]
            bct_ep_ = bct_ep.copy()
            bc_ = bc[ep, :, :].copy()
            bct_ep_[np.tile(bc_, [1, n_samples])] = False
            n_bad_channels = np.sum(bct_ep_*1, axis=0)
            if np.sum(~bc_*1) == 0:
                p_bad_channels = np.zeros_like(n_bad_channels)
            else:
                p_bad_channels = n_bad_channels / np.tile(np.sum(~bc_*1), [1, n_samples])

            # Number of bad samples per channel
            thresh_bad_times_i = thresh_bad_channels[i]
            bct_ep_ = bct_ep.copy()
            bt_ = bt[ep, :, :].copy()
            bct_ep_[np.tile(bt_, (n_channels, 1))] = False
            n_bad_samples = np.sum(bct_ep_*1, axis=1)
            if np.sum(~bt_*1) == 0:
                p_bad_samples = np.zeros_like(n_bad_samples)
            else:
                p_bad_samples = np.divide(n_bad_samples, np.tile(np.sum(~bt_*1), [n_channels]))

            # Reject bad data
            bt[ep, :, :] = bt[ep, :, :].copy() | (p_bad_channels > thresh_bad_channels_i)
            bc[ep, :, :] = bc[ep, :, :].copy() | bc_manual[:, np.newaxis]
            bc[ep, :, :] = bc[ep, :, :].copy() | np.reshape((p_bad_samples > thresh_bad_times_i), [n_channels, 1])

        # Test if the definition changes
        bcbt_old = bcbt.copy()
        bcbt[np.tile(bc, [1, 1, n_samples])] = True
        bcbt[np.tile(bt, [1, n_channels, 1])] = True
        change_in_def = np.not_equal(bcbt_old, bcbt)
        print('Cycle ', str(i),
              ': new rejected data ', np.round(np.sum(change_in_def) / np.size(change_in_def) * 100, 2), '%')

    if dimension_added_bct:
        bt = np.squeeze(np.transpose(bt, (1, 2, 0)), axis=2)
        bc = np.squeeze(np.transpose(bc, (1, 2, 0)), axis=2)

    return bc, bt, bcbt



def plot_artifact_structure(times, ch_names, bct, bc=None, bt=None, be=None, artifact='all', time_step=50, color_scheme='turbo', figsize=(12, 6), thresh_bad_channels=None, thresh_bad_times=None):
    """Plot artifact masks over channels and time/epochs.

    Parameters
    ----------
    times : numpy.ndarray
        Time vector in seconds.
    ch_names : list of str
        Channel names.
    bct : numpy.ndarray
        Bad-channel-time mask.
    bc : numpy.ndarray | None, default=None
        Bad-channel mask.
    bt : numpy.ndarray | None, default=None
        Bad-time mask.
    be : numpy.ndarray | None, default=None
        Bad-epoch mask.
    artifact : {'all', 'BCT', 'BT', 'BC', 'BE'}, default='all'
        Artifact layer to visualize.
    time_step : int, default=50
        Tick spacing for time axis in seconds.
    color_scheme : str, default='turbo'
        Matplotlib colormap name.
    figsize : tuple, default=(12, 6)
        Figure size in inches.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the artifact heatmap.
    """

    # Import necessary modules
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    # Functions
    def prepare_cmap(ax, data, artifact, color_scheme='turbo', cax=None):
        if artifact == 'all':
            # Define tick labels and colormap for all artifact types
            # tick_labels = ['good', 'bad', 'BT', 'BC', 'BE']
            tick_labels = ['Good Data', 'Bad Data', 'Bad Time Point', 'Bad Channel', 'Bad Epoch']
            cmap = plt.get_cmap(color_scheme, len(tick_labels))
            mat = ax.imshow(data, cmap=cmap, vmin=-0.5, vmax=4.5, aspect='auto')
            cb = plt.colorbar(mat, cax=cax, ticks=np.arange(5))
            cb.set_ticklabels(tick_labels)
        else:
            # Define tick labels and colormap for all artifact types
            cmap = plt.get_cmap(color_scheme, len(np.unique(data)))
            mat = ax.imshow(data, cmap=cmap, vmin=np.min(data) - 0.5, vmax=np.max(data) + 0.5, aspect='auto')
            cb = plt.colorbar(mat, cax=cax, ticks=np.asarray(np.unique(data), dtype=int))
            colorbar_ticks = np.unique(data)
            # labels = ['good', 'bad', 'BT', 'BC', 'BE']
            labels = ['Good Data', 'Bad Data', 'Bad Time Point', 'Bad Channel', 'Bad Epoch']
            tick_labels = []
            for i in colorbar_ticks:
                tick_labels.append(labels[i])
            cb.set_ticklabels(tick_labels)

    def set_ticks(ax, data, t, time_step, sfreq, n_epochs, n_channels, n_samples, ch_names):        
        # Set x-axis ticks and labels
        ax.tick_params(axis="x", bottom=True, top=False, labeltop=False, labelbottom=True)
        if n_epochs > 1:
            ax.set_xticks(np.arange(0, n_epochs * n_samples, n_samples * 5))
            ax.set_xticklabels(np.arange(0, n_epochs, 5))
            ax.set_xlabel('Epoch #')
        else:
            ax.set_xticks(np.arange(0, np.shape(data)[1], time_step * sfreq))
            xticks = np.asarray(ax.get_xticks(), dtype=int)
            ax.set_xticklabels(t[xticks])   
            ax.set_xlabel('Time (s)')    
        
        yticks = np.arange(n_channels)
        ax.set_yticks(yticks)
        ax.set_yticklabels([ch_names[i] for i in yticks], fontsize=5)  # Use channel names for labels
        
        # Set subplot title, x-axis label, and y-axis label
        # ax.set_title(artifact)
        ax.set_ylabel('Channel #')

    # get some information about the data
    sfreq = 1/(times[1]-times[0])
    n_channels = len(ch_names)
    n_samples = len(times)
    if np.size(np.shape(bct)) == 2:
        n_epochs = 1
    if np.size(np.shape(bct)) == 3:
        n_epochs = np.shape(bct)[0]

    # Initialize a matrix to store artifact occurrence information
    M = np.zeros(np.shape(bct))

    # Populate the matrix 'M' based on the specified artifacts.
    # Accept masks that may be stored as 1D/2D/3D and normalize them.
    if artifact in ['BCT', 'all']:
        M[bct == 1] = 1

    if bt is not None and artifact in ['BT', 'all']:
        bt_ = np.asarray(bt)
        if bt_.ndim == 1:  # (n_samples,)
            bt_ = bt_[np.newaxis, np.newaxis, :]
        elif bt_.ndim == 2:  # (n_epochs, n_samples) or (1, n_samples)
            bt_ = bt_[:, np.newaxis, :]
        if bt_.shape[0] == 1 and n_epochs > 1:
            bt_ = np.repeat(bt_, n_epochs, axis=0)
        bt_mask = np.tile(bt_ == 1, [1, n_channels, 1])
        if bt_mask.shape == M.shape:
            M[bt_mask] = 2

    if bc is not None and artifact in ['BC', 'all']:
        bc_ = np.asarray(bc)
        if bc_.ndim == 1:  # (n_channels,)
            bc_ = bc_[np.newaxis, :, np.newaxis]
        elif bc_.ndim == 2:  # (n_epochs, n_channels) or (n_channels, 1)
            if bc_.shape[0] == n_channels and bc_.shape[1] == 1:
                bc_ = bc_[np.newaxis, :, :]
            else:
                bc_ = bc_[:, :, np.newaxis]
        if bc_.shape[0] == 1 and n_epochs > 1:
            bc_ = np.repeat(bc_, n_epochs, axis=0)
        bc_mask = np.tile(bc_ == 1, [1, 1, n_samples])
        if bc_mask.shape == M.shape:
            M[bc_mask] = 3

    if be is not None and artifact in ['BE', 'all']:
        be_ = np.asarray(be)
        if be_.ndim == 1:  # (n_epochs,)
            be_ = be_[:, np.newaxis, np.newaxis]
        elif be_.ndim == 2:  # (n_epochs, 1)
            be_ = be_[:, :, np.newaxis]
        if be_.shape[0] == 1 and n_epochs > 1:
            be_ = np.repeat(be_, n_epochs, axis=0)
        be_mask = np.tile(be_ == 1, [1, n_channels, n_samples])
        if be_mask.shape == M.shape:
            M[be_mask] = 4

    # Convert the artifact matrix to an integer data type for consistent plotting
    M = np.asarray(M, dtype=int)

    # --- Compute per-channel bad-data percentages (bar plot, left panel) ---
    # Normalise bct to 3D (n_epochs, n_channels, n_samples) for indexing
    bct_3d = np.asarray(bct)
    if bct_3d.ndim == 2:
        bct_3d = bct_3d[np.newaxis, :, :]

    if bt is not None:
        bt_p = np.asarray(bt)
        if bt_p.ndim == 1:
            bt_p = bt_p[np.newaxis, np.newaxis, :]
        elif bt_p.ndim == 2:
            bt_p = bt_p[:, np.newaxis, :]
        if bt_p.shape[0] == 1 and n_epochs > 1:
            bt_p = np.repeat(bt_p, n_epochs, axis=0)
        good_time_mask = ~bt_p[:, 0, :].astype(bool)  # (n_epochs, n_samples)
    else:
        good_time_mask = np.ones((n_epochs, n_samples), dtype=bool)

    n_good_times = int(good_time_mask.sum())
    pct_channels = np.array([
        float(bct_3d[:, i, :][good_time_mask].sum()) / n_good_times * 100
        if n_good_times > 0 else 0.0
        for i in range(n_channels)
    ])

    # --- Compute per-sample bad-channel percentages (line plot, bottom panel) ---
    # Normalise bc to (n_epochs, n_channels, 1) for indexing
    if bc is not None:
        bc_p = np.asarray(bc)
        if bc_p.ndim == 1:
            bc_p = bc_p[np.newaxis, :, np.newaxis]
        elif bc_p.ndim == 2:
            if bc_p.shape[0] == n_channels and bc_p.shape[1] == 1:
                bc_p = bc_p[np.newaxis, :, :]
            else:
                bc_p = bc_p[:, :, np.newaxis]
        if bc_p.shape[0] == 1 and n_epochs > 1:
            bc_p = np.repeat(bc_p, n_epochs, axis=0)
        good_ch_mask = ~bc_p[:, :, 0].astype(bool)  # (n_epochs, n_channels)
    else:
        good_ch_mask = np.ones((n_epochs, n_channels), dtype=bool)

    pct_times_list = []
    for ep in range(n_epochs):
        good_ch = good_ch_mask[ep, :]
        n_good_ch = int(good_ch.sum())
        pct_ep = (
            bct_3d[ep, good_ch, :].sum(axis=0).astype(float) / n_good_ch * 100
            if n_good_ch > 0 else np.zeros(n_samples)
        )
        pct_times_list.append(pct_ep)
    pct_times = np.concatenate(pct_times_list)  # (n_epochs * n_samples,)

    # --- Build figure with 3-panel layout ---
    # Columns: bar | heatmap | colorbar   Rows: main | line
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 3,
                  figure=fig,
                  width_ratios=[1, 5, 0.15],
                  height_ratios=[5, 1],
                  hspace=0.04,
                  wspace=0.04)

    ax_heatmap = fig.add_subplot(gs[0, 1])
    ax_bar     = fig.add_subplot(gs[0, 0])
    ax_cbar    = fig.add_subplot(gs[0, 2])   # dedicated colorbar axis
    ax_line    = fig.add_subplot(gs[1, 1])
    ax_corner  = fig.add_subplot(gs[1, 0])
    ax_corner.axis('off')
    fig.add_subplot(gs[1, 2]).axis('off')    # empty corner next to colorbar

    # Build concatenated heatmap data
    if n_epochs == 1:
        data = M[0, :, :]
    else:
        data = M[0, :, :]
        for ep in np.arange(1, n_epochs):
            data = np.concatenate((data, M[ep, :, :]), axis=1)

    prepare_cmap(ax_heatmap, data, artifact, color_scheme=color_scheme, cax=ax_cbar)
    set_ticks(ax_heatmap, data, times, time_step, sfreq, n_epochs, n_channels, n_samples, ch_names)
    # Hide heatmap x-tick labels; they are reproduced on the line plot below
    ax_heatmap.tick_params(axis='x', labelbottom=False)
    ax_heatmap.set_xlabel('')

    # Derive bar/line color from the same 5-level colormap used by the heatmap.
    # index 1 → Bad Data (violet in turbo).
    _cmap5 = plt.get_cmap(color_scheme, 5)
    _norm5 = plt.Normalize(vmin=-0.5, vmax=4.5)
    color_bad = _cmap5(_norm5(1))   # Bad Data
    color_bt  = _cmap5(_norm5(2))   # Bad Time Point (threshold line, bottom panel)
    color_bc  = _cmap5(_norm5(3))   # Bad Channel (threshold line, left panel)

    # --- Bar plot (left panel) — channels on y, % bad data on x, inverted x ---
    ax_bar.barh(np.arange(n_channels), pct_channels, color=color_bad, height=0.8)
    if thresh_bad_channels is not None:
        t_pct = thresh_bad_channels * 100
        ax_bar.axvline(t_pct, color=color_bc, linestyle='--', linewidth=1.0,
                       label=f'{t_pct:.0f}%')
        ax_bar.legend(fontsize=6, loc='lower right')
    ax_bar.invert_xaxis()           # 0 % adjacent to heatmap
    ax_bar.set_xlabel('% bad data', fontsize=8)
    ax_bar.set_yticks([])
    ax_bar.set_ylim(ax_heatmap.get_ylim())  # align channel axis with heatmap

    # --- Line plot (bottom panel) — % bad channels over time / epochs ---
    x_line = np.arange(len(pct_times))
    ax_line.plot(x_line, pct_times, color=color_bad, linewidth=0.8)
    ax_line.fill_between(x_line, pct_times, alpha=0.3, color=color_bad)
    if thresh_bad_times is not None:
        t_pct = thresh_bad_times * 100
        ax_line.axhline(t_pct, color=color_bt, linestyle='--', linewidth=1.0,
                        label=f'{t_pct:.0f}%')
        ax_line.legend(fontsize=6, loc='upper right')
    if n_epochs > 1:
        for ep in range(1, n_epochs):
            ax_line.axvline(ep * n_samples - 0.5, color='gray', linestyle=':', linewidth=0.5)
    ax_line.set_ylabel('% bad ch', fontsize=8)
    ax_line.set_xlim(ax_heatmap.get_xlim())  # align time/epoch axis with heatmap
    # Reproduce x-axis ticks matching the heatmap
    if n_epochs > 1:
        ax_line.set_xticks(np.arange(0, n_epochs * n_samples, n_samples * 5))
        ax_line.set_xticklabels(np.arange(0, n_epochs, 5), fontsize=7)
        ax_line.set_xlabel('Epoch #', fontsize=8)
    else:
        xtick_pos = np.arange(0, n_samples, int(time_step * sfreq))
        xtick_pos = xtick_pos[xtick_pos < n_samples]
        ax_line.set_xticks(xtick_pos)
        ax_line.set_xticklabels(np.round(times[xtick_pos], 2), fontsize=7)
        ax_line.set_xlabel('Time (s)', fontsize=8)

    return fig
    


# %% CLASSE TO HOLD THE ARTIFACTS REJECTION MATRICES

class Artifacts:
    """Base container for artifact masks and related operations.

    Parameters
    ----------
    obj : mne.io.BaseRaw | mne.BaseEpochs
        Data object the masks are associated with.
    thresh_bad_channels : list of float, default=[0.7, 0.5, 0.3]
        Thresholds used to derive bad channels from bad-channel-time masks.
    thresh_bad_times : list of float, default=[0.7, 0.5, 0.3]
        Thresholds used to derive bad times from bad-channel-time masks.
    min_good_time : float, default=0
        Minimum good segment duration in seconds.
    min_bad_time : float, default=0
        Minimum bad segment duration in seconds.
    mask_time : float, default=0
        Buffer duration in seconds applied around bad segments.
    """

    def __init__(self, obj, 
                 thresh_bad_channels=[0.7, 0.5, 0.3], thresh_bad_times=[0.7, 0.5, 0.3], 
                 min_good_time=0, min_bad_time=0, mask_time=0):

        base_epochs_type = getattr(mne, "BaseEpochs", mne.epochs.BaseEpochs)
        if not isinstance(obj, (mne.io.BaseRaw, base_epochs_type)):
            raise ValueError("The object must be an instance of mne.io.Raw or mne.Epochs.")
        
        n_channels, n_samples, n_epochs = get_data_size(obj)

        if isinstance(obj, mne.io.BaseRaw):
            artifacts_types = {
                'BCT': np.full((n_channels, n_samples), False),  # Bad Channel Time
                'BC': np.full((n_channels, 1), False),  # Bad Channel
                'BCmanual': np.full((n_channels), False),  # Bad Channel (manual input)
                'BT': np.full((1, n_samples), False),  # Bad Time
                'CCT': np.full((n_channels, n_samples), False),  # Corrected Channel Time
            }
        elif isinstance(obj, base_epochs_type):
            artifacts_types = {
                'BCT': np.full((n_epochs, n_channels, n_samples), False),  # Bad Channel Time
                'BC': np.full((n_epochs, n_channels, 1), False),  # Bad Channel
                'BCmanual': np.full((n_channels), False),  # Bad Channel (manual input)
                'BT': np.full((n_epochs, 1, n_samples), False),  # Bad Time
                'CCT': np.full((n_epochs, n_channels, n_samples), False),  # Corrected Channel Time
                'BE': np.full((n_epochs, 1, 1), False),  # Bad Epoch
            }

        if not hasattr(obj, 'artifacts'):
            # If the EEG object does not contain an artifacts attribute, set it up.
            print("Initializing artifacts structure")
            if isinstance(obj, mne.io.BaseRaw):
                self.object_type = 'raw'
            elif isinstance(obj, base_epochs_type):
                self.object_type = 'epochs'
            self.params = dict(thresh_bad_channels=thresh_bad_channels, thresh_bad_times=thresh_bad_times,
                               min_good_time=min_good_time, min_bad_time=min_bad_time, mask_time=mask_time)
            self.n_epochs = n_epochs
            self.n_channels = n_channels
            self.n_samples = n_samples
            self.artifacts_types_names = list(artifacts_types.keys())
            self.ch_names = obj.ch_names
            self.sfreq = obj.info['sfreq']
            self.times = obj.times
            # Assign the initialized matrices to the attributes.
            for artifact_type in artifacts_types.keys():
                setattr(self, artifact_type, artifacts_types[artifact_type])
                
        else:
            print("Artifacts structure already exists. Nothing is initialized.")

    def print_summary(self):
        """Summarize mask occupancy percentages in a table.

        Returns
        -------
        summary : prettytable.PrettyTable
            Table with one column per artifact mask and percentage of True
            values.
        """
        
        # Initialize table
        summary = PrettyTable()
        
        # Calculate and add data to the summary table.
        for artifact_key in self.artifacts_types_names:
            if hasattr(self, artifact_key):
                total_elements = np.size(getattr(self, artifact_key))
                total_true_elements = np.sum(getattr(self, artifact_key))
                percentage = np.round(total_true_elements / total_elements * 100, 2)
                summary.add_column(artifact_key, [f"{percentage}%"])

        return summary
    
    def set_bcmanual(self, bc_manual):
        """Set manual bad-channel flags from channel names.

        Parameters
        ----------
        bc_manual : list of str
            Channel names to mark as manually bad.

        Returns
        -------
        None
        """
        self.BCmanual = np.asarray([ch in bc_manual for ch in self.ch_names])
    
    def update_params(self, **kwargs):
        """Update artifact-derivation parameters.

        Parameters
        ----------
        **kwargs
            Parameter names and new values.

        Returns
        -------
        None

        Raises
        ------
        KeyError
            If a parameter name is not recognized.
        """
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value
            else:
                raise KeyError(f"Parameter '{key}' is not a valid parameter. Valid parameters are: {list(self.params.keys())}")

    def copy(self):
        """Return a deep copy of the artifacts object and all rejection matrices."""
        return copy.deepcopy(self)

    
class ArtifactsRaw(Artifacts):
    """
    Class representing the artifacts in raw EEG data.
    
    This class holds and manages an artifact rejection matrix for an EEG dataset,
    providing facilities to access and update information about various types of artifacts.
       
    The artifact rejection matrix contains several fields that denote different aspects of data quality:
    - BCT: Bad Channel Time - samples that are bad for specific channels over time.
    - BC: Bad Channel - channels that are bad throughout the recording.
    - BCmanual: Bad Channel (manual) - manually specified bad channels.
    - BT: Bad Time - time segments that are bad across all channels.
    - CCT: Corrected Channel Time - samples that have been corrected over time.
    """

    def __init__(self, raw, **kwargs):
        """Create artifact masks for a raw recording.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            Raw EEG object.
        **kwargs
            Additional parameters forwarded to :class:`Artifacts`.

        Returns
        -------
        None
        """

        # Check if the raw object is an mne.io.BaseRaw object
        if not isinstance(raw,  mne.io.BaseRaw):
            raise ValueError("The raw object must be an instance of mne.io.BaseRaw.")
        
        super().__init__(raw, **kwargs)

    def update_bc(self, bc):
        """Merge a new bad-channel mask into ``BC``.

        Parameters
        ----------
        bc : array-like
            Channel-wise bad flags.

        Returns
        -------
        None
        """
        bc = np.reshape(bc,(self.n_channels, 1))
        self.BC = np.logical_or(bc, self.BC)

    def update_bt(self, bt):
        """Merge a new bad-time mask into ``BT``.

        Parameters
        ----------
        bt : array-like
            Time-wise bad flags.

        Returns
        -------
        None
        """
        bt = np.reshape(bt,(1, self.n_samples))
        self.BT = np.logical_or(bt, self.BT)
 
    def set_bc(self, bc):
        """Set the bad-channel mask.

        Parameters
        ----------
        bc : array-like
            Channel-wise bad flags.

        Returns
        -------
        None
        """
        self.BC = np.reshape(bc,(self.n_channels, 1))

    def set_bt(self, bt):
        """Set the bad-time mask.

        Parameters
        ----------
        bt : array-like
            Time-wise bad flags.

        Returns
        -------
        None
        """
        self.BT = np.reshape(bt,(1, self.n_samples))
        
    def reset_bc(self):
        """Reset ``BC`` to all-good values.

        Returns
        -------
        None
        """
        self.BC = np.full((self.n_channels, 1), False)

    def reset_bt(self):
        """Reset ``BT`` to all-good values.

        Returns
        -------
        None
        """
        self.BT = np.full((1, self.n_samples), False)

    def include_short_bad_segments(self, time_limit):
        """Restore very short bad segments as good.

        Parameters
        ----------
        time_limit : float
            Maximum bad-segment duration in seconds to restore.

        Returns
        -------
        None
        """
        samples_limit = int(np.round(time_limit*self.sfreq))
        bt, _ = include_short_bad_segments(self.BT.flatten(), samples_limit)
        self.set_bt(bt)
              
    def reject_short_good_segments(self, time_limit):
        """Reject very short good intervals.

        Parameters
        ----------
        time_limit : float
            Maximum good-segment duration in seconds to reject.

        Returns
        -------
        None
        """
        samples_limit = int(np.round(time_limit*self.sfreq))
        bt, _ = reject_short_good_segments(self.BT.flatten(), samples_limit)
        self.set_bt(bt)
    
    def mask_bad_segments(self, time_mask):
        """Expand bad segments with a temporal buffer.

        Parameters
        ----------
        time_mask : float
            Buffer duration in seconds added around bad segments.

        Returns
        -------
        None
        """
        mask_samples = int(np.round(time_mask*self.sfreq))
        bt, _ = mask_bad_segments(self.BT.flatten(), mask_samples)
        self.set_bt(bt)
        
    def define_bcbt(self, keep_rejected_previous=None, plot_rejection_matrix=False):
        """Recompute bad-channel and bad-time masks from ``BCT``.

        Parameters
        ----------
        keep_rejected_previous : {'bt', 'bc'} | None, default=None
            Preserve previously rejected times or channels before recomputing.
        plot_rejection_matrix : bool, default=False
            If True, display the resulting artifact structure figure.

        Returns
        -------
        None
        """
        
        print('Identifying bad samples and channels...')

       # Reject
        if keep_rejected_previous=='bt':
            bt_pre = self.BT==1
        else:
            bt_pre = None
        if keep_rejected_previous=='bc':
            bc_pre = self.BC==1
        else:
            bc_pre = None
        bc, bt, _ = define_bcbt(self.BCT, self.params['thresh_bad_times'], self.params['thresh_bad_channels'], bt=bt_pre, bc=bc_pre, bc_manual=self.BCmanual)
                
        # Remove too short artifacts
        samples_limit = int(np.round(self.params['min_bad_time']*self.sfreq))
        bt, _ = include_short_bad_segments(bt, samples_limit, axis=1)
        
        # Mask around short artifacts
        mask_samples = int(np.round(self.params['mask_time']*self.sfreq))
        bt, _ = mask_bad_segments(bt, mask_samples, axis=1)
        
        # Remove too short periods with non artifacts
        samples_limit = int(np.round(self.params['min_good_time']*self.sfreq))
        bt, _ = reject_short_good_segments(bt, samples_limit, axis=1)      
    
        # Update rejected data
        self.set_bc(bc)
        self.set_bt(bt)
                    
        # Display rejected data
        self.display_rejected_data()
        print('\n\nSUMMARY: Artifacts')
        print(self.print_summary())
        print('\n')

        # Plot rejection matrix
        if plot_rejection_matrix:
            self.plot_artifact_structure(artifact='all')
                   
    def display_rejected_data(self):
        """Print summary percentages of rejected time and channels.

        Returns
        -------
        None
        """
        print(f"Total BAD TIMES __________________________________ {np.sum(self.BT) / self.n_samples * 100:.2f}%")
        print(f"Total BAD CHANNELS _______________________________ {np.sum(self.BC) / self.n_channels * 100:.2f}%")
                  
    def plot_artifact_structure(self, artifact='all',time_step=50, color_scheme='turbo'):
        """Plot current raw artifact masks.

        Parameters
        ----------
        artifact : {'all', 'BCT', 'BT', 'BC', 'BE'}, default='all'
            Artifact layer to display.
        time_step : int, default=50
            Tick spacing for the time axis.
        color_scheme : str, default='turbo'
            Matplotlib colormap name.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Artifact structure figure.
        """
        bct = self.BCT.copy()
        bc = self.BC.copy()
        bt = self.BT.copy()
        bct = bct[np.newaxis, :, :]  # Add an epoch dimension for compatibility with the plotting function
        bc = bc[np.newaxis, :, :]  # Add an epoch dimension for compatibility with the plotting function
        bt = bt[np.newaxis, :, :]  # Add an epoch dimension for compatibility with the plotting function
        return plot_artifact_structure(self.times, self.ch_names, bct, bc=bc, bt=bt, be=None,
                       artifact=artifact, time_step=time_step, color_scheme=color_scheme,
                       thresh_bad_channels=self.params['thresh_bad_channels'][-1],
                       thresh_bad_times=self.params['thresh_bad_times'][-1])

    def plot_bad_channels_bar(self):
        """Bar plot of bad-data percentage per channel, excluding bad-time samples.

        For each channel the percentage is computed as the fraction of samples
        in ``BCT`` that are True, counted only over samples where ``BT`` is
        False (i.e. samples not already flagged as bad times).  A horizontal
        line marks the final ``thresh_bad_channels`` threshold.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Bar chart figure.
        """
        import matplotlib.pyplot as plt

        # Good-time mask: BT shape is (1, n_samples)
        idx_good = self.BT[0, :] == False  # noqa: E712
        n_good = idx_good.sum()

        pct = np.array([
            self.BCT[i, idx_good].sum() / n_good * 100 if n_good > 0 else 0.0
            for i in range(self.n_channels)
        ])

        fig, ax = plt.subplots(figsize=(max(8, self.n_channels * 0.15), 4))
        ax.bar(np.arange(self.n_channels), pct, color='steelblue', width=0.8)
        thresh = self.params['thresh_bad_channels'][-1] * 100
        ax.axhline(thresh, color='red', linestyle='--', linewidth=1.2,
                   label=f'Threshold ({thresh:.0f}%)')
        ax.set_xticks(np.arange(self.n_channels))
        ax.set_xticklabels(self.ch_names, rotation=90, fontsize=5)
        ax.set_xlabel('Channel')
        ax.set_ylabel('Bad data (%)')
        ax.set_title('Percentage of bad data per channel (excluding bad times)')
        ax.legend()
        fig.tight_layout()
        return fig

    def plot_bad_times_line(self):
        """Line plot of bad-channel percentage at each time sample, excluding bad channels.

        For each sample the percentage is computed as the fraction of channels
        in ``BCT`` that are True, counted only over channels where ``BC`` is
        False (i.e. channels not already flagged as bad channels).  A
        horizontal line marks the final ``thresh_bad_times`` threshold.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Line plot figure.
        """
        import matplotlib.pyplot as plt

        # Good-channel mask: BC shape is (n_channels, 1)
        good_ch = self.BC[:, 0] == False  # noqa: E712
        n_good_ch = good_ch.sum()

        if n_good_ch > 0:
            pct = self.BCT[good_ch, :].sum(axis=0) / n_good_ch * 100
        else:
            pct = np.zeros(self.n_samples)

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(self.times, pct, color='steelblue', linewidth=0.8)
        ax.fill_between(self.times, pct, alpha=0.3, color='steelblue')
        thresh = self.params['thresh_bad_times'][-1] * 100
        ax.axhline(thresh, color='red', linestyle='--', linewidth=1.2,
                   label=f'Threshold ({thresh:.0f}%)')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Bad channels (%)')
        ax.set_title('Percentage of bad channels over time (excluding bad channels)')
        ax.legend()
        fig.tight_layout()
        return fig


class ArtifactsEpochs(Artifacts):
    """
    Class representing the artifacts in Epochs EEG data.
    
    This class holds and manages an artifact rejection matrix for an EEG dataset,
    providing facilities to access and update information about various types of artifacts.
       
    The artifact rejection matrix contains several fields that denote different aspects of data quality:
    - BCT: Bad Channel Time - samples that are bad for specific channels over time.
    - BC: Bad Channel - channels that are bad throughout the recording.
    - BCmanual: Bad Channel (manual) - manually specified bad channels.
    - BT: Bad Time - time segments that are bad across all channels.
    - BE: Bad Epochs - epochs that are bad across all channels and times.
    - CCT: Corrected Channel Time - samples that have been corrected over time.
    """



    def __init__(self, epochs, **kwargs):
        """Create artifact masks for an epoched dataset.

        Parameters
        ----------
        epochs : mne.BaseEpochs
            Epoched EEG object.
        **kwargs
            Additional parameters forwarded to :class:`Artifacts`.

        Returns
        -------
        None
        """

        # Accept all MNE epochs containers (Epochs, EpochsArray, and subclasses).
        base_epochs_type = getattr(mne, "BaseEpochs", mne.epochs.BaseEpochs)
        if not isinstance(epochs, base_epochs_type):
            raise ValueError(
                f"The epochs object must be an instance of mne.BaseEpochs, got {type(epochs)}."
            )
        
        # if kwargs is empyt load default parameters
        if not kwargs:
            cfg_define_bcbt_epochs = get_cfg(None, 'define_bcbt_epochs_config.json')
        super().__init__(epochs, **cfg_define_bcbt_epochs)

    def update_bc(self, bc):
        """Merge a new epoch-wise bad-channel mask into ``BC``."""
        bc = np.reshape(bc,(self.n_epochs, self.n_channels, 1))
        self.BC = np.logical_or(bc, self.BC)

    def update_bt(self, bt):
        """Merge a new epoch-wise bad-time mask into ``BT``."""
        bt = np.reshape(bt,(self.n_epochs, 1, self.n_samples))
        self.BT = np.logical_or(bt, self.BT)
 
    def update_be(self, be):
        """Merge a new bad-epoch mask into ``BE``."""
        be = np.reshape(be, self.n_epochs)
        self.BE = np.logical_or(be, self.BE)
 
    def set_bc(self, bc):
        """Set the epoch-wise bad-channel mask."""
        self.BC = np.reshape(bc,(self.n_epochs, self.n_channels, 1))

    def set_bt(self, bt):
        """Set the epoch-wise bad-time mask."""
        self.BT = np.reshape(bt,(self.n_epochs, 1, self.n_samples))
 
    def set_be(self, be):
        """Set the bad-epoch mask."""
        self.BE = np.reshape(be, self.n_epochs)

    def reset_bc(self):
        """Reset ``BC`` to all-good values."""
        self.BC = np.full((self.n_epochs, self.n_channels, 1), False)

    def reset_bt(self):
        """Reset ``BT`` to all-good values."""
        self.BT = np.full((self.n_epochs, 1, self.n_samples), False)

    def reset_be(self):
        """Reset ``BE`` to all-good values."""
        self.BE = np.full(self.n_epochs, False)

    def include_short_bad_segments(self, time_limit):
        """Restore very short bad segments in each epoch."""
        bt = self.BT.copy()
        samples_limit = int(np.round(time_limit*self.sfreq))
        for ep in range(self.n_epochs): 
            bt_ep = self.BT[ep,:,:]
            bt_ep, _ = include_short_bad_segments(bt_ep.flatten(), samples_limit)
            bt[ep,:,:] = bt_ep            
        self.set_bt(bt)
              
    def reject_short_good_segments(self, time_limit):
        """Reject very short good intervals in each epoch."""
        bt = self.BT.copy()
        samples_limit = int(np.round(time_limit*self.sfreq))
        for ep in range(self.n_epochs): 
            bt_ep = self.BT[ep,:,:]
            bt_ep, _ = reject_short_good_segments(bt_ep.flatten(), samples_limit)
            bt[ep,:,:] = bt_ep            
        self.set_bt(bt)

    
    def mask_bad_segments(self, time_mask):
        """Expand bad segments in each epoch by a temporal buffer."""
        bt = self.BT.copy()
        mask_samples = int(np.round(time_mask*self.sfreq))
        for ep in range(self.n_epochs): 
            bt_ep = self.BT[ep,:,:]
            bt_ep, _ = mask_bad_segments(bt_ep.flatten(), mask_samples)
            bt[ep,:,:] = bt_ep  
        self.set_bt(bt)

    def define_bcbt(self, keep_rejected_previous=None, plot_rejection_matrix=False):
        """Recompute epoch-wise ``BC`` and ``BT`` masks from ``BCT``.

        Parameters
        ----------
        keep_rejected_previous : {'bt', 'bc'} | None, default=None
            Preserve previously rejected times or channels before recomputing.
        plot_rejection_matrix : bool, default=False
            If True, display the resulting artifact structure figure.

        Returns
        -------
        None
        """
        
        print('Identifying bad samples and channels...')

        # take the rejection matrix to define BC and BT
        bct = self.BCT.copy()
            
        # Reject
        if keep_rejected_previous=='bt':
            bt_pre = self.BT==1
        else:
            bt_pre = None
        if keep_rejected_previous=='bc':
            bc_pre = self.BC==1
        else:
            bc_pre = None
        bc, bt, _ = define_bcbt(self.BCT, self.params['thresh_bad_times'], self.params['thresh_bad_channels'], bt=bt_pre, bc=bc_pre, bc_manual=self.BCmanual)
        
        for ep in range(self.n_epochs): 
            bt_ep = bt[ep,0,:]
            bt_ep = np.reshape(bt_ep, np.size(bt_ep))
            
            # Remove too short artifacts
            samples_limit = int(np.round(self.params['min_bad_time']*self.sfreq))
            bt_ep, _ = include_short_bad_segments(bt_ep, samples_limit)
            
            # Mask around short artifacts
            mask_samples = int(np.round(self.params['mask_time']*self.sfreq))
            bt_ep, _ = mask_bad_segments(bt_ep, mask_samples)
            
            # Remove too short periods with non artifacts
            samples_limit = int(np.round(self.params['min_good_time']*self.sfreq))
            bt_ep, _ = reject_short_good_segments(bt_ep, samples_limit)
            
            bt[ep,0,:] = bt_ep
        
        # Update rejected data
        self.set_bc(bc)
        self.set_bt(bt)
        
        # Display rejected data
        self.display_rejected_data()
        print('\n\nSUMMARY: Artifacts')
        print(self.print_summary())
        print('\n')

        # Plot rejection matrix
        if plot_rejection_matrix:
            self.plot_artifact_structure(artifact='all')
       

    def display_rejected_data(self): 
        """Print summary percentages of rejected data across epochs."""
        print(f"Total BAD TIMES __________________________________ {np.sum(self.BT[:]) / (self.n_epochs * self.n_samples) * 100:.2f}%")
        print(f"Total BAD CHANNELS per epoch _____________________ {np.sum(self.BC[:]) / (self.n_epochs *self.n_channels) * 100:.2f}%")
        print(f"Total BAD CHANNELS _______________________________ {np.sum(np.all(self.BC, axis=0)) / self.n_channels * 100:.2f}%")
            


    def plot_artifact_structure(self, artifact='all',time_step=50, color_scheme='turbo'):
        """Plot current epoch artifact masks.

        Parameters
        ----------
        artifact : {'all', 'BCT', 'BT', 'BC', 'BE'}, default='all'
            Artifact layer to display.
        time_step : int, default=50
            Tick spacing for the x-axis.
        color_scheme : str, default='turbo'
            Matplotlib colormap name.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Artifact structure figure.
        """
        
        return plot_artifact_structure(self.times, self.ch_names, self.BCT, bc=self.BC, bt=self.BT, be=self.BE,
                       artifact=artifact, time_step=time_step, color_scheme=color_scheme,
                       thresh_bad_channels=self.params['thresh_bad_channels'][-1],
                       thresh_bad_times=self.params['thresh_bad_times'][-1])

    def plot_bad_channels_bar(self):
        """Bar plot of bad-data percentage per channel, excluding bad-time samples.

        For each channel the percentage is computed over samples where ``BT``
        is False across all epochs.  A horizontal line marks the final
        ``thresh_bad_channels`` threshold.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Bar chart figure.
        """
        import matplotlib.pyplot as plt

        # Good-time mask: BT shape is (n_epochs, 1, n_samples) → (n_epochs, n_samples)
        idx_good = self.BT[:, 0, :] == False  # noqa: E712  shape: (n_epochs, n_samples)
        n_good = idx_good.sum()

        pct = np.array([
            self.BCT[:, i, :][idx_good].sum() / n_good * 100 if n_good > 0 else 0.0
            for i in range(self.n_channels)
        ])

        fig, ax = plt.subplots(figsize=(max(8, self.n_channels * 0.15), 4))
        ax.bar(np.arange(self.n_channels), pct, color='steelblue', width=0.8)
        thresh = self.params['thresh_bad_channels'][-1] * 100
        ax.axhline(thresh, color='red', linestyle='--', linewidth=1.2,
                   label=f'Threshold ({thresh:.0f}%)')
        ax.set_xticks(np.arange(self.n_channels))
        ax.set_xticklabels(self.ch_names, rotation=90, fontsize=5)
        ax.set_xlabel('Channel')
        ax.set_ylabel('Bad data (%)')
        ax.set_title('Percentage of bad data per channel (excluding bad times)')
        ax.legend()
        fig.tight_layout()
        return fig

    def plot_bad_times_line(self):
        """Line plot of bad-channel percentage at each sample, concatenated across epochs.

        For each epoch, the per-sample bad-channel percentage is computed
        excluding channels flagged as bad (``BC == True``) in that epoch.
        The epoch vectors are concatenated into one continuous signal and
        vertical dashed lines mark the epoch boundaries.  A horizontal line
        marks the final ``thresh_bad_times`` threshold.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Line plot figure.
        """
        import matplotlib.pyplot as plt

        pct_all = []
        for ep in range(self.n_epochs):
            good_ch = self.BC[ep, :, 0] == False  # noqa: E712
            n_good_ch = good_ch.sum()
            if n_good_ch > 0:
                pct_ep = self.BCT[ep, good_ch, :].sum(axis=0) / n_good_ch * 100
            else:
                pct_ep = np.zeros(self.n_samples)
            pct_all.append(pct_ep)

        pct_concat = np.concatenate(pct_all)
        x = np.arange(len(pct_concat))

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(x, pct_concat, color='steelblue', linewidth=0.8)
        ax.fill_between(x, pct_concat, alpha=0.3, color='steelblue')

        # Vertical lines at epoch boundaries
        for ep in range(1, self.n_epochs):
            ax.axvline(ep * self.n_samples, color='gray', linestyle=':', linewidth=0.8)

        thresh = self.params['thresh_bad_times'][-1] * 100
        ax.axhline(thresh, color='red', linestyle='--', linewidth=1.2,
                   label=f'Threshold ({thresh:.0f}%)')

        # X-axis ticks at epoch centres labelled by epoch number
        centres = np.arange(self.n_epochs) * self.n_samples + self.n_samples // 2
        ax.set_xticks(centres)
        ax.set_xticklabels(np.arange(self.n_epochs))
        ax.set_xlabel('Epoch #')
        ax.set_ylabel('Bad channels (%)')
        ax.set_title('Percentage of bad channels over time (excluding bad channels)')
        ax.legend()
        fig.tight_layout()
        return fig


