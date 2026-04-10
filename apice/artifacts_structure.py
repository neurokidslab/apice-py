# %% LIBRARIES

# Import necessary modules
import copy
import mne  
import numpy as np 
from prettytable import PrettyTable 

# Import specific modules from your project's modules
from apice.utils import (get_data_size, include_short_bad_segments, reject_short_good_segments, mask_bad_segments)


# %% FUNCTIONS

def define_bcbt(bct, thresh_bad_times, thresh_bad_channels, bc=None, bt=None, bc_manual=None):
    
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



def plot_artifact_structure(times, ch_names, bct, bc=None, bt=None, be=None, artifact='all', time_step=50, color_scheme='gnuplot', figsize=(12, 6)):
    """
    This function plots a visual representation of the artifact structure within EEG data.
    It allows visualization of different types of artifacts and their occurrences over time or epochs.

    Args:
        times (numpy.ndarray): Array of time points corresponding to the EEG data.
        bct (numpy.ndarray): Binary matrix indicating bad channels and time points.
        bc (numpy.ndarray, optional): Binary matrix indicating bad channels. Defaults to None.
        bt (numpy.ndarray, optional): Binary matrix indicating bad time points. Defaults to None.
        be (numpy.ndarray, optional): Binary matrix indicating bad epochs. Defaults to None.
        artifact (str): Specifies the type of artifact to plot ('all', 'BCT', 'BT', 'BC', 'BE'). Defaults to 'all'.
        time_step (int): Time step for x-axis ticks, in seconds. Defaults to 50.
        color_scheme (str): The color scheme for plotting. Defaults to 'gnuplot'.
        figsize (tuple): Tuple specifying the figure size (width, height) in inches. Defaults to (8, 6).
        ch_names (list): List of channel names corresponding to the EEG data.

    Returns:
        matplotlib.figure.Figure: The figure object containing the artifact plot.
    """

    # Import necessary modules
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Functions
    def prepare_cmap(ax, data, artifact, color_scheme='gnuplot'):
        """
        Prepare colormap and colorbar for artifact matrix visualization.

        Args:
            ax (matplotlib.axes.Axes): The subplot where the artifact matrix will be displayed.
            data (numpy.ndarray): The artifact matrix to be visualized.
            artifact (str): Specifies the type of artifact ('all', 'BCT', 'BT', 'BC', 'BE').
            color_scheme (str): The color scheme for plotting. Defaults to 'gnuplot'.

        Returns:
            None
        """
        if artifact == 'all':
            # Define tick labels and colormap for all artifact types
            # tick_labels = ['good', 'bad', 'BT', 'BC', 'BE']
            tick_labels = ['Good Data', 'Bad Data', 'Bad Time Point', 'Bad Channel', 'Bad Epoch']
            cmap = plt.get_cmap(color_scheme, len(tick_labels))
            mat = ax.imshow(data, cmap=cmap, vmin=-0.5, vmax=4.5, aspect='auto')
            cax = plt.colorbar(mat, ticks=np.arange(5))
            cax.set_ticklabels(tick_labels)
        else:
            # Define tick labels and colormap for all artifact types
            cmap = plt.get_cmap(color_scheme, len(np.unique(data)))
            mat = ax.imshow(data, cmap=cmap, vmin=np.min(data) - 0.5, vmax=np.max(data) + 0.5, aspect='auto')
            cax = plt.colorbar(mat, ticks=np.asarray(np.unique(data), dtype=int))
            colorbar_ticks = np.unique(data)
            # labels = ['good', 'bad', 'BT', 'BC', 'BE']
            labels = ['Good Data', 'Bad Data', 'Bad Time Point', 'Bad Channel', 'Bad Epoch']
            tick_labels = []
            for i in colorbar_ticks:
                tick_labels.append(labels[i])
            cax.set_ticklabels(tick_labels)

    def set_ticks(ax, data, t, time_step, sfreq, n_epochs, n_channels, n_samples, ch_names):
        """
        Set ticks and labels for the x and y axes of a subplot.

        Args:
            ax (matplotlib.axes.Axes): The subplot where ticks and labels will be set.
            data (numpy.ndarray): The data matrix being displayed.
            time_step (int): Time step for x-axis ticks, in seconds.
            sfreq (float): The sampling frequency of the data.
            n_channels (int): Number of electrode channels.
            ch_names (list): List of channel names.
            x_label (str): Label for the x-axis. Defaults to 'Time (s)'.
            artifact (str): Specifies the type of artifact ('all' or specific type).

        Returns:
            None
        """
        
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
        ax.set_title(artifact)
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
    # Create a figure object with the specified figsize
    fig = plt.figure(figsize=figsize)
    
    # Plotting routine for a single epoch
    if n_epochs == 1:
        ax = fig.add_subplot(111)
        data = M[0, :, :]
        prepare_cmap(ax, data, artifact, color_scheme=color_scheme)
        set_ticks(ax, data, times, time_step, sfreq, n_epochs, n_channels, n_samples, ch_names)
    # Plotting routine for multiple epochs
    else:
        N = M[0, :, :]
        for ep in np.arange(1, n_epochs):
            N = np.concatenate((N.copy(), M[ep, :, :]), axis=1)
        ax = fig.add_subplot(111)
        data = N
        prepare_cmap(ax, data, artifact, color_scheme=color_scheme)
        set_ticks(ax, data, times, time_step, sfreq, n_epochs, n_channels, n_samples, ch_names)

    return fig
    


# %% CLASSE TO HOLD THE ARTIFACTS REJECTION MATRICES

class Artifacts:

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
        """
        Prints a summary of the bad data in the rejection matrices as a percentage of the total data.
        
        Returns:
        - summary: PrettyTable object representing the percentage of bad data for each artifact type.
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
        self.BCmanual = np.asarray([ch in bc_manual for ch in self.ch_names])
    
    def update_params(self, **kwargs):
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
        """
        Initializes the Artifacts object with artifact rejection matrices based on the EEG data.
        
        Parameters:
        - raw: An object containing the EEG data.
        """

        # Check if the raw object is an mne.io.BaseRaw object
        if not isinstance(raw,  mne.io.BaseRaw):
            raise ValueError("The raw object must be an instance of mne.io.BaseRaw.")
        
        super().__init__(raw, **kwargs)

    def update_bc(self, bc):
        bc = np.reshape(bc,(self.n_channels, 1))
        self.BC = np.logical_or(bc, self.BC)

    def update_bt(self, bt):
        bt = np.reshape(bt,(1, self.n_samples))
        self.BT = np.logical_or(bt, self.BT)
 
    def set_bc(self, bc):
        self.BC = np.reshape(bc,(self.n_channels, 1))

    def set_bt(self, bt):
        self.BT = np.reshape(bt,(1, self.n_samples))
        
    def reset_bc(self):
        self.BC = np.full((self.n_channels, 1), False)

    def reset_bt(self):
        self.BT = np.full((1, self.n_samples), False)

    def include_short_bad_segments(self, time_limit):
        samples_limit = int(np.round(time_limit*self.sfreq))
        bt, _ = include_short_bad_segments(self.BT.flatten(), samples_limit)
        self.set_bt(bt)
              
    def reject_short_good_segments(self, time_limit):
        samples_limit = int(np.round(time_limit*self.sfreq))
        bt, _ = reject_short_good_segments(self.BT.flatten(), samples_limit)
        self.set_bt(bt)
    
    def mask_bad_segments(self, time_mask):
        mask_samples = int(np.round(time_mask*self.sfreq))
        bt, _ = mask_bad_segments(self.BT.flatten(), mask_samples)
        self.set_bt(bt)
        
    def define_bcbt(self, keep_rejected_previous=None, plot_rejection_matrix=False):
        
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
        print(f"Total BAD TIMES __________________________________ {np.sum(self.BT) / self.n_samples * 100:.2f}%")
        print(f"Total BAD CHANNELS _______________________________ {np.sum(self.BC) / self.n_channels * 100:.2f}%")
                  
    def plot_artifact_structure(self, artifact='all',time_step=50, color_scheme='gnuplot'):
        bct = self.BCT.copy()
        bc = self.BC.copy()
        bt = self.BT.copy()
        bct = bct[np.newaxis, :, :]  # Add an epoch dimension for compatibility with the plotting function
        bc = bc[np.newaxis, :, :]  # Add an epoch dimension for compatibility with the plotting function
        bt = bt[np.newaxis, :, :]  # Add an epoch dimension for compatibility with the plotting function
        return plot_artifact_structure(self.times, self.ch_names, bct, bc=bc, bt=bt, be=None, 
                       artifact=artifact, time_step=time_step, color_scheme=color_scheme)
      


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
        """Initializes the Artifacts object with artifact rejection matrices based on the EEG data."""

        # Accept all MNE epochs containers (Epochs, EpochsArray, and subclasses).
        base_epochs_type = getattr(mne, "BaseEpochs", mne.epochs.BaseEpochs)
        if not isinstance(epochs, base_epochs_type):
            raise ValueError(
                f"The epochs object must be an instance of mne.BaseEpochs, got {type(epochs)}."
            )
        super().__init__(epochs, **kwargs)

    def update_bc(self, bc):
        bc = np.reshape(bc,(self.n_epochs, self.n_channels, 1))
        self.BC = np.logical_or(bc, self.BC)

    def update_bt(self, bt):
        bt = np.reshape(bt,(self.n_epochs, 1, self.n_samples))
        self.BT = np.logical_or(bt, self.BT)
 
    def update_be(self, be):
        be = np.reshape(be, self.n_epochs)
        self.BE = np.logical_or(be, self.BE)
 
    def set_bc(self, bc):
        self.BC = np.reshape(bc,(self.n_epochs, self.n_channels, 1))

    def set_bt(self, bt):
        self.BT = np.reshape(bt,(self.n_epochs, 1, self.n_samples))
 
    def set_be(self, be):
        self.BE = np.reshape(be, self.n_epochs)

    def reset_bc(self):
        self.BC = np.full((self.n_epochs, self.n_channels, 1), False)

    def reset_bt(self):
        self.BT = np.full((self.n_epochs, 1, self.n_samples), False)

    def reset_be(self):
        self.BE = np.full(self.n_epochs, False)

    def include_short_bad_segments(self, time_limit):
        bt = self.BT.copy()
        samples_limit = int(np.round(time_limit*self.sfreq))
        for ep in range(self.n_epochs): 
            bt_ep = self.BT[ep,:,:]
            bt_ep, _ = include_short_bad_segments(bt_ep.flatten(), samples_limit)
            bt[ep,:,:] = bt_ep            
        self.set_bt(bt)
              
    def reject_short_good_segments(self, time_limit):
        bt = self.BT.copy()
        samples_limit = int(np.round(time_limit*self.sfreq))
        for ep in range(self.n_epochs): 
            bt_ep = self.BT[ep,:,:]
            bt_ep, _ = reject_short_good_segments(bt_ep.flatten(), samples_limit)
            bt[ep,:,:] = bt_ep            
        self.set_bt(bt)

    
    def mask_bad_segments(self, time_mask):
        bt = self.BT.copy()
        mask_samples = int(np.round(time_mask*self.sfreq))
        for ep in range(self.n_epochs): 
            bt_ep = self.BT[ep,:,:]
            bt_ep, _ = mask_bad_segments(bt_ep.flatten(), mask_samples)
            bt[ep,:,:] = bt_ep  
        self.set_bt(bt)

    def define_bcbt(self, keep_rejected_previous=None, plot_rejection_matrix=False):
        
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
        print(f"Total BAD TIMES __________________________________ {np.sum(self.BT[:]) / (self.n_epochs * self.n_samples) * 100:.2f}%")
        print(f"Total BAD CHANNELS per epoch _____________________ {np.sum(self.BC[:]) / (self.n_epochs *self.n_channels) * 100:.2f}%")
        print(f"Total BAD CHANNELS _______________________________ {np.sum(np.all(self.BC, axis=0)) / self.n_channels * 100:.2f}%")
            


    def plot_artifact_structure(self, artifact='all',time_step=50, color_scheme='gnuplot'):
        
        return plot_artifact_structure(self.times, self.ch_names, self.BCT, bc=self.BC, bt=self.BT, be=self.BE, 
                       artifact=artifact, time_step=time_step, color_scheme=color_scheme)




