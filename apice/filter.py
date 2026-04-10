# %% CLASSES
class Filter:
    """
    Class for performing bandpass filtering on EEG data using Finite Impulse Response (FIR) filter.

    Attributes:
    ----------
        - raw (mne.io.Raw): Instance of MNE Raw object containing the EEG data.
        - low_pass_freq (float): The cutoff frequency for the low-pass filter in Hz.
        - high_pass_freq (float): The cutoff frequency for the high-pass filter in Hz.

    Args:
    -----
        - raw (mne.io.Raw): The raw EEG data object that contains the EEG signal and metadata.
        - low_pass_freq (float): The low-pass filter cutoff frequency in Hz. Defaults to 40.0 Hz, meaning frequencies above this will be attenuated.
        - high_pass_freq (float): The high-pass filter cutoff frequency in Hz. Defaults to 0.1 Hz, meaning frequencies below this will be attenuated.

    Methods:
    -------
        apply_high_pass(raw, f_cutoff=0.1):
            Applies a high-pass filter to the EEG data to remove frequencies below the cutoff frequency.
            Args:
                - raw (mne.io.Raw): The raw EEG data object to filter.
                - f_cutoff (float): The cutoff frequency for the high-pass filter in Hz; defaults to 0.1 Hz.
            Returns:
                mne.io.Raw: The filtered EEG data object.

        apply_low_pass(raw, f_cutoff=40.0):
            Applies a low-pass filter to the EEG data to remove frequencies above the cutoff frequency.
            Args:
                - raw (mne.io.Raw): The raw EEG data object to filter.
                - f_cutoff (float): The cutoff frequency for the low-pass filter in Hz; defaults to 40 Hz.
            Returns:
                mne.io.Raw: The filtered EEG data object.
    """
    def __init__(self, raw, h_freq=40.0, l_freq=0.1, h_trans_bandwidth=10, l_trans_bandwidth=0.1, n_jobs=-1):
        """
        Initializes the Filter object with specified low pass and high pass filter cutoff frequencies.

        Args:
            - raw: Object containing EEG data and information.
            - h_freq (float): Low pass filter cutoff frequency in Hz. Defaults to 40 Hz.
            - l_freq (float): High pass filter cutoff frequency in Hz. Defaults to 0.1 Hz.
            - h_trans_bandwidth (float): Transition bandwidth for the low-pass filter in Hz. Defaults to 10 Hz.
            - l_trans_bandwidth (float): Transition bandwidth for the high-pass filter in Hz. Defaults to 0.1 Hz.
        """

        if h_freq:
            self.low_pass(raw, f_cutoff=h_freq, h_trans_bandwidth=h_trans_bandwidth, n_jobs=n_jobs)
        if l_freq:
            self.high_pass(raw, f_cutoff=l_freq, l_trans_bandwidth=l_trans_bandwidth, n_jobs=n_jobs)

    @staticmethod
    def high_pass(raw, f_cutoff=0.1, l_trans_bandwidth=0.1, n_jobs=-1):
        """
        Applies high pass filtering on EEG data.

        Args:
            - raw: Object containing EEG data and information.
            - f_cutoff (float): High pass filter cutoff frequency in Hz; defaults to 0.1 Hz.
        """

        raw.load_data().filter(l_freq=f_cutoff, h_freq=None, l_trans_bandwidth=l_trans_bandwidth, n_jobs=n_jobs)
        return

    @staticmethod
    def low_pass(raw, f_cutoff=40.0, h_trans_bandwidth=10, n_jobs=-1):
        """
        Applies low pass filtering on EEG data.

        Args:
            - raw: Object containing EEG data and information.
            - f_cutoff (float): Low pass filter cutoff frequency in Hz; defaults to 40 Hz.
        """

        raw.load_data().filter(l_freq=None, h_freq=f_cutoff, h_trans_bandwidth=h_trans_bandwidth, n_jobs=n_jobs)
        return
