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
    def __init__(self, raw, low_pass_freq=40.0, high_pass_freq=0.1, n_jobs=-1):
        """
        Initializes the Filter object with specified low pass and high pass filter cutoff frequencies.

        Args:
            - raw: Object containing EEG data and information.
            - low_pass_freq (float): Low pass filter cutoff frequency in Hz. Defaults to 40.0 Hz.
            - high_pass_freq (float): High pass filter cutoff frequency in Hz. Defaults to 0.1 Hz.
        """

        if low_pass_freq:
            self.remove_high_freq(raw, f_cutoff=low_pass_freq, n_jobs=n_jobs)
        if high_pass_freq:
            self.remove_low_freq(raw, f_cutoff=high_pass_freq, n_jobs=n_jobs)

    @staticmethod
    def remove_low_freq(raw, f_cutoff=0.1, n_jobs=-1):
        """
        Applies high pass filtering on EEG data.

        Args:
            - raw: Object containing EEG data and information.
            - f_cutoff (float): High pass filter cutoff frequency in Hz; defaults to 0.1 Hz.
        """

        raw.load_data().filter(l_freq=f_cutoff, h_freq=None, l_trans_bandwidth=0.1, n_jobs=n_jobs)
        return

    @staticmethod
    def remove_high_freq(raw, f_cutoff=40.0, n_jobs=-1):
        """
        Applies low pass filtering on EEG data.

        Args:
            - raw: Object containing EEG data and information.
            - f_cutoff (float): Low pass filter cutoff frequency in Hz; defaults to 40 Hz.
        """

        raw.load_data().filter(l_freq=None, h_freq=f_cutoff, h_trans_bandwidth=10, n_jobs=n_jobs)
        return
