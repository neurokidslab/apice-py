"""Signal filtering utilities for APICE preprocessing.

This module defines a thin wrapper around MNE filtering methods for applying
high-pass and low-pass FIR filters to raw EEG recordings.
"""

# %% CLASSES
class Filter:
    """Apply low-pass and high-pass filters to MNE raw data.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Raw EEG object to filter in place.
    h_freq : float | None, default=40.0
        Low-pass cutoff frequency in Hz. If ``None`` or falsy, low-pass
        filtering is skipped.
    l_freq : float | None, default=0.1
        High-pass cutoff frequency in Hz. If ``None`` or falsy, high-pass
        filtering is skipped.
    h_trans_bandwidth : float, default=10
        Transition bandwidth for low-pass filtering.
    l_trans_bandwidth : float, default=0.1
        Transition bandwidth for high-pass filtering.
    n_jobs : int, default=-1
        Number of parallel jobs passed to MNE filtering.

    Returns
    -------
    None
        Filtering is applied in place to ``raw`` during initialization.
    """
    def __init__(self, raw, h_freq=40.0, l_freq=0.1, h_trans_bandwidth=10, l_trans_bandwidth=0.1, n_jobs=-1):
        """Initialize filtering and apply configured filters in place.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            Raw EEG object to filter.
        h_freq : float | None, default=40.0
            Low-pass cutoff frequency in Hz.
        l_freq : float | None, default=0.1
            High-pass cutoff frequency in Hz.
        h_trans_bandwidth : float, default=10
            Transition bandwidth for low-pass filtering.
        l_trans_bandwidth : float, default=0.1
            Transition bandwidth for high-pass filtering.
        n_jobs : int, default=-1
            Number of parallel jobs used by MNE.

        Returns
        -------
        None
            Filters are applied directly to ``raw``.
        """

        if h_freq:
            self.low_pass(raw, f_cutoff=h_freq, h_trans_bandwidth=h_trans_bandwidth, n_jobs=n_jobs)
        if l_freq:
            self.high_pass(raw, f_cutoff=l_freq, l_trans_bandwidth=l_trans_bandwidth, n_jobs=n_jobs)

    @staticmethod
    def high_pass(raw, f_cutoff=0.1, l_trans_bandwidth=0.1, n_jobs=-1):
        """Apply a high-pass FIR filter to raw data in place.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            Raw EEG object to filter.
        f_cutoff : float, default=0.1
            High-pass cutoff frequency in Hz.
        l_trans_bandwidth : float, default=0.1
            Low transition bandwidth passed to ``mne.io.Raw.filter``.
        n_jobs : int, default=-1
            Number of jobs used by MNE.

        Returns
        -------
        None
            The operation mutates ``raw``.
        """

        raw.load_data().filter(l_freq=f_cutoff, h_freq=None, l_trans_bandwidth=l_trans_bandwidth, n_jobs=n_jobs)
        return

    @staticmethod
    def low_pass(raw, f_cutoff=40.0, h_trans_bandwidth=10, n_jobs=-1):
        """Apply a low-pass FIR filter to raw data in place.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            Raw EEG object to filter.
        f_cutoff : float, default=40.0
            Low-pass cutoff frequency in Hz.
        h_trans_bandwidth : float, default=10
            High transition bandwidth passed to ``mne.io.Raw.filter``.
        n_jobs : int, default=-1
            Number of jobs used by MNE.

        Returns
        -------
        None
            The operation mutates ``raw``.
        """

        raw.load_data().filter(l_freq=None, h_freq=f_cutoff, h_trans_bandwidth=h_trans_bandwidth, n_jobs=n_jobs)
        return
