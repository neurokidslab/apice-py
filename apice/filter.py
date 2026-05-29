"""Signal filtering utilities for APICE preprocessing.

This module defines a thin wrapper around MNE filtering methods for applying
high-pass and low-pass FIR filters to raw EEG recordings.
It defines also the class filter to remove the 50Hz noise following the zapline introduced by De Chevigne (2020).
"""
import numpy as np
import mne
import matplotlib.pyplot as plt
from scipy import linalg, signal
from joblib import Parallel, delayed

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
    n_jobs : int, default=-1
        Number of parallel jobs passed to MNE filtering.

    Returns
    -------
    None
        Filtering is applied in place to ``raw`` during initialization.
    """
    def __init__(self, raw, h_freq=40.0, l_freq=0.1, n_jobs=-1):
        """Initialize filtering and apply configured filters in place.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            Raw EEG object to filter.
        h_freq : float | None, default=40.0
            Low-pass cutoff frequency in Hz.
        l_freq : float | None, default=0.1
            High-pass cutoff frequency in Hz.
        n_jobs : int, default=-1
            Number of parallel jobs used by MNE.

        Returns
        -------
        None
            Filters are applied directly to ``raw``.
        """

        if h_freq:
            self.low_pass(raw, f_cutoff=h_freq, n_jobs=n_jobs)
        if l_freq:
            self.high_pass(raw, f_cutoff=l_freq, n_jobs=n_jobs)

    @staticmethod
    def high_pass(raw, f_cutoff=0.1, n_jobs=-1):
        """Apply a high-pass FIR filter to raw data in place.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            Raw EEG object to filter.
        f_cutoff : float, default=0.1
            High-pass cutoff frequency in Hz.
        n_jobs : int, default=-1
            Number of jobs used by MNE.

        Returns
        -------
        None
            The operation mutates ``raw``.
        """

        raw.load_data().filter(l_freq=f_cutoff, h_freq=None, n_jobs=n_jobs)
        return

    @staticmethod
    def low_pass(raw, f_cutoff=40.0, n_jobs=-1):
        """Apply a low-pass FIR filter to raw data in place.

        Parameters
        ----------
        raw : mne.io.BaseRaw
            Raw EEG object to filter.
        f_cutoff : float, default=40.0
            Low-pass cutoff frequency in Hz.
        n_jobs : int, default=-1
            Number of jobs used by MNE.

        Returns
        -------
        None
            The operation mutates ``raw``.
        """

        raw.load_data().filter(l_freq=None, h_freq=f_cutoff, n_jobs=n_jobs)
        return


class ZapLine:
    def __init__(self, raw, fline=50, chunk_duration=30, n_jobs=-1):
        self.fline = fline
        self.chunk_duration = chunk_duration
        self.n_jobs = n_jobs
        self.report_stats = {"harmonics": []}
        # We don't auto-apply in __init__ anymore to give you control over the fig
        
    def apply(self, raw):
        """Executes cleaning and returns (raw, fig)"""
        print("\n" + "="*30)
        print("ZAPLINE SPATIAL FILTERING")
        print("="*30)

        if not raw.preload:
            raw.load_data()
            
        eeg_indices = mne.pick_types(raw.info, eeg=True)
        sfreq = raw.info['sfreq']
        data = raw.get_data(picks=eeg_indices).T
        
        # 1. Pre-processing Metrics for the Log
        n_fft = int(sfreq * 2)
        freqs, psd_before = signal.welch(data.T, sfreq, nperseg=n_fft)
        psd_before_avg = np.mean(psd_before, axis=0)

        max_h = int(np.floor((sfreq/2) / self.fline))
        freqs_to_clean = [self.fline * i for i in range(1, max_h + 1) if self.fline * i < (sfreq/2) - 2]
        
        print(f"Targeting fundamental {self.fline}Hz and {len(freqs_to_clean)-1} harmonics.")
        
        current_data = data.copy()
        chunk_size = int(self.chunk_duration * sfreq)
        starts = range(0, data.shape[0], chunk_size)
        
        # To store stats for the final text summary
        summary_stats = []

        # 2. Iterative Cleaning
        for f in freqs_to_clean:
            print(f" - Processing {f}Hz...", end="\r")
            
            # Run ZapLine core
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self._zapline_core)(
                    current_data[s:min(s+chunk_size, data.shape[0])], sfreq, f
                ) for s in starts
            )
            current_data = np.concatenate(results, axis=0)
            
            # Calculate SNR reduction for this harmonic
            _, psd_tmp = signal.welch(current_data.T, sfreq, nperseg=n_fft)
            psd_after_tmp = np.mean(psd_tmp, axis=0)
            
            snr_pre = self._calculate_snr(psd_before_avg, freqs, f)
            snr_post = self._calculate_snr(psd_after_tmp, freqs, f)
            summary_stats.append((f, snr_pre, snr_post))

        # 3. Final Text Report (printed to console/log file)
        print(f"\n\n{'Freq (Hz)':<10} | {'Pre-SNR':<10} | {'Post-SNR':<10} | {'Reduction':<10}")
        print("-" * 50)
        for f, pre, post in summary_stats:
            reduction = pre - post
            print(f"{f:<10} | {pre:<8.1f} dB | {post:<9.1f} dB | {reduction:<9.1f} dB")
            self.report_stats["harmonics"].append({
                "freq": f, "reduction": reduction
            })

        # Update the raw object in place
        raw._data[eeg_indices, :] = current_data.T
        
        # Generate the figure for the HTML/PDF report
        psd_final_avg = np.mean(psd_tmp, axis=0)
        fig = self._generate_diagnostic_fig(freqs, psd_before_avg, psd_final_avg)
        
        return raw, fig

    def _generate_diagnostic_fig(self, freqs, pre, post):
        """Creates and returns the matplotlib figure for the report."""
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.semilogy(freqs, pre, label='Pre-ZapLine', color='red', alpha=0.5)
        ax.semilogy(freqs, post, label='Post-ZapLine', color='black', linewidth=1)
        ax.set_xlim(0, min(450, freqs[-1]))
        ax.set_title(f"ZapLine Spectral Impact ({self.fline}Hz + Harmonics)")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power (dB)")
        ax.legend()
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        return fig

    @staticmethod
    def _zapline_core(x, sfreq, fline):
        n_samples, n_chans = x.shape
        T_samples = int(np.round(sfreq / fline))
        kernel = np.ones(T_samples) / T_samples
        x_prime = signal.convolve(x, kernel[:, np.newaxis], mode='same')
        x_res = x - x_prime
        b, a = signal.butter(4, [fline-1, fline+1], btype='bandpass', fs=sfreq)
        x_biased = signal.filtfilt(b, a, x_res, axis=0)
        c0 = np.dot(x_res.T, x_res) + np.eye(n_chans) * (np.trace(np.dot(x_res.T, x_res)) * 1e-6 / n_chans)
        c1 = np.dot(x_biased.T, x_biased)
        evals, M = linalg.eigh(c1, c0)
        idx = np.argsort(evals)[::-1]
        evals, M = evals[idx], M[:, idx]
        
        # Use the Z-score logic from before
        n_rem = ZapLine._find_n_remove(evals)
        
        M_inv = linalg.pinv(M)
        D = np.dot(M[:, n_rem:], M_inv[n_rem:, :])
        return x_prime + np.dot(x_res, D)
    
    @staticmethod
    def _calculate_snr(psd, freqs, f):
        idx_f = (freqs >= f - 0.5) & (freqs <= f + 0.5)
        idx_n = (freqs >= f - 2.5) & (freqs <= f + 2.5) & ~idx_f
        return 10 * np.log10(np.mean(psd[idx_f]) / np.mean(psd[idx_n]))

    @staticmethod
    def _find_n_remove(evals):
        """Determines how many spatial components represent line noise."""
        ratios = evals[:-1] / np.maximum(evals[1:], 1e-12)
        med = np.median(ratios[int(len(ratios)*0.3):])
        mad = np.median(np.abs(ratios - med)) + 1e-6
        thresh = med + (3.0 * (mad / 0.6745))
        
        candidates = np.where(ratios[:int(len(evals)*0.15)] > thresh)[0]
        return candidates[-1] + 1 if len(candidates) > 0 else 0
