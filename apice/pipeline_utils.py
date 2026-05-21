
import numpy as np
import pandas as pd
import sys
from pathlib import Path

import mne
from mne import BaseEpochs



# %% CLASSES DEFINITIONS
class Summary():
    """Base helper for CSV-backed processing summaries.

    Parameters
    ----------
    output_folder : str | pathlib.Path
        Directory where the summary CSV will be stored.
    output_file : str
        Summary filename.
    columns : list of str, default=['file_id', 'step', 'length', 'corrected_data', 'bad_data', 'bad_channels', 'bad_times']
        Columns used when initializing a new summary dataframe.
    try_loading : bool, default=True
        If True, load an existing summary file when present.
    """

    def __init__(self, 
                 output_folder,
                 output_file,
                 columns=['file_id', 'step', 'length', 'corrected_data', 'bad_data', 'bad_channels', 'bad_times'],
                 try_loading=True):
        
        if 'file_id' not in columns or 'step' not in columns:
            raise ValueError("Columns must include 'file_id' and 'step'")
        
        self.output_folder = Path(output_folder)
        self.output_file = output_file
        self.output_full_path = self.output_folder / output_file
        if try_loading and self.output_full_path.exists():
            self.load()
        else:
            self.summary_df = pd.DataFrame(columns=columns)
        
    def load(self):
        """Load summary data from disk if the CSV exists.

        Returns
        -------
        None
        """
        if self.output_full_path.exists():
            self.summary_df = pd.read_csv(self.output_full_path)
        else:
            print(f"File {self.output_full_path} does not exist. Summary could not be loaded.")

    def save(self):
        """Persist the current summary dataframe to disk.

        Returns
        -------
        None
        """
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.summary_df.to_csv(self.output_full_path, index=False)
    
    def remove_file_from_summary(self, file_id):
        """Remove all rows associated with one file identifier.

        Parameters
        ----------
        file_id : str
            File identifier to remove.

        Returns
        -------
        None
        """
        self.summary_df = self.summary_df[self.summary_df['file_id'] != file_id]

    def remove_file_step_from_summary(self, file_id, step):
        """Remove rows associated with a specific file and step.

        Parameters
        ----------
        file_id : str
            File identifier to filter.
        step : str
            Processing step label to remove.

        Returns
        -------
        None
        """
        self.summary_df = self.summary_df[~((self.summary_df['file_id'] == file_id) & (self.summary_df['step'] == step))]


class SummaryPreprocessing(Summary):
    """Summary table specialized for continuous preprocessing outputs."""

    def __init__(self, 
                 output_folder, 
                 file_name,
                 file_id=None,
                 outputfile_subfix="-summary-preproc.csv", 
                 try_loading=True,
                 ):
        """Initialize preprocessing summary storage.

        Parameters
        ----------
        output_folder : str | pathlib.Path
            Directory where the summary CSV is stored.
        file_name : str | pathlib.Path
            Source filename used to derive the summary filename.
        file_id : str | None, default=None
            Custom file identifier. If None, derived from ``file_name``.
        outputfile_subfix : str, default='-summary-preproc.csv'
            Filename suffix for the summary CSV.
        try_loading : bool, default=True
            If True, load an existing summary file if available.

        Returns
        -------
        None
        """
        print("Initializing SummaryPreprocessing object")
        file_name = Path(file_name).stem
        outputfile = file_name + outputfile_subfix
        super().__init__(output_folder=output_folder, 
                         output_file=outputfile, 
                         columns=[
                             "file_id", 
                             "step", 
                             "length", 
                             "%_corrected_data", 
                             "%_bad_data", 
                             "%_bad_channels", 
                             "%_bad_times",
                             ],
                         try_loading=try_loading)
        if file_id is None:
            file_id = file_name
        self.file_id = file_id

    def add_to_summary(self, step, raw, overwrite=False):
        """Append one preprocessing summary row.

        Parameters
        ----------
        step : str
            Processing step label.
        raw : mne.io.BaseRaw
            Raw object whose artifact metrics will be summarized.
        overwrite : bool, default=False
            If True, replace an existing row for the same file and step.

        Returns
        -------
        None
        """
        
        if not isinstance(raw, mne.io.BaseRaw):
            raise TypeError("raw must be an instance of mne.io.Raw")
        
        if any(((self.summary_df['file_id'] == self.file_id) & (self.summary_df['step'] == step))) and not overwrite:
            print(f"File {self.file_id} and step {step} already exist in the summary. Use overwrite=True to overwrite the existing entry.")
            return
        
        if any(((self.summary_df['file_id'] == self.file_id) & (self.summary_df['step'] == step))) and overwrite:
            self.remove_file_step_from_summary(self.file_id, step)
        
        length = raw.times.max()
        if hasattr(raw, 'artifacts'):
            corrected_data = np.round(np.sum(raw.artifacts.CCT) / np.size(raw.artifacts.CCT) * 100, 2)
            bad_data = np.round(np.sum(raw.artifacts.BCT) / np.size(raw.artifacts.BCT) * 100, 2)
            bad_channels = np.round(np.sum(raw.artifacts.BC) / np.size(raw.artifacts.BC) * 100, 2)
            bad_times = np.round(np.sum(raw.artifacts.BT) / np.size(raw.artifacts.BT) * 100, 2)
        else:
            corrected_data = np.nan
            bad_data = np.nan
            bad_channels = np.nan
            bad_times = np.nan

        self.summary_df.loc[len(self.summary_df)] = [self.file_id, step, length, corrected_data, bad_data, bad_channels, bad_times]

class SummaryEpochs(Summary):
    """Summary table specialized for segmented/epoched outputs."""

    def __init__(self, 
                 output_folder, 
                 file_name,
                 file_id=None,
                 outputfile_subfix="summary-epo.csv", 
                 try_loading=True,
                 ):
        """Initialize epochs summary storage.

        Parameters
        ----------
        output_folder : str | pathlib.Path
            Directory where the summary CSV is stored.
        file_name : str | pathlib.Path
            Source filename used to derive the summary filename.
        file_id : str | None, default=None
            Custom file identifier. If None, derived from ``file_name``.
        outputfile_subfix : str, default='summary-epo.csv'
            Filename suffix for the summary CSV.
        try_loading : bool, default=True
            If True, load an existing summary file if available.

        Returns
        -------
        None
        """
        print("Initializing SummaryEpochs object")
        file_name = Path(file_name).stem
        outputfile = file_name + outputfile_subfix
        super().__init__(output_folder=output_folder, 
                         output_file=outputfile, 
                         columns=[
                             "file_id",
                             "step",
                             "n_epochs",
                             "n_remaining_epochs",
                             "n_rejected_epochs",
                             "length", 
                             "%_corrected_data", 
                             "%_bad_data", 
                             "%_bad_channels", 
                             "%_bad_times",
                             "%_bad_epochs",
                             ],
                         try_loading=try_loading)
        if file_id is None:
            file_id = file_name
        self.file_id = file_id

    def add_to_summary(self, step, epochs, overwrite=False):
        """Append one epochs summary row.

        Parameters
        ----------
        step : str
            Processing step label.
        epochs : mne.BaseEpochs
            Epochs object whose artifact metrics will be summarized.
        overwrite : bool, default=False
            If True, replace an existing row for the same file and step.

        Returns
        -------
        None
        """
        
        if not isinstance(epochs, BaseEpochs):
            raise TypeError("epochs must be an instance of mne.BaseEpochs")
        
        if any(((self.summary_df['file_id'] == self.file_id) & (self.summary_df['step'] == step))) and not overwrite:
            print(f"File {self.file_id} and step {step} already exist in the summary. Use overwrite=True to overwrite the existing entry.")
            return
        
        if any(((self.summary_df['file_id'] == self.file_id) & (self.summary_df['step'] == step))) and overwrite:
            self.remove_file_step_from_summary(self.file_id, step)
        
        length = epochs.times.max() - epochs.times.min()
        drop_log = np.asarray(epochs.drop_log, dtype=list)
        no_of_epochs = np.shape(drop_log)[0]
        no_of_remaining_epochs = np.shape(epochs._data)[0]
        if hasattr(epochs, 'artifacts'):
            corrected_data = np.round(np.sum(epochs.artifacts.CCT) / np.size(epochs.artifacts.CCT) * 100, 2)
            bad_data = np.round(np.sum(epochs.artifacts.BCT) / np.size(epochs.artifacts.BCT) * 100, 2)
            bad_channels = np.round(np.sum(epochs.artifacts.BC) / np.size(epochs.artifacts.BC) * 100, 2)
            bad_times = np.round(np.sum(epochs.artifacts.BT) / np.size(epochs.artifacts.BT) * 100, 2)
            bad_epochs = np.round(np.sum(epochs.artifacts.BE) / np.size(epochs.artifacts.BE) * 100, 2)
            no_of_rejected_epochs = np.sum(epochs.artifacts.BE)
        else:
            corrected_data = np.nan
            bad_data = np.nan
            bad_channels = np.nan
            bad_times = np.nan
            bad_epochs = np.nan
            no_of_rejected_epochs = np.nan
        self.summary_df.loc[len(self.summary_df)] = [self.file_id, step, no_of_epochs, no_of_remaining_epochs, no_of_rejected_epochs,
                                            length, corrected_data, bad_data, bad_channels, bad_times, bad_epochs]
        

class StdOutLogger():
    """Redirect standard output to a log file during pipeline execution.

    Parameters
    ----------
    output_folder : str | pathlib.Path
        Directory that will store the log file.
    file_name : str | pathlib.Path
        Base filename used to derive the log filename.
    """

    def __init__(self, output_folder, file_name):
        """Initialize logger paths.

        Parameters
        ----------
        output_folder : str | pathlib.Path
            Destination folder for log output.
        file_name : str | pathlib.Path
            Base name used to create the log filename.

        Returns
        -------
        None
        """
        file_name = Path(file_name).stem
        self.output_folder = Path(output_folder)
        self.output_file = f"{file_name}_log.txt"
        self.output_full_path = self.output_folder / self.output_file

    def restore_stdout(self):
        """Reset the log file contents before a new run.

        Returns
        -------
        None
        """
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.output_full_path.write_text("")

    def redirect_stdout_to_file(self, restore=False):
        """Redirect ``sys.stdout`` to the configured log file.

        Parameters
        ----------
        restore : bool, default=False
            If True, clear the existing log file before redirecting.

        Returns
        -------
        None
        """
        self.output_folder.mkdir(parents=True, exist_ok=True)
        if restore:
            self.restore_stdout()
        self._original_stdout = sys.stdout
        sys.stdout = open(self.output_full_path, "w")

    def close(self):
        """Close the redirected standard-output stream and restore the original stdout.

        Returns
        -------
        None
        """
        sys.stdout.close()
        sys.stdout = self._original_stdout
