"""High-level preprocessing and segmentation workflows for APICE.

This module orchestrates the end-to-end APICE pipelines for continuous EEG
preprocessing, epoch segmentation, artifact summaries, logging, and report
generation.
"""

import json
import re
import numpy as np
import pandas as pd
import time
from pathlib import Path
from datetime import datetime
from tabulate import tabulate
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime, timezone


import mne

from apice.pipeline_utils import (SummaryPreprocessing, SummaryEpochs, StdOutLogger)
from apice.data_structures import RawAPICE
from apice.io import load_rawapice
from apice.utils import (get_onset_and_duration, get_cfg)
from apice.filter import (Filter, ZapLine)
from apice.ica import clean_ica




# %% Function to run the full APICE for multiple data files

def run_preprocessing(input_dir, 
                      output_dir, 
                      input_dir_bids=False,
                      bids_session=None,
                      bids_task=None,
                      bids_run=None,
                      bids_subject=None,
                      bids_extension='.vhdr',
                      bids_datatype='eeg',
                      bids_suffix='eeg',
                      processed_file_pattern='*-preproc-raw.fif',
                      data_selection_method="all",
                      drop_electrodes=None,
                      picks='eeg',
                      reference_channels=None,
                      crop_times=None,
                      crop_from_beginning=None,
                      crop_from_end=None,
                      resample_freq=None,
                      stim_channels_to_annotations=True,
                      montage=None,
                      save_log=True,
                      save_report=True,
                      save_summary=True,
                      show_figures=False,
                      apply_ica=False,
                      ica_parameters=None,
                      l_freq_initial=0.10,
                      h_freq_initial=100,
                      line_noise_freq_initial=50,
                      l_freq_apice=0.10,
                      h_freq_apice=40,
                      line_noise_freq_apice=None,
                      l_freq_artifacts=None,
                      h_freq_artifacts=None,
                      cfg_bad_channels_detection=None,
                      cfg_glitches_detection=None,
                      cfg_target_pca=None,
                      cfg_artifacts_detection=None,
                      cfg_spline_segments=None,
                      cfg_spline_channels=None,
                      n_jobs=-1,
                      ):
    """Run the default APICE preprocessing workflow over multiple files.

    Parameters
    ----------
    input_dir : str | pathlib.Path
        Input folder containing raw EEG files, or the BIDS root when
        ``input_dir_bids`` is True.
    output_dir : str | pathlib.Path
        Output folder for preprocessed data, reports, logs, and summaries.
    input_dir_bids : bool, default=False
        If True, discover input files as BIDS entries instead of plain files.
    bids_session : str | None, default=None
        Optional BIDS session filter.
    bids_task : str | None, default=None
        Optional BIDS task filter.
    bids_run : str | None, default=None
        Optional BIDS run filter.
    bids_subject : str | None, default=None
        Optional BIDS subject filter.
    bids_extension : str, default='.vhdr'
        BIDS file extension to select.
    bids_datatype : str, default='eeg'
        BIDS datatype selector.
    bids_suffix : str, default='eeg'
        BIDS suffix selector.
    processed_file_pattern : str, default='*-preproc-raw.fif'
        Pattern used to detect already processed outputs.
    data_selection_method : str | list[str], default='all'
        File-selection strategy forwarded to APICE I/O helpers.
    drop_electrodes : list[str] | None, default=None
        Channels to remove before preprocessing.
    picks : str | list | slice, default='eeg'
        Channels to keep during preprocessing.
    reference_channels : list[str] | str | None, default=None
        Reference channels to preserve in artifact handling.
    crop_times : tuple[float, float] | None, default=None
        Optional ``(tmin, tmax)`` crop window in seconds.
    crop_from_beginning : float | None, default=None
        Seconds to trim from the start of the recording.
    crop_from_end : float | None, default=None
        Seconds to trim from the end of the recording.
    resample_freq : float | None, default=None
        Target sampling frequency in Hz.
    stim_channels_to_annotations : bool, default=True
        If True, convert stim-channel events to annotations before processing.
    montage : mne.channels.DigMontage | str | pathlib.Path | None, default=None
        Montage object, built-in montage name, or path to a montage file.
    save_log : bool, default=True
        If True, write one log file per processed recording.
    save_report : bool, default=True
        If True, write an HTML report per processed recording.
    save_summary : bool, default=True
        If True, write a preprocessing summary CSV.
    show_figures : bool, default=False
        If False, disable interactive Matplotlib rendering while each file is
        processed. This prevents figure windows from popping up during
        multi-file runs while still allowing report figures to be created.
    l_freq_initial : float, default=0.10
        High-pass cutoff frequency in Hz apply in the initial steps (before ICA).
    h_freq_initial : float | None, default=40
        Low-pass cutoff frequency in Hz apply in the initial steps (before ICA).
    l_freq_apice : float, default=0.10
        High-pass cutoff frequency in Hz apply in the APICE default pipeline.
    h_freq_apice : float | None, default=40
        Low-pass cutoff frequency in Hz apply in the APICE default pipeline.
    l_freq_artifacts : float | None, default=None
        High-pass cutoff frequency for the copy of the data used in artifact detection steps. 
        None to use the same filtering as the data in the current step.
    h_freq_artifacts : float | None, default=None
        Low-pass cutoff frequency for the copy of the data used in artifact detection steps. 
        None to use the same filtering as the data in the current step.
    line_noise_freq_initial : float | None, default=None
        Line noise frequency to be removed in the initial steps (before ICA).
    line_noise_freq_apice : float | None, default=None
        Line noise frequency to be removed in the APICE default pipeline.
    cfg_bad_channels_detection : None | str | pathlib.Path | dict, default=None
        Configuration source for bad-channel detection.
    cfg_glitches_detection : None | str | pathlib.Path | dict, default=None
        Configuration source for glitch detection.
    cfg_target_pca : None | str | pathlib.Path | dict, default=None
        Configuration source for target-PCA correction.
    cfg_artifacts_detection : None | str | pathlib.Path | dict, default=None
        Configuration source for artifact detection.
    cfg_spline_segments : None | str | pathlib.Path | dict, default=None
        Configuration source for segment-wise spline correction.
    cfg_spline_channels : None | str | pathlib.Path | dict, default=None
        Configuration source for channel-wise spline correction.
    n_jobs : int, default=-1
        Number of parallel jobs for compute-intensive steps.

    Returns
    -------
    None
        Writes preprocessing outputs for each selected file.
    """

    # Check the if apply_ica flag is True then ica_parameters must be provided
    if apply_ica and ica_parameters is None:
        raise ValueError("If apply_ica is True, ica_parameters must be provided.")
    
    # Initialize output folders
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Get all files to process
    if not input_dir_bids:
        from apice.io import get_files_to_process
        files = get_files_to_process(input_dir=input_dir,
                                    output_dir=output_dir,
                                    processed_file_pattern=processed_file_pattern,
                                    data_selection_method=data_selection_method,
                                    )
    else:
        import mne_bids
        from apice.io import get_bids_files_to_process
        files = get_bids_files_to_process(bids_root=input_dir, 
                                          session=bids_session,
                                          task=bids_task,
                                          run=bids_run,
                                          subject=bids_subject,
                                          datatype=bids_datatype,
                                          suffix=bids_suffix,
                                          extension=bids_extension,
                                          output_dir=output_dir,
                                          processed_file_pattern=processed_file_pattern,
                                          data_selection_method=data_selection_method,
                                          )
    print(f"\nNumber of files to process: {len(files)}\n")

    # Loop over the subjects for data processing
    for file in files:

        print(f"Processing file: {file}")

        try:
        
            # Load raw data
            if not input_dir_bids:
                raw = mne.io.read_raw(file, preload=False, verbose=False)
                file_name = file.stem
            else:
                raw = mne_bids.read_raw_bids(file)
                file_name = file.basename.replace(bids_extension, '')

            # Create a shared report and logger for all steps of this file
            if save_report:
                shared_report = mne.Report(title=file_name)
            else:
                shared_report = None
            if save_log:
                output_dir_logs = output_dir / "logs"
                output_dir_logs.mkdir(exist_ok=True)
                shared_logger = StdOutLogger(output_dir_logs, file_name)
                shared_logger.redirect_stdout_to_file(restore=True)
            else:
                shared_logger = None

            # Run initial preprocessing steps
            raw, _ = preprocess_initial_steps(
                                        raw,
                                        drop_electrodes=drop_electrodes,
                                        picks=picks,
                                        crop_times=crop_times,
                                        crop_from_beginning=crop_from_beginning,
                                        crop_from_end=crop_from_end,
                                        resample_freq=resample_freq,
                                        stim_channels_to_annotations=stim_channels_to_annotations,
                                        montage=montage,
                                        l_freq=l_freq_initial,
                                        h_freq=h_freq_initial,
                                        line_noise_freq=line_noise_freq_initial,
                                        n_jobs=n_jobs,
                                        create_report=save_report,
                                        save_report=False,
                                        save_cfg=True,
                                        save_log=False,
                                        output_dir=output_dir,
                                        file_name=file_name,
                                        report=shared_report,
                                        logger=shared_logger,
                                        )
            
            # Run APICE ICA correction (remove big artifacted segments and bad channels before ICA fitting)
            if apply_ica:
                raw, _, _ = clean_ica(raw, 
                                      **ica_parameters,
                                        create_report=True,
                                        save_data=False,
                                        save_report=False,
                                        save_ica=True,
                                        save_cfg=True,
                                        save_log=False,
                                        output_dir=output_dir,
                                        file_name=file_name,
                                        report=shared_report,
                                        logger=shared_logger,
                                        )

            # Run APICE default preprocessing pipeline
            was_interactive = plt.isinteractive()
            original_backend = plt.get_backend()
            if not show_figures:
                plt.close('all')
                if original_backend.lower() != 'agg':
                    plt.switch_backend('Agg')
                plt.ioff()
            try:
                _ = preprocess_apice_default(
                                            raw,
                                            file_name=file_name,
                                            output_dir=output_dir,
                                            create_report=save_report,
                                            save_log=False,
                                            save_data=True,
                                            save_report=False,
                                            save_summary=save_summary,
                                            reference_channels=reference_channels,
                                            l_freq=l_freq_apice,
                                            h_freq=h_freq_apice,
                                            line_noise_freq=line_noise_freq_apice,
                                            l_freq_artifacts=l_freq_artifacts,
                                            h_freq_artifacts=h_freq_artifacts,
                                            cfg_bad_channels_detection=cfg_bad_channels_detection,
                                            cfg_glitches_detection=cfg_glitches_detection,
                                            cfg_target_pca=cfg_target_pca,
                                            cfg_artifacts_detection=cfg_artifacts_detection,
                                            cfg_spline_segments=cfg_spline_segments,
                                            cfg_spline_channels=cfg_spline_channels,
                                            n_jobs=n_jobs,
                                            report=shared_report,
                                            logger=shared_logger,
                                            )

            finally:
                plt.close('all')
                if not show_figures and original_backend.lower() != 'agg':
                    plt.switch_backend(original_backend)
                if not show_figures and was_interactive:
                    plt.ion()

            # Save the shared report and close the shared logger
            if save_report and shared_report is not None:
                output_dir_reports = output_dir / "reports"
                output_dir_reports.mkdir(exist_ok=True)
                report_path = output_dir_reports / f"{file_name}-preproc-report.html"
                print(f"Saving unified report")
                shared_report.save(fname=report_path, open_browser=False, overwrite=True)
            if save_log and shared_logger is not None:
                shared_logger.close()
            
        except Exception as e:
            print(f"Error processing file {file}: {e}")
            if save_log and 'shared_logger' in dir() and shared_logger is not None:
                try:
                    shared_logger.close()
                except Exception:
                    pass
            continue
            

def run_segmentation(input_dir, 
                     output_dir, 
                     kwargs_events_from_annotations_for_segmentation, 
                     event_time_window,
                     processed_file_pattern='*-epo.fif',
                     data_selection_method="all",
                     l_freq=None,
                     h_freq=None,
                     baseline=None, 
                     kwargs_events_from_annotations_for_metadata=None,
                     kwargs_make_metadata=None,                             
                     evoked_by=True,
                     save_log=True,
                     save_epochs=True,
                     save_only_good_epochs=False,
                     save_evoked=True,
                     save_report=True,
                     save_summary=True,
                     show_figures=False,
                     save_cfg=True,
                     set_reference=None,
                     cfg_define_bcbt_epochs=None,
                     cfg_spline_channels=None,  
                     cfg_bad_epochs=None,              
                     n_jobs=-1,
                     ):
    """Run the default APICE segmentation workflow over multiple files.

    Parameters
    ----------
    input_dir : str | pathlib.Path
        Directory containing preprocessed raw FIF files.
    output_dir : str | pathlib.Path
        Output folder for epoch files, ERP files, reports, and summaries.
    kwargs_events_from_annotations_for_segmentation : dict
        Keyword arguments passed to ``mne.events_from_annotations`` for epoching.
    event_time_window : tuple[float, float]
        Epoch time window ``(tmin, tmax)`` in seconds.
    processed_file_pattern : str, default='*-epo.fif'
        Pattern used to detect existing segmentation outputs.
    data_selection_method : str | list[str], default='all'
        File-selection strategy forwarded to APICE I/O helpers.
    l_freq : float | None, default=None
        Optional high-pass cutoff used before segmentation.
    h_freq : float | None, default=None
        Optional low-pass cutoff used before segmentation.
    baseline : tuple[float, float] | None, default=None
        Baseline window passed to MNE epoching.
    kwargs_events_from_annotations_for_metadata : dict | None, default=None
        Keyword arguments used to derive metadata events.
    kwargs_make_metadata : dict | None, default=None
        Keyword arguments used when creating epoch metadata.
    evoked_by : bool | str | list[str], default=True
        Event grouping used when computing evoked responses.
    save_log : bool, default=True
        If True, write one segmentation log per processed recording.
    save_epochs : bool, default=True
        If True, export epochs to FIF.
    save_only_good_epochs : bool, default=False
        If True, export only epochs that survive rejection.
    save_evoked : bool, default=True
        If True, export evoked responses.
    save_report : bool, default=True
        If True, write an HTML segmentation report.
    save_summary : bool, default=True
        If True, write a segmentation summary CSV.
    show_figures : bool, default=False
        If False, disable interactive Matplotlib rendering while each file is
        processed. This prevents figure windows from popping up during
        multi-file runs while still allowing report figures to be created.
    save_cfg : bool, default=True
        If True, save the effective segmentation configurations to JSON.
    set_reference : dict | None, default=None
        Reference parameters passed to ``epochs.set_eeg_reference``.
    cfg_define_bcbt_epochs : None | str | pathlib.Path | dict, default=None
        Configuration source for epoch-level BC/BT derivation.
    cfg_spline_channels : None | str | pathlib.Path | dict, default=None
        Configuration source for spline interpolation of epochs.
    cfg_bad_epochs : None | str | pathlib.Path | dict, default=None
        Configuration source for bad-epoch detection.
    n_jobs : int, default=-1
        Number of parallel jobs for compute-intensive steps.

    Returns
    -------
    None
        Writes segmentation outputs for each selected file.
    """

    # Initialize output folders
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Get all files to process
    from apice.io import get_files_to_process
    files = get_files_to_process(input_dir=input_dir,
                                output_dir=output_dir,
                                processed_file_pattern=processed_file_pattern,
                                data_selection_method=data_selection_method,
                                )

    print(f"\nNumber of files to process: {len(files)}\n")

    for file in files:

        print(f"Processing file: {file}")

        try:
        
            # Load raw data
            raw = load_rawapice(file)

            # Run segmentation pipeline
            was_interactive = plt.isinteractive()
            original_backend = plt.get_backend()
            if not show_figures:
                plt.close('all')
                if original_backend.lower() != 'agg':
                    plt.switch_backend('Agg')
                plt.ioff()
            try:
                _ = segment_default_pipeline(raw, 
                            kwargs_events_from_annotations_for_segmentation, 
                            event_time_window,
                            file_name=file.stem,
                            l_freq=l_freq,
                            h_freq=h_freq,
                            baseline=baseline, 
                            kwargs_events_from_annotations_for_metadata=kwargs_events_from_annotations_for_metadata,
                            kwargs_make_metadata=kwargs_make_metadata,                             
                            evoked_by=evoked_by,
                            output_dir=output_dir,
                            save_log=save_log,
                            save_epochs=save_epochs,
                            save_only_good_epochs=save_only_good_epochs,
                            save_evoked=save_evoked,
                            save_report=save_report,
                            save_summary=save_summary,
                            save_cfg=save_cfg,
                            set_reference=set_reference,
                            cfg_define_bcbt_epochs=cfg_define_bcbt_epochs,
                            cfg_spline_channels=cfg_spline_channels,  
                            cfg_bad_epochs=cfg_bad_epochs,              
                            n_jobs=n_jobs,
                            )
            finally:
                plt.close('all')
                if not show_figures and original_backend.lower() != 'agg':
                    plt.switch_backend(original_backend)
                if not show_figures and was_interactive:
                    plt.ion()
            
        except Exception as e:
            print(f"Error processing file {file}: {e}")
            continue


# %% Sub Functions


def preprocess_initial_steps(raw,
                          drop_electrodes=None,
                          picks='eeg',
                          crop_times=None,
                          crop_from_beginning=None,
                          crop_from_end=None,
                          resample_freq=None,
                          stim_channels_to_annotations=True,
                          montage=None,
                          head_size=None,
                          l_freq=0.10,
                          h_freq=40,
                          line_noise_freq=None,
                          n_jobs=-1,
                          create_report=False,
                          save_report=False,
                          save_log=False,
                          save_data=False,
                          save_cfg=False,
                          preprocessed_data_suffix='-initial-raw',
                          output_dir=None,
                          file_name=None,
                          report=None,
                          logger=None,
                         ):
    """Apply structural preprocessing steps before APICE artifact handling.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Raw object to process.
    drop_electrodes : list[str] | None, default=None
        Channels to drop before further processing.
    picks : str | list | slice, default='eeg'
        Channels to keep using MNE picking semantics.
    crop_times : tuple[float, float] | None, default=None
        Optional ``(tmin, tmax)`` crop window in seconds.
    crop_from_beginning : float | None, default=None
        Seconds to trim from the beginning of the recording.
    crop_from_end : float | None, default=None
        Seconds to trim from the end of the recording.
    resample_freq : float | None, default=None
        Target sampling frequency in Hz.
    stim_channels_to_annotations : bool, default=True
        If True, convert stim-channel events to annotations.
    montage : mne.channels.DigMontage | str | pathlib.Path | None, default=None
        Montage object, built-in montage name, or path to a montage file.
    head_size : float | None, default=None
        Head-size parameter forwarded to MNE montage creation when relevant.
    l_freq : float, default=0.10
        High-pass cutoff frequency in Hz.
    h_freq : float | None, default=40
        Low-pass cutoff frequency in Hz.
    line_noise_freq : float | None, default=None
        Line noise frequency to be removed.
    n_jobs : int, default=-1
        Number of parallel jobs for compute-intensive steps.
    create_report : bool, default=False
        If True, build an in-memory MNE report with raw overview, PSD before
        and after filtering, and the ZapLine figure (if applicable).
    save_report : bool, default=False
        If True and *this function owns the report* (i.e. ``report`` was not
        injected), save the report to
        ``output_dir/reports/{file_name}-initial-report.html``.
    save_log : bool, default=False
        If True and *this function owns the logger* (i.e. ``logger`` was not
        injected), redirect ``sys.stdout`` to
        ``output_dir/logs/{file_name}_log.txt``.
    save_data : bool, default=False
        If True, export the filtered raw object to
        ``output_dir/{file_name}{preprocessed_data_suffix}.fif``.
    save_cfg : bool, default=False
        If True, save the effective configuration of this function to
        ``output_dir/{file_name}_cfg.json``.
    preprocessed_data_suffix : str, default='-initial-raw'
        Suffix appended to ``file_name`` when saving the raw file.
    output_dir : str | pathlib.Path | None, default=None
        Destination folder.  Required when ``save_report``, ``save_log``, or
        ``save_data`` is True and the corresponding object is not injected.
    file_name : str | None, default=None
        Base name for output files.  Required when ``save_report``,
        ``save_log``, or ``save_data`` is True and the corresponding object is
        not injected.
    report : mne.Report | None, default=None
        An existing report to add figures to.  When provided the function will
        add its content but will **not** save the report — saving is the
        caller's responsibility.
    logger : StdOutLogger | None, default=None
        An existing logger whose stdout redirect is already active.  When
        provided the function will not create, redirect, or close a new logger.

    Returns
    -------
    raw : mne.io.BaseRaw
        Updated raw object after structural preprocessing.
    report : mne.Report | None
        The report built (or extended) during this call.  ``None`` when both
        ``create_report`` is False and no ``report`` was injected.
    """

    # INITIALIZATION -----------------------------------------------------------------------------------------------

    # Check if raw is an instance of mne.io.Raw
    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError("raw must be an instance of mne.io.Raw")

    # Determine ownership of report and logger
    _owns_report = (report is None) and create_report
    _owns_logger = (logger is None) and save_log

    # Validate output_dir / file_name when saving is requested by this function
    if _owns_report and save_report and (output_dir is None or file_name is None):
        raise ValueError("output_dir and file_name must be provided when save_report=True")
    if _owns_logger and (output_dir is None or file_name is None):
        raise ValueError("output_dir and file_name must be provided when save_log=True")
    if save_data and (output_dir is None or file_name is None):
        raise ValueError("output_dir and file_name must be provided when save_data=True")

    # Create output directories if needed
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        if _owns_report and save_report:
            output_dir_reports = output_dir / "reports"
            output_dir_reports.mkdir(exist_ok=True)
        if _owns_logger:
            output_dir_logs = output_dir / "logs"
            output_dir_logs.mkdir(exist_ok=True)
        if save_cfg:
            output_dir_cfg = output_dir / "cfgs"
            output_dir_cfg.mkdir(exist_ok=True)

    # Initialise the logger owned by this function
    if _owns_logger:
        logger = StdOutLogger(output_dir_logs, file_name)

    # Initialise the report owned by this function
    if _owns_report:
        report = mne.Report(title=file_name)

    # Start logging
    if _owns_logger:
        logger.redirect_stdout_to_file(restore=True)

    # Preprocessing start time
    sim_time_start = time.time()
    print('=============================================')
    print('Starting initial preprocessing steps\n')
    print(f"Processing date and time: {datetime.now()}\n\n")

    if raw.info['meas_date'] is None:
        print("raw.info['meas_date'] is None. Setting it to the current date and time.")
        raw.set_meas_date(datetime.now(tz=timezone.utc))

    # STIM CHANNELS TO ANNOTATIONS ------------------------------------------------------------------------------------------
    if stim_channels_to_annotations and "stim" in np.unique(raw.get_channel_types()):
        convert_stim_channels_to_annotations(raw)

    # DROPPING ELECTRODES -----------------------------------------------------------------------------------------------
    if drop_electrodes is not None:
        raw.drop_channels(list(drop_electrodes), on_missing='warn')

    # PICKING ELECTRODES -----------------------------------------------------------------------------------------------
    if picks is not None:
        raw.pick(picks)

    # CROP TIMES -----------------------------------------------------------------------------------------------
    if crop_times is not None:
        raw.crop(tmin=crop_times[0], tmax=crop_times[1])
    if crop_from_beginning is not None:
            raw.crop(tmin=crop_from_beginning, tmax=None)
    if crop_from_end is not None:
            raw.crop(tmin=0, tmax=raw.times[-1]-crop_from_end)    

    # RESAMPLE -----------------------------------------------------------------------------------------------
    if resample_freq is not None:
        raw.resample(resample_freq)

    # SET THE MONTAGE -----------------------------------------------------------------------------------------------
    if montage is not None:
        if isinstance(montage, mne.channels.DigMontage):
            raw.set_montage(montage)
        elif isinstance(montage, (str, Path)):
            if Path(montage).exists() and Path(montage).suffix is not None:
                try:
                    montage = mne.channels.read_custom_montage(montage, head_size=head_size)
                    raw.set_montage(montage)
                except Exception as e:
                    raise ValueError(f"Could not read montage from {montage}: {e}")
            elif montage in mne.channels.get_builtin_montages():
                montage = mne.channels.make_standard_montage(montage, head_size=head_size)
                raw.set_montage(montage)
            else:
                raise ValueError(f"Montage {montage} not recognized as a built-in montage or a valid file path.")
    
    if raw.get_montage() is None:
        raise ValueError("raw must have a montage. Please set the montage before preprocessing.")
    

    # FILTER -----------------------------------------------------------------------------------------------

    # High pass filter to remove the slow drifts
    Filter(raw, l_freq=l_freq, h_freq=None, n_jobs=n_jobs)    

    # Optional line noise filter
    if line_noise_freq is not None: 
        zap_worker = ZapLine(raw, fline=line_noise_freq, chunk_duration=30, n_jobs=n_jobs)
        raw, zap_fig = zap_worker.apply(raw)
        
    # Low pass filter
    Filter(raw, l_freq=None, h_freq=h_freq, n_jobs=n_jobs)


    # REPORT -----------------------------------------------------------------------------------------------
    if report is not None:
        try:
            report.add_raw(raw, title="Preprocess Initial Steps - Output Data", psd=False, butterfly=False, replace=True)
        except Exception as e:
            print(f"Warning: Could not add raw overview to report: {e}")
        try:
            report.add_figure(zap_fig, "ZapLine Spectral Power",
                                section="Preprocess Initial Steps - Output Data", replace=True)
            plt.close(zap_fig)
        except Exception as e:
            print(f"Warning: Could not add ZapLine figure to report: {e}")
        try:
            fmax_post = raw.info['sfreq'] / 2
            fig_psd_post = raw.compute_psd(method='welch', fmax=fmax_post).plot(show=False)
            report.add_figure(fig_psd_post, "PSD — After Filtering",
                              section="Preprocess Initial Steps - Output Data", replace=True)
            plt.close(fig_psd_post)
        except Exception as e:
            print(f"Warning: Could not add post-filter PSD to report: {e}")


    # END TIME ------------------------------------------------------------------------------------------------
    sim_time_end = timedelta(seconds=np.round(time.time() - sim_time_start))
    print(f"\nInitial preprocessing steps completed in: {sim_time_end}, in hh:mm:ss")
    print('=============================================\n')

    # Save raw data
    if save_data:
        print("Saving initial preprocessed raw data")
        full_path = output_dir / (file_name + preprocessed_data_suffix + ".fif")
        raw.export(full_path, overwrite=True)

    # Save configuration
    if save_cfg:
        cfg = {
            "drop_electrodes": drop_electrodes,
            "picks": picks,
            "crop_times": crop_times,
            "crop_from_beginning": crop_from_beginning,
            "crop_from_end": crop_from_end,
            "resample_freq": resample_freq,
            "stim_channels_to_annotations": stim_channels_to_annotations,
            "montage": montage if not isinstance(montage, mne.channels.DigMontage) else "custom_dig_montage",
            "head_size": head_size,
            "l_freq": l_freq,
            "h_freq": h_freq,
            "line_noise_freq": line_noise_freq,
        }
        with open(output_dir_cfg / f"{file_name}_initial_preprocessing_cfg.json", 'w') as f:
            json.dump(cfg, f, indent=4)

    # Save report and close logger if owned by this function
    if _owns_report and save_report:
        report.save(output_dir_reports / f"{file_name}-initial-report.html",
                    open_browser=False, overwrite=True)
    if _owns_logger:
        logger.close()

    return raw, report
    

def preprocess_apice_default(raw, 
                             preprocessed_data_suffix='-preproc-raw',
                             output_dir=None,
                             file_name=None,
                             create_report=True,
                             save_log=True,
                             save_data=True,
                             save_report=True,
                             save_summary=True,
                             save_cfg=True,
                             reference_channels=None,
                             l_freq=0.10,
                             h_freq=40,
                             l_freq_artifacts=None,
                             h_freq_artifacts=None,
                             line_noise_freq=None,
                             cfg_define_bcbt_raw=None,
                             cfg_bad_channels_detection=None,
                             cfg_glitches_detection=None,
                             cfg_target_pca=None,
                             cfg_artifacts_detection=None,
                             cfg_spline_segments=None,
                             cfg_spline_channels=None,
                             n_jobs=-1,
                             report=None,
                             logger=None,
                             ):
    """Run the default APICE artifact-detection and correction pipeline.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Raw object to preprocess. It must already contain a montage.
    preprocessed_data_suffix : str, default='-preproc-raw'
        Suffix appended to exported preprocessed raw files.
    output_dir : str | pathlib.Path | None, default=None
        Destination folder for exported artifacts, reports, logs, and summaries.
    file_name : str | None, default=None
        Base name used for output files.
    create_report : bool, default=True
        If True, build an in-memory MNE report during processing.
    save_log : bool, default=True
        If True, write pipeline logs to disk.
    save_data : bool, default=True
        If True, export the preprocessed raw file.
    save_report : bool, default=True
        If True, save the generated HTML report.
    save_summary : bool, default=True
        If True, save preprocessing summary metrics.
    save_cfg : bool, default=True
        If True, save the effective preprocessing configurations to JSON.
    reference_channels : list[str] | str | None, default=None
        Reference channels to preserve during bad-channel handling.
    l_freq : float, default=0.10
        High-pass cutoff frequency in Hz.
    h_freq : float | None, default=40
        Low-pass cutoff frequency in Hz.
    l_freq_artifacts : float, default=None
        High-pass cutoff frequency used for artifact detection.
    h_freq_artifacts : float, default=None
        Low-pass cutoff frequency used for artifact detection.
    line_noise_freq : float | None, default=None
        Line noise frequency to be removed.
    cfg_define_bcbt_raw : None | str | pathlib.Path | dict, default=None
        Configuration source for deriving BC/BT masks from raw BCT masks.
    cfg_bad_channels_detection : None | str | pathlib.Path | dict, default=None
        Configuration source for bad-channel detection.
    cfg_glitches_detection : None | str | pathlib.Path | dict, default=None
        Configuration source for glitch detection.
    cfg_target_pca : None | str | pathlib.Path | dict, default=None
        Configuration source for target-PCA correction.
    cfg_artifacts_detection : None | str | pathlib.Path | dict, default=None
        Configuration source for artifact detection.
    cfg_spline_segments : None | str | pathlib.Path | dict, default=None
        Configuration source for segment-wise spline correction.
    cfg_spline_channels : None | str | pathlib.Path | dict, default=None
        Configuration source for channel-wise spline correction.
    n_jobs : int, default=-1
        Number of parallel jobs for compute-intensive steps.
    report : mne.Report | None, default=None
        An existing report to add figures to.  When provided the function will
        add its content but will **not** save the report — saving is the
        caller's responsibility.
    logger : StdOutLogger | None, default=None
        An existing logger whose stdout redirect is already active.  When
        provided the function will not create, redirect, or close a new logger.

    Returns
    -------
    raw : RawAPICE
        Preprocessed raw object with artifact masks and corrections applied.
    summary : SummaryPreprocessing
        Summary object tracking preprocessing metrics.
    report : mne.Report | None
        Generated report, or None when ``create_report`` is False and no
        ``report`` was injected.
    """

        
    # INITIALIZATION -----------------------------------------------------------------------------------------------

    # Check if raw is an instance of mne.io.Raw
    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError("raw must be an instance of mne.io.Raw")
    
    # Check that raw has a montage
    if raw.get_montage() is None:
        raise ValueError("raw must have a montage. Please set the montage before preprocessing.")

    # Determine ownership of report and logger
    _owns_report = (report is None) and create_report
    _owns_logger = (logger is None) and save_log

    # Check that output_dir is provided if any of the saving options is True
    if output_dir is None and (save_data or save_summary or (_owns_report and save_report) or _owns_logger):
        raise ValueError("output_dir must be provided if any of the saving options is True")
    
    # Check that file_name is provided if any of the saving options is True, to use as part of the file name for the saved files
    if file_name is None and (save_data or save_summary or (_owns_report and save_report) or _owns_logger):
        raise ValueError("file_name must be provided if any of the saving options is True, to use as part of the file name for the saved files")
    
    # Check the filter frequencies
    if l_freq is None:
        l_freq=0.1
        print(f"Warning: High pass frequency not provided. Using default value: {l_freq} Hz")
        
    if isinstance(reference_channels, str):
        reference_channels = [reference_channels]
    # check that reference channels are in raw
    if reference_channels is not None:
        for ch in reference_channels:
            if ch not in raw.ch_names:
                print(f"Warning:Reference channel {ch} not found in raw data channels")

    # Get default configurations if not provided or load it if provided as path to a json file
    cfg_define_bcbt_raw = get_cfg(cfg_define_bcbt_raw, 'define_bcbt_raw_config.json')
    cfg_bad_channels_detection = get_cfg(cfg_bad_channels_detection, 'detect_bad_channels_config.json')
    cfg_glitches_detection = get_cfg(cfg_glitches_detection, 'detect_artifacts_glitches_config.json')
    cfg_target_pca = get_cfg(cfg_target_pca, 'correction_target_pca_config.json')
    cfg_artifacts_detection = get_cfg(cfg_artifacts_detection, 'detect_artifacts_motion_config.json')
    cfg_spline_segments = get_cfg(cfg_spline_segments, 'correction_spline_segments_config.json')
    cfg_spline_channels = get_cfg(cfg_spline_channels, 'correction_spline_channels_config.json')    
    
    # for each configuration for correction set n_jobs to the value provided in the function argument
    cfg_spline_segments['n_jobs'] = n_jobs
    cfg_spline_channels['n_jobs'] = n_jobs
    
    # Create output folder if it does not exist
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        if save_summary:
            output_dir_summary = output_dir / "summary"
            output_dir_summary.mkdir(exist_ok=True)
        if _owns_report and save_report:
            output_dir_reports = output_dir / "reports"
            output_dir_reports.mkdir(exist_ok=True)
        if save_cfg:
            output_dir_cfgs = output_dir / "cfgs"
            output_dir_cfgs.mkdir(exist_ok=True)
        if _owns_logger:
            output_dir_logs = output_dir / "logs"
            output_dir_logs.mkdir(exist_ok=True)
    
    # Initialize logger owned by this function
    if _owns_logger:
        logger = StdOutLogger(output_dir_logs, file_name)
        
    # Preprocessing start time
    sim_time_start = time.time()
    print('=============================================')
    print('Starting APICE default preprocessing pipeline')
    print(f"Processing date and time: {datetime.now()}\n\n")

    # Initialize object tracking the summary of artifacts
    summary = SummaryPreprocessing(output_dir_summary, file_name, try_loading=False)

    # Initialize report owned by this function
    if _owns_report:
        report = mne.Report(title=file_name)

    # Start logging (only when this function owns it)
    if _owns_logger: logger.redirect_stdout_to_file(restore=True)

    # FILTER -----------------------------------------------------------------------------------------------
    
    # High pass filter to remove the slow drifts
    Filter(raw, l_freq=l_freq, h_freq=None, n_jobs=n_jobs)    

    # Optional line noise filter
    if line_noise_freq is not None: 
        zap_worker = ZapLine(raw, fline=line_noise_freq, chunk_duration=30, n_jobs=n_jobs)
        raw, zap_fig = zap_worker.apply(raw)
        if report is not None:
            report.add_figure(zap_fig, "ZapLine Spectral Power", section="Raw Data", replace=True)
            plt.close(zap_fig)

    # Low pass filter
    Filter(raw, l_freq=None, h_freq=h_freq, n_jobs=n_jobs)


    # ARTIFACT DETECTION AND CORRECTION -----------------------------------------------------------------------------

    # Initialize artifacts structure
    raw = RawAPICE(raw, **cfg_define_bcbt_raw)

    if report is not None:
        try:
            report.add_raw(raw, 
                            title="Preprocess APICE - Raw Data", 
                            psd=False, 
                            butterfly=False, 
                            replace=True, 
                            )
        except Exception as e:
            print(f"Warning: Could not add raw data to report: {e}")
        
        # Add PSD
        if h_freq is None:
            fmax = raw.info['sfreq'] / 2
        else:
            fmax = h_freq
        if reference_channels is not None:
            exclude = reference_channels
        else:
            exclude = []
        try:
            fig = raw.compute_psd(method='welch',fmax=fmax,exclude=exclude).plot(show=False)
            report.add_figure(fig, "PSD", section="Preprocess APICE - Raw Data", replace=True)
        except Exception as e:
            print(f"Warning: Could not add raw PSD to report: {e}")

        # Add events
        try:
            events, event_id = mne.events_from_annotations(raw)
            if (len(events) > 0) and (len(events)<30):  # only add events to the report if there are events and if there are not too many events (otherwise the report gets too heavy and does not open properly)
                report.add_events(events=events, title='Events from "annotations"', sfreq=raw.info['sfreq'], section="Preprocess APICE - Raw Data", replace=True)
            else:                
                print(f"Warning: Not adding events to report because there are {len(events)} events, which is more than the threshold of 30 events.")
        except Exception as e:
            print(f"Warning: Could not add events to report: {e}")


    # Detect bad channels
    raw.detect_bad_channels(cfg=cfg_bad_channels_detection, l_freq=l_freq_artifacts, h_freq=h_freq_artifacts)
    raw.deal_with_reference_channels(reference_channels)
    summary.add_to_summary('artifacts_detection_BadElectrodes', raw, overwrite=True)

    # Detect glitches
    raw.detect_glitches(cfg=cfg_glitches_detection, l_freq=l_freq_artifacts, h_freq=h_freq_artifacts)
    raw.deal_with_reference_channels(reference_channels)
    summary.add_to_summary('artifacts_detection_Glitches', raw, overwrite=True)

    # Correct glitches
    raw.correct_target_pca(cfg=cfg_target_pca)
    Filter(raw, l_freq=l_freq, h_freq=None, n_jobs=n_jobs)
    summary.add_to_summary('artifacts_correction_TargetPCA', raw, overwrite=True)

    # Detect artifacts
    raw.detect_artifacts(cfg=cfg_artifacts_detection, l_freq=l_freq_artifacts, h_freq=h_freq_artifacts)
    raw.deal_with_reference_channels(reference_channels)
    summary.add_to_summary('artifacts_detection_Artifacts', raw, overwrite=True)

    # Create a figure to visualize the artifact structure
    if report is not None:
        try:
            fig_bct = raw.plot_artifact_structure(color_scheme='jet', artifact='BCT')    
            report.add_figure(fig_bct, "Artifacts Matrix (BCT)", section="Preprocess APICE - Raw Data", replace=True)
        except Exception as e:
            print(f"Warning: Could not add raw artifacts matrix to report: {e}")
        try:
            fig_all = raw.plot_artifact_structure(color_scheme='jet', artifact='all')    
            report.add_figure(fig_all, "Artifacts Matrix (All)", section="Preprocess APICE - Raw Data", replace=True)
        except Exception as e:
            print(f"Warning: Could not add raw artifacts matrix to report: {e}")

    # Add topomap of bad electrodes
    if report is not None:
        try:
            fig_bad = raw.plot_percentage_of_bad_data_across_sensors()
            report.add_figure(fig_bad, "Bad data across electrodes", section="Preprocess APICE - Raw Data", replace=True)
        except Exception as e:
            print(f"Warning: Could not add raw bad-data topomap to report: {e}")
    
    # Correct artifacts using spherical spline interpolation per segment
    raw.correct_spline_segments(cfg=cfg_spline_segments)
    Filter(raw, l_freq=l_freq, h_freq=None, n_jobs=n_jobs)
    summary.add_to_summary('artifacts_correction_Segments', raw, overwrite=True)

    # Correct channels using spherical spline interpolation
    raw.correct_spline_channels(cfg=cfg_spline_channels)
    summary.add_to_summary('artifacts_correction_BadChannels', raw, overwrite=True)

    # Re-detect bad data after correction to check if there are still bad channels or time segments that need to be marked as bad after the correction
    raw.detect_artifacts(cfg=cfg_artifacts_detection, l_freq=l_freq_artifacts, h_freq=h_freq_artifacts)
    raw.deal_with_reference_channels(reference_channels)    
    summary.add_to_summary('artifacts_detection_ArtifactsPostCorrection', raw, overwrite=True)

    if report is not None:
        try:
            report.add_raw(raw, 
                            title="Preprocess APICE - Preprocessed Data", 
                            psd=False, 
                            butterfly=True, 
                            scalings=50e-6, 
                            replace=True,
                            topomap_kwargs={"color_scheme": "jet"}
                            )
        except Exception as e:
            print(f"Warning: Could not add preprocessed raw data to report: {e}")
    
        # Add PSD
        if h_freq is None:
            fmax = raw.info['sfreq'] / 2
        else:
            fmax = h_freq
        if reference_channels is not None:
            exclude = reference_channels
        else:
            exclude = []
        try:
            fig = raw.compute_psd(method='welch',fmax=fmax,exclude=exclude).plot(show=False)
            report.add_figure(fig, "PSD", section="Preprocess APICE - Preprocessed Data", replace=True)
        except Exception as e:
            print(f"Warning: Could not add preprocessed PSD to report: {e}")
    
    # Create a figure to visualize the artifact structure
    if report is not None:
        try:
            fig_bct = raw.plot_artifact_structure(color_scheme='jet', artifact='BCT')    
            report.add_figure(fig_bct, "Artifacts Matrix (BCT)", section="Preprocess APICE - Preprocessed Data", replace=True)
        except Exception as e:
            print(f"Warning: Could not add raw artifacts matrix to report: {e}")
        try:
            fig_all = raw.plot_artifact_structure(color_scheme='jet', artifact='all')    
            report.add_figure(fig_all, "Artifacts Matrix (All)", section="Preprocess APICE - Preprocessed Data", replace=True)
        except Exception as e:
            print(f"Warning: Could not add raw artifacts matrix to report: {e}")

    # Add topomap of bad electrodes
    if report is not None:
        try:
            fig = raw.plot_percentage_of_bad_data_across_sensors()
            report.add_figure(fig, "Bad data across electrodes", section="Preprocess APICE - Preprocessed Data", replace=True)
        except Exception as e:
            print(f"Warning: Could not add preprocessed bad-data topomap to report: {e}")

    # Add a plot showing the amount of rejected data
    if report is not None:
        try:
            fig = plot_summary(summary.summary_df, metrics = ["%_corrected_data", "%_bad_data", "%_bad_channels", "%_bad_times"])
            report.add_figure(fig, "Rejected Data Summary", section="Preprocess APICE - Preprocessed Data", replace=True)
        except Exception as e:
            print(f"Warning: Could not add preprocessed rejected-data summary to report: {e}")

    # EXPORT DATA -----------------------------------------------------------------------------------------------
    
    # Save preprocessed raw
    if save_data:
        print("Saving preprocessed raw data")
        full_path = output_dir / (file_name + preprocessed_data_suffix + ".fif")
        raw.export(full_path, overwrite=True)

    # Save summary file
    if save_summary:
        print("Saving preprocessing summary")
        output_dir_summary.mkdir(exist_ok=True)
        summary.save()

    # Save report
    if _owns_report and save_report:
        output_dir_reports.mkdir(exist_ok=True)
        print("Saving report")
        full_path = output_dir_reports / (file_name + preprocessed_data_suffix + "-report.html")
        report.save(fname=full_path, open_browser=False, overwrite=True)

    # Save the configurations used for preprocessing
    if save_cfg:
        print("Saving preprocessing configurations")
        cfg_to_save = {
            "cfg_bad_channels_detection": cfg_bad_channels_detection,
            "cfg_glitches_detection": cfg_glitches_detection,
            "cfg_target_pca": cfg_target_pca,
            "cfg_artifacts_detection": cfg_artifacts_detection,
            "cfg_spline_segments": cfg_spline_segments,
            "cfg_spline_channels": cfg_spline_channels,
            }
        for cfg_name, cfg in cfg_to_save.items():
            with open(output_dir_cfgs / f"{file_name}_{cfg_name}.json", 'w') as f:
                json.dump(cfg, f, indent=4)

    # Preprocessing end time
    sim_time_end = timedelta(seconds=np.round(time.time() - sim_time_start))
    print(f"\nAPICE default preprocessing pipeline completed in: {sim_time_end}, in hh:mm:ss")
    print('=============================================\n')
    
    if _owns_logger:
        logger.close()

    return raw, summary, report


def segment_default_pipeline(raw, 
                   kwargs_events_from_annotations_for_segmentation, 
                   event_time_window,
                   epoch_data_suffix='-epo',
                   file_name=None,
                   l_freq=None,
                   h_freq=None,
                   baseline=None, 
                   kwargs_events_from_annotations_for_metadata=None,
                   kwargs_make_metadata=None,                             
                   evoked_by="all",
                   output_dir=None,
                   create_report=True,
                   save_log=True,
                   save_epochs=True,
                   save_only_good_epochs=False,
                   save_evoked=True,
                   save_report=True,
                   save_summary=True,
                   save_cfg=True,
                   set_reference=None,
                   cfg_define_bcbt_epochs=None,
                   cfg_spline_channels=None,  
                   cfg_bad_epochs=None,              
                   n_jobs=-1,
                   report=None,
                   logger=None,
                   ):
    """Run epoching, interpolation, bad-epoch marking, and ERP computation.

    Parameters
    ----------
    raw : RawAPICE
        Preprocessed raw object ready for segmentation.
    kwargs_events_from_annotations_for_segmentation : dict
        Keyword arguments passed to ``mne.events_from_annotations`` for epoching.
    event_time_window : tuple[float, float]
        Epoch time window ``(tmin, tmax)`` in seconds.
    file_name : str | None, default=None
        Base name used for exported files.
    l_freq : float | None, default=None
        Optional high-pass cutoff before segmentation.
    h_freq : float | None, default=None
        Optional low-pass cutoff before segmentation.
    baseline : tuple[float, float] | None, default=None
        Baseline correction window passed to MNE epoching.
    kwargs_events_from_annotations_for_metadata : dict | None, default=None
        Event selection kwargs used to build metadata.
    kwargs_make_metadata : dict | None, default=None
        Keyword arguments forwarded to metadata generation.
    evoked_by : str | list[str] | None, default='all'
        Grouping used when computing evoked responses.
    output_dir : str | pathlib.Path | None, default=None
        Destination folder for exported segmentation outputs.
    create_report : bool, default=True
        If True, build an in-memory MNE report during processing.
    save_log : bool, default=True
        If True, write pipeline logs to disk.
    save_epochs : bool, default=True
        If True, export epochs to FIF.
    save_only_good_epochs : bool, default=False
        If True, export only epochs that remain after rejection.
    save_evoked : bool, default=True
        If True, export evoked responses.
    save_report : bool, default=True
        If True, save the generated HTML report.
    save_summary : bool, default=True
        If True, save segmentation summary metrics.
    save_cfg : bool, default=True
        If True, save the effective segmentation configurations to JSON.
    set_reference : dict | None, default=None
        Reference parameters passed to ``epochs.set_eeg_reference``.
    cfg_define_bcbt_epochs : None | str | pathlib.Path | dict, default=None
        Configuration source for epoch-level BC/BT derivation.
    cfg_spline_channels : None | str | pathlib.Path | dict, default=None
        Configuration source for channel-wise spline interpolation.
    cfg_bad_epochs : None | str | pathlib.Path | dict, default=None
        Configuration source for bad-epoch detection.
    n_jobs : int, default=-1
        Number of parallel jobs for compute-intensive steps.
    report : mne.Report | None, default=None
        An existing report to add figures to.  When provided the function will
        add its content but will **not** save the report — saving is the
        caller's responsibility.
    logger : StdOutLogger | None, default=None
        An existing logger whose stdout redirect is already active.  When
        provided the function will not create, redirect, or close a new logger.

    Returns
    -------
    epochs : EpochsAPICE
        Segmented data with updated artifact masks.
    evokeds : mne.Evoked | dict | None
        Evoked response object(s) computed from good epochs.
    summary : SummaryEpochs
        Summary object tracking segmentation metrics.
    report : mne.Report | None
        Generated report, or None when ``create_report`` is False and no
        ``report`` was injected.
    """

    # INITIALIZATION -----------------------------------------------------------------------------------------------

    # Check if raw is an instance of mne.io.Raw
    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError("raw must be an instance of mne.io.Raw")
    
    # Check that raw is an instance of RawAPICE, which is required for the segmentation pipeline
    if not isinstance(raw, RawAPICE):
        raise TypeError("raw must be an instance of RawAPICE. Please run the preprocess_apice_default function before segmentation to ensure raw is an instance of RawAPICE and has the necessary attributes for segmentation.")    
    
    # Check that raw has a montage
    if raw.get_montage() is None:
        raise ValueError("raw must have a montage. Please set the montage before preprocessing.")

    # Determine ownership of report and logger
    _owns_report = (report is None) and create_report
    _owns_logger = (logger is None) and save_log

    # Check that output_dir is provided if any of the saving options is True
    if output_dir is None and (save_epochs or save_evoked or save_summary or (_owns_report and save_report) or _owns_logger):
        raise ValueError("output_dir must be provided if any of the saving options is True")

    # Check that file_name is provided if any of the saving options is True, to use as part of the file name for the saved files
    if file_name is None and (save_epochs or save_evoked or save_summary or (_owns_report and save_report) or _owns_logger):
        raise ValueError("file_name must be provided if any of the saving options is True, to use as part of the file name for the saved files")

    # Get the default configurations if not provided or load it if provided as path to a json file
    cfg_define_bcbt_epochs = get_cfg(cfg_define_bcbt_epochs, 'define_bcbt_epochs_config.json')
    cfg_spline_channels = get_cfg(cfg_spline_channels, 'correction_spline_channels_config.json')
    cfg_bad_epochs = get_cfg(cfg_bad_epochs, 'detect_bad_epochs_config.json')
    
    # for each configuration for correction set n_jobs to the value provided in the function argument
    cfg_spline_channels['n_jobs'] = n_jobs

    
    # Create output folder if it does not exist
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        if save_cfg:
            output_dir_cfgs = output_dir / "cfgs"
            output_dir_cfgs.mkdir(exist_ok=True)
        if _owns_report and save_report:
            output_dir_reports = output_dir / "reports"
            output_dir_reports.mkdir(exist_ok=True)
        if _owns_logger:
            output_dir_logs = output_dir / "logs"
            output_dir_logs.mkdir(exist_ok=True)
        if save_summary:
            output_dir_summary = output_dir / "summary"
            output_dir_summary.mkdir(exist_ok=True)

    # Initialize object tracking the summary of artifacts
    summary = SummaryEpochs(output_dir_summary, file_name, try_loading=False)

    # Initialize logger owned by this function
    if _owns_logger:
        logger = StdOutLogger(output_dir_logs, file_name)
        
    # Initialize report owned by this function
    if _owns_report:
        report = mne.Report(title=file_name)

    # Start logging (only when this function owns it)
    if _owns_logger: logger.redirect_stdout_to_file(restore=True)

    # Segmentation start time
    sim_time_start = time.time()
    print('=============================================')
    print('Starting APICE default segmentation pipeline')
    print(f"Segmentation date and time: {datetime.now()}\n\n")


    # FILTER -----------------------------------------------------------------------------------------------
    Filter(raw, l_freq=l_freq, h_freq=h_freq, n_jobs=n_jobs)

    # SEGMENTATION ----------------------------------------------------------------------------------------------
    
    # raw_=raw.copy()
    # raw_.annotate_bads(data=False, corrected=False)
    # raw_.plot(n_channels=raw_.info['nchan'], duration=100)
    
    if kwargs_events_from_annotations_for_metadata:
        if kwargs_make_metadata is None:
            kwargs_make_metadata = {}
        tmin_metadata = kwargs_make_metadata.get('tmin',event_time_window[0]) 
        tmax_metadata = kwargs_make_metadata.get('tmax',event_time_window[1]) 
        keep_first = kwargs_make_metadata.get('keep_first',None) 
        keep_last = kwargs_make_metadata.get('keep_last',None) 
        columns_events_to_keep = kwargs_make_metadata.get('columns_events_to_keep', None) 
        metadata = generate_metadata_for_epochs(raw, 
                                                kwargs_events_from_annotations_for_metadata=kwargs_events_from_annotations_for_metadata,
                                                kwargs_events_from_annotations_for_segmentation=kwargs_events_from_annotations_for_segmentation,
                                                tmin=tmin_metadata, 
                                                tmax=tmax_metadata, 
                                                keep_first=keep_first, 
                                                keep_last=keep_last,
                                                columns_events_to_keep=columns_events_to_keep,
                                                )
    else:
        metadata = None
    
    # Segment data into epochs, apply artifact correction, and compute evoked responses
    epochs, evokeds, summary = compute_epochs_and_evoked(raw, 
                                   kwargs_events_from_annotations=kwargs_events_from_annotations_for_segmentation,
                                   tmin=event_time_window[0], 
                                   tmax=event_time_window[1],
                                   baseline=baseline,
                                   metadata=metadata,
                                   evoked_by=evoked_by,
                                   set_reference=set_reference,
                                   summary=summary,
                                   cfg_define_bcbt_epochs=cfg_define_bcbt_epochs,
                                   cfg_spline_channels=cfg_spline_channels,
                                   cfg_bad_epochs=cfg_bad_epochs,
                                   )
    
    epochs_good = epochs.copy()
    epochs_good.remove_bad_epochs()

    # n_channels = epochs.info['nchan']
    # epoch_colors = [['black' for i in range(n_channels)] for i in range(len(epochs))]
    # for i in range(len(epochs)):
    #     if epochs.artifacts.BE[i]:
    #         epoch_colors[i] = ['red' for i in range(n_channels)]
    #     else:
    #         epoch_colors[i] = ['black' if not epochs.artifacts.BC[i][j,0] else 'blue' for j in range(n_channels)]
    # epochs.plot(epoch_colors=epoch_colors)

    # Add epochs artifacts matrix
    if report is not None:
        try:
            fig = epochs.plot_artifact_structure(color_scheme='jet')
            report.add_figure(fig, "Artifacts Matrix", section="Epochs All", replace=True)
        except Exception as e:
            print(f"Warning: Could not add epochs artifacts matrix to report: {e}")

    # Add epoch rejection summary (all epochs, with rejection reasons)
    if report is not None:
        try:
            fig = epochs.plot_rejection_summary()
            report.add_figure(fig, "Rejection Summary", section="Epochs All", replace=True)
        except Exception as e:
            print(f"Warning: Could not add epoch rejection summary to report: {e}")

    # Add epochs in report
    if report is not None:
        try:
            report.add_epochs(epochs_good, "Epochs Good", psd=False, replace=True)
        except Exception as e:
            print(f"Warning: Could not add epochs to report: {e}")
    
    # Add topomap of bad electrodes
    if report is not None:
        try:
            fig = epochs_good.plot_percentage_of_bad_data_across_sensors()
            report.add_figure(fig, "Bad data across electrodes", section="Epochs Good", replace=True)
        except Exception as e:
            print(f"Warning: Could not add epochs bad-data topomap to report: {e}")

    # Add a plot showing the amount of rejected data
    if report is not None:
        try:
            fig = plot_summary(summary.summary_df, metrics = ["%_corrected_data", "%_bad_data", "%_bad_channels", "%_bad_times", "%_bad_epochs"])
            report.add_figure(fig, "Rejected Data Summary", section="Epochs Good", replace=True)
        except Exception as e:
            print(f"Warning: Could not add rejected data summary to report: {e}")

    # Add evokeds in the report
    if report is not None:
        if evoked_by is None:
            evokeds_to_add = []
        elif evoked_by == "all":
            evokeds_to_add = dict(all=evokeds)
        else:                
            evokeds_to_add = evokeds
        for key, evoked in evokeds_to_add.items():
            try:
                report.add_evokeds(evoked, titles=key, replace=True)
            except Exception as e:
                print(f"Warning: Could not add evoked responses {key} to report: {e}")
                try:
                    fig = evoked.plot(show=False)
                    report.add_figure(fig, f"Evoked responses {key}", section="Evoked responses", replace=True)
                except Exception as fig_err:
                    print(f"Warning: Could not add fallback evoked figure {key} to report: {fig_err}")


    # EXPORT DATA -----------------------------------------------------------------------------------------------
    
    # Save epochs 
    if save_epochs:
        fullpath_epochs = output_dir / (file_name + epoch_data_suffix + '.fif')
        if save_only_good_epochs:
            epochs_good.export(fullpath_epochs, overwrite=True)
        else:
            epochs.export(fullpath_epochs, overwrite=True)

    # Save evoked responses
    if save_evoked:

        if evoked_by is None:
            print(f"No evoked responses to save for {file_name} since evoked_by is set to None.")
        elif evoked_by == "all":
            file_name_evoked = (file_name + '-ave.fif')
            folder_path = output_dir / 'erp'
            folder_path.mkdir(parents=True, exist_ok=True)
            full_path = folder_path / file_name_evoked
            print(f"Writing {full_path}")
            evokeds.save(full_path, overwrite=True)
            print(f"Closing {full_path}")
        else:
            for key, ev in evokeds.items():
                folder_path = output_dir / 'erp'
                folder_path.mkdir(parents=True, exist_ok=True)
                # make the key name a valid file name by removing or replacing characters that are not allowed in file names
                key_ = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in key)
                full_path = folder_path / f"{file_name}_{key_}-ave.fif"
                print(f"Writing {full_path}")
                ev.save(full_path, overwrite=True)
                print(f"Closing {full_path}")
            print('[done]')

    # Save summary file
    if save_summary:
        output_dir_summary.mkdir(exist_ok=True)
        summary.save()

    # Save report
    if _owns_report and save_report:
        output_dir_reports.mkdir(exist_ok=True)
        print("Saving report")
        full_path = output_dir_reports / (file_name + "-epo.html")
        report.save(fname=full_path, open_browser=False, overwrite=True)

    # Save configurations used for segmentation
    if save_cfg:    
        cfg_to_save = {
            "cfg_define_bcbt_epochs": cfg_define_bcbt_epochs,
            "cfg_spline_channels": cfg_spline_channels,
            "cfg_bad_epochs": cfg_bad_epochs,
            }
        for cfg_name, cfg in cfg_to_save.items():
            with open(output_dir_cfgs / f"{file_name}_{cfg_name}.json", 'w') as f:
                json.dump(cfg, f, indent=4)
    
    # Preprocessing end time
    sim_time_end = timedelta(seconds=np.round(time.time() - sim_time_start))
    print(f"\nAPICE default segmentation pipeline completed in: {sim_time_end}, in hh:mm:ss")
    print('=============================================\n')
    
    if _owns_logger:
        logger.close()

    return epochs, evokeds, summary, report


def get_stim_duration(raw, threshold=0.01):
    """Estimate a default annotation duration from stim channels.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Raw object containing one or more stim channels.
    threshold : float, default=0.01
        Minimum stim amplitude considered active.

    Returns
    -------
    duration : float
        Maximum detected stim duration in seconds.
    """

    stim_data = raw.copy().pick('stim').get_data()    
    above_baseline = np.where(stim_data > threshold, 1, 0)
    onsets, durations = get_onset_and_duration(above_baseline)

    return np.max(durations)
    
def convert_stim_channels_to_annotations(raw):
    """Convert stim-channel events into MNE annotations.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Raw object whose stim channels will be translated into annotations.

    Returns
    -------
    None
        Updates ``raw.annotations`` in place.
    """
    # Get a copy of the original annotations
    df_annotations = raw.annotations.to_data_frame(time_format=None)
    
    # Convert stim channels to annotations
    print("\nConverting STIMs to annotations...")
    
    stims = raw.copy().pick('stim')
    
    if stims:

        # Detect all events
        from mne import find_events
        events = find_events(raw, stim_channel=stims.ch_names, verbose=False)
        
        # Assuming event IDs are directly the data values in the stim channel
        event_ids = np.unique(events[:, 2])  # Unique event identifiers
        
        onsets = events[:, 0] / raw.info['sfreq']  # Convert sample indices to times
        
        ch_names = [()] * len(events)
        
        durations = [get_stim_duration(raw)] * len(events) 
        
        event_map = {event_id: f"{stims.ch_names[event_id - 1]}" for event_id in event_ids if event_id < len(raw.ch_names)}
        
        descriptions = [event_map[event_id] for event_id in events[:, 2]]

        # Create Annotations object
        df_annotations_from_stims = pd.DataFrame(dict(onset=onsets, duration=durations, description=descriptions, ch_names=ch_names))
        
        # List of DataFrames to combine
        dfs = [df_annotations, df_annotations_from_stims]
        
        # Filter out empty DataFrames
        dfs = [df for df in dfs if not df.empty]
        
        if dfs:
            # Concatenating DataFrames
            df_combined = pd.concat(dfs, ignore_index=True)

            # Dropping duplicates and sorting by 'onset'
            df_final = df_combined.drop_duplicates().sort_values(by="onset").reset_index(drop=True)

            from mne import Annotations 
            annotations = Annotations(onset=list(df_final["onset"]),
                                        duration=list(df_final["duration"]),
                                        description=list(df_final["description"]),
                                        ch_names=list(df_final["ch_names"]))
            
            raw.set_annotations(annotations)
    

def generate_metadata_for_epochs(raw, 
                 kwargs_events_from_annotations_for_metadata={},
                 kwargs_events_from_annotations_for_segmentation={},
                 columns_events_to_keep=None,
                 tmin=-0.5, 
                 tmax=0.5, 
                 keep_first=None, 
                 keep_last=None,
                 ):
    """Generate an epoch metadata dataframe from raw annotations.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Raw object providing annotations and sampling information.
    kwargs_events_from_annotations_for_metadata : dict, default={}
        Keyword arguments used to derive metadata events.
    kwargs_events_from_annotations_for_segmentation : dict, default={}
        Keyword arguments used to derive segmentation events.
    columns_events_to_keep : list[str] | None, default=None
        Optional list of regex patterns. Any metadata column whose name matches
        at least one pattern is retained. Pass ``None`` to keep all columns.
    tmin : float, default=-0.5
        Metadata window start in seconds.
    tmax : float, default=0.5
        Metadata window end in seconds.
    keep_first : str | None, default=None
        Forwarded to ``mne.epochs.make_metadata``.
    keep_last : str | None, default=None
        Forwarded to ``mne.epochs.make_metadata``.

    Returns
    -------
    metadata : pandas.DataFrame
        Metadata table aligned with segmentation events.
    """
    
    # Get events, and event ids for segmentation
    events_segm, event_id_segm = mne.events_from_annotations(raw, **kwargs_events_from_annotations_for_segmentation)
    
    # get events and event ids for metadata
    events_metadata, event_ids_metadata = mne.events_from_annotations(raw, **kwargs_events_from_annotations_for_metadata)
    
    # make metadata
    metadata_, events, event_id = mne.epochs.make_metadata(
                                                    events_metadata,
                                                    event_ids_metadata,
                                                    tmin,
                                                    tmax,
                                                    raw.info["sfreq"],
                                                    row_events=list(event_id_segm.keys()),
                                                    keep_first=keep_first, 
                                                    keep_last=keep_last
                                                    )
    # keep relevant columns (columns_events_to_keep is a list of regex patterns)
    if columns_events_to_keep is not None:
        col = [c for c in metadata_.columns
               if any(re.search(pattern, c) for pattern in columns_events_to_keep)]
        metadata = metadata_[col]
    metadata = metadata.reset_index(drop=True)
    
    return metadata
    

def compute_epochs_and_evoked(raw, 
                 kwargs_events_from_annotations={},
                 metadata=None,
                 tmin=-0.2, 
                 tmax=0.5, 
                 baseline=None, 
                 evoked_by="all", 
                 set_reference=None,
                 cfg_define_bcbt_epochs=None,
                 cfg_spline_channels=None,
                 cfg_bad_epochs=None,
                 summary=None):
    """Segment continuous data, mark bad epochs, and compute evoked responses.

    Parameters
    ----------
    raw : RawAPICE
        Preprocessed raw object containing annotations and artifact masks.
    kwargs_events_from_annotations : dict, default={}
        Keyword arguments passed to ``mne.events_from_annotations``.
    metadata : pandas.DataFrame | None, default=None
        Optional metadata aligned with the requested epochs.
    tmin : float, default=-0.2
        Epoch start relative to event onset in seconds.
    tmax : float, default=0.5
        Epoch end relative to event onset in seconds.
    baseline : tuple[float, float] | None, default=None
        Baseline correction window for MNE epoching.
    evoked_by : str | list[str] | None, default='all'
        Grouping used when computing evoked responses.
    set_reference : dict | None, default=None
        Reference parameters passed to ``epochs.set_eeg_reference``.
    cfg_define_bcbt_epochs : None | str | pathlib.Path | dict, default=None
        Configuration source for epoch-level BC/BT derivation.
    cfg_spline_channels : None | str | pathlib.Path | dict, default=None
        Configuration source for channel-wise spline interpolation.
    cfg_bad_epochs : None | str | pathlib.Path | dict, default=None
        Configuration source for bad-epoch detection.
    summary : SummaryEpochs | None, default=None
        Summary object updated with segmentation metrics.

    Returns
    -------
    epochs : EpochsAPICE
        Segmented epochs with updated artifact masks.
    evokeds : mne.Evoked | dict | None
        Evoked response object(s) computed from good epochs.
    summary : SummaryEpochs | None
        Updated summary object, or None when no summary was provided.
    """
    # get the configurations 
    cfg_define_bcbt_epochs = get_cfg(cfg_define_bcbt_epochs, 'define_bcbt_epochs_config.json')
    cfg_spline_channels = get_cfg(cfg_spline_channels, 'correction_spline_channels_config.json')
    cfg_bad_epochs = get_cfg(cfg_bad_epochs, 'detect_bad_epochs_config.json')

    # Get events, and event ids
    events_segm, event_id_segm = mne.events_from_annotations(raw, **kwargs_events_from_annotations)
    
    # Segment the continuous data into epochs
    karg = dict(reject_by_annotation=False,
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            preload=True,
            metadata=metadata,
    )
    epochs = raw.segment_continuous_data(events_segm, event_id_segm, karg)
    epochs.update_artifacts_params(**cfg_define_bcbt_epochs)
    
    # Define BadTimes and BadChannels for the segmented data
    epochs.define_bcbt()
    if summary is not None:
        summary.add_to_summary('segmentation_Initial', epochs, overwrite=True)

    # Apply spherical spline interpolation for artifact correction and re-define BT and BC after interpolation
    epochs.correct_spline_channels(cfg=cfg_spline_channels)
    
    # Identify and define bad epochs
    epochs.define_bad_epochs(bad_data=cfg_bad_epochs['bad_data'], 
                             bad_time=cfg_bad_epochs['bad_time'], 
                             bad_channel=cfg_bad_epochs['bad_channel'], 
                             lim_dist=cfg_bad_epochs['lim_dist'], 
                             lim_gfp=cfg_bad_epochs['lim_gfp'])
    

    # Re-reference the data if specified by the user
    if set_reference is not None:
        epochs.set_eeg_reference(**set_reference)
    if summary is not None:
        summary.add_to_summary('segmentation_Final', epochs, overwrite=True)
    
    # Print summary of artifacts and processing
    print(f"\nSummary: {print(epochs.artifacts.print_summary())}\n")

    # Compute the evoked responses
    print(
        f"\nGetting evoked responses...",
        f"-\n\t-- by event type: {evoked_by}"
        )
    # Remove bad epochs from the data 
    epochs_good = epochs.copy()
    epochs_good.remove_bad_epochs()
    if summary is not None:
        summary.add_to_summary('segmentation_GoodEpochs', epochs_good, overwrite=True)
    if evoked_by is not None:
        if evoked_by == "all":
            evokeds = epochs_good.average()
        else:
            evokeds = {}
            for ev in evoked_by:
                try:
                   evokeds[ev] = epochs_good[ev].average()
                except Exception as e:
                    print(f"Warning: Could not compute evoked response for event type {ev}: {e}")
    else:
        evokeds = None

    if summary is not None:
        return epochs, evokeds, summary
    else:
        return epochs, evokeds, None




def plot_summary(summary_df, metrics = ["%_corrected_data", "%_bad_data", "%_bad_channels", "%_bad_times"]):

    preproc_df = summary_df.melt(id_vars='step', value_vars=metrics, var_name='metric', value_name='value')

    # Do a bar plot using the metrics as hues for each preprocessing step
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=preproc_df, x='step', y='value', hue='metric', ax=ax)
    # draw vertical line to separate steps  
    for i in range(len(preproc_df['step'].unique()) - 1):
        ax.axvline(x=i + 0.5, color='gray', linestyle='--')
    ax.set_title("Preprocessing Metrics by Step")
    ax.set_ylabel("Percentage")
    ax.set_xlabel("Preprocessing Step")
    ax.legend(title="Metric")
    ax.tick_params(axis='x', rotation=45)

    return fig
