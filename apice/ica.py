from pathlib import Path
import numpy as np
import json
import matplotlib.pyplot as plt  # Import Matplotlib for data visualization
import mne  # Import MNE-Python for EEG data handling
from mne.preprocessing import ICA

from mne_icalabel import label_components



from apice.data_structures import RawAPICE
from apice.utils import get_cfg
from apice.filter import Filter
from apice.utils import get_cfg
from apice.pipeline_utils import StdOutLogger


def find_ica_components_correlation(ica, raw,
                                     eog_channels=None,
                                     eog_bipolar_anodes=None,
                                     eog_bipolar_cathodes=None,
                                     ecg_channels=[]):
    """Identify ICA artifact components using correlation-based detectors.

    Parameters
    ----------
    ica : mne.preprocessing.ICA
        Fitted ICA object.
    raw : mne.io.BaseRaw
        Raw data used for correlation analysis.  A copy is made internally so
        the original is not modified.
    eog_channels : list of str or None
        Existing channel names used as EOG proxies with ``find_bads_eog``.
        Passed after any bipolar channels derived from
        ``eog_bipolar_anodes`` / ``eog_bipolar_cathodes``.
    eog_bipolar_anodes : list of str or None
        Anode channel names for creating bipolar EOG reference channels.
        Must be the same length as ``eog_bipolar_cathodes``.  Each pair
        produces a channel named '<anode>-<cathode>' that is prepended to the
        EOG channel list.
    eog_bipolar_cathodes : list of str or None
        Cathode channel names paired with ``eog_bipolar_anodes``.
    ecg_channels : list of str or None
        Channel names used as ECG proxies with ``find_bads_ecg``.  Each
        channel is tested independently and the detected indices are unioned.
        If empty, ECG detection is skipped.
        If None, find_bads_ecg will create a synthetic ECG channel. Only for MEG data.

    Returns
    -------
    components_to_exclude : list of int
        Indices of ICA components identified as artifacts.
    ic_labels : dict
        Dictionary with a ``'labels'`` key mapping to a list of length
        ``n_components``.  Each entry is ``'eye artifact'``, ``'heart beat'``,
        or ``'brain'`` for unlabelled components.
    """
    raw_correlation = raw.copy()

    # Create bipolar channels from anodes/cathodes pairs if provided
    bipolar_ch_names = []
    if eog_bipolar_anodes and eog_bipolar_cathodes:
        for anode, cathode in zip(eog_bipolar_anodes, eog_bipolar_cathodes):
            bp_ch_name = f'{anode}-{cathode}'
            raw_correlation = mne.set_bipolar_reference(
                raw_correlation, [anode], [cathode],
                ch_name=bp_ch_name, drop_refs=False, copy=False, on_bad='warn'
            )
            bipolar_ch_names.append(bp_ch_name)

    # Build EOG channels list: bipolar channels first, then regular channels
    channels_eog = bipolar_ch_names + (eog_channels if eog_channels else [])

    # Perform EOG artifact detection using all channels one by one
    eog_indices = {}
    eog_scores = {}
    for ch_eog in channels_eog:
        eog_idx, scores = ica.find_bads_eog(raw_correlation, ch_name=ch_eog,
                                             threshold=0.90, measure='correlation',
                                             start=None, stop=None, l_freq=1, h_freq=10,
                                             reject_by_annotation=True, verbose='WARNING')
        eog_indices[ch_eog] = eog_idx
        eog_scores[ch_eog] = scores

        print(f"EOG indices for {ch_eog}:")
        for idx in eog_idx:
            print(f"  Component {idx} with score {scores[idx]:.2f}")

    # Get a unique list of EOG indices across all channels
    unique_eog_indices = set()
    for idx_list in eog_indices.values():
        unique_eog_indices.update(idx_list)
    print(f"Unique EOG indices across all channels: {list(map(int, unique_eog_indices))}")

    # Perform ECG artifact detection
    ecg_indices_all = set()
    if ecg_channels is None:
        ecg_idx, ecg_scores_ch = ica.find_bads_ecg(
                raw_correlation, ch_name=None,
                threshold=0.90, start=None, stop=None,
                l_freq=8, h_freq=16, method='ctps',
                reject_by_annotation=True, measure='correlation', verbose=None
            )
        ecg_indices_all.update(ecg_idx)
        print("ECG indices (synthetic channel):")
        for idx in ecg_idx:
                print(f"  Component {idx}")
    else:
        for ch_ecg in ecg_channels:
            ecg_idx, ecg_scores_ch = ica.find_bads_ecg(
                raw_correlation, ch_name=ch_ecg,
                threshold=0.90, start=None, stop=None,
                l_freq=8, h_freq=16, method='ctps',
                reject_by_annotation=True, measure='correlation', verbose=None
            )
            ecg_indices_all.update(ecg_idx)
            print(f"ECG indices for {ch_ecg}:")
            for idx in ecg_idx:
                print(f"  Component {idx}")
    ecg_indices = list(ecg_indices_all)

    # Combine EOG and ECG indices
    components_to_exclude = list(set(unique_eog_indices) | set(ecg_indices))
    print(f"Components to exclude based on EOG and ECG detection: {list(map(int, components_to_exclude))}")

    # Build ic_labels dict (mirrors the iclabel output structure)
    labels = ['brain'] * ica.n_components_
    for idx in unique_eog_indices:
        labels[idx] = 'eye artifact'
    for idx in ecg_indices:
        if labels[idx] == 'brain':   
            labels[idx] = 'heart beat'
    ic_labels = {'labels': labels}

    return components_to_exclude, ic_labels




def clean_ica(raw,
              cfg_artifacts_detection=None,
              cfg_bcbt=None,
              l_freq_ica=1,
              h_freq_ica=100,
              l_freq_artifacts=None,
              h_freq_artifacts=None,
              picks_ica=None,
              exclude_ica=None,
              n_components='auto',
              noise_cov=None,
              random_state=42,
              method='picard',
              fit_params=dict(ortho=False, extended=True),
              max_iterint ='auto',
              start_fit=None,
              stop_fit=None,
              label_components_method='iclabel',
              iclabel_lim_probability=0.9,
              iclabel_labels_to_exclude = ['eye blink', 'muscle artifact', 'heart beat', 'line noise', 'channel noise'],
              eog_channels=None,
              eog_bipolar_anodes=None,
              eog_bipolar_cathodes=None,
              ecg_channels=[],
              create_report=True,
              save_data=False,
              save_report=False,
              save_ica=False,
              save_cfg=False,
              save_log=False,
              output_dir=None,
              file_name=None,
              report=None,
              logger=None,
              ):
    """Run ICA on EEG data to identify and remove artifact components.

    A copy of the raw data is made, artifact-contaminated segments are detected
    and annotated (using the APICE pipeline) so they are excluded from the ICA
    fit, and the resulting ICA decomposition is applied to clean the original
    (unfiltered) data.  Components to exclude are identified either with the
    ICLabel classifier or with correlation-based methods (find_bads_eog /
    find_bads_ecg).

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Raw EEG data to clean.  Must have a montage set.
    cfg_artifacts_detection : dict or None
        Configuration dictionary for artifact detection before ICA fitting.
        If None, the default 'detect_for_ica_config.json' configuration is used.
    cfg_bcbt : dict or None
        Configuration dictionary for the BCBT (bad channel / bad time) 
        parameters used before ICA fitting.
        If None, the default 'define_bcbt_raw_ica_config.json' configuration is used.
    l_freq_ica : float
        High-pass cut-off frequency (Hz) applied to the ICA fitting copy of
        the data.  Default is 1 Hz.
    h_freq_ica : float or None
        Low-pass cut-off frequency (Hz) applied to the ICA fitting copy of
        the data.  None means no low-pass filter is applied.
    l_freq_artifacts : float or None
        High-pass cut-off frequency (Hz) applied to the copy of the data used for artifact detection before ICA fitting.  
        If None, the data is not further high-pass filtered for artifacts detection.
    h_freq_artifacts : float or None
        Low-pass cut-off frequency (Hz) applied to the copy of the data used for artifact detection before ICA fitting.
        If None, the data is not further low-pass filtered for artifacts detection.
    picks_ica : list of str or None
        List of channel names to include in ICA fitting. If None, all channels are used, excluding those marked as bad and those in ``exclude_ica``.
    exclude_ica : list of str or None
        List of channel names to exclude from ICA fitting (e.g., reference channels).
    n_components : int, float, None, or 'auto'
        Number of principal components (PCA) passed to the ICA decomposition.
        If 'auto', the number of components is automatically determined considering m ≥ 30 x n2, where m is the number of samples and n is the number of channels.
    noise_cov : mne.Cov or None
        Noise covariance used for pre-whitening. If None, no pre-whitening is applied.
    random_state : int
        Seed for the random number generator, ensuring reproducibility. Default is 42.
    method : str
        ICA algorithm. Any method supported by mne.preprocessing.ICA
        (e.g. 'picard', 'fastica', 'infomax'). Default is 'picard'.
    fit_params : dict
        Additional keyword arguments passed to the ICA fitting algorithm.
        Default is dict(ortho=False, extended=True).
    max_iterint : int or 'auto'
        Maximum number of iterations during ICA fitting. Default is 'auto'.
    start_fit : int, float or None
        First sample (int) or time in seconds (float) to use for ICA fitting.
        None means the beginning of the recording.
    stop_fit : int, float or None
        Last sample (int) or time in seconds (float) to use for ICA fitting.
        None means the end of the recording.
    label_components_method : str
        Method used to identify artifact components. Options:

        * ``'iclabel'`` — Uses the ICLabel classifier (mne_icalabel).
          Requires an average reference; it is set automatically on the ICA
          fitting copy.  Components labeled as 'eye blink', 'muscle artifact',
          'heart beat', or 'line noise' above ``iclabel_lim_probability`` are excluded.
        * ``'correlation'`` — Uses mne correlation-based detectors
          (find_bads_eog / find_bads_ecg) via
          :func:`find_ica_components_correlation`.  Requires
          ``eog_channels`` and/or ``eog_bipolar_anodes`` /
          ``eog_bipolar_cathodes`` to be set.

    iclabel_lim_probability : float
        Minimum ICLabel probability for a component to be excluded (only used
        when ``label_components_method='iclabel'``).  Default is 0.9.
    iclabel_labels_to_exclude : list of str
        List of ICLabel labels to exclude (only used when ``label_components_method='iclabel'``).
        Default is ['eye blink', 'muscle artifact', 'heart beat', 'line noise'].
        Possible labels include: 'eye blink', 'muscle artifact', 'heart beat', 'line noise', 'channel noise', 'other'.
    eog_channels : list of str or None
        Existing channel names to use as EOG proxies with find_bads_eog
        (only used when ``label_components_method='correlation'``).
        These channels are used in addition to any bipolar channels created
        from ``eog_bipolar_anodes`` / ``eog_bipolar_cathodes``.
    eog_bipolar_anodes : list of str or None
        Anode channel names for creating bipolar EOG reference channels
        (only used when ``label_components_method='correlation'``).
        Must be paired with ``eog_bipolar_cathodes`` of the same length.
        Each pair produces a channel named '<anode>-<cathode>' that is
        prepended to the EOG channel list before calling find_bads_eog.
    eog_bipolar_cathodes : list of str or None
        Cathode channel names paired with ``eog_bipolar_anodes``.
        See ``eog_bipolar_anodes`` for details.
    ecg_channels : list of str or None
        Channel names to use as ECG proxies with find_bads_ecg
        (only used when ``label_components_method='correlation'``).
        Each channel is tested independently; detected indices are unioned.
        If None or empty, ECG detection is skipped.
    create_report : bool
        If True, an mne.Report is created with diagnostic figures (artifact
        matrix, bad-electrode topomap, ICA topomaps, and properties of
        excluded components).  Default is True.
    save_data : bool
        If True, the cleaned raw data with ICA components removed is saved to
        ``output_dir/<file_name>_cleaned_raw.fif``.  Requires ``output_dir`` and
        ``file_name`` to be provided. Default is False.
    save_report : bool
        If True, the report is saved to ``output_dir/reports/<file_name>_report.html``.
        Requires ``output_dir`` and ``file_name`` to be provided. Default is False.
    save_ica : bool
        If True, the fitted ICA object is saved to
        ``output_dir/<file_name>_raw-ica.fif``.
        Requires ``output_dir`` and ``file_name`` to be provided. Default is False.
    save_cfg : bool
        If True, the preprocessing configurations are saved to
        ``output_dir/cfgs/<file_name>_<cfg_name>.json``.
        Requires ``output_dir`` and ``file_name`` to be provided. Default is False.
    save_log : bool
        If True, the preprocessing log is printed to the console and saved to
        ``output_dir/logs/<file_name>_log.txt``.  Requires ``output_dir`` and
        ``file_name`` to be provided. Default is False.
    output_dir : str, Path or None
        Directory where output files (ICA, report, configurations) are saved.
        Required when ``save_ica=True``, ``save_report=True``, or ``save_cfg=True``.
    file_name : str or None
        Base name used for output file names (without extension).
        Required when ``save_ica=True``, ``save_report=True``, or ``save_cfg=True``.

    Returns
    -------
    raw_clean : mne.io.BaseRaw
        Copy of the input ``raw`` with the artifact ICA components subtracted.
    ica : mne.preprocessing.ICA
        Fitted ICA object.  ``ica.exclude`` is *not* set; the exclusion is
        applied directly to ``raw_clean`` via component subtraction.
    report : mne.Report or None
        Diagnostic report with figures illustrating the ICA results.
        None if ``create_report=False``.
    """

    # Check the n_components parameter    
    if isinstance(n_components, str) and n_components != 'auto':
        raise ValueError("n_components must be an int, float, None, or 'auto'")

    # Check if raw is an instance of mne.io.Raw
    if not isinstance(raw, mne.io.BaseRaw):
        raise TypeError("raw must be an instance of mne.io.Raw")
    
    # Check that raw has a montage
    if raw.get_montage() is None:
        raise ValueError("raw must have a montage. Please set the montage before preprocessing.")
    
    # Check that output_dir is provided if any of the saving options is True
    if output_dir is None and (save_ica or save_report or save_cfg or save_data or save_log):
        raise ValueError("output_dir must be provided if any of the saving options is True")

    # Check that file_name is provided if any of the saving options is True, to use as part of the file name for the saved files
    if file_name is None and (save_ica or save_report or save_cfg or save_data):
        raise ValueError("file_name must be provided if any of the saving options is True, to use as part of the file name for the saved files")

    # Check that file_name ends with 'ica', '_ica', '-ica'. If it does not, add '_ica' to the end of the file name to indicate that this is the result of ICA cleaning.  
    if file_name is not None and not (file_name.endswith('ica') or file_name.endswith('_ica') or file_name.endswith('-ica')):
        file_name = file_name + '_ica'

    # Create output folder if it does not exist
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        if save_report:
            output_dir_reports = output_dir / "reports"
            output_dir_reports.mkdir(exist_ok=True)
        if save_cfg:
            output_dir_cfgs = output_dir / "cfgs"
            output_dir_cfgs.mkdir(exist_ok=True)
        if save_log:
            output_dir_logs = output_dir / "logs"
            output_dir_logs.mkdir(exist_ok=True)
        if save_ica:
            output_dir_ica = output_dir / "ica"
            output_dir_ica.mkdir(exist_ok=True)
        
    # get the configurations for rejecting data before ICA
    cfg_artifacts_detection = get_cfg(cfg_artifacts_detection, 'detect_for_ica_config.json')
    cfg_bcbt = get_cfg(cfg_bcbt, 'define_bcbt_raw_ica_config.json')

    # Determine ownership of report and logger
    _owns_report = (report is None) and create_report
    _owns_logger = (logger is None) and save_log

    # Initialize object for logging (only when this function owns it)
    if _owns_logger:
        logger = StdOutLogger(output_dir_logs, file_name)

    # Initialize reports (only when this function owns it)
    if _owns_report:
        report = mne.Report(title=file_name)

    # Start logging (only when this function owns it)
    if _owns_logger: logger.redirect_stdout_to_file(restore=True)

    print('ICA cleaning')
    print('============')
    
    # Print the configuration parameters for ICA cleaning
    print("\nICA cleaning configuration:")    
    print("------------------------------------")
    print(f"  l_freq_ica: {l_freq_ica}")
    print(f"  h_freq_ica: {h_freq_ica}")    
    print(f"  l_freq_artifacts: {l_freq_artifacts}")
    print(f"  h_freq_artifacts: {h_freq_artifacts}")
    print(f"  picks_ica: {picks_ica}")
    print(f"  exclude_ica: {exclude_ica}")
    print(f"  n_components: {n_components}")
    print(f"  method: {method}")
    print(f"  fit_params: {fit_params}")
    print(f"  label_components_method: {label_components_method}")
    print(f"  iclabel_lim_probability: {iclabel_lim_probability}")
    print(f"  iclabel_labels_to_exclude: {iclabel_labels_to_exclude}")
    print(f"  eog_channels: {eog_channels}")
    print(f"  eog_bipolar_anodes: {eog_bipolar_anodes}")
    print(f"  eog_bipolar_cathodes: {eog_bipolar_cathodes}")
    print(f"  ecg_channels: {ecg_channels}")           

    # Make a copy of the raw data to filter, mark bad data and then apply ICA on it
    raw_ica = raw.copy()

    # Detect artifacts to remove them before ICA
    print("\nDetecting artifacts and marking bad data for ICA fitting...")
    print("------------------------------------")
    raw_ica = RawAPICE(raw_ica, **cfg_bcbt)
    raw_ica.run_algorithms(cfg_artifacts_detection, l_freq = l_freq_artifacts, h_freq = h_freq_artifacts)
    raw_ica.define_bcbt()

    # Mark bad data before ICA so that it is not used for ICA decomposition
    raw_ica.annotate_bads(channels=True, times=True, data=False, corrected=False)

    # Filter
    Filter(raw_ica, l_freq=l_freq_ica, h_freq=h_freq_ica) 

    # Get the picks for ICA fitting
    if picks_ica is None:
        picks_ica = mne.pick_types(raw_ica.info, meg=False, eeg=True, eog=True, ecg=True)
    picks_ica = [i for i in picks_ica if raw_ica.ch_names[i] not in raw_ica.info['bads']]
    if exclude_ica is not None:
        picks_ica = [i for i in picks_ica if raw_ica.ch_names[i] not in exclude_ica]
    
    # Power spectrum of the data used for ICA fitting, to check the effect of filtering and artifact marking before ICA
    if report is not None:
        fmax = raw_ica.info['sfreq'] / 2
        try:
            fig_psd = raw_ica.compute_psd(method='welch', fmax=fmax, picks=picks_ica, reject_by_annotation=True, verbose='ERROR').plot(show=False)
            report.add_figure(fig_psd, "Power Spectrum for ICA fitting", section="Raw Data ICA", replace=True)
            plt.close(fig_psd)
        except Exception as e:
            print(f"Warning: Could not add power spectrum plot to report: {e}")
    # Create a figure to visualize the artifact structure
    if report is not None:
        try:
            fig = raw_ica.plot_artifact_structure(color_scheme='jet')    
            report.add_figure(fig, "Artifacts Matrix", section="Raw Data ICA", replace=True)
            plt.close(fig)
        except Exception as e:
            print(f"Warning: Could not add raw artifacts matrix to report: {e}")

    # Add topomap of bad electrodes
    if report is not None:
        try:
            fig = raw_ica.plot_percentage_of_bad_data_across_sensors()
            report.add_figure(fig, "Bad data across electrodes", section="Raw Data ICA", replace=True)
            plt.close(fig)
        except Exception as e:
            print(f"Warning: Could not add raw bad-data topomap to report: {e}")

    # Fit ICA
    print("\nFitting ICA...")
    print("------------------------------------")
    
    # define the number of n_components
    if n_components == 'auto':
        n_samples = (raw_ica.artifacts.BT == False).sum()  # number of samples not marked as bad for ICA fitting
        n_channels = len(picks_ica)
        n_components_auto = min(n_channels, int((n_samples / 30) ** 0.5))  # n_samples ≥ 30 x n_channels^2 => n_channels ≤ sqrt(n_samples / 30)
        print(f"Automatically setting n_components to {n_components_auto} based on the number of samples and channels.")
        n_components = n_components_auto

    if label_components_method == 'iclabel':
        # ICLabel requires an average reference, so we set it here for ICA fitting
        raw_ica.set_eeg_reference("average")
        ica = ICA(n_components=n_components, random_state=random_state, noise_cov=noise_cov, max_iter=max_iterint, method=method, fit_params=fit_params)
        ica.fit(raw_ica, 
            start=start_fit, stop=stop_fit, 
            picks=picks_ica,
            reject=None, flat=None, reject_by_annotation=True, verbose=None)
    else:
        ica = ICA(n_components=n_components, random_state=random_state, noise_cov=noise_cov, max_iter=max_iterint, method=method, fit_params=fit_params)
        ica.fit(raw_ica, 
                start=start_fit, stop=stop_fit, 
                picks=picks_ica,
                reject=None, flat=None, reject_by_annotation=True, verbose=None)
    

    # Get the components to exclude 
    print("\nFinding components to exclude...")
    print("------------------------------------")
    if label_components_method == 'iclabel':
        ic_labels = label_components(raw_ica, ica, method="iclabel")
        components_to_exclude = [i for i, label in enumerate(ic_labels["labels"]) if label in iclabel_labels_to_exclude and ic_labels["y_pred_proba"][i]>=iclabel_lim_probability]
        print(f"Components to exclude based on ICLabel:")
        for i in components_to_exclude:
            print(f"  Component {i} with label {ic_labels['labels'][i]} and probability {ic_labels['y_pred_proba'][i]:.2f}")
        # print all the components order by decreasing probability and the label for each component
        print(f"\nAll components with ICLabel probabilities:")
        for i in range(ica.n_components_):
            print(f"  Component {i}: label={ic_labels['labels'][i]}, probability={ic_labels['y_pred_proba'][i]:.2f}")   
    
    elif label_components_method == 'correlation':
        components_to_exclude, ic_labels = find_ica_components_correlation(
            ica, raw_ica,
            eog_channels=eog_channels,
            eog_bipolar_anodes=eog_bipolar_anodes,
            eog_bipolar_cathodes=eog_bipolar_cathodes,
            ecg_channels=ecg_channels
        )
        print(f"Components to exclude based on correlation:")
        for i in components_to_exclude:
            print(f"  Component {i} with label {ic_labels['labels'][i]}")

    # Remove the bad components from the original raw data
    raw_clean = raw.copy()
    # ica.apply(raw_clean, include=None, exclude = components_to_exclude, n_pca_components=None)
    artifacts = raw.copy()
    ica.apply(artifacts, include=components_to_exclude, exclude = [], n_pca_components=None)
    raw_clean._data = raw._data - artifacts._data

    # Add some plots to the report to visualize the results of ICA
    if report is not None:
        # Plot the ICA components 
        fig_components = ica.plot_components(inst=raw_ica, show=False)
        try:
            for i, f in enumerate(fig_components):
                report.add_figure(f, "ICA component", section="ICA", replace=False)
                plt.close(f)
        except Exception as e:
            print(f"Warning: Could not add components topomap to report: {e}")

        # Add a table with the labels and probabilities for each component if ICLabel was used. Order by label and then by decreasing probability of being an artifact, and include the label for each component.
        if label_components_method == 'iclabel':
            try:
                import pandas as pd
                df = pd.DataFrame({
                    "Component": np.arange(ica.n_components_),
                    "Label": ic_labels['labels'],
                    "Probability": ic_labels['y_pred_proba']
                }).sort_values(by=['Label', 'Probability'], ascending=[True, False])
                fig_table, ax = plt.subplots(figsize=(8, 0.5 * len(df)))
                ax.axis('off')
                table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.auto_set_column_width(col=list(range(len(df.columns))))
                report.add_figure(fig_table, "ICLabel Component Classification", section="ICA", replace=False)
                plt.close(fig_table)
            except Exception as e:
                print(f"Warning: Could not add ICLabel classification table to report: {e}")
        # Plot the properties of the components to exclude
        for idx in components_to_exclude:
            fig_properties = ica.plot_properties(raw_ica, picks=idx, show=False)
            try:
                comp_label = ic_labels['labels'][idx]
                report.add_figure(fig_properties, f"ICA component {idx}, type {comp_label}", section="ICA", replace=False)
                for f in fig_properties:
                    plt.close(f)
            except Exception as e:
                print(f"Warning: Could not add component {idx} properties to report: {e}")

        # # Plot the sources of the components to exclude
        # fig_sources = ica.plot_sources(raw, picks=components_to_exclude, show=False)
        # try:
        #     report.add_figure(fig_sources, "ICA component sources", section="ICA", replace=False)
        # except Exception as e:
        #     print(f"Warning: Could not add ICA component sources to report: {e}")

        # Plot the original and clean signal
        # select 30 seconds of data around the midpoint of the recording, to avoid edge effects
        mid_point = raw.times[0] + ((raw.times[-1] - raw.times[0]) / 2)
        start = max(raw.times[0], mid_point - 15)  # 15 seconds before the midpoint
        stop = min(raw.times[-1], start + 30)  # 30 seconds window
        fig_overlay = ica.plot_overlay(raw, exclude=components_to_exclude, picks="eeg", start=start, stop=stop, show=False)
        try:
            report.add_figure(fig_overlay, "Original and Clean Signal", section="ICA", replace=False)
            plt.close(fig_overlay)
        except Exception as e:
            print(f"Warning: Could not add overlay plot to report: {e}")

        # Power spectrum of the clean data after ICA, to check the effect of ICA cleaning on the power spectrum
        fmax = raw_clean.info['sfreq'] / 2
        try:
            fig_psd = raw_clean.compute_psd(method='welch', fmax=fmax, picks=picks_ica, reject_by_annotation=True).plot(show=False)
            report.add_figure(fig_psd, "Power Spectrum after ICA", section="Clean Data", replace=True)
            plt.close(fig_psd)
        except Exception as e:
            print(f"Warning: Could not add power spectrum plot to report: {e}")

    # Save the ICA object, the report, and the configurations if requested
    if save_data:
        raw_clean.save(output_dir / f"{file_name}-raw.fif", overwrite=True)
    if save_ica:
        ica.save(output_dir_ica / f"{file_name}-decomposition.fif", overwrite=True)
    if _owns_report and save_report:
        report.save(output_dir_reports / f"{file_name}-report.html", overwrite=True, open_browser=False)
    if save_cfg:
        print("Saving preprocessing configurations")
        cfg_to_save = {
            "cfg_artifacts_detection": cfg_artifacts_detection,
            "cfg_bcbt": cfg_bcbt,
            "cfg": {
                "l_freq_ica": l_freq_ica,
                "h_freq_ica": h_freq_ica,
                "n_components": n_components,
                "noise_cov": str(noise_cov) if noise_cov is not None else None,  # Convert noise_cov to string for JSON serialization
                "random_state": random_state,
                "method": method,
                "fit_params": fit_params,
                "max_iterint": max_iterint,
                "start_fit": start_fit,
                "stop_fit": stop_fit,
                "label_components_method": label_components_method,
                "iclabel_lim_probability": iclabel_lim_probability,
                "eog_channels": eog_channels,
                "eog_bipolar_anodes": eog_bipolar_anodes,
                "eog_bipolar_cathodes": eog_bipolar_cathodes,
                "ecg_channels": ecg_channels
                    }
                }
        for cfg_name, cfg in cfg_to_save.items():
            with open(output_dir_cfgs / f"{file_name}_{cfg_name}.json", 'w') as f:
                json.dump(cfg, f, indent=4)

    if _owns_logger:
        logger.close()

    return raw_clean, ica, report