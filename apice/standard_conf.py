import json
from pathlib import Path
from apice.artifacts_rejection import ArtifactsConfiguration, concatenate_configurations

# Expected minimum rejection rate for a 3-sigma threshold (normal distribution)
_MIN_REJECTION_3_0_IQ = 0.00011


def min_rejection_from_thresh(thresh):
    """Estimate the expected minimum rejection rate for an IQR-based threshold.

    Assuming the data follow a normal distribution, computes the fraction of
    data (in percent) that would fall outside the rejection bounds

    .. code-block:: text

        lower: data < Q1 + thresh[0] * (Q3 - Q1)
        upper: data > Q3 + thresh[1] * (Q3 - Q1)

    where Q1 and Q3 are the first and third quartiles of the distribution.
    A ``None`` entry means that direction is not rejected.

    Parameters
    ----------
    thresh : list of length 2
        ``[thresh_lower, thresh_upper]``. Each element is either a float
        (the IQR multiplier for that tail) or ``None`` (tail not rejected).

    Returns
    -------
    min_rejection : float
        Expected percentage of data rejected under a Gaussian assumption,
        expressed in the same units as ``min_rejection`` in
        ``ArtifactsConfiguration`` (i.e. percent, not fraction).

    Examples
    --------
    >>> min_rejection_from_thresh([None, 3])   # one-sided, ~0.000118 %
    >>> min_rejection_from_thresh([-3, 3])     # two-sided, ~0.000236 %
    """
    from scipy.stats import norm

    Q1 = norm.ppf(0.25)   # ≈ -0.6745
    Q3 = norm.ppf(0.75)   # ≈  0.6745
    IQR = Q3 - Q1         # ≈  1.3490

    min_rej = 0.0

    if thresh[0] is not None:
        t_l = Q1 + thresh[0] * IQR   # thresh[0] is typically negative
        min_rej += norm.cdf(t_l)      # P(Z < t_l)

    if thresh[1] is not None:
        t_u = Q3 + thresh[1] * IQR
        min_rej += norm.sf(t_u)       # P(Z > t_u)

    result = min_rej * 100            # convert to percent
    if result == 0.0:
        return 0.0
    import math
    ndigits = -int(math.floor(math.log10(result))) + 1   # 2 significant figures
    return round(result, ndigits)


def cfg_bad_epochs(bad_data=1.00, bad_time=0, bad_channel=0.30, lim_dist=2, lim_gfp=2, filename=None):
    cfg = {}
    cfg['bad_data'] = bad_data
    cfg['bad_time'] = bad_time
    cfg['bad_channel'] = bad_channel
    cfg['lim_dist'] = lim_dist
    cfg['lim_gfp'] = lim_gfp

    if filename is not None:
        with open(filename, 'w') as f:
            json.dump(cfg, f, indent=4)

    return cfg


def cfg_define_bcbt_epochs(bcbt_method='functional', thresh_bad_channels=0.10,
                            thresh_bad_times=0.30, min_good_time=1.000,
                            min_bad_time=0.100, mask_time=0, filename=None):
    cfg = {}
    cfg['bcbt_method'] = bcbt_method
    cfg['thresh_bad_channels'] = thresh_bad_channels
    cfg['thresh_bad_times'] = thresh_bad_times
    cfg['min_good_time'] = min_good_time
    cfg['min_bad_time'] = min_bad_time
    cfg['mask_time'] = mask_time

    if filename is not None:
        with open(filename, 'w') as f:
            json.dump(cfg, f, indent=4)

    return cfg


def cfg_define_bcbt_raw(bcbt_method='functional', thresh_bad_channels=0.30,
                         thresh_bad_times=0.30, min_good_time=1.000,
                         min_bad_time=0.100, mask_time=0, filename=None):
    cfg = {}
    cfg['bcbt_method'] = bcbt_method
    cfg['thresh_bad_channels'] = thresh_bad_channels
    cfg['thresh_bad_times'] = thresh_bad_times
    cfg['min_good_time'] = min_good_time
    cfg['min_bad_time'] = min_bad_time
    cfg['mask_time'] = mask_time

    if filename is not None:
        with open(filename, 'w') as f:
            json.dump(cfg, f, indent=4)

    return cfg


def cfg_define_bcbt_raw_ica(bcbt_method='functional', thresh_bad_channels=0.50,
                             thresh_bad_times=0.50, min_good_time=2.000,
                             min_bad_time=0.020, mask_time=0.500, filename=None):
    cfg = {}
    cfg['bcbt_method'] = bcbt_method
    cfg['thresh_bad_channels'] = thresh_bad_channels
    cfg['thresh_bad_times'] = thresh_bad_times
    cfg['min_good_time'] = min_good_time
    cfg['min_bad_time'] = min_bad_time
    cfg['mask_time'] = mask_time

    if filename is not None:
        with open(filename, 'w') as f:
            json.dump(cfg, f, indent=4)

    return cfg


def cfg_correction_target_pca(max_time=0.125, components_to_remove=None,
                               variance_to_remove=0.99, mask_time=0.050,
                               all_time='all', all_channel='no_bad_channel',
                               all_epochs='all', splice_method=1,
                               save_corrected=True, filename=None):
    cfg = {}
    cfg['max_time'] = max_time
    cfg['components_to_remove'] = components_to_remove
    cfg['variance_to_remove'] = variance_to_remove
    cfg['mask_time'] = mask_time
    cfg['all_time'] = all_time
    cfg['all_channel'] = all_channel
    cfg['all_epochs'] = all_epochs
    cfg['splice_method'] = splice_method
    cfg['save_corrected'] = save_corrected

    if filename is not None:
        with open(filename, 'w') as f:
            json.dump(cfg, f, indent=4)

    return cfg


def cfg_correction_spline_segments(p=0.5, p_neighbors=1, min_good_time=1.00,
                                    min_intertime=0.050, mask_time=0.100,
                                    min_segment_time=0.250, splice_method=1,
                                    parallelize_mode='auto', save_corrected=True,
                                    filename=None):
    cfg = {}
    cfg['p'] = p
    cfg['p_neighbors'] = p_neighbors
    cfg['min_good_time'] = min_good_time
    cfg['min_intertime'] = min_intertime
    cfg['mask_time'] = mask_time
    cfg['min_segment_time'] = min_segment_time
    cfg['splice_method'] = splice_method
    cfg['parallelize_mode'] = parallelize_mode
    cfg['save_corrected'] = save_corrected

    if filename is not None:
        with open(filename, 'w') as f:
            json.dump(cfg, f, indent=4)

    return cfg


def cfg_correction_spline_channels(p=0.4, p_neighbors=1, save_corrected=True,
                                    filename=None):
    cfg = {}
    cfg['p'] = p
    cfg['p_neighbors'] = p_neighbors
    cfg['save_corrected'] = save_corrected

    if filename is not None:
        with open(filename, 'w') as f:
            json.dump(cfg, f, indent=4)

    return cfg


def cfg_detect_bad_channels(filename=None):
    artcfg = ArtifactsConfiguration()

    artcfg.add_algorithm_group('bad_channels_basic', max_loops=1, min_rejection=0, define_bcbt=True)
    artcfg.add_algorithm('bad_channels_basic', 'FlatChannel', {
        'bad_data': None,
        'do_reference_data': False,
        'do_zscore': False,
        'time_window': 10,
        'time_window_step': 5,
        'min_change': 1e-7,
        'thresh': 5,
        'mask': 0,
    })
    artcfg.add_algorithm('bad_channels_basic', 'ChannelCorr', {
        'bad_data': None,
        'do_reference_data': False,
        'do_zscore': False,
        'time_window': 10,
        'time_window_step': 5,
        'top_channel_corr': 5,
        'thresh': 0.4,
        'mask': 0,
    })
    artcfg.add_algorithm('bad_channels_basic', 'ShortGoodSegments', {
        'time_limit': 5,
    }, post_detection=True)
    artcfg.add_algorithm('bad_channels_basic', 'ShortBadSegments', {
        'time_limit': 0.500,
    }, post_detection=True)

    if filename is not None:
        artcfg.save_to_json(filename)

    return artcfg.cfg


def cfg_detect_power(rejection_level=3, min_rejection=None, max_loops=5, filename=None):
    thresh_upper = [None, rejection_level]               # upper tail only
    min_rej      = min_rejection if min_rejection is not None else min_rejection_from_thresh(thresh_upper)

    artcfg = ArtifactsConfiguration()
    artcfg.add_algorithm_group('bad_channels_power', max_loops=max_loops, min_rejection=min_rej, define_bcbt=True)
    artcfg.add_algorithm('bad_channels_power', 'Power', {
        'bad_data': None,
        'do_reference_data': False,
        'do_zscore': False,
        'time_window': 10,
        'time_window_step': 5,
        'freq_band': [20, 40],
        'thresh': thresh_upper,
        'remove_bct': True,
        'remove_bt': True,
        'remove_bc': True,
        'mask': 0,
    }, algorithm_name='PowerHighFreq')
    artcfg.add_algorithm('bad_channels_power', 'Power', {
        'bad_data': None,
        'do_reference_data': False,
        'do_zscore': False,
        'time_window': 10,
        'time_window_step': 5,
        'freq_band': [0.5, 5],
        'thresh': thresh_upper,
        'remove_bct': True,
        'remove_bt': True,
        'remove_bc': True,
        'mask': 0,
    }, algorithm_name='PowerLowFreq')

    artcfg.add_algorithm_group('bad_channels_modify', max_loops=1, min_rejection=0, define_bcbt=True)
    artcfg.add_algorithm('bad_channels_modify', 'ShortGoodSegments', {
        'time_limit': 1,
    }, post_detection=True)
    artcfg.add_algorithm('bad_channels_modify', 'ShortBadSegments', {
        'time_limit': 0.500,
    }, post_detection=True)

    if filename is not None:
        artcfg.save_to_json(filename)

    return artcfg.cfg


def cfg_detect_artifacts_huge(filename=None, rejection_level=4, abs_thresh_amp = 1000 * 1e-6):
    thresh_sym      = [-rejection_level, rejection_level]   # both tails
    thresh_upper    = [None, rejection_level]               # upper tail only
    
    artcfg = ArtifactsConfiguration()

    artcfg.add_algorithm_group('huge_amplitude_abs', max_loops=1, min_rejection=0, define_bcbt=True)
    artcfg.add_algorithm('huge_amplitude_abs', 'Amplitude', {
        'bad_data': None,
        'do_reference_data': False,
        'do_zscore': False,
        'thresh_type': 'absolute',
        'thresh': abs_thresh_amp,
        'mask': 0,
        'remove_bct': True,
        'remove_bt': True,
        'remove_bc': True,
    }, algorithm_name='HugeAmplitudeAbsolute')

    artcfg.add_algorithm_group('huge_artifacts', max_loops=1, min_rejection=0, define_bcbt=True)
    artcfg.add_algorithm('huge_artifacts', 'Amplitude', {
        'bad_data': None,
        'do_reference_data': False,
        'do_zscore': False,
        'thresh_type': 'outliers_per_channel',
        'thresh': thresh_sym,
        'mask': 0,
        'remove_bct': True,
        'remove_bt': True,
        'remove_bc': True,
    }, algorithm_name='HugeAmplitude')
    artcfg.add_algorithm('huge_artifacts', 'MaxChange', {
        'bad_data': None,
        'do_reference_data': False,
        'do_zscore': False,
        'thresh_type': 'outliers_per_channel',
        'thresh': thresh_upper,
        'time_window': 0.500,
        'time_window_step': 0.100,
        'mask': 0,
        'remove_bct': True,
        'remove_bt': True,
        'remove_bc': True,
    }, algorithm_name='HugeMaxChange500ms')
    artcfg.add_algorithm('huge_artifacts', 'ShortBadSegments', {
        'time_limit': 0.100,
    }, post_detection=True)
    artcfg.add_algorithm('huge_artifacts', 'ShortGoodSegments', {
        'time_limit': 0.500,
    }, post_detection=True)
    artcfg.add_algorithm('huge_artifacts', 'Mask', {
        'mask_length': 0.200,
    }, post_detection=True)
    artcfg.add_algorithm('huge_artifacts', 'ShortGoodSegments', {
        'time_limit': 0.500,
    }, post_detection=True)

    if filename is not None:
        artcfg.save_to_json(filename)

    return artcfg.cfg


def cfg_detect_artifacts_glitches(rejection_level=2, min_rejection=0, max_loops=2, filename=None):
    thresh_upper = [None, rejection_level]               # upper tail only
    min_rej      = min_rejection if min_rejection is not None else min_rejection_from_thresh(thresh_upper)

    artcfg_glitches = ArtifactsConfiguration()

    artcfg_glitches.add_algorithm_group('glitches', max_loops=max_loops, min_rejection=min_rej, define_bcbt=True)
    artcfg_glitches.add_algorithm('glitches', 'MaxChange', {
        'bad_data': None,
        'do_reference_data': False,
        'do_zscore': False,
        'time_window': 0.020,
        'time_window_step': 0.005,
        'thresh_type': 'outliers_per_channel',
        'thresh': thresh_upper,
        'mask': 0,
        'remove_bct': True,
        'remove_bt': True,
        'remove_bc': True,
    }, algorithm_name='GlitchesMaxChange')
    artcfg_glitches.add_algorithm('glitches', 'ShortBadSegments', {
        'time_limit': 0.010,
    }, post_detection=True)

    if filename is not None:
        artcfg_glitches.save_to_json(filename)

    return artcfg_glitches.cfg


def cfg_detect_artifacts_motion(rejection_level=3, min_rejection=None, max_loops=5, abs_thresh_amp = 1000 * 1e-6, filename=None):
    thresh_sym      = [-rejection_level, rejection_level]   # both tails
    thresh_upper    = [None, rejection_level]               # upper tail only
    min_rej_sym     = min_rejection if min_rejection is not None else min_rejection_from_thresh(thresh_sym)
    min_rej_upper   = min_rejection if min_rejection is not None else min_rejection_from_thresh(thresh_upper)

    artcfg = ArtifactsConfiguration()

    artcfg.add_algorithm_group('huge_amplitude_abs', max_loops=1, min_rejection=0, define_bcbt=True)
    artcfg.add_algorithm('huge_amplitude_abs', 'Amplitude', {
        'bad_data': None,
        'do_reference_data': False,
        'do_zscore': False,
        'thresh_type': 'absolute',
        'thresh': abs_thresh_amp,
        'mask': 0,
        'remove_bct': True,
        'remove_bt': True,
        'remove_bc': True,
    }, algorithm_name='HugeAmplitudeAbsolute')

    artcfg.add_algorithm_group('artifacts_amplitude', max_loops=max_loops, min_rejection=min_rej_sym, define_bcbt=True)
    artcfg.add_algorithm('artifacts_amplitude', 'Amplitude', {
        'bad_data': None,
        'do_reference_data': False,
        'do_zscore': False,
        'thresh_type': 'outliers_per_channel',
        'thresh': thresh_sym,
        'mask': 0,
        'remove_bct': True,
        'remove_bt': True,
        'remove_bc': True,
    }, algorithm_name='ArtifactsAmplitude')

    artcfg.add_algorithm_group('artifacts_maxchange500', max_loops=max_loops, min_rejection=min_rej_upper, define_bcbt=True)
    artcfg.add_algorithm('artifacts_maxchange500', 'MaxChange', {
        'bad_data': None,
        'do_reference_data': False,
        'do_zscore': False,
        'thresh_type': 'outliers_per_channel',
        'thresh': thresh_upper,
        'time_window': 0.500,
        'time_window_step': 0.100,
        'mask': 0,
        'remove_bct': True,
        'remove_bt': True,
        'remove_bc': True,
    }, algorithm_name='MaxChange500ms')

    artcfg.add_algorithm_group('artifacts_maxchange100', max_loops=max_loops, min_rejection=min_rej_upper, define_bcbt=True)
    artcfg.add_algorithm('artifacts_maxchange100', 'MaxChange', {
        'bad_data': None,
        'do_reference_data': False,
        'do_zscore': False,
        'thresh_type': 'outliers_per_channel',
        'thresh': thresh_upper,
        'time_window': 0.100,
        'time_window_step': 0.020,
        'mask': 0,
        'remove_bct': True,
        'remove_bt': True,
        'remove_bc': True,
    }, algorithm_name='MaxChange100ms')

    artcfg.add_algorithm_group('artifacts_amplitude_avgref', max_loops=max_loops, min_rejection=min_rej_sym, define_bcbt=True)
    artcfg.add_algorithm('artifacts_amplitude_avgref', 'Amplitude', {
        'bad_data': 'replace by nan',
        'do_reference_data': True,
        'do_zscore': False,
        'thresh_type': 'outliers_all',
        'thresh': thresh_sym,
        'mask': 0,
        'remove_bct': True,
        'remove_bt': True,
        'remove_bc': True,
    }, algorithm_name='ArtifactsAmplitudeAvgRef')

    artcfg.add_algorithm_group('artifacts_maxchange500_avgref', max_loops=max_loops, min_rejection=min_rej_upper, define_bcbt=True)
    artcfg.add_algorithm('artifacts_maxchange500_avgref', 'MaxChange', {
        'bad_data': 'replace by nan',
        'do_reference_data': True,
        'do_zscore': False,
        'thresh_type': 'outliers_all',
        'thresh': thresh_upper,
        'time_window': 0.500,
        'time_window_step': 0.100,
        'mask': 0,
        'remove_bct': True,
        'remove_bt': True,
        'remove_bc': True,
    }, algorithm_name='MaxChange500msAvgRef')

    artcfg.add_algorithm_group('artifacts_maxchange100_avgref', max_loops=max_loops, min_rejection=min_rej_upper, define_bcbt=True)
    artcfg.add_algorithm('artifacts_maxchange100_avgref', 'MaxChange', {
        'bad_data': 'replace by nan',
        'do_reference_data': True,
        'do_zscore': False,
        'thresh_type': 'outliers_all',
        'thresh': thresh_upper,
        'time_window': 0.100,
        'time_window_step': 0.020,
        'mask': 0,
        'remove_bct': True,
        'remove_bt': True,
        'remove_bc': True,
    }, algorithm_name='MaxChange100msAvgRef')

    artcfg.add_algorithm_group('artifacts_modify', max_loops=1, min_rejection=0, define_bcbt=True)
    artcfg.add_algorithm('artifacts_modify', 'ShortBadSegments', {
        'time_limit': 0.020,
    }, algorithm_name='VeryShortBadSegments', post_detection=True)
    artcfg.add_algorithm('artifacts_modify', 'ShortGoodSegments', {
        'time_limit': 0.020,
    }, algorithm_name='VeryShortGoodSegments', post_detection=True)
    artcfg.add_algorithm('artifacts_modify', 'ShortBadSegments', {
        'time_limit': 0.050,
    }, algorithm_name='ShortBadSegments', post_detection=True)
    artcfg.add_algorithm('artifacts_modify', 'ShortGoodSegments', {
        'time_limit': 0.500,
    }, algorithm_name='ShortGoodSegments', post_detection=True)

    if filename is not None:
        artcfg.save_to_json(filename)

    return artcfg.cfg


