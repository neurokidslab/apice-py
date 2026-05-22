"""Build and save default APICE JSON configuration files.

This script generates default detection, correction, and summary configuration
files in ``apice/default_cfg``.
"""

import json

from apice.artifacts_rejection import ArtifactsConfiguration, concatenate_configurations
from pathlib import Path


def main():
    """Create and persist the package default configuration set.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Writes multiple JSON files into ``apice/default_cfg``.
    """
        
    # %% Create a default configuration for defining bad epochs
    bad_epochs_cfg = {}
    bad_epochs_cfg['bad_data'] = 1.00
    bad_epochs_cfg['bad_time'] = 0
    bad_epochs_cfg['bad_channel'] = 0.30
    bad_epochs_cfg['lim_dist'] = 2
    bad_epochs_cfg['lim_gfp'] = 2
    # save to json
    with open(Path(__file__).parent / "default_cfg" / 'detect_bad_epochs_config.json', 'w') as f:
        json.dump(bad_epochs_cfg, f, indent=4)


    # %% Create default configuration for BC and BT definition on epochs
    # define_bcbt_epoch_cfg = {}
    # define_bcbt_epoch_cfg['bcbt_method'] = 'fix'
    # define_bcbt_epoch_cfg['thresh_bad_channels'] = [0.90, 0.70, 0.50, 0.30, 0.10]
    # define_bcbt_epoch_cfg['thresh_bad_times'] = [0.90, 0.70, 0.50, 0.30, 0.30]
    # define_bcbt_epoch_cfg['min_good_time'] = 1.000
    # define_bcbt_epoch_cfg['min_bad_time'] = 0.100
    # define_bcbt_epoch_cfg['mask_time'] = 0
    
    define_bcbt_epoch_cfg = {}
    define_bcbt_epoch_cfg['bcbt_method'] = 'functional'
    define_bcbt_epoch_cfg['thresh_bad_channels'] = 0.10
    define_bcbt_epoch_cfg['thresh_bad_times'] = 0.30
    define_bcbt_epoch_cfg['min_good_time'] = 1.000
    define_bcbt_epoch_cfg['min_bad_time'] = 0.100
    define_bcbt_epoch_cfg['mask_time'] = 0
    # save to json
    with open(Path(__file__).parent / "default_cfg" / 'define_bcbt_epochs_config.json', 'w') as f:
        json.dump(define_bcbt_epoch_cfg, f, indent=4)


    # %% Create default configuration for BC and BT definition
    # define_bcbt_raw_cfg = {}
    # define_bcbt_raw_cfg['bcbt_method'] = 'fix'
    # define_bcbt_raw_cfg['thresh_bad_channels'] = [0.90, 0.70, 0.50, 0.30]
    # define_bcbt_raw_cfg['thresh_bad_times'] = [0.90, 0.70, 0.50, 0.30]
    # define_bcbt_raw_cfg['min_good_time'] = 1.000
    # define_bcbt_raw_cfg['min_bad_time'] = 0.100
    # define_bcbt_raw_cfg['mask_time'] = 0
    
    define_bcbt_raw_cfg = {}
    define_bcbt_raw_cfg['bcbt_method'] = 'functional'
    define_bcbt_raw_cfg['thresh_bad_channels'] = 0.30
    define_bcbt_raw_cfg['thresh_bad_times'] = 0.30
    define_bcbt_raw_cfg['min_good_time'] = 1.000
    define_bcbt_raw_cfg['min_bad_time'] = 0.100
    define_bcbt_raw_cfg['mask_time'] = 0
    # save to json
    with open(Path(__file__).parent / "default_cfg" / 'define_bcbt_raw_config.json', 'w') as f:
        json.dump(define_bcbt_raw_cfg, f, indent=4)


    # %% Create default configuration for artifacts correction
    pca_cfg = {}
    pca_cfg['max_time'] = 0.125
    pca_cfg['components_to_remove'] = None
    pca_cfg['variance_to_remove'] = 0.99
    pca_cfg['mask_time'] = 0.050
    pca_cfg['all_time'] = 'all'
    pca_cfg['all_channel'] = 'no_bad_channel'
    pca_cfg['all_epochs'] = 'all'
    pca_cfg['splice_method'] = 1
    pca_cfg['save_corrected'] = True
    # save to json
    with open(Path(__file__).parent / "default_cfg" / 'correction_target_pca_config.json', 'w') as f:
        json.dump(pca_cfg, f, indent=4)

    spline_segments_cgf = {}
    spline_segments_cgf['p'] = 0.5
    spline_segments_cgf['p_neighbors'] = 1
    spline_segments_cgf['min_good_time'] = 1.00 
    spline_segments_cgf['min_intertime'] = 0.050
    spline_segments_cgf['mask_time'] = 0.100
    spline_segments_cgf['min_segment_time'] = 0.250
    spline_segments_cgf['splice_method'] = 1
    spline_segments_cgf['parallelize_mode'] = 'auto'
    spline_segments_cgf['save_corrected'] = True
    # save to json
    with open(Path(__file__).parent / "default_cfg" / 'correction_spline_segments_config.json', 'w') as f:
        json.dump(spline_segments_cgf, f, indent=4) 

    spline_channels_cgf = {}
    spline_channels_cgf['p'] = 0.5
    spline_channels_cgf['p_neighbors'] = 1
    spline_channels_cgf['save_corrected'] = True
    # save to json
    with open(Path(__file__).parent / "default_cfg" / 'correction_spline_channels_config.json', 'w') as f:
        json.dump(spline_channels_cgf, f, indent=4)


    # %% Define min_rejection expected based on normal distribution 
    # ---------------------------------------------------------------------------------
    # with a 2.0 threshold we expect 0.089% of the data to be above or below the treshold
    min_rejection_thresh_2_0_IQ = 0.089
    # with a 2.5 threshold we expect 0.0042% of the data to be above or below the treshold
    min_rejection_thresh_2_5_IQ = 0.0042
    # with a 3.0 threshold we expect 0.00011% of the data to be above or below the treshold
    min_rejection_thresh_3_0_IQ = 0.00011


    # %% Create the default configuration for bad channels detection and rejection
    # ---------------------------------------------------------------------------------
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
        'time_limit': 2,
    }, post_detection=True)


    artcfg.add_algorithm_group('bad_channels_power', max_loops=5, min_rejection=min_rejection_thresh_2_0_IQ, define_bcbt=True)
    artcfg.add_algorithm('bad_channels_power', 'Power', {
        'bad_data': None,
        'do_reference_data': False,
        'do_zscore': False,
        'time_window': 10,
        'time_window_step': 5,
        'freq_band': [20, 40],
        'thresh': [None, 3],
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
        'thresh': [None, 3],
        'remove_bct': True,
        'remove_bt': True,
        'remove_bc': True,
        'mask': 0,
    }, algorithm_name='PowerLowFreq')
    artcfg.add_algorithm_group('bad_channels_modify', max_loops=1, min_rejection=0, define_bcbt=True)
    artcfg.add_algorithm('bad_channels_modify', 'ShortGoodSegments', {
        'time_limit': 10,
    }, post_detection=True)
    artcfg.add_algorithm('bad_channels_modify', 'ShortBadSegments', {
        'time_limit': 0.500,
    }, post_detection=True)


    artcfg.save_to_json(Path(__file__).parent / "default_cfg" / 'detect_bad_channels_config.json')


    # %% Create a default configuration for detecting very bad data segments
    # ---------------------------------------------------------------------------------
    artcfg = ArtifactsConfiguration()   
    
    artcfg.add_algorithm_group('huge_amplitude_abs', max_loops=1, min_rejection=0, define_bcbt=True)
    artcfg.add_algorithm('huge_amplitude_abs', 'Amplitude', {
        'bad_data': None,
        'do_reference_data': False,
        'do_zscore': False,
        'thresh_type': 'absolute',
        'thresh': 1000*1e-6,
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
        'thresh': [-4, 4],
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
        'thresh': [None, 3.0],
        'time_window': 0.500,
        'time_window_step': 0.100,
        'mask': 0,
        'remove_bct': True,
        'remove_bt': True,
        'remove_bc': True,
    }, algorithm_name='HugeMaxChange500ms')
    artcfg.add_algorithm('huge_artifacts', 'ShortGoodSegments', {
        'time_limit': 1.000,
    }, post_detection=True)
    artcfg.add_algorithm('huge_artifacts', 'Mask', {
        'mask_length': 0.500,
    }, post_detection=True)

    artcfg.save_to_json(Path(__file__).parent / "default_cfg" / 'detect_artifacts_huge_config.json')

    # %% Create a default configuration for detecting glitches
    # ---------------------------------------------------------------------------------
    
    artcfg_huge = ArtifactsConfiguration()   
    artcfg_huge.load_from_json(Path(__file__).parent / "default_cfg" / 'detect_artifacts_huge_config.json')

    artcfg_glitches = ArtifactsConfiguration()
    
    artcfg_glitches.add_algorithm_group('glitches', max_loops=2, min_rejection=0, define_bcbt=True)
    artcfg_glitches.add_algorithm('glitches', 'MaxChange', {
        'bad_data': None,
        'do_reference_data': False,
        'do_zscore': False,
        'time_window': 0.020,
        'time_window_step': 0.005,
        'thresh_type': 'outliers_per_channel',
        'thresh': [None, 2],
        'mask': 0,
        'remove_bct': True,
        'remove_bt': True,
        'remove_bc': True,
    }, algorithm_name='GlitchesMaxChange')
    
    artcfg_glitches.add_algorithm('glitches', 'ShortBadSegments', {
        'time_limit': 0.010,
    }, post_detection=True)

    artcfg = concatenate_configurations([artcfg_huge.cfg, artcfg_glitches.cfg])
    artcfg.save_to_json(Path(__file__).parent / "default_cfg" / 'detect_artifacts_glitches_config.json')

    # %% Create a default configuration for detecting artifacts
    # ---------------------------------------------------------------------------------
    artcfg = ArtifactsConfiguration()

    artcfg.add_algorithm_group('huge_amplitude_abs', max_loops=1, min_rejection=0, define_bcbt=True)
    artcfg.add_algorithm('huge_amplitude_abs', 'Amplitude', {
        'bad_data': None,
        'do_reference_data': False,
        'do_zscore': False,
        'thresh_type': 'absolute',
        'thresh': 1000*1e-6,
        'mask': 0,
        'remove_bct': True,
        'remove_bt': True,
        'remove_bc': True,
    }, algorithm_name='HugeAmplitudeAbsolute')

    artcfg.add_algorithm_group('artifacts_amplitude', max_loops=5, min_rejection=min_rejection_thresh_2_0_IQ, define_bcbt=True)
    artcfg.add_algorithm('artifacts_amplitude', 'Amplitude', {
        'bad_data': None,
        'do_reference_data': False,
        'do_zscore': False,
        'thresh_type': 'outliers_per_channel',
        'thresh': [-2, 2],
        'mask': 0,
        'remove_bct': True,
        'remove_bt': True,
        'remove_bc': True,
    }, algorithm_name='ArtifactsAmplitude')

    artcfg.add_algorithm_group('artifacts_maxchange500', max_loops=5, min_rejection=min_rejection_thresh_2_0_IQ, define_bcbt=True)
    artcfg.add_algorithm('artifacts_maxchange500', 'MaxChange', {
        'bad_data': None,
        'do_reference_data': False,
        'do_zscore': False,
        'thresh_type': 'outliers_per_channel',
        'thresh': [None, 2.0],
        'time_window': 0.500,
        'time_window_step': 0.100,
        'mask': 0,
        'remove_bct': True,
        'remove_bt': True,
        'remove_bc': True,
    }, algorithm_name='MaxChange500ms')

    artcfg.add_algorithm_group('artifacts_maxchange100', max_loops=5, min_rejection=min_rejection_thresh_2_0_IQ, define_bcbt=True)
    artcfg.add_algorithm('artifacts_maxchange100', 'MaxChange', {
        'bad_data': None,
        'do_reference_data': False,
        'do_zscore': False,
        'thresh_type': 'outliers_per_channel',
        'thresh': [None, 2.0],
        'time_window': 0.100,
        'time_window_step': 0.020,
        'mask': 0,
        'remove_bct': True,
        'remove_bt': True,
        'remove_bc': True,
    }, algorithm_name='MaxChange100ms')

    # artcfg.add_algorithm_group('artifacts_runningaverage', max_loops=3, min_rejection=min_rejection_thresh_2_0_IQ, define_bcbt=True)
    # artcfg.add_algorithm('artifacts_runningaverage', 'RunningAverage', {
    #     'bad_data': None,
    #     'do_reference_data': False,
    #     'do_zscore': False,
    #     'thresh_type': 'outliers_per_channel',
    #     'thresh_fast': [-2, 2],
    #     'thresh_diff': [-3, 3],
    #     'mask': 0,
    #     'remove_bct': True,
    #     'remove_bt': True,
    #     'remove_bc': True,
    # }, algorithm_name='ArtifactsRunningAverage')

    # artcfg.add_algorithm_group('artifacts_crosselectrodesoutlier', max_loops=1, min_rejection=0, define_bcbt=True)
    # artcfg.add_algorithm('artifacts_crosselectrodesoutlier', 'CrossElectrodesOutlier', {
    #     'bad_data': None,
    #     'time_window':0.020,
    #     'time_window_step': 0.005,
    #     'thresh': 2.5,
    #     'mask': 0,
    #     'remove_bct': True,
    #     'remove_bt': True,
    #     'remove_bc': True,
    # }, algorithm_name='CrossElectrodesOutlier')

    artcfg.add_algorithm_group('artifacts_amplitude_avgref', max_loops=5, min_rejection=min_rejection_thresh_2_0_IQ, define_bcbt=True)
    artcfg.add_algorithm('artifacts_amplitude_avgref', 'Amplitude', {
        'bad_data': 'replace by nan',
        'do_reference_data': True,
        'do_zscore': False,
        'thresh_type': 'outliers_all',
        'thresh': [-2, 2],
        'mask': 0,
        'remove_bct': True,
        'remove_bt': True,
        'remove_bc': True,
    }, algorithm_name='ArtifactsAmplitudeAvgRef')

    artcfg.add_algorithm_group('artifacts_maxchange500_avgref', max_loops=5, min_rejection=min_rejection_thresh_2_0_IQ, define_bcbt=True)
    artcfg.add_algorithm('artifacts_maxchange500_avgref', 'MaxChange', {
        'bad_data': 'replace by nan',
        'do_reference_data': True,
        'do_zscore': False,
        'thresh_type': 'outliers_all',
        'thresh': [None, 2.0],
        'time_window': 0.500,
        'time_window_step': 0.100,
        'mask': 0,
        'remove_bct': True,
        'remove_bt': True,
        'remove_bc': True,
    }, algorithm_name='MaxChange500msAvgRef')

    artcfg.add_algorithm_group('artifacts_maxchange100_avgref', max_loops=5, min_rejection=min_rejection_thresh_2_0_IQ, define_bcbt=True)
    artcfg.add_algorithm('artifacts_maxchange100_avgref', 'MaxChange', {
        'bad_data': 'replace by nan',
        'do_reference_data': True,
        'do_zscore': False,
        'thresh_type': 'outliers_all',
        'thresh': [None, 2.0],
        'time_window': 0.100,
        'time_window_step': 0.020,
        'mask': 0,
        'remove_bct': True,
        'remove_bt': True,
        'remove_bc': True,
    }, algorithm_name='MaxChange100msAvgRef')

    # artcfg.add_algorithm_group('artifacts_runningaverage_avgref', max_loops=3, min_rejection=min_rejection_thresh_2_0_IQ, define_bcbt=True)
    # artcfg.add_algorithm('artifacts_runningaverage_avgref', 'RunningAverage', {
    #     'bad_data': 'replace by nan',
    #     'do_reference_data': True,
    #     'do_zscore': False,
    #     'thresh_type': 'outliers_all',
    #     'thresh_fast': [-2, 2],
    #     'thresh_diff': [-3, 3],
    #     'mask': 0,
    #     'remove_bct': True,
    #     'remove_bt': True,
    #     'remove_bc': True,
    # }, algorithm_name='ArtifactsRunningAverage')

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

    artcfg.save_to_json(Path(__file__).parent / "default_cfg" / 'detect_artifacts_all_config.json')

    # %% Create the default configuration for detecting bad data before ICA
    # ---------------------------------------------------------------------------------
    artcfg_bad_cha = ArtifactsConfiguration()

    artcfg_bad_cha.add_algorithm_group('bad_channels_basic', max_loops=1, min_rejection=0, define_bcbt=True)
    artcfg_bad_cha.add_algorithm('bad_channels_basic', 'FlatChannel', {
        'bad_data': None,
        'do_reference_data': False,
        'do_zscore': False,
        'time_window': 10,
        'time_window_step': 5,
        'min_change': 1e-7,
        'thresh': 5,
        'mask': 0,
    })
    artcfg_bad_cha.add_algorithm('bad_channels_basic', 'ChannelCorr', {
        'bad_data': None,
        'do_reference_data': False,
        'do_zscore': False,
        'time_window': 10,
        'time_window_step': 5,
        'top_channel_corr': 5,
        'thresh': 0.4,
        'mask': 0,
    })
    
    artcfg_bad_cha.add_algorithm_group('bad_channels_modify', max_loops=1, min_rejection=0, define_bcbt=True)
    artcfg_bad_cha.add_algorithm('bad_channels_modify', 'ShortGoodSegments', {
        'time_limit': 10,
    }, post_detection=True)
    artcfg_bad_cha.add_algorithm('bad_channels_modify', 'ShortBadSegments', {
        'time_limit': 0.500,
    }, post_detection=True)


    artcfg_huge = ArtifactsConfiguration()   
    
    artcfg_huge.add_algorithm_group('huge_amplitude_abs', max_loops=1, min_rejection=0, define_bcbt=True)
    artcfg_huge.add_algorithm('huge_amplitude_abs', 'Amplitude', {
        'bad_data': None,
        'do_reference_data': False,
        'do_zscore': False,
        'thresh_type': 'absolute',
        'thresh': 1000*1e-6,
        'mask': 0,
        'remove_bct': True,
        'remove_bt': True,
        'remove_bc': True,
    }, algorithm_name='HugeAmplitudeAbsolute')

    artcfg_huge.add_algorithm_group('huge_artifacts', max_loops=1, min_rejection=0, define_bcbt=True)
    artcfg_huge.add_algorithm('huge_artifacts', 'Amplitude', {
        'bad_data': None,
        'do_reference_data': False,
        'do_zscore': False,
        'thresh_type': 'outliers_per_channel',
        'thresh': [-4, 4],
        'mask': 0,
        'remove_bct': True,
        'remove_bt': True,
        'remove_bc': True,
    }, algorithm_name='HugeAmplitude')
    artcfg_huge.add_algorithm('huge_artifacts', 'MaxChange', {
        'bad_data': None,
        'do_reference_data': False,
        'do_zscore': False,
        'thresh_type': 'outliers_per_channel',
        'thresh': [None, 3.0],
        'time_window': 0.500,
        'time_window_step': 0.100,
        'mask': 0,
        'remove_bct': True,
        'remove_bt': True,
        'remove_bc': True,
    }, algorithm_name='HugeMaxChange500ms')
    
    artcfg_huge.add_algorithm('huge_artifacts', 'ShortGoodSegments', {
        'time_limit': 0.500,
    }, post_detection=True)
    artcfg_huge.add_algorithm('huge_artifacts', 'ShortBadSegments', {
        'time_limit': 0.100,
    }, post_detection=True)


    artcfg = concatenate_configurations([artcfg_bad_cha.cfg, artcfg_huge.cfg])
    artcfg.save_to_json(Path(__file__).parent / "default_cfg" / 'detect_for_ica_config.json')

    # %% Create default configuration for BC and BT definition for ICA
    # define_bcbt_raw_cfg = {}
    # define_bcbt_raw_cfg['bcbt_method'] = 'fix'
    # define_bcbt_raw_cfg['thresh_bad_channels'] = [0.7, 0.5]
    # define_bcbt_raw_cfg['thresh_bad_times'] = [0.7, 0.5]
    # define_bcbt_raw_cfg['min_good_time'] = 2.000
    # define_bcbt_raw_cfg['min_bad_time'] = 0.020
    # define_bcbt_raw_cfg['mask_time'] = 0.500
    
    define_bcbt_raw_cfg = {}
    define_bcbt_raw_cfg['bcbt_method'] = 'functional'
    define_bcbt_raw_cfg['thresh_bad_channels'] = 0.50
    define_bcbt_raw_cfg['thresh_bad_times'] = 0.50
    define_bcbt_raw_cfg['min_good_time'] = 2.000
    define_bcbt_raw_cfg['min_bad_time'] = 0.020
    define_bcbt_raw_cfg['mask_time'] = 0.500
    # save to json
    with open(Path(__file__).parent / "default_cfg" / 'define_bcbt_raw_ica_config.json', 'w') as f:
        json.dump(define_bcbt_raw_cfg, f, indent=4)


if __name__ == '__main__':
    main()