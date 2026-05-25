"""Build and save default APICE JSON configuration files.

This script generates default detection, correction, and summary configuration
files in ``apice/default_cfg``.
"""

from pathlib import Path
import json

from apice.artifacts_rejection import concatenate_configurations

from apice.standard_conf import (
    cfg_bad_epochs,
    cfg_define_bcbt_epochs,
    cfg_define_bcbt_raw,
    cfg_define_bcbt_raw_ica,
    cfg_correction_target_pca,
    cfg_correction_spline_segments,
    cfg_correction_spline_channels,
    cfg_detect_bad_channels,
    cfg_detect_power,
    cfg_detect_artifacts_huge,
    cfg_detect_artifacts_glitches,
    cfg_detect_artifacts_motion,
)


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

    cfg_dir = Path(__file__).parent / "default_cfg"

    # artifacts detection - bad channels 
    cfg_badcha = cfg_detect_bad_channels(filename=None)
    cfg_power = cfg_detect_power(filename=None)
    artcfg = concatenate_configurations([cfg_badcha, cfg_power])
    filename=cfg_dir / 'detect_bad_channels_config.json'
    with open(filename, 'w') as f:
        json.dump(artcfg.cfg, f, indent=4)

    # artifacts detection - glitches
    cfg_huge = cfg_detect_artifacts_huge(filename=None)
    cfg_glitches = cfg_detect_artifacts_glitches(filename=None)
    artcfg = concatenate_configurations([cfg_huge, cfg_glitches])
    filename = cfg_dir / 'detect_artifacts_glitches_config.json'
    with open(filename, 'w') as f:
        json.dump(artcfg.cfg, f, indent=4)

    # artifacts detection - motion
    cfg_detect_artifacts_motion(filename=cfg_dir / 'detect_artifacts_motion_config.json')


    # define BC and BT for raw and epochs
    cfg_define_bcbt_epochs(filename=cfg_dir / 'define_bcbt_epochs_config.json')
    cfg_define_bcbt_raw(filename=cfg_dir / 'define_bcbt_raw_config.json')
    
    # artifacts correction
    cfg_correction_target_pca(filename=cfg_dir / 'correction_target_pca_config.json')
    cfg_correction_spline_segments(filename=cfg_dir / 'correction_spline_segments_config.json')
    cfg_correction_spline_channels(filename=cfg_dir / 'correction_spline_channels_config.json')
    
    # define bad epochs
    cfg_bad_epochs(filename=cfg_dir / 'detect_bad_epochs_config.json')
    
    

    # artifacts detection - for ICA
    cfg_badcha = cfg_detect_bad_channels(filename=None)
    cfg_huge = cfg_detect_artifacts_huge(filename=None)
    artcfg = concatenate_configurations([cfg_badcha, cfg_huge])
    filename=cfg_dir / 'detect_for_ica_config.json'
    with open(filename, 'w') as f:
        json.dump(artcfg.cfg, f, indent=4)

    # define BC and BT for raw ICA
    cfg_define_bcbt_raw_ica(filename=cfg_dir / 'define_bcbt_raw_ica_config.json')


if __name__ == '__main__':
    main()