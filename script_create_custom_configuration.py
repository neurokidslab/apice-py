"""
This script creates custom configuration files for different steps of the artifacts rejection pipeline.
The configuration files are saved in the specified output directory in JSON format.

Modify the parameters passed to each function to create configurations that suit your specific needs.
Comment out any section you do not need.

The script creates configurations for:
- Detecting bad epochs
- Defining bad channels and bad times
- Correcting artifacts using PCA and spline methods
- Detecting bad channels based on basic and power features
- Detecting huge artifacts and glitches
- Detecting motion artifacts
"""

import json
from pathlib import Path

from apice.artifacts_rejection import concatenate_configurations
from apice.standard_conf import (
    cfg_bad_epochs,
    cfg_define_bcbt_epochs,
    cfg_define_bcbt_raw,
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

    # Input / Output parameters
    # ============================================================================================

    # Directory for output configuration files
    OUTPUT_DIR = r"cfg_output_folder"
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # ============================================================================================
    # %% Bad epochs
    cfg_bad_epochs(
        bad_data=1.00,
        bad_time=0,
        bad_channel=0.30,
        lim_dist=2,
        lim_gfp=2,
        filename=Path(OUTPUT_DIR) / 'detect_bad_epochs_config.json',
    )

    # ============================================================================================
    # %% BC and BT definition (raw and epochs)
    cfg_define_bcbt_raw(
        bcbt_method='functional',
        thresh_bad_channels=0.30,
        thresh_bad_times=0.30,
        min_good_time=1.000,
        min_bad_time=0.100,
        mask_time=0,
        filename=Path(OUTPUT_DIR) / 'define_bcbt_raw_config.json',
    )
    cfg_define_bcbt_epochs(
        bcbt_method='functional',
        thresh_bad_channels=0.10,
        thresh_bad_times=0.30,
        min_good_time=1.000,
        min_bad_time=0.100,
        mask_time=0,
        filename=Path(OUTPUT_DIR) / 'define_bcbt_epochs_config.json',
    )

    # ============================================================================================
    # %% Artifact correction
    cfg_correction_target_pca(
        max_time=0.125,
        components_to_remove=None,
        variance_to_remove=0.99,
        mask_time=0.050,
        all_time='all',
        all_channel='no_bad_channel',
        all_epochs='all',
        splice_method=1,
        save_corrected=True,
        filename=Path(OUTPUT_DIR) / 'correction_target_pca_config.json',
    )
    cfg_correction_spline_segments(
        p=0.5,
        p_neighbors=1,
        min_good_time=1.00,
        min_intertime=0.050,
        mask_time=0.100,
        min_segment_time=0.250,
        splice_method=1,
        parallelize_mode='auto',
        save_corrected=True,
        filename=Path(OUTPUT_DIR) / 'correction_spline_segments_config.json',
    )
    cfg_correction_spline_channels(
        p=0.4,
        p_neighbors=1,
        save_corrected=True,
        filename=Path(OUTPUT_DIR) / 'correction_spline_channels_config.json',
    )

    # ============================================================================================
    # %% Bad channel detection (basic + power)
    cfg_badcha = cfg_detect_bad_channels(filename=None)
    cfg_power = cfg_detect_power(
        rejection_level=3,
        min_rejection=None,
        max_loops=5,
        filename=None,
    )
    artcfg = concatenate_configurations([cfg_badcha, cfg_power])
    with open(Path(OUTPUT_DIR) / 'detect_bad_channels_config.json', 'w') as f:
        json.dump(artcfg.cfg, f, indent=4)

    # ============================================================================================
    # %% Huge artifacts and glitches
    cfg_huge = cfg_detect_artifacts_huge(
        rejection_level=4,
        abs_thresh_amp=1000 * 1e-6,
        filename=None,
    )
    cfg_glitches = cfg_detect_artifacts_glitches(
        rejection_level=2,
        min_rejection=0,
        max_loops=2,
        filename=None,
    )
    artcfg = concatenate_configurations([cfg_huge, cfg_glitches])
    with open(Path(OUTPUT_DIR) / 'detect_artifacts_glitches_config.json', 'w') as f:
        json.dump(artcfg.cfg, f, indent=4)

    # ============================================================================================
    # %% Motion artifacts
    cfg_detect_artifacts_motion(
        rejection_level=3,
        min_rejection=None,
        max_loops=5,
        abs_thresh_amp=1000 * 1e-6,
        filename=Path(OUTPUT_DIR) / 'detect_artifacts_motion_config.json',
    )


if __name__ == '__main__':
    main()

