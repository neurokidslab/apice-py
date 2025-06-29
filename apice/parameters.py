"""
This file contains user input or default parameters.
Change the parameters values if necessary.

- Parameters are initialized as classes to be inherited in specific
  classes rather than setting them globally.

"""

# Import libraries
import numpy as np

# %% Parameters

"""Artifacts Detection and Correction Threshold"""
thresh = 3


"""Filters"""


class Filters:
    high_pass_freq = 0.1
    low_pass_freq = 40

class Filters_epochs:
    high_pass_freq = 0.2
    low_pass_freq = 40


"""Parameters defining the overall artifacts"""


class Artifacts:
    PARAMS = dict(
        low_pass_freq=None,
        high_pass_freq=None,
        keep_rejected_previous=True,
        keep_rejection_cause=False,
        max_loop_num=None,
        rejection_tolerance=0,
        limit_rejection_subject=True,
    )


# %%
""" ARTIFACTS: BAD ELECTRODES """


class BadElectrodes:
    """Parameters"""

    """Parameters for rejecting electrodes based on correlation between channels"""
    ChannelCorr = dict(
        algorithm='ChannelCorr',
        loop_num=[1],
        # Sliding time window
        time_window=4,  # seconds
        time_window_step=2,  # seconds
        # Rejection parameters
        top_channel_corr=5,
        # Correlation threshold
        thresh=0.4,
        # Other options in performing the rejection algorithm
        do_reference_data=False,
        do_zscore=False,
        use_relative_thresh=False
    )

    """Parameters for rejecting electrodes based on the power spectrum"""
    Power = dict(
        algorithm='Power',
        loop_num=[2],
        # Sliding time window
        time_window=4,  # seconds
        time_window_step=2,  # seconds
        # Filter characteristics
        freq_band=[[1, 10], [20, 40]],  # Hz
        thresh=[[-thresh, np.inf], [-np.inf, thresh]],
        # Other options in performing the rejection algorithm
        do_reference_data=False,
        do_zscore=True,
        use_relative_thresh=[True, True]
    )

    """Parameters for rejecting short but good segments"""
    ShortGoodSegments = dict(
        algorithm='ShortGoodSegments',
        loop_num=[2],
        time_limit=2  # seconds
    )

    """Algorithms for rejecting bad electrodes"""
    PARAMS = dict(ChannelCorr=ChannelCorr,
                  Power=Power,
                  ShortGoodSegments=ShortGoodSegments)


# %%
""" ARTIFACTS: MOTION ARTIFACTS (TYPE1) """


class Motion1:
    """Parameters"""

    """ABSOLUTE threshold for ALL electrodes"""

    """Algorithm for rejecting data where the amplitude is higher than an ABSOLUTE threshold"""
    AmplitudeAbsoluteThresh = dict(
        algorithm='Amplitude',
        loop_num=[1],
        # Rejection parameters
        thresh=500 * 1e-6,  # uV
        # Mask bad segments
        mask=0.500,  # seconds
        # Other options in performing the rejection algorithm
        do_reference_data=False,
        do_zscore=False,
        use_relative_thresh=False,
        use_relative_thresh_per_electrode=False
    )

    """NON average reference data, RELATIVE threshold for EACH electrode"""

    """Algorithm for rejecting data where the amplitude is higher than an RELATIVE threshold"""
    AmplitudeRelativeThresh = dict(
        algorithm='Amplitude',
        loop_num=[2, 3],
        # Rejection parameters
        thresh=thresh,  # upper threshold
        # Mask bad segments
        mask=0.050,  # seconds
        # Other options in performing the rejection algorithm
        do_reference_data=False,
        do_zscore=False,
        use_relative_thresh=True,
        use_relative_thresh_per_electrode=True
    )

    """Algorithm for rejecting data based on time variance"""
    TimeVariance = dict(
        algorithm='TimeVariance',
        loop_num=[2, 3],
        # Sliding time window
        time_window=0.50,  # seconds
        time_window_step=0.10,  # seconds
        # Rejection Parameters
        thresh=[-thresh, thresh],  # lower and upper threshold
        # Other options in performing the rejection algorithm
        do_reference_data=False,
        do_zscore=False,
        use_relative_thresh=True,
        use_relative_thresh_per_electrode=True
    )

    """Algorithm for rejecting data based on the weighted running average"""
    RunningAverage = dict(
        algorithm='RunningAverage',
        loop_num=[2, 3],
        # Rejection parameters
        thresh_fast=thresh,
        thresh_diff=thresh,
        # Mask bad segments
        mask=0.05,  # seconds
        # Other options in performing the rejection algorithm
        do_reference_data=False,
        do_zscore=False,
        use_relative_thresh=True,
        use_relative_thresh_per_electrode=True
    )

    """Average reference data, RELATIVE threshold for ALL electrodes"""

    """Algorithm for rejecting data where the amplitude is higher than an RELATIVE threshold"""
    AmplitudeRelativeThresh_AverageReferenced = dict(
        algorithm='Amplitude',
        loop_num=[4, 5],
        # Rejection parameters
        thresh=thresh,  # upper threshold
        # Replace bad data by
        bad_data='replace by nan',
        # Mask bad segments
        mask=0.050,  # seconds
        # Other options in performing the rejection algorithm
        do_reference_data=True,
        do_zscore=False,
        use_relative_thresh=True,
        use_relative_thresh_per_electrode=False
    )

    """Algorithm for rejecting data based on time variance"""
    TimeVariance_AverageReferenced = dict(
        algorithm='TimeVariance',
        loop_num=[4, 5],
        # Sliding time window
        time_window=0.50,  # seconds
        time_window_step=0.10,  # seconds
        # Replace bad data by
        bad_data='replace by nan',
        # Rejection Parameters
        thresh=[-thresh, thresh],  # lower and upper threshold
        # Other options in performing the rejection algorithm
        do_reference_data=True,
        do_zscore=False,
        use_relative_thresh=True,
        use_relative_thresh_per_electrode=False
    )

    """Algorithm for rejecting data based on the weighted running average"""
    RunningAverage_AverageReferenced = dict(
        algorithm='RunningAverage',
        loop_num=[4, 5],
        # Rejection parameters
        thresh_fastrunning=thresh,
        thresh_diffrunninh=thresh,
        # Replace bad data by
        bad_data='replace by nan',
        # Mask bad segments
        mask=0.05,  # seconds
        # Other options in performing the rejection algorithm
        do_reference_data=True,
        do_zscore=False,
        use_relative_thresh=True,
        use_relative_thresh_per_electrode=False
    )

    """Algorithm for rejecting data based on the variance across electrodes"""
    AmplitudeVariance = dict(
        algorithm='AmplitudeVariance',
        loop_num=[5],
        # Rejection parameters
        thresh=thresh,
        # Replace bad data by
        bad_data='replace by nan',
        # Mask bad segments
        mask=0.05,  # seconds
        # Other options in performing the rejection algorithm
        do_reference_data=True,
        do_zscore=False
    )

    """Include/Reject data based on rejected data"""

    """Parameters for including short but bad segments"""
    ShortBadSegments = dict(
        algorithm='ShortBadSegments',
        loop_num=[5],
        time_limit=0.02  # seconds
    )

    """Parameters for rejecting short but good segments"""
    ShortGoodSegments = dict(
        algorithm='ShortGoodSegments',
        loop_num=[5],
        time_limit=2  # seconds
    )

    PARAMS = dict(
        AmplitudeAbsoluteThresh=AmplitudeAbsoluteThresh,
        AmplitudeRelativeThresh=AmplitudeRelativeThresh,
        TimeVariance=TimeVariance,
        RunningAverage=RunningAverage,
        AmplitudeRelativeThresh_AverageReferenced=AmplitudeRelativeThresh_AverageReferenced,
        TimeVariance_AverageReferenced=TimeVariance_AverageReferenced,
        RunningAverage_AverageReferenced=RunningAverage_AverageReferenced,
        AmplitudeVariance=AmplitudeVariance,
        ShortBadSegments=ShortBadSegments,
        ShortGoodSegments=ShortGoodSegments
    )


# %%
""" ARTIFACTS: MOTION ARTIFACTS (TYPE1) """


class Motion2:
    """Parameters"""

    """ABSOLUTE threshold for ALL electrodes"""

    """Algorithm for rejecting data where the amplitude is higher than an ABSOLUTE threshold"""
    AmplitudeAbsoluteThresh = dict(
        algorithm='Amplitude',
        loop_num=[1],
        # Rejection parameters
        thresh=500 * 1e-6,  # uV
        # Mask bad segments
        mask=0.500,  # seconds
        # Other options in performing the rejection algorithm
        do_reference_data=False,
        do_zscore=False,
        use_relative_thresh=False,
        use_relative_thresh_per_electrode=False
    )

    """NON average reference data, RELATIVE threshold for EACH electrode"""

    """Algorithm for rejecting data where the amplitude is higher than an RELATIVE threshold"""
    AmplitudeRelativeThresh = dict(
        algorithm='Amplitude',
        loop_num=[2, 3],
        # Rejection parameters
        thresh=thresh,  # upper threshold
        # Mask bad segments
        mask=0.050,  # seconds
        # Other options in performing the rejection algorithm
        do_reference_data=False,
        do_zscore=False,
        use_relative_thresh=True,
        use_relative_thresh_per_electrode=True
    )

    """Algorithm for rejecting data based on time variance"""
    TimeVariance = dict(
        algorithm='TimeVariance',
        loop_num=[2, 3],
        # Sliding time window
        time_window=0.50,  # seconds
        time_window_step=0.10,  # seconds
        # Rejection Parameters
        thresh=[-thresh, thresh],  # lower and upper threshold
        # Other options in performing the rejection algorithm
        do_reference_data=False,
        do_zscore=False,
        use_relative_thresh=True,
        use_relative_thresh_per_electrode=True
    )

    """Algorithm for rejecting data based on the weighted running average"""
    RunningAverage = dict(
        algorithm='RunningAverage',
        loop_num=[2, 3],
        # Rejection parameters
        thresh_fast=thresh,
        thresh_diff=thresh,
        # Mask bad segments
        mask=0.05,  # seconds
        # Other options in performing the rejection algorithm
        do_reference_data=False,
        do_zscore=False,
        use_relative_thresh=True,
        use_relative_thresh_per_electrode=True
    )

    """Average reference data, RELATIVE threshold for ALL electrodes"""

    """Algorithm for rejecting data where the amplitude is higher than an RELATIVE threshold"""
    AmplitudeRelativeThresh_AverageReferenced = dict(
        algorithm='Amplitude',
        loop_num=[4, 5],
        # Rejection parameters
        thresh=thresh,  # upper threshold
        # Replace bad data by
        bad_data='replace by nan',
        # Mask bad segments
        mask=0.050,  # seconds
        # Other options in performing the rejection algorithm
        do_reference_data=True,
        do_zscore=False,
        use_relative_thresh=True,
        use_relative_thresh_per_electrode=False
    )

    """Algorithm for rejecting data based on time variance"""
    TimeVariance_AverageReferenced = dict(
        algorithm='TimeVariance',
        loop_num=[4, 5],
        # Sliding time window
        time_window=0.50,  # seconds
        time_window_step=0.10,  # seconds
        # Replace bad data by
        bad_data='replace by nan',
        # Rejection Parameters
        thresh=[-thresh, thresh],  # lower and upper threshold
        # Other options in performing the rejection algorithm
        do_reference_data=True,
        do_zscore=False,
        use_relative_thresh=True,
        use_relative_thresh_per_electrode=False
    )

    """Algorithm for rejecting data based on the weighted running average"""
    RunningAverage_AverageReferenced = dict(
        algorithm='RunningAverage',
        loop_num=[4, 5],
        # Rejection parameters
        thresh_fastrunning=thresh,
        thresh_diffrunninh=thresh,
        # Replace bad data by
        bad_data='replace by nan',
        # Mask bad segments
        mask=0.05,  # seconds
        # Other options in performing the rejection algorithm
        do_reference_data=True,
        do_zscore=False,
        use_relative_thresh=True,
        use_relative_thresh_per_electrode=False
    )

    """Algorithm for rejecting data based on the variance across electrodes"""
    AmplitudeVariance = dict(
        algorithm='AmplitudeVariance',
        loop_num=[5],
        # Rejection parameters
        thresh=thresh,
        # Replace bad data by
        bad_data='replace by nan',
        # Mask bad segments
        mask=0.05,  # seconds
        # Other options in performing the rejection algorithm
        do_reference_data=True,
        do_zscore=False
    )

    """Include/Reject data based on rejected data"""

    """Parameters for including short but bad segments"""
    ShortBadSegments = dict(
        algorithm='ShortBadSegments',
        loop_num=[5],
        time_limit=0.1  # seconds
    )

    """Parameters for rejecting short but good segments"""
    ShortGoodSegments = dict(
        algorithm='ShortGoodSegments',
        loop_num=[5],
        time_limit=2  # seconds
    )

    PARAMS = dict(
        AmplitudeAbsoluteThresh=AmplitudeAbsoluteThresh,
        AmplitudeRelativeThresh=AmplitudeRelativeThresh,
        TimeVariance=TimeVariance,
        RunningAverage=RunningAverage,
        AmplitudeRelativeThresh_AverageReferenced=AmplitudeRelativeThresh_AverageReferenced,
        TimeVariance_AverageReferenced=TimeVariance_AverageReferenced,
        RunningAverage_AverageReferenced=RunningAverage_AverageReferenced,
        AmplitudeVariance=AmplitudeVariance,
        ShortBadSegments=ShortBadSegments,
        ShortGoodSegments=ShortGoodSegments
    )


# %%
""" ARTIFACTS: JUMP ARTIFACTS """


class Jump:
    """Parameters"""

    """NON average reference data, RELATIVE threshold for EACH electrode"""

    """Algorithm for rejecting data if a big change occurs in a small interval"""

    FastChange = dict(
        algorithm='FastChange',
        loop_num=[1],
        # Rejection parameters
        thresh=thresh,
        # Replace bad data by
        bad_data='none',
        # Time window
        time_window=0.020,
        # Other options in performing the rejection algorithm
        do_reference_data=False,
        do_zscore=False,
        use_relative_thresh=True,
        use_relative_thresh_per_electrode=True
    )

    """Average reference data, RELATIVE threshold for ALL electrode"""

    """Algorithm for rejecting data if a big change occurs in a small interval"""
    FastChange_AverageReferenced = dict(
        algorithm='FastChange',
        loop_num=[2],
        # Rejection parameters
        thresh=thresh,
        # Replace bad data by
        bad_data='replace by nan',
        # Time window
        time_window=0.020,
        # Other options in performing the rejection algorithm
        do_reference_data=True,
        do_zscore=False,
        use_relative_thresh=True,
        use_relative_thresh_per_electrode=False
    )

    """Include/Reject data based on rejected data"""

    """Parameters for including short but bad segments"""
    ShortBadSegments = dict(
        algorithm='ShortBadSegments',
        loop_num=[2],
        time_limit=0.020  # seconds
    )

    """Parameters for rejecting short but good segments"""
    ShortGoodSegments = dict(
        algorithm='ShortGoodSegments',
        loop_num=[2],
        time_limit=0.100  # seconds
    )

    PARAMS = dict(
        FastChange=FastChange,
        FastChange_AverageReferenced=FastChange_AverageReferenced,
        ShortBadSegments=ShortBadSegments,
        ShortGoodSegments=ShortGoodSegments
    )


# %% DEFINE BAD SAMPLES AND CHANNELS

class BTBC_Definition:
    """Parameters to define Bad Times (BT) and Bad Channels (BC) """

    """Bad Times (BT) parameters"""
    # Limits for the proportion of BT to define a BC (the last value is the final/effective one)
    thresh_bad_times = [0.70, 0.50, 0.30]

    """Bad Channels (BC) parameters"""
    # Limits for the proportion of BC to define a BT (the last value is the final/effective one)
    thresh_bad_channels = [0.70, 0.50, 0.30]
    # Shorter intervals between bad segments will be marked as bad
    min_good_time = 1.000
    # Shorter periods will not be considered as bad
    min_bad_time = 0.100
    # Also mark as bad surrounding samples within this value
    mask_time = 0.500

class BTBC_Definition_Epochs:
    """Parameters to define Bad Times (BT) and Bad Channels (BC) """
    """Bad Times (BT) parameters"""
    # Limits for the proportion of BT to define a BC (the last value is the final/effective one)
    thresh_bad_times = [0.7000, 0.5000, 0.3000, 0.1000]
    """Bad Channels (BC) parameters"""
    # Limits for the proportion of BC to define a BT (the last value is the final/effective one)
    thresh_bad_channels = [0.7000, 0.5000, 0.3000, 0.300]
    # Shorter intervals between bad segments will be marked as bad
    min_good_time = 1
    # Shorter periods will not be considered as bad
    min_bad_time = 0.100
    # Also mark as bad surrounding samples within this value
    mask_time = 0

class BE_Definition:
    """Parameters to define Bad Epochs (BE) """
    # Maximum proportion of bad data per epoch
    bad_data = 1.00
    # Maximum proportion of bad times per epoch
    bad_time = 0.00
    # Maximum proportion of bad channels per epoch
    bad_channel = 0.30
    # Maximum proportion of interpolated data per epoch
    corrected_data = 0.50


# %% ARTIFACT CORRECTIONS

class PCA:
    max_time = 0.100
    # Number of principal components to remove from the data
    components_to_remove = []
    # Proportion of variance to remove from the data
    variance_to_remove = 0.9
    splice_method = 1  # how the interpolated segments are splice in the data
    # 0 = none
    # 1 = segments are aligned with the previous segment
    mask_time = 0.05
    all_time = 'no_bad_time'  # 'all' | 'no_bad_time' | 'bad_time'
    all_channels = 'no_bad_channel'  # 'all' | 'no_bad_channel' | 'bad_channel'


class Spherical_Spline_Interpolation:
    p = 0.5  # maximum proportion of bad channels in a segment to interpolate
    p_neighbors = 1  # maximum proportion of bad neighbor channels
    min_good_time = 2  # minimum length of segments to be considered as good
    min_intertime = 0.10  # minimum length of a segment in order to be interpolated
    mask_time = 1  # time to mask bad segments before interpolation in samples
    splice_method = 1  # how the interpolated segments are splice in the data
    # 0 = none
    # 1 = segments are aligned with the previous segment
