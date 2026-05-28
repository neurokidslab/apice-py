# APICE Pipeline — Documentation

> **Companion to the [tutorial notebook](../apice_tutorial.ipynb).** For installation and quickstart see the [README](../README.md).

---

## Index

**Part A — Conceptual Pipeline Overview**

1. [Preprocessing Pipeline](#1-preprocessing-pipeline)
   - [1.1 Filtering](#11-filtering)
   - [1.2 Artifact Detection](#12-artifact-detection)
     - [The Rejection Matrices](#the-rejection-matrices)
     - [Algorithm Groups and Cycles](#algorithm-groups-and-cycles)
     - [Detection Algorithms](#detection-algorithms)
     - [Inferring Bad Channels and Bad Times](#inferring-bad-channels-and-bad-times)
   - [1.3 ICA (Optional)](#13-ica-optional)
   - [1.4 Artifact Correction](#14-artifact-correction)
     - [Target PCA](#target-pca)
     - [Segment-wise Spline Interpolation](#segment-wise-spline-interpolation)
     - [Channel-wise Spline Interpolation](#channel-wise-spline-interpolation)
   - [1.5 Default Preprocessing Sequence](#15-default-preprocessing-sequence)
2. [Segmentation Pipeline](#2-segmentation-pipeline)
   - [2.1 Epoching and Artifact Transfer](#21-epoching-and-artifact-transfer)
   - [2.2 Epoch-level BC/BT Derivation](#22-epoch-level-bcbt-derivation)
   - [2.3 Per-epoch Channel Interpolation](#23-per-epoch-channel-interpolation)
   - [2.4 Bad Epoch Definition](#24-bad-epoch-definition)
   - [2.5 Evoked Response Computation](#25-evoked-response-computation)

**Part B — Technical Reference**

3. [Data Structures](#3-data-structures)
   - [RawAPICE](#rawapice)
   - [EpochsAPICE](#epochsapice)
4. [Artifact Structure Classes](#4-artifact-structure-classes)
5. [Detection Algorithm Classes](#5-detection-algorithm-classes)
6. [Configuration and Execution](#6-configuration-and-execution)
   - [ArtifactsConfiguration](#artifactsconfiguration)
   - [JSON Configuration Schema](#json-configuration-schema)
   - [run_algorithms Execution Flow](#run_algorithms-execution-flow)
7. [Artifact Correction Classes](#7-artifact-correction-classes)
8. [IO Functions](#8-io-functions)
9. [ERP Statistics](#9-erp-statistics)

---

## Part A — Conceptual Pipeline Overview

---

### 1. Preprocessing Pipeline

The APICE preprocessing pipeline converts raw EEG into clean, artifact-labelled data ready for segmentation. When called via `run_preprocessing` (batch mode) the pipeline proceeds in three phases: (1) initial structural preprocessing and filtering; (2) optional ICA-based component removal; (3) the APICE artifact detection and correction loop. `preprocess_apice_default` covers phase 3 only and can be called directly on already-preprocessed data.

---

#### 1.1 Filtering

Three filter stages run before artifact processing:

1. **High-pass filter** (default 0.1 Hz) — removes slow baseline drifts. A minimal high-pass is always applied because many detection algorithms assume a reasonably stationary signal.
2. **ZapLine notch filter** (optional) — removes line noise (50 or 60 Hz) and its harmonics using a spatial filter ([de Cheveigné 2020](https://doi.org/10.1016/j.neuroimage.2019.116356)). Preferred over a classic notch filter because it does not distort neighbouring frequencies.
3. **Low-pass filter** (default 40 Hz) — removes high-frequency content unrelated to EEG.

> Within `preprocess_apice_default`, detection algorithms can optionally operate on a separately filtered copy of the data (see `l_freq_artifacts`, `h_freq_artifacts`), allowing detection at a different bandwidth than the final preprocessed output.

---

#### 1.2 Artifact Detection

##### The Rejection Matrices

All artifact information is stored in boolean matrices attached to the data object (`raw.artifacts`):

| Matrix | Shape (raw) | Shape (epochs) | Meaning |
|--------|-------------|----------------|---------|
| **BCT** | `(n_channels, n_samples)` | `(n_epochs, n_channels, n_samples)` | A specific channel at a specific time sample is artifactual |
| **BC**  | `(n_channels, 1)` | `(n_epochs, n_channels, 1)` | A channel is globally bad |
| **BT**  | `(1, n_samples)` | `(n_epochs, 1, n_samples)` | A time sample is bad across all (functional) channels |
| **BE**  | — | `(n_epochs, 1, 1)` | An entire epoch is rejected (epochs only) |
| **CCT** | `(n_channels, n_samples)` | `(n_epochs, n_channels, n_samples)` | A sample was repaired by a correction algorithm |

BCT is the primary mask produced directly by detection algorithms. BC and BT are derived from BCT as described [below](#inferring-bad-channels-and-bad-times). All algorithms accumulate results into BCT using a logical OR — a sample, once marked bad, stays bad unless the matrix is explicitly reset.

---

##### Algorithm Groups and Cycles

Detection algorithms are organised into named **groups** run in sequence. Within each group the execution proceeds in three phases:

**1 — Compute (once per group)**

Every detection algorithm calls `compute(raw)` to build a feature matrix from the current data (amplitude, variance, spectral power, etc.). This expensive step runs only once per group, before the rejection loop.

**2 — Rejection loop (up to `max_loops` iterations)**

At each iteration every algorithm calls `reject(raw)`, which applies its threshold to the pre-computed feature and returns a per-sample BCT mask. Results of all algorithms within the group are OR-combined into `bct_new`, which is then OR-merged into the cumulative `raw.artifacts.BCT`.

Iterating matters because rejection interacts with itself: once some bad channels and bad times are identified, later rejection passes operate on a cleaner signal and may flag additional artifacts that were masked before. The loop terminates early once the newly rejected fraction falls below `min_rejection` (provided `min_loops` have already run), avoiding unnecessary passes on converged data.

**3 — Post-detection (once per group)**

A separate set of algorithms (`post_detection=True`) runs once after the loops. These do not detect new artifacts but clean up the rejection matrix — for example, filling short good intervals surrounded by bad data, or applying a temporal buffer around bad segments.

**4 — Derive BC and BT** (if `define_bcbt=True`)

After a group completes, BC and BT are recomputed from BCT using the fixed-point algorithm described below, making the updated masks available to the next group.

This design makes detection adaptive: early groups typically run lenient detectors that flag obvious artifacts; derived BC/BT exclude these from subsequent computations, making later or stricter detectors more precise.

---

##### Detection Algorithms

All algorithms share a common interface via the `DetectionMethod` base class: `compute(raw)` pre-computes the feature, `reject(raw)` applies the threshold. Each can optionally average-reference the data (`do_reference_data=True`) or z-score across channels (`do_zscore=True`) before computing the feature.

| Class | Detects | How |
|-------|---------|-----|
| `Amplitude` | Samples with abnormally large or small absolute amplitude | Thresholds raw amplitude; supports absolute or IQR-relative thresholds per channel |
| `RunningAverage` | Local amplitude anomalies and rapid fluctuations | Computes two sliding averages with different window widths (fast and slow); flags samples where the fast average deviates from the slow baseline, or where the fast average itself is too large |
| `TimeVariance` | Time windows with unusually high variance | Slides a window across time; flags windows where variance is a statistical outlier across channels and epochs |
| `MaxChange` | Rapid transient jumps (electrode pops, glitches) | Computes the maximum consecutive sample-to-sample amplitude difference within a short window; flags outlier windows |
| `CrossElectrodesOutlier` | Samples that are spatially atypical across channels | Z-scores each time window across all channels; flags channel-time points that deviate too far from the cross-channel distribution — sensitive to localised but temporally extended bursts |
| `Power` | Anomalous spectral power in a specified frequency band | Estimates power in short windows; flags windows exceeding a threshold — useful for residual line noise, muscle activity, or broadband bursts |
| `ChannelCorr` | Channels poorly correlated with their neighbours | Computes mean top-*k* correlation with spatially adjacent channels in sliding windows; flags channels that are consistently uncorrelated — detects dead, shorted, or malfunctioning electrodes |
| `FlatChannel` | Flat or dead channels | Computes the proportion of near-zero-change samples within sliding windows; flags channels with an anomalously high flat-sample proportion |

Post-detection modifiers (not detection algorithms):

| Class | Purpose |
|-------|---------|
| `Mask` | Extends bad segments by a temporal buffer on each side |
| `ShortGoodSegments` | Absorbs short good intervals surrounded by bad data into the rejection |
| `ShortBadSegments` | Removes (un-marks) bad segments shorter than a minimum duration |

---

##### Inferring Bad Channels and Bad Times

After each detection group, `define_bcbt_functional` converts the cumulative BCT into BC and BT via **fixed-point iteration**:

1. Initialise: BC = manually forced bad channels, BT = empty.
2. Repeat until convergence:
   - **Update BT**: for each time sample, compute the fraction of *non-BC channels* flagged in BCT. Flag the sample as BT if the fraction exceeds `thresh_bad_times`.
   - **Update BC**: for each channel, compute the fraction of *non-BT samples* flagged in BCT. Flag the channel as BC if the fraction exceeds `thresh_bad_channels`.
3. Stop when neither BC nor BT changes.

The convergence guarantee is important: at the fixed point, every BC channel truly exceeds the threshold when computed over non-BT time, and every BT sample truly exceeds the threshold when computed over non-BC channels. This mutual consistency is not guaranteed by multi-cycle approaches that apply progressively stricter thresholds without iterating to convergence.

---

#### 1.3 ICA (Optional)

ICA-based component removal is an optional step (`apply_ica=True` in `run_preprocessing`) that runs **before** the main APICE artifact detection and correction pipeline. It is implemented in `clean_ica` (in `apice/ica.py`).

**Purpose:** Remove physiological noise sources (eye blinks, muscle, cardiac, line noise) that are pervasive across the recording and therefore not well handled by the sample-level rejection approach — particularly systematic blink activity and cardiac artifacts.

**Pre-ICA artifact detection (on a working copy)**

Before fitting ICA, a copy of the data is made and a dedicated detection pass is run on it to identify segments that should be *excluded from the ICA fit*. The algorithms used (defined by `detect_for_ica_config.json`) are:

| Group | Algorithms | Purpose |
|-------|-----------|--------|
| `bad_channels_basic` | `FlatChannel`, `ChannelCorr` | Identify flat or spatially uncorrelated channels |
| `huge_amplitude_abs` | `Amplitude` (absolute threshold 1 mV) | Catch saturations and amplifier clip artefacts |
| `huge_artifacts` | `Amplitude` (4 IQR outliers per channel) + `MaxChange` (4 IQR outliers, 500 ms window) | Detect large-amplitude transients and rapid-change events |

Post-detection modifiers (`ShortBadSegments`, `ShortGoodSegments`, `Mask`) clean up and extend the resulting rejection mask. Bad channels are added to `raw.info['bads']` so the ICA fitting step can exclude them.

**ICA fitting**

The working copy is high-pass filtered at **1 Hz** (default `l_freq_ica=1`) to remove slow drifts that can destabilise the ICA decomposition, while keeping frequencies relevant for identifying ocular and cardiac components. ICA is then fitted on this copy with `reject_by_annotation=True` so that all bad segments flagged in the detection pass above are excluded from the covariance estimate. The number of components is set automatically (`n_components='auto'`) using the rule $n \leq \sqrt{m/30}$ where $m$ is the number of clean samples and $n$ is the number of channels.

**Component labelling**

Components are labelled using [mne-icalabel](https://mne.tools/mne-icalabel) (ICLabel, default) or correlation-based detectors (`find_bads_eog` / `find_bads_ecg`). Components classified as eye blink, muscle artifact, heart beat, or line noise above a probability threshold (`iclabel_lim_probability=0.9`) are selected for removal.

**Application to original data**

The selected artifact components are subtracted from a **copy of the original (unfiltered) data** — not from the 1 Hz filtered copy. The subtraction is performed by reconstructing only the artifact components and subtracting them from the raw signal (`raw._data -= artifact_components._data`). This preserves the broadband characteristics of the data.

**Important:** the artifact detection performed on the working copy before ICA is used only to guide the ICA fit and is **not transferred** to the output. The output `raw_clean` is a clean copy of the original data with no APICE artifact matrices. Artifact detection starts fresh in the subsequent APICE default pipeline.

---

#### 1.4 Artifact Correction

APICE implements three correction methods that repair artifacts in the data, reducing the amount that must ultimately be rejected.

---

##### Target PCA

**What it corrects:** Short-duration transient artifacts (default max 100 ms) where a large-amplitude component is shared across multiple channels — electrode pops, amplifier resets, brief EMG bursts.

**How it works:** For each bad segment identified in BCT, PCA is applied to the selected channels over the affected time window. The leading components — which capture most variance and represent the common artifact — are removed, and the remaining components are back-projected to reconstruct a corrected signal. The number of components removed can be fixed or determined by a cumulative variance threshold (`variance_to_remove`, default 0.98). A splice step tapers the transitions at segment boundaries to minimise discontinuities.

After correction a high-pass re-filter is applied to remove low-frequency artefacts that may have been introduced at the boundaries.

---

##### Segment-wise Spline Interpolation

**What it corrects:** Longer bad channel-time segments (BCT entries) where the artifact is spatially localised — affecting only a minority of channels at a given time. Rather than rejecting the entire time window, the affected channels are repaired by interpolation from their good neighbours.

**How it works:** For each contiguous bad segment, the correction checks whether the proportion of bad channels is below a threshold (`p`, default 0.5). If so, each affected channel is reconstructed using **spherical spline interpolation** (Perrin et al. method): the signal is estimated from nearby good-quality electrodes weighted by an inverse-distance spline on the scalp sphere. A splice step reduces boundary discontinuities.

This is the most effective correction for localised artifacts confined to a few electrodes per time window (e.g., sweat, single-electrode movement). After correction a high-pass re-filter is applied.

---

##### Channel-wise Spline Interpolation

**What it corrects:** Channels that are globally bad throughout the recording (in BC) and cannot be recovered by segment-wise correction because the artifact is persistent.

**How it works:** For each BC channel, the entire time series is replaced with a spherical spline reconstruction from the surrounding good channels, subject to the same proportional constraints on bad neighbours. Conceptually equivalent to MNE's standard interpolation, but operating on the APICE artifact matrices and parallelised across channels.

---

#### 1.5 Default Preprocessing Sequence

`run_preprocessing` orchestrates three phases. The table below shows the full default step sequence:

**Phase 1 — Initial steps** (`preprocess_initial_steps`)

| # | Step | Notes |
|---|------|-------|
| 1 | Drop / pick channels (optional) | Remove or subset channels |
| 2 | Crop / resample (optional) | Time-domain trimming or resampling |
| 3 | Stim channels → annotations (optional) | Events transferred to annotation track |
| 4 | Set montage (optional) | Electrode positions required for spline interpolation |
| 5 | High-pass filter (0.1 Hz) | Removes slow drifts |
| 6 | ZapLine 50 Hz (default) | Removes line noise |

**Phase 2 — ICA** (`clean_ica`, only when `apply_ica=True`)

| # | Step | Notes |
|---|------|-------|
| 7 | Detect flat channels + channel correlation + huge amplitude artifacts on a working copy | `FlatChannel`, `ChannelCorr`, `Amplitude`, `MaxChange` (500 ms); bad data annotated on copy |
| 8 | Filter working copy to 1 Hz HP | Stabilises the ICA decomposition |
| 9 | Fit ICA excluding annotated bad segments | `n_components` set automatically; Picard algorithm by default |
| 10 | Label artifact components (ICLabel / correlation) | Eye blink, muscle, cardiac, line noise |
| 11 | Subtract artifact components from **original data** | Pre-ICA detection is **not kept** — output has no artifact matrices |

**Phase 3 — APICE default pipeline** (`preprocess_apice_default`)

| # | Step | Notes |
|---|------|-------|
| 12 | High-pass filter (0.1 Hz) | Re-applied to ICA-cleaned data |
| 13 | ZapLine (optional) | Configured separately from Phase 1 |
| 14 | Low-pass filter (40 Hz) | Removes high-frequency content |
| 15 | Initialise artifact structures | BCT / BC / BT / CCT matrices created |
| 16 | **Detect bad channels** | Flags globally broken/noisy channels (BC) |
| 17 | Protect reference channels | Reference electrode excluded from rejection |
| 18 | **Detect glitches** | BCT updated with transient spikes and pops |
| 19 | Protect reference channels | |
| 20 | **Correct glitches — Target PCA** | Short artifact segments removed; high-pass re-applied |
| 21 | **Detect artifacts** | BCT updated with motion, muscle, and other artifacts |
| 22 | Protect reference channels | |
| 23 | **Correct segments — Segment-wise spline** | Localised bad segments repaired; high-pass re-applied |
| 24 | **Correct channels — Channel-wise spline** | Globally bad channels interpolated |
| 25 | **Re-detect artifacts** | Final sweep to flag remaining bad data after correction |
| 26 | Protect reference channels | |
| 27 | Export | `.fif`, HTML report, summary CSV, JSON configs |

---

### 2. Segmentation Pipeline

The segmentation pipeline converts a preprocessed `RawAPICE` object into epochs, identifies bad epochs, and computes evoked responses. Entry points are `run_segmentation` (batch) and `segment_apice_default` / `compute_epochs_and_evoked` (single file).

---

#### 2.1 Epoching and Artifact Transfer

Events are extracted from MNE annotations using `mne.events_from_annotations`. The raw data is then cut into epochs around each event. During segmentation, the raw-level artifact matrices (BCT, BC, BT) are transferred to the corresponding time windows of the epoch-level matrices so that no prior rejection information is lost.

---

#### 2.2 Epoch-level BC/BT Derivation

After transfer, `define_bcbt` is applied **per epoch independently** using the same fixed-point algorithm. This means that a channel that was globally bad in the raw recording will be BC in every epoch, but a channel that was transiently bad in only a subset of epochs will be BC only in those epochs — enabling more granular data recovery. Thresholds are set via `cfg_define_bcbt_epochs`.

---

#### 2.3 Per-epoch Channel Interpolation

For each epoch, channels flagged as BC in that epoch (but not necessarily across all epochs) are candidates for spherical spline interpolation from good neighbours. This allows recovery of epochs where only a few channels are locally bad — preserving the epoch rather than rejecting it outright. The same proportional constraints as the channel-wise spline apply.

After interpolation, BC/BT are re-derived to reflect the corrected data.

---

#### 2.4 Bad Epoch Definition

An epoch is flagged as bad (BE) if it fails **any** of the following criteria:

| Criterion | Parameter | Default | What is checked |
|-----------|-----------|---------|-----------------|
| Artifact ratio | `bad_data` | 1.0 | Fraction of BCT entries that are True in the epoch |
| Bad-time ratio | `bad_time` | 0 | Fraction of BT time points in the epoch |
| Bad-channel ratio | `bad_channel` | 0.3 | Fraction of BC channels in the epoch |
| Mahalanobis distance | `lim_dist` | 2.0 | Distance of the epoch's ERP from the average ERP (z-score units); flags epochs with an unusual topography |
| Global Field Power | `lim_gfp` | 2.0 | Z-score of the epoch's mean GFP relative to all epochs; flags epochs with abnormally large responses |

The first three criteria operate on artifact masks alone. The distance and GFP criteria operate on the EEG signal itself and can catch epochs that look clean by mask criteria but contain unusual waveforms (e.g., sporadic high-amplitude responses not captured by detection).

---

#### 2.5 Evoked Response Computation

Evoked responses are computed from epochs not flagged as bad (BE = False). The `evoked_by` parameter controls grouping:

- `"all"` — one grand-average across all good epochs.
- A list of event labels — separate averages per event type.
- `None` — skip evoked computation.

---

## Part B — Technical Reference

---

### 3. Data Structures

#### RawAPICE

`RawAPICE` (in `apice/data_structures.py`) extends `mne.io.RawArray` and adds APICE artifact structures and convenience methods. Created by wrapping any `mne.io.BaseRaw`:

```python
from apice import RawAPICE
raw_apice = RawAPICE(raw, thresh_bad_channels=0.3, thresh_bad_times=0.3)
```

**Key attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `artifacts` | `ArtifactsRaw` | BCT, BC, BT, CCT matrices and configuration |
| `_data` | `ndarray (n_channels, n_samples)` | EEG data; also accessible via `.get_data()` |
| `info` | `mne.Info` | Channel info, sampling frequency, montage |

**Key methods:**

| Method | Description |
|--------|-------------|
| `get_data_size()` | Returns `(n_channels, n_samples, n_epochs=1)` |
| `detect_bad_channels(cfg)` | Run bad-channel detection |
| `detect_glitches(cfg)` | Run glitch detection |
| `detect_artifacts(cfg)` | Run full artifact detection |
| `correct_target_pca(cfg)` | Apply Target PCA correction |
| `correct_spline_segments(cfg)` | Apply segment-wise spline interpolation |
| `correct_spline_channels(cfg)` | Apply channel-wise spline interpolation |
| `define_bcbt()` | Recompute BC/BT from BCT |
| `deal_with_reference_channels(chs)` | Exclude reference channels from rejection |
| `plot_artifact_structure(...)` | Heatmap visualisation of BCT/BC/BT |
| `plot_percentage_of_bad_data_across_sensors()` | Topomap of bad-data percentages |
| `segment_continuous_data(events, event_id)` | Create `EpochsAPICE` and transfer artifact masks |
| `export(path, overwrite)` | Save to FIF with artifact annotations embedded |
| `annotations_to_rejection_matrix()` | Reload artifact matrices from FIF annotations |

---

#### EpochsAPICE

`EpochsAPICE` extends `mne.EpochsArray`. Produced by `raw.segment_continuous_data(...)` or loaded from disk via `load_epochapice`.

**Key attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `artifacts` | `ArtifactsEpochs` | BCT, BC, BT, BE, CCT matrices and `rejection_reasons` |
| `_data` | `ndarray (n_epochs, n_channels, n_samples)` | Epoched EEG data |

**Key methods:**

| Method | Description |
|--------|-------------|
| `get_data_size()` | Returns `(n_channels, n_samples, n_epochs)` |
| `define_bcbt()` | Recompute per-epoch BC/BT from BCT |
| `define_bad_epochs(bad_data, bad_time, bad_channel, lim_dist, lim_gfp)` | Flag bad epochs |
| `correct_spline_channels(cfg)` | Per-epoch channel interpolation |
| `remove_bad_epochs()` | Drop epochs flagged in BE |
| `rejection_matrix_to_data_frame()` | Export artifact masks as a long-form DataFrame |
| `dataframe_to_rejection_matrix()` | Import artifact masks from a DataFrame |
| `plot_artifact_structure(...)` | Heatmap visualisation |
| `plot_rejection_summary()` | Summary figure of rejection reasons |
| `export(path, overwrite)` | Save epochs FIF; artifact CSV saved alongside |

---

### 4. Artifact Structure Classes

The `Artifacts` base class (in `apice/artifacts_structure.py`) initialises and holds the rejection matrices. Two concrete subclasses cover the two data types:

| Class | Attached to | Extra attributes |
|-------|-------------|-----------------|
| `ArtifactsRaw` | `RawAPICE` | BCT `(n_ch, n_s)`, BC `(n_ch, 1)`, BT `(1, n_s)`, CCT `(n_ch, n_s)` |
| `ArtifactsEpochs` | `EpochsAPICE` | All of the above + BE `(n_ep, 1, 1)`, `rejection_reasons` (list of sets per epoch) |

Common methods: `update_bc`, `update_bt`, `set_bc`, `set_bt`, `reset_bc`, `reset_bt`, `include_short_bad_segments`, `reject_short_good_segments`, `mask_bad_segments`, `define_bcbt`, `print_summary`.

**BC/BT derivation algorithm** (selected at initialisation via `bcbt_method`):

| Value | Algorithm | Description |
|-------|-----------|-------------|
| `'functional'` *(default)* | `define_bcbt_functional` | Single threshold, fixed-point iteration, convergence guaranteed |
| `'fix'` | `define_bcbt_fix` | Multi-cycle with progressively stricter thresholds (legacy) |

---

### 5. Detection Algorithm Classes

All detection algorithms are in `apice/artifacts_detection.py` and inherit from `DetectionMethod`.

**`DetectionMethod` interface:**

| Method | Called by | Description |
|--------|-----------|-------------|
| `compute(raw)` | `run_algorithms` (once per group) | Computes and caches the feature matrix |
| `reject(raw)` | `run_algorithms` (each loop iteration) | Applies threshold; returns `(raw, bct)` |

**Detection classes:**

| Class | Feature | Threshold |
|-------|---------|-----------|
| `Amplitude` | Raw EEG amplitude | Absolute or IQR-relative per channel |
| `RunningAverage` | Fast running average; fast − slow difference | Two independent IQR-relative thresholds |
| `TimeVariance` | Rolling-window variance | Upper IQR-relative threshold per channel |
| `MaxChange` | Max sample-to-sample difference in sliding windows | Upper IQR-relative threshold (cube-root transformed) |
| `CrossElectrodesOutlier` | Z-scored amplitude across channels in sliding windows | Absolute threshold on z-score (mandatory z-scoring) |
| `Power` | Mean spectral power in a specified frequency band in sliding windows | Upper IQR-relative threshold per channel |
| `ChannelCorr` | Mean top-*k* correlation with spatial neighbours in sliding windows | Lower absolute threshold (low correlation = bad) |
| `FlatChannel` | Proportion of near-zero-change samples in sliding windows | Upper absolute threshold (high flat-proportion = bad) |

**Post-detection modifiers (`ModifyRejection` subclasses):**

| Class | Purpose |
|-------|---------|
| `Mask` | Extends each bad segment by a temporal buffer |
| `ShortGoodSegments` | Marks short good intervals flanked by bad data as bad |
| `ShortBadSegments` | Removes bad segments shorter than a minimum duration |

---

### 6. Configuration and Execution

#### ArtifactsConfiguration

`ArtifactsConfiguration` (in `apice/artifacts_rejection.py`) builds and validates detection pipeline configurations as a nested dictionary that round-trips to/from JSON.

```python
from apice.artifacts_rejection import ArtifactsConfiguration

cfg = ArtifactsConfiguration()

cfg.add_algorithm_group("glitches", min_loops=1, max_loops=3,
                        min_rejection=0.01, define_bcbt=True)
cfg.add_algorithm("glitches", "MaxChange",
                  {"thresh": [None, 2.0], "time_window": 0.1})
cfg.add_algorithm("glitches", "ShortBadSegments",
                  {"min_bad_time": 0.05}, post_detection=True)

cfg.save_to_json("my_glitch_config.json")
```

**Key methods:**

| Method | Description |
|--------|-------------|
| `add_algorithm_group(name, min_loops, max_loops, min_rejection, position, define_bcbt)` | Add a named algorithm group |
| `add_algorithm(add_to, class_name, parameters, position, algorithm_name, post_detection)` | Add an algorithm to a group |
| `save_to_json(path)` | Serialise configuration to JSON |
| `load_from_json(path)` | Load configuration from JSON |
| `check_configuration()` | Validate class names and parameter signatures |
| `concatenate_configurations(list_of_cfgs)` | Merge multiple configuration objects |

Ready-made configuration builders for standard use cases are in `apice/standard_conf.py` (e.g., `cfg_detect_bad_channels()`, `cfg_detect_artifacts_motion()`). The script `script_create_custom_configuration.py` demonstrates how to build and export custom configurations.

Default configurations shipped with the package live in `apice/default_cfg/`.

---

#### JSON Configuration Schema

```json
{
  "group_name": {
    "position": 1,
    "min_loops": 1,
    "max_loops": 3,
    "min_rejection": 0.01,
    "define_bcbt": true,
    "algorithms": {
      "algorithm_label": {
        "position": 1,
        "class_name": "MaxChange",
        "post_detection": false,
        "parameters": {
          "thresh": [null, 2.0],
          "time_window": 0.1
        }
      }
    }
  }
}
```

**Group-level fields:** `position` (execution order among groups), `min_loops`, `max_loops`, `min_rejection`, `define_bcbt`.

**Algorithm-level fields:** `position` (order within the group), `class_name` (must match a class in `apice.artifacts_detection`), `post_detection` (bool), `parameters` (constructor keyword arguments for that class).

---

#### run_algorithms Execution Flow

```
run_algorithms(raw, cfg)
│
├─ Load and validate configuration
│
└─ For each group (sorted by position):
    │
    ├─ [Compute phase]  for each detection algorithm:
    │      algorithm.compute(raw)           ← runs once per group
    │
    ├─ [Rejection loop]  for loop = 1 … max_loops:
    │      bct_new = all-False
    │      for each detection algorithm:
    │          raw, bct = algorithm.reject(raw)
    │          bct_new |= bct
    │      new_rejection% = (bct_new AND NOT BCT).sum() / BCT.size
    │      if new_rejection% < min_rejection AND loop > min_loops:
    │          break   ← discard this loop's results, do not update BCT
    │      raw.artifacts.BCT |= bct_new
    │
    ├─ [Post-detection]  for each post_detection algorithm:
    │      raw, bct = algorithm.reject(raw)
    │
    └─ if define_bcbt:
           raw.artifacts.define_bcbt()     ← update BC and BT from BCT
```

---

### 7. Artifact Correction Classes

All correction classes are in `apice/artifacts_correction.py` and inherit from `ArtCorrection`, which manages parameter loading, logging, and bookkeeping of corrected samples (CCT update).

| Class | Corrects | Key parameters |
|-------|---------|----------------|
| `TargetPCA` | Short transient segments via PCA component removal | `max_time` (max segment duration, default 0.1 s), `variance_to_remove` (default 0.98), `splice_method` |
| `SegmentSphericalSplineInterpolation` | BCT segments where a minority of channels are bad | `p` (max bad-channel proportion, default 0.5), `min_intertime` (min bad-segment duration), `parallelize_mode` (`'auto'`, `'channels'`, `'segments'`) |
| `ChannelsSphericalSplineInterpolation` | Globally bad channels (BC) for the whole recording or per epoch | `p` (max bad-channel proportion, default 0.3), `p_neighbors` (max bad-neighbour proportion, default 1) |

Correction classes are invoked through `RawAPICE.correct_target_pca(cfg)`, `correct_spline_segments(cfg)`, `correct_spline_channels(cfg)`, and `EpochsAPICE.correct_spline_channels(cfg)`, which load parameters from JSON configuration files.

---

### 8. IO Functions

All IO utilities are in `apice/io.py`.

**Loading data back into APICE objects:**

| Function | Purpose |
|----------|---------|
| `load_rawapice(fname, ...)` | Load a preprocessed `.fif` and reconstruct `RawAPICE` with artifact matrices from embedded annotations |
| `load_epochapice(fname, ...)` | Load an epochs `.fif` and reconstruct `EpochsAPICE` with artifact matrices from the associated `.csv` file |

**Discovering files to process:**

| Function | Purpose |
|----------|---------|
| `get_files_to_process(input_dir, output_dir, data_selection_method, processed_file_pattern)` | Return raw files from a flat directory. `data_selection_method` accepts `'all'`, `'new'` (skip already processed), or a list of glob patterns (e.g. `['*sub-01*', '*sub-02*']`) |
| `get_bids_files_to_process(bids_root, ...)` | Return `mne_bids.BIDSPath` objects from a BIDS dataset, with optional filters for subject, session, task, run, datatype, and extension |

**Supported raw file formats:** `.fif`, `.mat`, `.vhdr`, `.bdf`, `.cnt`, `.edf`, `.set`, `.egi`, `.mff`, `.nxe`, `.gdf`, `.data`, `.lay`, `.raw`.

---

### 9. ERP Statistics

`compute_sme` (in `apice/statistics.py`, exposed as `apice.compute_sme`) computes the **Standardised Measurement Error** (SME; [Luck et al. 2021](https://doi.org/10.1111/psyp.13793)) — a bootstrap-based index of how reliably a mean-amplitude ERP measure can be estimated from the available trials. Lower SME indicates more stable ERP estimates.

```python
from apice import compute_sme

# Per-channel SME for a single condition
sme = compute_sme(epochs, condition='animal/bird',
                  start=0.3, stop=0.6,
                  n_iter=1000, random_state=42)   # shape (n_channels,)

# SME for a difference wave (cond1 − cond2)
sme_diff = compute_sme(epochs,
                       condition=['animal/bird', 'animal/mammal'],
                       start=0.3, stop=0.6)        # shape (n_channels,)

# ROI average → scalar
sme_roi = compute_sme(epochs, roi=['E55', 'E62', 'E79'],
                      start=0.3, stop=0.6)          # float
```

See the `compute_sme` docstring for full parameter descriptions.
