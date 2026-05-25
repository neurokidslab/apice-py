# APICE-Py: An Open-Source MNE-Python Pipeline for Scalable EEG Preprocessing

## Authors
- **Jhunlyn Lorenzo**
- **Nicolò Formento Moletta**
- **Ana Fló**
- **Ghislaine Dehaene-Lambertz**

---

APICE-Py is a modular and scalable EEG preprocessing pipeline built on top of [MNE-Python](https://mne.tools). It is designed for researchers and practitioners who require reproducible, customizable, and efficient preprocessing of EEG datasets. With support for batch processing, clear logging, and flexible parameters, APICE-Py accelerates EEG analysis workflows in both academic and clinical settings.

---

## � What's New (branch `feature/major-update`)

This branch contains a major update relative to `main` and will become version **0.2.0**. The changes below summarise the differences.

### Artifact detection engine — complete rewrite

The artifact detection system was redesigned from scratch to be simpler, more transparent and easier to extend:

- Configurations are now built with `ArtifactsConfiguration`, which organises detection into named **algorithm groups** that run independently.
- Each group iterates in a loop with an explicit `max_loops` ceiling. Looping stops early once the new rejection per pass falls below `min_rejection`, which is now computed automatically from the IQR threshold assuming a Gaussian distribution (`min_rejection_from_thresh`). This replaces the ad-hoc stopping criteria from the previous version.
- New algorithms can be added to any group by calling `add_algorithm()` without touching existing code.
- Bad channels (BC) and bad times (BT) are now derived from the bad-channel-time (BCT) matrix using `define_bcbt_functional`, a fixed-point iteration algorithm that alternates between updating BC and BT until convergence. This guarantees that every channel flagged as bad truly exceeds the threshold among non-bad time points, and vice versa — a consistency property that the previous multi-cycle approach did not guarantee.
- A new `FlatChannel` detector was added — flat channels (zero or near-zero variance) were often missed by the previous correlation-based approach.
- All detection results are stored in structured `Artifacts` matrices and round-trip correctly when loading/exporting `.fif` files (bug fixes for artifact import/export).

### Default and custom configurations

- Ready-made JSON configuration files are shipped inside `apice/default_cfg/` and are loaded automatically when no configuration is supplied.
- Users can generate their own JSON configs by calling the functions in `apice/standard_conf.py` (e.g. `cfg_detect_artifacts_motion`, `cfg_detect_bad_channels`) with custom parameters, or by editing `script_create_custom_configuration.py`.
- Artifact-detection configurations can also be built entirely from scratch in Python using `ArtifactsConfiguration` (`apice/artifacts_rejection.py`): call `add_algorithm_group()` to define groups and `add_algorithm()` to populate them, then export with `save_to_json()`.

### ZapLine line-noise removal

- A `ZapLine` class (`apice/filter.py`) implements the ZapLine spatial filter ([de Cheveigné 2020](https://doi.org/10.1016/j.neuroimage.2019.116356)) for removing line noise and its harmonics.

### Parallelised spline interpolation

- `correct_spline_segments` now supports three parallelisation modes controlled by `parallelize_mode`:
  - `'channels'` — original behaviour (parallelise across channels).
  - `'segments'` — parallelise across bad segments (faster for long recordings with many segments).
  - `'auto'` — chooses the faster mode automatically.
- For typical long EEG recordings, segment-level parallelisation roughly halves interpolation time.

### ICA

- A full ICA sub-pipeline was implemented (`apice/ica.py`):
  1. Strong artifacts are detected first (`cfg_detect_artifacts_huge`) to avoid ICA being driven by outlier epochs.
  2. ICA is fitted on the clean segments.
  3. Artifactual components are identified automatically using the [mne-icalabel](https://mne.tools/mne-icalabel) classifier.
- ICA is integrated into `run_preprocessing` via the `run_ica` parameter.

### Ready-to-run pipeline scripts

- `script_pipeline_preprocessing.py` — edit parameters at the top and run. Supports BIDS input (`input_dir_bids=True`), batch and new-files-only modes, and saves HTML reports, log files, per-step rejection summaries, and configuration snapshots — all controllable from the script parameters.
- `script_pipeline_segmentation.py` — equivalent script for bulk epoching, evoked-response computation, and segmentation output.

### BIDS format support

- `run_preprocessing` can read directly from a BIDS dataset via `input_dir_bids=True` with optional filters for subject, session, task, run, and file extension.

### ERP statistics

- `compute_sme` (`apice/erp_statistics.py`) computes the Standard Measurement Error (SME) on `EpochsAPICE` objects.

---

## 🚀 Features


- 🧠 Built with [MNE-Python](https://mne.tools)
- 🧩 Modular pipeline structure
- ⚙️ Configurable via CLI or direct file editing
- 🗂️ Batch-friendly design
- 🧼 Includes filtering, epoching, artifact rejection, and more
- � ICA with automatic component labeling via [mne-icalabel](https://mne.tools/mne-icalabel)
- 📄 Generates HTML preprocessing and segmentation reports
- 🔧 Programmatic configuration API (`standard_conf.py`) for custom pipelines
- 📐 ERP statistics: computes the Standard Measurement Error (SME)
- 📦 Optional BIDS format support
- 🧾 Outputs summary tables using PrettyTable and Tabulate
- 📊 Plots EEG data using Matplotlib and Seaborn
- ✅ Python 3.12+ support

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/neurokidslab/apice-py.git
cd apice-py
git switch feature/major-update
```

If your Git version does not support `git switch`, use:

```bash
git checkout feature/major-update
```

### 2. Create and activate a Python environment (recommended)

Using conda (recommended) — install into the [MNE conda environment](https://mne.tools/stable/install/index.html):

```bash
conda activate mne
```

Or using venv:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows
```

### 3. Install APICE in editable mode

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
```

This installs the package from `pyproject.toml` and links the local source, so changes in the `apice/` folder are immediately available without reinstalling.

Optional: if you only want to install dependency pins from the requirements file, run:

```bash
python -m pip install -r requirements.txt
```

Note: Python >= 3.12 is required.

---
## 📂 Sample Data

For sample input raw data, use the data in `test_data/raw`

---
## Project Structure

```text
apice-py/
├── apice/                          # Package source code
│   ├── __init__.py
│   ├── pipeline.py                 # High-level preprocessing pipeline entry points
│   ├── data_structures.py          # RawAPICE and EpochsAPICE custom containers
│   ├── artifacts_detection.py      # Artifact detection methods
│   ├── artifacts_correction.py     # Artifact correction methods
│   ├── artifacts_rejection.py      # Artifact rejection methods
│   ├── artifacts_structure.py      # Artifact matrices and management classes
│   ├── io.py                       # I/O helpers
│   ├── filter.py                   # Filtering utilities
│   ├── ica.py                      # ICA helpers (component correlation & cleaning)
│   ├── electrode_positions.py      # Electrode position helpers
│   ├── pipeline_utils.py           # Summary classes and pipeline logging
│   ├── erp_statistics.py           # ERP statistics (compute_sme)
│   ├── standard_conf.py            # Programmatic configuration builder functions
│   ├── _create_default_configuration.py  # Regenerates default JSON configs
│   ├── utils.py                    # Utility functions and config loading
│   └── default_cfg/                # Default JSON configurations bundled with the package
├── test_data/                      # Example recording for testing
├── electrode_layout/               # Example montage files (.sfp)
├── apice_tutorial.ipynb            # Tutorial notebook
├── script_create_custom_configuration.py
├── script_pipeline_preprocessing.py
├── script_pipeline_segmentation.py
├── pyproject.toml                  # Build system and package metadata
├── requirements.txt                # Optional dependency install list
├── LICENSE
└── README.md
```

---
## Usage

### 1. Run the default preprocessing script

Edit paths and parameters in `script_pipeline_preprocessing.py`, then run:

```bash
python script_pipeline_preprocessing.py
```

This script calls `run_preprocessing(...)` from `apice.pipeline` and writes preprocessed outputs to your configured output directory.

### 2. Run the default segmentation script

Edit paths and segmentation settings in `script_pipeline_segmentation.py`, then run:

```bash
python script_pipeline_segmentation.py
```

This script calls `run_segmentation(...)` from `apice.pipeline` and can save epochs, evokeds, report, summary, and config files.

### 3. Use APICE from your own Python code

Check the example notebook `apice_tutorial.ipynb`

```python
from apice.pipeline import run_preprocessing, run_segmentation

# Run preprocessing on your dataset
run_preprocessing(input_dir, output_dir, ...)

# Run segmentation on preprocessed data
run_segmentation(input_dir, output_dir, kwargs_events_from_annotations, event_time_window, ...)
```

### 4. Customize pipeline configurations

`script_create_custom_configuration.py` uses the `standard_conf` API to build configurations with custom parameters and save them as JSON files. Edit the parameter values in the script, set `OUTPUT_DIR` to your target folder, and run it:

```bash
python script_create_custom_configuration.py
```

You can also call the configuration functions directly in your own code:

```python
from apice.standard_conf import cfg_detect_artifacts_motion, cfg_correction_spline_channels

# Build a custom motion artifact detection config
cfg = cfg_detect_artifacts_motion(rejection_level=3, max_loops=5)
```

### 5. Compute ERP statistics

```python
from apice import compute_sme

# Compute the Standard Measurement Error on an EpochsAPICE object
sme = compute_sme(epochs, ...)
```

---
## 📖 Documentation

Click [here](https://zenodo.org/records/17151631) for the documentation (examples, customization guide, pipeline structure).

---
## 📖 Citation

If you use this repository in your research, please cite:

```bibtex
@misc{lorenzo_2025_17151631,
  author       = {Lorenzo, Jhunlyn and
                  Formento Moletta, Nicolò and
                  Fló, Ana and
                  Dehaene-Lambertz, Ghislaine},
  title        = {APICE-Py: An Open-Source MNE-Python Pipeline for
                   Scalable EEG Preprocessing
                  },
  month        = sep,
  year         = 2025,
  publisher    = {Zenodo},
  version      = {0.1.0},
  doi          = {10.5281/zenodo.17151631},
  url          = {https://doi.org/10.5281/zenodo.17151631},
}
```


---
## 🤝 Contributing
Contributions are welcome! Feel free to:

- Open an issue to report bugs or request features
- Suggest ideas for pipeline extensions

---
## 📜 License
This project is licensed under the Apache-2.0 license. See `LICENSE` for details.

---
## ✨ Acknowledgements
- Built with MNE-Python, an open-source EEG/MEG analysis package
- Inspired by best practices in open neuroscience workflows

