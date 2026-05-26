# APICE-Py: An Open-Source MNE-Python Pipeline for Scalable EEG Preprocessing

## Authors
- **Jhunlyn Lorenzo**
- **Nicolò Formento Moletta**
- **Ana Fló**
- **Ghislaine Dehaene-Lambertz**

---

APICE-Py is a modular and scalable EEG preprocessing pipeline built on top of [MNE-Python](https://mne.tools). It is designed for researchers and practitioners who require reproducible, customizable, and efficient preprocessing of EEG datasets. With support for batch processing, clear logging, and flexible parameters, APICE-Py accelerates EEG analysis workflows in both academic and clinical settings.

---

## Contents

- [Features](#-features)
- [What’s New (v0.2.0)](#-whats-new-v020)
- [Installation](#installation)
- [Sample Data](#-sample-data)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Documentation](#-documentation)
- [Citation](#-citation)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)

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

## 🆕 What’s New (v0.2.0)

Key changes from v0.1.0:

- **Artifact detection engine rewrite** — named algorithm groups, loop-based iteration with early stopping, new `FlatChannel` detector, fixed artifact import/export. → [pipeline docs §5–6](docs/pipeline.md#5-detection-algorithm-classes)
- **Configurations** — default JSON configs in `apice/default_cfg/`; custom configs via `ArtifactsConfiguration` API. → [pipeline docs §6](docs/pipeline.md#6-configuration-and-execution)
- **`define_bcbt_functional`** — fixed-point BC/BT derivation guaranteeing mutual consistency. → [pipeline docs §1.2](docs/pipeline.md#inferring-bad-channels-and-bad-times)
- **ZapLine** — spatial line-noise filter (de Chevigné 2020) replacing classic notch.
- **Parallelised spline interpolation** — three modes: `'channels'`, `'segments'`, `'auto'`.
- **ICA sub-pipeline** — pre-ICA detection on a working copy, 1 Hz-filtered fit, automatic component labelling via [mne-icalabel](https://mne.tools/mne-icalabel). → [pipeline docs §1.3](docs/pipeline.md#13-ica-optional)
- **Ready-to-run scripts** — `script_pipeline_preprocessing.py` and `script_pipeline_segmentation.py` with BIDS support.
- **ERP statistics** — `compute_sme` computes the Standard Measurement Error (SME) on `EpochsAPICE` objects.

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

- **[Pipeline documentation](docs/pipeline.md)** — conceptual overview of all processing steps (filtering, artifact detection, correction, segmentation) and a technical reference for all main classes, configuration files, and IO functions.
- **[Published documentation](https://zenodo.org/records/17151631)** — examples and customisation guide for the published release (v0.1.0).

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

