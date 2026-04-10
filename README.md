# APICE-Py: An Open-Source MNE-Python Pipeline for Scalable EEG Preprocessing

## Authors
- **Jhunlyn Lorenzo**
- **Nicolò Formento Moletta**
- **Ana Fló**
- **Ghislaine Dehaene-Lambertz**

---

APICE-Py is a modular and scalable EEG preprocessing pipeline built on top of [MNE-Python](https://mne.tools). It is designed for researchers and practitioners who require reproducible, customizable, and efficient preprocessing of EEG datasets. With support for batch processing, clear logging, and flexible parameters, APICE-Py accelerates EEG analysis workflows in both academic and clinical settings.

---

## 🚀 Features

- 🧠 Built with [MNE-Python](https://mne.tools)
- 🧩 Modular pipeline structure
- ⚙️ Configurable via CLI or direct file editing
- 🗂️ Batch-friendly design
- 🧼 Includes filtering, epoching, artifact rejection, and more
- 🧾 Outputs summary tables using PrettyTable and Tabulate
- 📊 Plots EEG data using Matplotlib
- ✅ Python 3.12+ support

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/neurokidslab/apice-py.git
cd apice-py
```

### 2. Create and activate a Python environment (recommended)

```bash
python -m venv .venv
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

For sample input raw data, use the [sample files](https://github.com/neurokidslab/eeg_preprocessing/tree/main/examples/example_original/DATA/set).

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
│   ├── electrode_positions.py      # Electrode position helpers
│   ├── utils.py                    # Utility functions and config loading
│   └── default_cfg/                # Default JSON configurations bundled with the package
├── electrode_layout/               # Example montage files (.sfp)
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

```python
from apice.pipeline import run_preprocessing, run_segmentation

# Run preprocessing on your dataset
run_preprocessing(input_dir, output_dir, ...)

# Run segmentation on preprocessed data
run_segmentation(input_dir, output_dir, kwargs_events_from_annotations, event_time_window, ...)
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

