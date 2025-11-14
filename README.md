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

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/neurokidslab/apice-py.git
cd apice-py
```

### 2. Install Dependencies

Install required packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```
**Note**: Requires Python >= 3.12

---
## 📂 Sample Data

For sample input raw data, use the [sample files](https://github.com/neurokidslab/eeg_preprocessing/tree/main/examples/example_original/DATA/set).

---
## 📁 Project Structure

```bash
APICE-Py/
├── main_terminal.py # CLI version: accepts arguments
├── main_config.py # Manual config version: edit params in code
├── preprocessing/ # Core EEG processing modules
│ ├── argparser_main.py
│ ├── artifacts_correction.py
│ └── ...
├── data/ # EEG input data
├── output/ # Processed data and logs
│ ├── artifacts
│ ├── epochs
│ └── ...
├── electrode_layout/ # montage (optional)
├── requirements.txt # Dependencies list
└── README.md # Project info
```

---
## 🛠️ Usage

### Option 1: Command-Line Interface

Use `main_terminal.py` to preprocess EEG data via command-line arguments:

```bash
python main_terminal.py \
  --input "input" \
  --output "output" \
  --selection_method 1\
  --montage "electrode_layout/montage_file \
  --event_keys_for_segmentation Event1, Event2, Event3 \
  --event_tile_window -1.6, 2.0 \
  --baseline_tile_window -1.6, 0
```
### Option 2: Manual Configuration

Edit parameters directly in `main_config.py` and run:
```bash
python main_config.py
```

---
## 📖 Documentation

Click [here](https://zenodo.org/records/17151631) fo the documentation (examples, customization guide, pipeline structure).

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
This project is licensed under the Apache-2.0 license. See `LICENSE`for details.

---
## ✨ Acknowledgements
- Built with MNE-Python, an open-source EEG/MEG analysis package
- Inspired by best practices in open neuroscience workflows

