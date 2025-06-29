# APICE-Py: An Open-Source MNE-Python Pipeline for Scalable EEG Preprocessing

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
cd APICE-Py
```

### 2. Install Dependencies

Install required packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```
**Note**: Requires Python >= 3.12

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
│ ├── artifacts.py
│ ├── epochs.py
│ └── ...
├── electrode_layout/ # montage (optional)
├── requirements.txt # Dependencies list
└── README.md # Project info
```

