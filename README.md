# 🌌 Galaxy Redshift Prediction using CNN & XGBoost

![Galaxy Redshift](https://upload.wikimedia.org/wikipedia/commons/thumb/3/3d/Redshift_blueshift.svg/800px-Redshift_blueshift.svg.png)

## 🚀 Project Overview
This project aims to **predict the redshift of galaxies** using **deep learning**. The process involves:
1. **Data Collection & Preprocessing** – Automating the download of galaxy images and cleaning them.
2. **Spectrum Reconstruction** – Training a CNN to reconstruct galaxy spectra from filter images.
3. **Redshift Prediction** – Using an XGBoost model to estimate the redshift.

## 📂 Dataset
- **Source:** SDSS Galaxy Survey
- **Size:** ~3,500 galaxies
- **Filters Used:** (u, g, r, i, z)
- **Spectral Data:** Corresponding spectra for each galaxy

## 📜 Methodology
1. **Automated Data Extraction:** Using Selenium to fetch galaxy images and metadata.
2. **Preprocessing:** Applying `galmask`, WCS coordinate conversion, and noise filtering.
3. **CNN-Based Spectrum Reconstruction:** Training a model to reconstruct missing spectra.
4. **XGBoost for Redshift Prediction:** Using reconstructed spectra to estimate redshift.

## ⚙️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/galaxy-redshift-prediction.git
cd galaxy-redshift-prediction
pip install -r requirements.txt

## ▶️ Usage

### 1️⃣ Preprocess Data
```python
from preprocessing import clean_data
clean_data("path/to/your/dataset")
