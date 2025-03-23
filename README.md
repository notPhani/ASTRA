🚀 Project Overview

This project aims to approximate the redshift of galaxies by analyzing their filter images and reconstructing their spectra using deep learning. The pipeline consists of:

1️⃣ Data Collection & Preprocessing: Automating galaxy data downloads, isolating galaxies, and cleaning images.

2️⃣ Spectrum Reconstruction: Using a CNN model to reconstruct missing spectra from filter images.

3️⃣ Redshift Prediction: Feeding reconstructed spectra into an XGBoost model to estimate redshift values.

📂 Dataset

Source: SDSS Galaxy Survey

Size: ~3,500 galaxies

Filters Used: (u, g, r, i, z)

Spectral Data: Corresponding spectra for each galaxy

📜 Methodology

Automated Data Extraction: Using web scrapers (like Selenium) to fetch galaxy images and metadata.

Preprocessing: Applying galmask, WCS coordinate conversion, and filtering noisy data.

CNN-Based Spectrum Reconstruction: Training a neural network to estimate missing spectra from galaxy filters.

XGBoost Model for Redshift Prediction: Using reconstructed spectra as input for redshift classification/regression.

⚙️ Installation

To set up the project, clone this repository and install dependencies:

bash
Copy
Edit
git clone https://github.com/yourusername/galaxy-redshift-prediction.git
cd galaxy-redshift-prediction
pip install -r requirements.txt
▶️ Usage
1️⃣ Preprocess Data

python
Copy
Edit
from preprocessing import clean_data
clean_data("path/to/your/dataset")
2️⃣ Train CNN for Spectrum Reconstruction

python
Copy
Edit
from train_cnn import train_model
train_model(epochs=100, batch_size=32)
3️⃣ Predict Redshift using XGBoost

python
Copy
Edit
from redshift_predictor import predict_redshift
predict_redshift("path/to/reconstructed/spectra")
📊 Results
CNN Accuracy on Spectrum Reconstruction: XX%

XGBoost RMSE on Redshift Prediction: XX

❗ Issues & Contributions
If you encounter any issues, report them in the Issues Tab. Contributions are welcome! 🚀

📜 License
This project is licensed under the MIT License
