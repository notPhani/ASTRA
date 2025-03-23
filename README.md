# ASTRA
Astronomical Spectral Transformer for Redshift Approximation

This project focuses on reconstructing the spectrum of galaxies using deep learning techniques. The dataset used for this project is ASTROPIXEL, sourced from Kaggle. The model employs a Transformer-based architecture with patch embeddings, attention mechanisms, and feed-forward networks to learn spectral features from multi-channel astronomical images.

Dataset
Source: ASTROPIXEL (Kaggle)

The dataset contains FITS images and corresponding spectrum files for multiple galaxies.

Each galaxy has images in different filters, which serve as input to the model.

The target output is a continuous spectrum of the galaxy, interpolated to a fixed length of 4000 data points.

Model Architecture
The model follows a structured pipeline:

Data Preprocessing

FITS images are extracted and normalized using an arcsin transformation.

Images are split into small patches for feature extraction.

The corresponding spectrum is interpolated to 4000 points.

Feature Extraction

A Static Patch CNN extracts features from image patches and converts them into 1280-dimensional embeddings.

Positional encodings are added to retain spatial relationships.

Transformer Encoder

Uses Multi-Head Self-Attention and Feed-Forward Networks (MLP layers) to learn meaningful representations.

A dummy token is introduced to store global information.

Spectrum Reconstruction

A decoder network maps the transformed embeddings to the final 4000-point spectrum.

Code Structure
extract_images(): Reads FITS files and extracts image data.

stack_clean(): Normalizes image pixel values.

make_patches(): Splits images into 16x16 patches.

StaticPatchCNN: CNN for feature extraction from patches.

TransformerBlock: Implements self-attention and MLP layers.

TransformerEncoder: Stacks multiple transformer layers and introduces a dummy token.

SpectrumDecoder: Maps transformer outputs to spectral data.

GalaxyDataset: Custom dataset class for PyTorch, handling data loading and preprocessing.

Requirements
Install dependencies using:

bash
Copy
Edit
pip install numpy pandas torch torchvision astropy matplotlib torchinfo
Usage
Training the Model
Set the dataset path in the folder_name variable.

Run the script to initialize and train the model:

bash
Copy
Edit
python main.py
The trained model will be saved as transformer.h.

Results & Next Steps
The model learns galaxy spectral features from image patches.

Future work includes optimizing transformer layers and testing on larger datasets.

Acknowledgments
Dataset: ASTROPIXEL (Kaggle)

Libraries: PyTorch, Astropy, NumPy, Pandas

Special thanks to SDSS for astronomical data
