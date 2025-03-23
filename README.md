# ASTRA - Astronomical Spectrum Transformer for Redshift Approximation

This repository contains a PyTorch-based implementation of a Transformer model designed to process galaxy images and predict their corresponding spectra. The model leverages a combination of convolutional neural networks (CNNs) for patch extraction and a Transformer encoder for feature learning, followed by a decoder to generate the spectrum.

## Table of Contents
- [Overview](#overview)
- [Key Components](#key-components)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [License](#license)

## Overview

The goal of this project is to predict the spectrum of a galaxy from its multi-band images. The model processes the images by dividing them into patches, encoding these patches using a CNN, and then applying a Transformer encoder to learn spatial and spectral relationships. Finally, a decoder generates the predicted spectrum.

## Key Components

1. **Patch Extraction**: Images are divided into patches, which are then normalized and encoded using a CNN.
2. **Transformer Encoder**: The encoded patches are passed through a Transformer encoder to learn complex relationships.
3. **Spectrum Decoder**: The output of the Transformer encoder is decoded to produce the predicted spectrum.
4. **Dataset Handling**: The `GalaxyDataset` class handles loading and preprocessing of galaxy images and spectra.

## Installation

To use this code, you need to have Python 3.x installed along with the following libraries:

```bash
pip install numpy matplotlib astropy torch torchinfo pandas
```
## Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/galaxy-spectrum-transformer.git
   cd galaxy-spectrum-transformer
   ```
## Prepare your dataset

Ensure your dataset is organized in the following structure:
```python
# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Set the path to the file you'd like to load
file_path = ""

# Load the latest version
df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "sneakyrat/galaxy-image-filters-with-spectra",
  file_path,
  # Provide any additional arguments like 
  # sql_query or pandas_kwargs. See the 
  # documenation for more information:
  # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
)

print("First 5 records:", df.head())
```

## Run the model

1. Modify the `folder_name` variable in the script to point to your dataset directory.
2. Execute the script:

```bash
python main_v2.py
```
## Model Architecture

The model consists of the following key components:

**StaticPatchCNN**: A CNN that processes image patches and generates embeddings.

**Attention_block**: Implements multi-head self-attention.

**MLPBlock**: A multi-layer perceptron block used in the Transformer.

**TransformerBlock**: Combines attention and MLP blocks.

**TransformerEncoder**: A stack of Transformer blocks.

**SpectrumDecoder**: Decodes the Transformer output to produce the spectrum.

## Example model summary
```python
model = StaticPatchCNN(embed_dim=1280).to("cuda")
summary(model, input_size=(324, 5, 256))
```
## Dataset

The dataset should contain galaxy images in FITS format and corresponding spectra in CSV format. The `GalaxyDataset` class handles the loading and preprocessing of this data.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
