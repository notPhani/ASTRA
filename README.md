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
