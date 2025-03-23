import math
import os
import numpy as np
import matplotlib.pyplot as plt 
from astropy.io import fits
from astropy.wcs import WCS
import sys
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from astropy.coordinates import SkyCoord
from astropy import units as u
from torch.utils.data import DataLoader, Dataset,random_split
from astropy.table import Table
import pandas as pd
# Unpack the data into tuples
# Then load the previous model
# Test with 100 samples
# Build the adaptive width transformer next in the same one
# Load the model into main_v2.py
# Save the model as transformer.h

""" Define the transformer model - 
        - Patch embeddings ( Making a 5d array -  Make patches - Add postional encodings)
        - Attentions Block (Make dummy Q k V learnable matrices. Make vertical divisions (Pacth_size/n))
        - Multi perceptron layer (Basic Feed forward network)
        - Collective Transformer block (Collection of a attention block and MLP. )
        - Add a dummy vector (325,1280). Assuming this vector learns the parameters. 
        - This vector is scaled and learned under a FFN to the spectrum
"""
folder_name = "D:/Dataset/Galaxy_299490502078654464"

def extract_images(folder_name):
    # Folder name - ./Galaxy_specid
    main_folder = os.path.join(folder_name,"images")
    folder = os.listdir(main_folder)
    image_data = []
    for file_name in folder:
        if file_name.endswith('.fits'):
            try:
                file_path = os.path.join(main_folder,file_name)
                with fits.open(file_path) as hdul:
                    image_data.append(hdul[0].data)
            except Exception as e:
                print(f"Error loading {file_name}: {e}")
                print(folder_name)
                continue  # Skip the problematic file
    return image_data

def stack_clean(image_data):
    stack_images = []

    for image  in image_data:
        # Normalize before asin normalization - since asin only works in (-1,1)
        image = 2 * (image - np.min(image)) / (np.max(image) - np.min(image)) - 1
        image = np.asin(image)/np.asin(np.max(image))
        stack_images.append(image)
    return stack_images

def make_patches(image,patch_size=16):
    image= np.array(image)
    dim,H,W = np.array(image).shape
    num_patches = int((H*W/(patch_size**2)))
    patches = np.zeros((num_patches,patch_size*patch_size,dim)) #(324,256,5)
    patch_idx = 0
    for i in range(0, H, patch_size):
        for j in range(0, W, patch_size):
            patch = image[:, i:i+patch_size, j:j+patch_size]  # Shape (5, 16, 16)
            patches[patch_idx] = patch.reshape(dim, -1).T  # (256, 5)
            patch_idx += 1
    
    return patches  

def final_encoding(tokens,patch_length,dim):
    encoding = torch.zeros(patch_length,dim)
    position = torch.arange(0, patch_length, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim))
    encoding[:, 0::2] = torch.sin(position * div_term)  # Apply sine to even indices
    encoding[:, 1::2] = torch.cos(position * div_term) 
    return tokens+encoding

def get_spectrum(folder_name,target_length=4000):
    spectrum_folder = os.path.join(folder_name,"spectrum")
    spectrum = os.path.join(spectrum_folder,os.listdir(spectrum_folder)[0])
    df = pd.read_csv(spectrum, delimiter=",")  # Ensure tab-separated format

    # Extract wavelength and flux
    wavelength = df["wavelength"].to_numpy(dtype=np.float32)
    flux = df["flux"].to_numpy(dtype=np.float32)
    target_wavelength = np.linspace(wavelength.min(), wavelength.max(), target_length)

    # Interpolate flux values to match 4000 wavelength points
    interp_flux = np.interp(target_wavelength, wavelength, flux)

    return target_wavelength, interp_flux

class StaticPatchCNN(nn.Module):
    def __init__(self, embed_dim=1280):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=5, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * 256, embed_dim)  

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        x = x.to(torch.float32)
        x = x.view(324, 5, 256)  # Reshape to (batch=324, channels=5, width=256)
        x = F.relu(self.conv1(x))  
        x = F.relu(self.conv2(x))  
        x = x.view(324, -1)  # Flatten
        x = self.fc(x)  # Final transformation to 1280-dim
        return x
     
class Attention_block(nn.Module):
    def __init__ (self,encoding_len = 1280,num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_len = encoding_len//num_heads

        self.Q_linear = nn.Linear(encoding_len,encoding_len)
        self.K_linear = nn.Linear(encoding_len,encoding_len)
        self.V_linear = nn.Linear(encoding_len,encoding_len)

        self.out_linear = nn.Linear(encoding_len,encoding_len)
    
    def forward(self, x):  # x shape: (324, 1280)
        #print(x.shape)
        seq_len, embed_dim = x.shape  

    # Step 1: Linear projections for Q, K, V
        Q = self.Q_linear(x)  # (324, 1280)
        K = self.K_linear(x)  # (324, 1280)
        V = self.V_linear(x)  # (324, 1280)

    # Step 2: Reshape for multi-head attention
        Q = Q.view(seq_len, self.num_heads, self.head_len).transpose(0, 1)  # (8, 324, 160)
        K = K.view(seq_len, self.num_heads, self.head_len).transpose(0, 1)  # (8, 324, 160)
        V = V.view(seq_len, self.num_heads, self.head_len).transpose(0, 1)  # (8, 324, 160)

        # Step 3: Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_len, dtype=torch.float32))  # (8, 324, 324)
        attn_weights = F.softmax(scores, dim=-1)  # Attention map
        attn_output = torch.matmul(attn_weights, V)  # Weighted sum (8, 324, 160)

    # Step 4: Concatenate heads back
        attn_output = attn_output.transpose(0, 1).contiguous().view(seq_len, embed_dim)  # (324, 1280)

    # Step 5: Final linear layer
        output = self.out_linear(attn_output)

        return output  # (324, 1280)

class MLPBlock(nn.Module):
    def __init__(self, embed_dim, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim * mlp_ratio)  # Expand dimension
        self.activation = nn.GELU()  # Non-linearity
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(embed_dim * mlp_ratio, embed_dim)  # Project back
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self,encoding_len=1280,mlp_ratio=4,dropout=0.1,num_heads=8):
        super().__init__()
        self.attention = Attention_block(encoding_len,num_heads)
        self.norm1 = nn.LayerNorm(encoding_len)
        self.mlp = MLPBlock(encoding_len,mlp_ratio,dropout)
        self.norm2 = nn.LayerNorm(encoding_len)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        x = self.attention(x)
        x = x + self.dropout(x)
        x = self.norm1(x)
        x = self.mlp(x)
        x = x + self.dropout(x)
        x = self.norm2(x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers=6, encoding_len=1280, mlp_ratio=4, dropout=0.1, num_heads=8):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerBlock(encoding_len, mlp_ratio, dropout, num_heads) for _ in range(num_layers)]
        )
        # Adding a dummy vector a=hoping it will store all the important info
        self.dummy_token = nn.Parameter(torch.randn(1,encoding_len))

    def forward(self, x):  # Make sure it has (1, 1280)
        #print(x.shape,self.dummy_token.shape)
        x = torch.cat([x,self.dummy_token], dim=0)
        for layer in self.layers:
            x = layer(x) 
        output = x[-1]
        return output
    
class SpectrumDecoder(nn.Module):
    def __init__(self, input_dim=1280, output_dim=4000):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, output_dim)  # Output matches spectrum length (3836,)
        )
    def forward(self, x):
        return self.decoder(x)  # Output: (3836,)

class Model(nn.Module):
    def __init__(self,transformer_encoder,spec_decoder):
        super().__init__()
        self.encoder  = transformer_encoder
        self.decoder = spec_decoder

    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class GalaxyDataset(Dataset):
    def __init__(self,root_dir,target_length=4000,patch_size=16,dim=256):
        self.root_dir = root_dir
        self.target_length = target_length
        self.patch_size = patch_size
        self.dim = dim

        self.galaxy_folders = os.listdir(root_dir)
    
    def __len__(self):
        return len(self.galaxy_folders)
    
    def __getitem__(self, index):
        folder = os.path.join(self.root_dir,self.galaxy_folders[index])
        images = extract_images(folder)
        images = stack_clean(images)
        tokens = make_patches(images,self.patch_size)
        model_cnn = StaticPatchCNN()
        encodings = model_cnn(tokens)
        encodings = final_encoding(encodings,324,1280)

        wavelength,spectrum = get_spectrum(folder,target_length=self.target_length)

        return encodings,wavelength,spectrum


import torch
from torchinfo import summary

model = StaticPatchCNN(embed_dim=1280).to("cuda")  # Move to GPU
x = torch.randn(324, 5, 256).to("cuda")  # Simulated input
output = model(x)
summary(model, input_size=(324, 5, 256))