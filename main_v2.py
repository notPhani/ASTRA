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
import time
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from scipy.ndimage import median_filter
import cv2
from skimage import exposure 
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
                    datam = hdul[0].data
                    datam += 0.01

                    image_data.append(datam)
            except Exception as e:
                print(f"Error loading {file_name}: {e}")
                print(folder_name)
                continue  # Skip the problematic file
    return image_data

import numpy as np
from scipy.ndimage import median_filter
import cv2

def stack_clean(image_data):
    stack_images = []
    
    for image in image_data:
        image = median_filter(image, size=3)
        image = cv2.bilateralFilter(image, d=5, sigmaColor=0.5, sigmaSpace=5)
        lower_percentile = 1
        upper_percentile = 99
        low, high = np.percentile(image, [lower_percentile, upper_percentile])
        image = np.clip(image, low, high)
        image = np.arcsinh(image)
        image = (image - image.min()) / (image.max() - image.min())
        def mask_background(image, threshold=0.01):
            image[image < threshold] = 0  
            return image
        image = mask_background(image, threshold=0.01)
        stack_images.append(image)
    
    return np.stack(stack_images)

def make_patches(image, patch_size=16):
    """Splits an image into non-overlapping patches with safety checks."""
    image = np.array(image)  # Ensure NumPy array
    dim, H, W = image.shape
    
    # Ensure valid patching (crop to nearest multiple of patch_size)
    H_crop = (H // patch_size) * patch_size
    W_crop = (W // patch_size) * patch_size
    image = image[:, :H_crop, :W_crop]  # Crop the image to fit exact patches
    
    num_patches = (H_crop // patch_size) * (W_crop // patch_size)
    patches = np.zeros((num_patches, patch_size * patch_size, dim), dtype=np.float32)

    patch_idx = 0
    for i in range(0, H_crop, patch_size):
        for j in range(0, W_crop, patch_size):
            patch = image[:, i:i+patch_size, j:j+patch_size]  # Shape (dim, 16, 16)
            patches[patch_idx] = patch.reshape(dim, -1).T  # (256, dim)
            patch_idx += 1

    return patches  
class OptimizedStaticPatchCNN(nn.Module):
    def __init__(self, embed_dim=1280, kernel_size=3, sigma=1.0):
        super().__init__()

        self.gaussian_weight = self._create_gaussian_kernel(kernel_size, sigma)
        self.depthwise = nn.Conv1d(in_channels=5, out_channels=5, kernel_size=kernel_size, padding=1, groups=5)
        self.pointwise = nn.Conv1d(in_channels=5, out_channels=32, kernel_size=1)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, embed_dim)

    def _create_gaussian_kernel(self, size, sigma):
        x = torch.arange(-size // 2 + 1, size // 2 + 1).float()
        kernel_1d = torch.exp(-x**2 / (2 * sigma**2))
        kernel_1d /= kernel_1d.sum()
        kernel_1d = kernel_1d.view(1, 1, size)
        kernel_1d = kernel_1d.expand(5, 1, size)  # (5,1,3)
        return nn.Parameter(kernel_1d, requires_grad=False)

    def forward(self, x):
        x = torch.nan_to_num(x, nan=0.0)
        x = x.to(dtype=torch.float32, device=self.gaussian_weight.device)
        x = x.permute(0, 2, 1).contiguous()  #  Change to (patch_length, channels, patch_size²)

        x = F.conv1d(x, self.gaussian_weight, padding=1, groups=5)
        x = F.relu(self.depthwise(x))
        x = F.relu(self.pointwise(x))

        x = self.global_avg_pool(x).view(x.shape[0], -1)  # Flatten (patch_length, 32)
        x = self.fc(x)  # Fully connected layer (patch_length, 1280)

        return x  

def final_encoding(tokens,patch_length,dim):
    encoding = torch.zeros(patch_length,dim)
    position = torch.arange(0, patch_length, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim))
    encoding[:, 0::2] = torch.sin(position * div_term)  # Apply sine to even indices
    encoding[:, 1::2] = torch.cos(position * div_term)
    tokens = tokens.to(device='cuda') 
    encoding = encoding.to(device='cuda')
    return tokens+encoding

def get_spectrum(folder_name,target_length=4000,eps=1e-6):
    spectrum_folder = os.path.join(folder_name,"spectrum")
    spectrum = os.path.join(spectrum_folder,os.listdir(spectrum_folder)[0])
    df = pd.read_csv(spectrum, delimiter=",")  # Ensure tab-separated format

    # Extract wavelength and flux
    wavelength = df["wavelength"].to_numpy(dtype=np.float32)
    flux = df["flux"].to_numpy(dtype=np.float32)
    target_wavelength = np.linspace(wavelength.min(), wavelength.max(), target_length)
    
    # Interpolate flux values to match 4000 wavelength points
    interp_flux = np.interp(target_wavelength, wavelength, flux)
    interp_flux = np.clip(interp_flux,eps,None)
    flux = np.log1p(interp_flux)
    
    flux = 2 * (flux- flux.min()) / (flux.max() - flux.min() + 1e-6) - 1

    return target_wavelength, flux



def pad_to_power_of_2(tensor):
    """Pads the tensor along sequence length (dim=1) to the next power of 2."""
    batch_size, seq_len, embed_dim = tensor.shape
    next_power_2 = 2**((seq_len - 1).bit_length())  # Find next power of 2
    pad_size = next_power_2 - seq_len
    
    if pad_size > 0:
        return F.pad(tensor, (0, 0, 0, pad_size))  # Pad along seq_len dimension
    return tensor  # No padding needed

class RobustAttentionBlock(nn.Module):
    def __init__(self, embed_dim=1280, num_heads=8, rank=325, use_flash=False, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # 1280 / 8 = 160
        self.use_flash = use_flash  # Flag for FlashAttention support

        # Step 1: Low-Rank Factorized QKV Projection
        self.qkv_linear = nn.Linear(embed_dim, 3 * rank, bias=False)  # (1280, 3*325)
        self.q_proj = nn.Linear(rank, embed_dim, bias=False)  # (325, 1280)
        self.k_proj = nn.Linear(rank, embed_dim, bias=False)  # (325, 128)
        self.v_proj = nn.Linear(rank, embed_dim, bias=False)  # (325, 128)

        # Step 3: Final Output Projection
        self.out_linear = nn.Linear(embed_dim, embed_dim, bias=False)  # (1280, 1280)
        self.dropout = nn.Dropout(dropout)

        # Step 4: LayerNorm for Stability
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):  # x shape: (batch, 325, 1280)
        batch_size, seq_len, embed_dim = x.shape

        # Step 1: Normalize input
        x_norm = self.norm(x)
        qkv = self.qkv_linear(x_norm)  # (batch, 325, 3*325)
        qkv = qkv.view(batch_size, seq_len, 3, -1).unbind(dim=2)  # Unbind Q, K, V
        q, k, v = self.q_proj(qkv[0]), self.k_proj(qkv[1]), self.v_proj(qkv[2])  

        # Step 2: Pad Q, K, V to the nearest power of 2
        q_padded = pad_to_power_of_2(q)
        k_padded = pad_to_power_of_2(k)
        v_padded = pad_to_power_of_2(v)

        # Step 3: Apply FFT on the padded tensors
        q_fft, k_fft, v_fft = map(lambda t: torch.fft.fft(t, dim=1), (q_padded, k_padded, v_padded))
        q_fft = q_fft.real
        k_fft = k_fft.real
        v_fft = v_fft.real

        # Step 4: Optimized Matrix Multiplication (Replaces einsum)
        attn_scores = torch.einsum('bqi,bki->bqk', q_fft, k_fft.conj())
        attn_scores /= torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))  
        attn_weights = F.softmax(attn_scores.abs(), dim=-1)  # Convert to real values

        # Step 5: Attention Output
        attn_output_fft = torch.einsum('bqk,bkd->bqd', attn_weights, v_fft)
        attn_output = torch.fft.ifft(attn_output_fft, dim=1).real  # Convert back to real domain

        # Step 6: Output Projection + Residual Connection
        output = self.dropout(self.out_linear(attn_output))

        # Step 7: Crop back to original sequence length
        return x + output[:, :seq_len, :]
    
class SharedMLPWithSkipConnection(nn.Module):
    def __init__(self, embed_dim=1280, mlp_ratio=4, dropout=0.1):
        super().__init__()
        # Define shared weight linear transformation
        self.shared_fc = nn.Linear(embed_dim, embed_dim * mlp_ratio)
        self.shared_fc_2 = nn.Linear(embed_dim * mlp_ratio, embed_dim)   # Shared Linear layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # First pass through the MLP (parameter shared)
        x_mlp = self.shared_fc(x)  # Shape: (b, 325, embed_dim * mlp_ratio)
        x_mlp = F.gelu(x_mlp)  # Apply GELU activation
        x_mlp = self.dropout(x_mlp)

        # Apply shared weights again
        x_mlp = self.shared_fc_2(x_mlp)  # Shape: (b, 325, embed_dim * mlp_ratio)
        x_mlp = self.dropout(x_mlp)

        # Skip connection (add original input to the output)
        return x + x_mlp 
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=1280, num_heads=8, rank=325, dropout=0.1):
        super(TransformerBlock, self).__init__()

        # Initialize Attention and MLP blocks
        self.attention_block = RobustAttentionBlock(embed_dim=embed_dim, num_heads=num_heads, rank=rank, dropout=dropout)
        self.mlp_block = SharedMLPWithSkipConnection(embed_dim=embed_dim, dropout=dropout)

        # LayerNorms for the overall transformer block
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # First pass through the attention block with residual connection
        attn_out = self.attention_block(self.norm1(x))
        
        # Add the attention output to the original input (skip connection after attention)
        x = x + attn_out

        # Pass through MLP block with residual connection
        mlp_out = self.mlp_block(self.norm2(x))

        # Add the MLP output to the result after the attention block
        out = x + mlp_out
        return out

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers=8, embed_dim=1280, num_heads=8, rank=325, dropout=0.1):
        super(TransformerEncoder, self).__init__()

        # Initialize a list of transformer blocks
        self.layers = nn.ModuleList([TransformerBlock(embed_dim, num_heads, rank, dropout).to('cuda') for _ in range(num_layers)])

        # Randomly initialized dummy token
        self.dummy_token = nn.Parameter(torch.randn(1, 1, embed_dim))  # Random initialization

        # Final linear layer to output the token's embedding at index 0
        self.final_linear = nn.Linear(embed_dim, embed_dim).to('cuda')

    def forward(self, x):
        # Step 1: Add the dummy token to the input
        batch_size = x.shape[0]
        dummy_token = self.dummy_token.expand(batch_size, -1, -1).to(device='cuda')  # Expand to match batch size
        x = torch.cat([dummy_token, x], dim=1).to(device='cuda')  # Shape: (batch_size, 325, embed_dim)

        # Step 2: Pass through multiple transformer blocks
        for layer in self.layers:
            x = layer(x)

        # s 3: Extract the output for the dummy token (index 0)
        output = x[:, 0:1, :]  # We get the embedding for the dummy token

        # Optional: Pass through a final linear layer if needed
        output = self.final_linear(output)

        return output
    
import torch.nn.utils.prune as prune

class SpectraUpscaler(nn.Module):
    def __init__(self, input_dim=1280, output_dim=4000):
        super(SpectraUpscaler, self).__init__()

        # Interpolation layer (non-learnable)
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Learnable Linear Layer
        self.linear = nn.Linear(output_dim, output_dim).to('cuda')

        # Apply pruning (for example, 30% pruning)
        prune.l1_unstructured(self.linear, name="weight", amount=0.1)

    def forward(self, x):
        # x shape is (batch_size, channels, input_dim) = (16, 1, 1280)
        
        # Step 1: Use linear interpolation to upscale to the desired output dimension (4000)
        x = F.interpolate(x, size=self.output_dim, mode='linear', align_corners=False)
        
        # Step 2: Pass through a learnable linear layer
        x = x.squeeze(1)  # shape becomes (16, 4000)
        x = self.linear(x)  # shape becomes (16, 4000)
        
        # Return back to the required shape (16, 1, 4000)
        x = x.unsqueeze(1)  # shape becomes (16, 1, 4000)
        
        return x
    
class Model(nn.Module):
    def __init__(self, embed_dim=1280, num_heads=8, rank=325, dropout=0.1, mlp_ratio=4):
        super(Model, self).__init__()
        self.dropout_rate = dropout
        # Transformer Encoder
        self.input_proj = nn.Sequential(
            nn.Linear(1280, 1280),
            nn.LayerNorm(1280),
            nn.GELU(),
            nn.Dropout(dropout/2)
        )
        self.encoder = TransformerEncoder(embed_dim=embed_dim, num_heads=num_heads, rank=rank, dropout=dropout)
        
        # Spectra Upscaler (for upscaling to the desired output dimension)
        self.spectra_upscaler = SpectraUpscaler(input_dim=1280, output_dim=4000)
        self.dropout_layers = []
        for name, module in self.named_modules():
            if isinstance(module, nn.Dropout):
                self.dropout_layers.append(module)

    def forward(self, x):
        # Pass through the Transformer Encoder
        x = self.input_proj(x)
        encoder_output = self.encoder(x)

        # Now pass the output through the SpectraUpscaler
        output = self.spectra_upscaler(encoder_output)
        return output
    def adjust_dropout(self, new_rate):
        """Adjust all dropout layers in the model"""
        self.dropout_rate = min(0.5, new_rate)
        for layer in self.dropout_layers:
            layer.p = self.dropout_rate
        return self.dropout_rate

class GalaxyDataset(Dataset):
    def __init__(self,root_dir,target_length=4000,patch_size=16,dim=256,size=1000):
        self.root_dir = root_dir
        self.target_length = target_length
        self.patch_size = patch_size
        self.dim = dim

        self.galaxy_folders = os.listdir(root_dir)[0:size]
    
    def __len__(self):
        return len(self.galaxy_folders)
    
    def __getitem__(self, index):
        folder = os.path.join(self.root_dir,self.galaxy_folders[index])
        images = extract_images(folder)
        tokens = torch.tensor(make_patches(stack_clean(images),self.patch_size)).clone().detach().to(device='cuda')
        model_cnn = OptimizedStaticPatchCNN().to(device='cuda')
        encodings = model_cnn(tokens)
        encodings = final_encoding(encodings,324,1280)
        encodings = 2 * (encodings - encodings.min()) / (encodings.max() - encodings.min() + 1e-6) - 1
        wavelength,spectrum = get_spectrum(folder,target_length=self.target_length)
        return encodings,wavelength,spectrum

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.optim.lr_scheduler import OneCycleLR

class ASTRA_Trainer:
    def __init__(self, model, train_loader,test_loader, val_loader, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = torch.nn.MSELoss()
        # Adjust loss function as needed
        self.optimizer = optim.Adafactor(self.model.parameters(), lr=0.0005, eps=(1e-30, 1e-3), weight_decay=0.0)
        self.scheduler = OneCycleLR(
        self.optimizer,
        max_lr=5e-4,                     # Upper bound
        steps_per_epoch=len(train_loader),
        epochs=50,                       # Total epochs
        pct_start=0.3,                   # Warmup percentage
        div_factor=25,                   # Initial lr = max_lr/25
        final_div_factor=1e4,             # Final lr = max_lr/1e4
        cycle_momentum=False
        )
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.writer = SummaryWriter()
        self.best_loss = float("inf")
    
    def train_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0.0
        for images,wavelenth,spectrum in tqdm(self.train_loader, desc=f"Epoch {epoch} [Training]"):
            images, spectrum = images.to(self.device).float(), spectrum.to(self.device).float()
            spectrum = spectrum.unsqueeze(1)
            print(
                f"Input: μ={images.mean():.3f} var={images.std():.3f} range=[{images.min():.3f}, {images.max():.3f}]\n"
                f"Target: μ={spectrum.mean():.3f} var={spectrum.std():.3f} range=[{spectrum.min():.3f}, {spectrum.max():.3f}]"
                ) 
            self.optimizer.zero_grad()
            preds = self.model(images)
            loss = self.criterion(preds, spectrum)
            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), 0.05)
            self.optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(self.train_loader)
        self.writer.add_scalar("Loss/Train", avg_loss, epoch)
        return avg_loss
    
    def validate(self, epoch):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images,wavelenth,spectrum in tqdm(self.train_loader, desc=f"Epoch {epoch} [Training]"):
                images, spectrum = images.to(self.device).float(), spectrum.to(self.device).float()
                spectrum = spectrum.unsqueeze(1)
                preds = self.model(images)
                loss = self.criterion(preds, spectrum)
                val_loss += loss.item()
        
        avg_loss = val_loss / len(self.val_loader)
        self.writer.add_scalar("Loss/Validation", avg_loss, epoch)
        return avg_loss
    
    def train(self, epochs=100):
        loss_data_train = []
        loss_data_val = []
        for epoch in range(epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)
            if val_loss > 2.0 * train_loss:
                current = self.model.dropout_rate
                new_rate = min(0.5, current + 0.05)
                actual_new = self.model.adjust_dropout(new_rate)
                print(f"🚨 Overfitting detected! Dropout: {current:.2f} → {actual_new:.2f}")
            self.scheduler.step()  
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
            print("Current LR:", self.optimizer.param_groups[0]['lr'])
            print("Last LR:", self.scheduler.get_last_lr())
            loss_data_train.append(train_loss)
            loss_data_val.append(val_loss)
            self.scheduler.step()
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                torch.save(self.model.state_dict(), "best_model.pth")
                print("[INFO] Best model saved.")

        self.writer.close()
        return loss_data_train,loss_data_val

dataset = GalaxyDataset("D:/Dataset")

# Define split sizes
total_size = len(dataset)
train_size = int(0.8 * total_size)
val_size = int(0.1 * total_size)
test_size = total_size - train_size - val_size  # Ensures full dataset usage

# Perform split
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

batch_size = 8  # Your chosen batch size
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = Model().to('cuda')
def init_weights(m):
    if isinstance(m, nn.Linear):
        print(f"Initializing {m}")
        nn.init.xavier_uniform_(m.weight, gain=0.1)  # Small initial scale
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

model.apply(init_weights)
model.eval()

with torch.no_grad():
    test_input, _, test_target = next(iter(train_loader))
    test_input = test_input.to('cuda')
    test_target = test_target.to('cuda')
    
    # Test untrained model
    pred = model(test_input)
    print("\n=== Untrained Model Test ===")
    print(f"Prediction Mean: {pred.mean():.4f}, Std: {pred.std():.4f}")
    print(f"Target Mean: {test_target.mean():.4f}, Std: {test_target.std():.4f}")
    
    # Calculate loss components
    mse = (pred - test_target).pow(2).mean()
    print(f"\nMSE Breakdown:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Mean Absolute Error: {(pred - test_target).abs().mean():.4f}")
    print(f"Max Error: {(pred - test_target).abs().max():.4f}")
trainer = ASTRA_Trainer(model, train_loader, test_loader, val_loader)
#train_loss,val_loss = trainer.train(epochs=100)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize model architecture
model = Model().to(device)

# Load trained weights
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval() 
test_sample, test_wavelengths, test_spectrum = next(iter(test_loader))

# Move to device
test_sample = test_sample.to(device)
test_spectrum = test_spectrum.to(device)
with torch.no_grad():
    # Forward pass
    pred = model(test_sample).squeeze(1)
    
    # Calculate metrics
    mse = torch.nn.functional.mse_loss(pred, test_spectrum)
    mae = torch.nn.functional.l1_loss(pred, test_spectrum)
    
    print(f"\nTest Results:")
    print(f"MSE: {mse.item():.4f}")
    print(f"MAE: {mae.item():.4f}")

pred_np = pred[0].squeeze(0).cpu().numpy()  # First sample
true_np = test_spectrum[0].cpu().numpy()

plt.figure(figsize=(12, 6))
plt.plot(test_wavelengths[2], true_np, label='True Spectrum', color='blue', alpha=0.7)
plt.plot(test_wavelengths[2], pred_np, label='Predicted', color='red', alpha=0.5)
plt.xlabel('Wavelength (Å)')
plt.ylabel('Flux')
plt.legend()
plt.title(f'Spectrum Prediction (MSE: {mse.item():.2f})')
plt.show()
