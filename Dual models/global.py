import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import random
import pandas as pd
import numpy as np

# Import your custom functions
from my_utils import load_energy_data_feather
from Models import model_fn

# ============================
# Configuration
# ============================
MODEL_NAME = "dual_simple_ann_fcnn"
NUM_CLIENTS = 1410
TOTAL_EPOCHS = 30
LR = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_FILE = "train_final.feather"

# Set seeds
random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ============================
# Setup
# ============================
model_dir = os.path.join("results", MODEL_NAME)
os.makedirs(model_dir, exist_ok=True)

# Initialize model and optimizer (persistent across all training)
global_model = model_fn(MODEL_NAME).to(DEVICE)
global_model.train()

optimizer = optim.Adam(global_model.parameters(), lr=LR)
criterion = nn.MSELoss()  # Adjust if your task uses different loss

print(f"Starting Centralized Training with {MODEL_NAME} for {TOTAL_EPOCHS} epochs")
print(f"Training on data from all {NUM_CLIENTS} clients sequentially")

all_clients = list(range(NUM_CLIENTS))

# ============================
# Centralized Training Loop
# ============================
for epoch in range(1, TOTAL_EPOCHS + 1):
    print(f"\n=== Epoch {epoch}/{TOTAL_EPOCHS} ===")
    epoch_losses = []
    
    for cid in tqdm(all_clients, desc=f"Epoch {epoch} - Training on clients"):
        train_loader, _ = load_energy_data_feather(cid, filepath=DATA_FILE)
        
        client_losses = []
        for batch in train_loader:
            # Correctly unpack the batch based on your data structure
            # From your logs: 3 inputs + 1 target
            X_ts, X_air_temp, X_primary_use, y = batch
            
            # Move to device
            X_ts = X_ts.to(DEVICE)              # shape: (batch, 168, 4)
            X_air_temp = X_air_temp.to(DEVICE)  # shape: (batch, 168, 1)
            X_primary_use = X_primary_use.to(DEVICE).long()  # likely categorical, convert to long
            y = y.to(DEVICE)                    # shape: (batch, 24)
            
            optimizer.zero_grad()
            output = global_model(X_ts, X_air_temp, X_primary_use)  # Model expects 3 inputs
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
            client_losses.append(loss.item())
        
        if client_losses:
            epoch_losses.append(np.mean(client_losses))
    
    avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
    print(f"Epoch {epoch} - Average Loss: {avg_loss:.6f}")
    
    # Save checkpoint
    checkpoint_path = os.path.join(model_dir, f"{MODEL_NAME}_central_epoch_{epoch:02d}.pt")
    torch.save({
        'epoch': epoch,
        'model_state_dict': global_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'avg_loss': avg_loss,
    }, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")

# Save final model
final_path = os.path.join(model_dir, f"{MODEL_NAME}_final_{TOTAL_EPOCHS}epochs.pt")
torch.save(global_model.state_dict(), final_path)
print(f"\nCentralized training completed! Final model saved to {final_path}")