import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

# Your custom imports
from Models import model_fn
from my_utils import train_model, load_energy_data_feather

# ============================
# Configuration
# ============================
MODEL_NAMES = ["simple_cnn"]  # You can add more: ["lstm", "dual_cnn_gru_fcnn"]
NUM_CLIENTS = 1410
TOTAL_EPOCHS = 30
LR = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_FILE = "train_final.feather"

# Set seeds for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# List of all clients (no sampling)
all_clients = list(range(NUM_CLIENTS))

# ============================
# Centralized Training Loop
# ============================
for model_name in MODEL_NAMES:
    print(f"\n{'='*60}")
    print(f"Starting Centralized Training with model: {model_name}")
    print(f"Training on all {NUM_CLIENTS} clients for {TOTAL_EPOCHS} epochs")
    print(f"{'='*60}")

    # Setup save directory
    model_dir = os.path.join("results", model_name)
    os.makedirs(model_dir, exist_ok=True)

    # Initialize model
    model = model_fn(model_name).to(DEVICE)
    model.train()

    # Main training loop over epochs
    for epoch in range(1, TOTAL_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{TOTAL_EPOCHS}")
        epoch_losses = []

        # Train on every client sequentially
        for cid in tqdm(all_clients, desc=f"Epoch {epoch} - Training clients"):
            train_loader, _ = load_energy_data_feather(cid, filepath=DATA_FILE)

            # Use your existing train_model function
            # It creates its own optimizer internally (Adam with LR)
            # Trains for 1 epoch on this client's data and updates the model in-place
            _, fin_loss = train_model(
                model=model,
                train_loader=train_loader,
                device=DEVICE,
                learning_rate=LR,
                loss_fn=None,           # assumes handled inside
                optimizer_class=optim.Adam,
                epochs=1                # one full pass over client's data
            )
            epoch_losses.append(fin_loss)

        # Average loss across all clients for this epoch
        avg_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch} - Average Loss: {avg_loss:.6f}")

        # Save checkpoint after each epoch
        checkpoint_path = os.path.join(model_dir, f"{model_name}_central_epoch_{epoch:02d}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'avg_loss': avg_loss,
        }, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

    # Save final model after all epochs
    final_path = os.path.join(model_dir, f"{model_name}_final_{TOTAL_EPOCHS}epochs.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'total_epochs': TOTAL_EPOCHS,
    }, final_path)

    print(f"\nCentralized training completed for {model_name}!")
    print(f"Final model saved to: {final_path}")
    print(f"All epoch checkpoints saved in: {model_dir}")
    print(f"{'='*60}\n")