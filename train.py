"""
Training loop script for the ocean turbulence forecaster Deep learning models.

Usage:
    python train.py

Outputs:
    checkpoints/best_model.pt   ← saved whenever val loss improves
    checkpoints/last_model.pt   ← saved every epoch
"""

import os
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from Unet import UNet
from dataset import (get_dataloaders, DATA_PATH,
                     INPUT_VARS, TARGET_VARS, LEVEL, WINDOW_SIZE, OUT_STEPS)

# ─── TRAINING CONFIG, Hyperparameters──────────────────────────────────────────────────────
BATCH_SIZE    = 16
LR            = 1e-3
MAX_EPOCHS    = 100
PATIENCE      = 10       # early stopping patience
#BASE_FILTERS  = 32
#DEPTH         = 3        # U-Net depth (3 works well for 64×64)
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_DIR      = "checkpoints"
# ─────────────────────────────────────────────────────────────────────────────


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:            #loop for each x, y pair in the loader (input,output)
        x, y = x.permute(0, 2, 3, 1).to(device), y.to(device) #Chatgpt helped to add permute 
        optimizer.zero_grad() #reset of gradients
        print(x.shape) #  Chatgpt helpedadded to check the shape of x as it gives error
        pred = model(x)  #forwards passing
        loss = criterion(pred, y) #compute and storage loss
        loss.backward() #compute gradients
        optimizer.step() #update weights 
        total_loss += loss.item() * x.size(0) #Accumulation of models loss
    return total_loss / len(loader.dataset) #avrage loss per batch


@torch.no_grad() #disable gradient calculation for evalutation of model. 
def evaluate(model, loader, criterion, device):
    model.eval()  #Evaluation mode
    total_loss = 0.0 
    for x, y in loader: #rucn for each x, y pair 
        x, y = x.permute(0, 2, 3, 1).to(device), y.to(device) #Chatgpt added permute to change from batch size, channels, H, W to batch size, H, channels, W, as the model expects it.
        pred = model(x)
        total_loss += criterion(pred, y).item() * x.size(0)
    return total_loss / len(loader.dataset) # Return validation loss


def train():
    os.makedirs(CKPT_DIR, exist_ok=True)

    # ── get data from dataloader─────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, in_ch, out_ch, norm_stats = get_dataloaders(
        data_path   = DATA_PATH,
        input_vars  = INPUT_VARS,
        target_vars = TARGET_VARS,
        level       = LEVEL,
        window      = WINDOW_SIZE,
        out_steps   = OUT_STEPS,
        batch_size  = BATCH_SIZE,
    )

    # ── define imported model to use for traning─────────────────────────────────────────────────────────────────
    model = UNet(
        in_channels  = in_ch,
        out_channels = out_ch,
        device=DEVICE,
        kernel_size=3,
        selected_dim=5,  #chat recommended to start with 0 as selected dim.
        #base_filters = BASE_FILTERS,
        #depth        = DEPTH
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[train] Device: {DEVICE}")
    print(f"[train] in_channels: {in_ch}  out_channels: {out_ch}")
    print(f"[train] Parameters: {n_params:,}\n")

    # ── Loss / Optimiser / Scheduler ─────────────────────────────────────────
    criterion = nn.MSELoss() #mean square error used as loss function
    optimizer = Adam(model.parameters(), lr=LR) #Defualt adam optimizer
    scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5) #Removed , verbose=True, aslo.. half the learning rate if loss plateaus with 5 epochs

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val   = float("inf") #assign high valude, so first epoch will be saved as best model.
    no_improve = 0 #counter how many epochs without improvment, used for early stopping
    history    = {"train": [], "val": []} #store log of train av validation loss epoch wise

    for epoch in range(1, MAX_EPOCHS + 1): #Loop for a epoch
        t0 = time.time() #start time count for each epoch
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss   = evaluate(model, val_loader, criterion, DEVICE)
        elapsed    = time.time() - t0 #stop time for epoch

        history["train"].append(train_loss)
        history["val"].append(val_loss)
        scheduler.step(val_loss) #reduce learning rate if no improvement

        #Logs for epoch
        print(f"Epoch {epoch:03d}/{MAX_EPOCHS}  "
              f"train={train_loss:.6f}  val={val_loss:.6f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}  "
              f"[{elapsed:.1f}s]")

        # Save best model from given epoch
        if val_loss < best_val:
            best_val   = val_loss #update best validation
            no_improve = 0 #restart coutner
            torch.save({            #save checkpoint and model
                "epoch":        epoch,
                "model_state":  model.state_dict(),
                "val_loss":     best_val,
                "in_channels":  in_ch,
                "out_channels": out_ch,
                "norm_stats":   norm_stats,   # x_mean, x_std, y_mean, y_std
            }, os.path.join(CKPT_DIR, "best_model.pt"))
            print(f" !!!Best model saved  (val={best_val:.6f})!!!")
        else:
            no_improve += 1

        # Save last, incase of interruption
        torch.save(model.state_dict(), os.path.join(CKPT_DIR, "last_model.pt"))

        # Early stopping if plateau after 5 epochs
        if no_improve >= PATIENCE:
            print(f"\nEarly stopping after {PATIENCE} epochs without improvement for last 5 epochs.")
            break

    # ── Test evaluation ───────────────────────────────────────────────────────
    ckpt = torch.load(os.path.join(CKPT_DIR, "best_model.pt"), map_location=DEVICE) #laod best model saved
    model.load_state_dict(ckpt["model_state"]) #load best weights into model
    test_loss = evaluate(model, test_loader, criterion, DEVICE) #evaludatemodel on test set and print results
    print(f"\n[train] Test MSE (best model): {test_loss:.6f}")

    return model, history


if __name__ == "__main__":
    train()
