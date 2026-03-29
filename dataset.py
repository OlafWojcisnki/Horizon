"""
Dataset for sequential ocean turbulence prediction using U-Net.

Data shape from train.nc:
  (run=300, time=86, lev=2, y=64, x=64)

We use only the upper level (lev=0) to start.
Variables: q, psi, q_forcing_advection  →  stacked as channels.

For sequential prediction:
  Input  : (batch, in_channels,  H, W)  where in_channels = n_vars * window_size
  Target : (batch, out_channels, H, W)  where out_channels = n_vars * out_steps
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr
from typing import List


# ─── CONFIG ──────────────────────────────────────────────────────────────────
DATA_PATH      = "train.nc"                              # ← Path
WINDOW_SIZE    = 8                                                # past timesteps fed to the model
OUT_STEPS      = 5                                                # future q timesteps to predict
LEVEL          = 0                                                # 0 = upper layer, 1 = lower layer
INPUT_VARS     = ["q", "psi", "q_forcing_advection"]              # all vars used as input channels
TARGET_VARS    = ["q"]                                            # only q is predicted

# Train / val / test split (by run index, NOT shuffled → no leakage)
TRAIN_END = 210
VAL_END   = 255
# runs 255-299 → test
# ─────────────────────────────────────────────────────────────────────────────


def _normalise(arr, mean, std):
    return (arr - mean) / std


def load_and_split(
    data_path:    str,
    input_vars:   List[str], #Edited to List[str]from list[str], to fix en error? 29/3/2026
    target_vars:  List[str], #Edited to List[str]from list[str]
    level:        int,
):
    """
    Load the NetCDF file, select one vertical level, split by run,
    and normalise input and target arrays separately.

    Input  variables: all of q, psi, q_forcing_advection  → context for the U-Net
    Target variables: q only                               → what we predict

    Returns
    -------
    x_train, x_val, x_test  : (n_runs, time, n_input_vars,  H, W)
    y_train, y_val, y_test  : (n_runs, time, n_target_vars, H, W)
    norm_stats : dict with mean/std for inputs and targets (needed for denorm at eval)
    """
    ds     = xr.open_dataset(data_path)
    ds_lev = ds.isel(lev=level)                    # drop lev dim

    # ── Stack variables ───────────────────────────────────────────────────────
    # shape: (300, 86, n_vars, 64, 64)  float32
    inputs  = np.stack([ds_lev[v].values for v in input_vars],  axis=2).astype(np.float32)
    targets = np.stack([ds_lev[v].values for v in target_vars], axis=2).astype(np.float32)

    # ── Split by run (no shuffle → no temporal leakage) ───────────────────────
    x_train, x_val, x_test = inputs[:TRAIN_END],  inputs[TRAIN_END:VAL_END],  inputs[VAL_END:]
    y_train, y_val, y_test = targets[:TRAIN_END], targets[TRAIN_END:VAL_END], targets[VAL_END:]

    # ── Normalise — fit only on training set ──────────────────────────────────
    # mean/std: (1, 1, n_vars, 1, 1)  so they broadcast over runs, time, H, W
    x_mean = x_train.mean(axis=(0, 1, 3, 4), keepdims=True) #Compute one average per varbile using, all snapshots grids. Gives one one mean values per variable.
    x_std  = x_train.std (axis=(0, 1, 3, 4), keepdims=True) #Then standard deviation for each variable.
    x_std  = np.where(x_std == 0, 1.0, x_std) #In case a varaible is zero, we set std rather to 1, do furter avoid divtion by zero.


    y_mean = y_train.mean(axis=(0, 1, 3, 4), keepdims=True) #Same for the target variable, but only q, as we predict only q.
    y_std  = y_train.std (axis=(0, 1, 3, 4), keepdims=True) 
    y_std  = np.where(y_std == 0, 1.0, y_std)

    x_train = _normalise(x_train, x_mean, x_std) # Normalize each split of dataset.
    x_val   = _normalise(x_val,   x_mean, x_std)
    x_test  = _normalise(x_test,  x_mean, x_std)

    y_train = _normalise(y_train, y_mean, y_std)
    y_val   = _normalise(y_val,   y_mean, y_std)
    y_test  = _normalise(y_test,  y_mean, y_std)

    norm_stats = dict(x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)   #store normalization stats for later denormalization at evaluation of the model.

    print(f"[dataset] inputs  shape : {x_train.shape}  vars={input_vars}")
    print(f"[dataset] targets shape : {y_train.shape}  vars={target_vars}")
    print(f"[dataset] window={WINDOW_SIZE}  out_steps={OUT_STEPS}  level={level}")

    return x_train, x_val, x_test, y_train, y_val, y_test, norm_stats


class OceanDataset(Dataset):
    """
    Sliding-window dataset for sequential forecasting.

    Input  (X): all variables over WINDOW_SIZE past steps → channels-first
    Target (Y): q only over OUT_STEPS future steps        → channels-first

    Each sample:
      X : (n_input_vars  * WINDOW_SIZE, H, W)
      Y : (n_target_vars * OUT_STEPS,   H, W)
    """

    def __init__(
        self,               # inputs to be provided.
        x_data:    np.ndarray,   # (n_runs, T, n_input_vars,  H, W)
        y_data:    np.ndarray,   # (n_runs, T, n_target_vars, H, W)
        window:    int,     #number of timesteps fed to the model at a time
        out_steps: int,     #Nubmer of predicted timestamp sequence.
    ):
        n_runs, T, n_in,  H, W = x_data.shape #number of sequencs runs, total timesptems per run, number of input,  snapshot dimentions.
        _,      _, n_out, _, _ = y_data.shape   # number of traget variables.
        self.window    = window             #Save parameter
        self.out_steps = out_steps           

        xs, ys = [], []                    # Sliding window over each run (r), and time (t) inside of runs.
        for r in range(n_runs):
            for t in range(T - window - out_steps + 1):
                # ── Input: all vars, past window ──────────────────────────────
                x_win = x_data[r, t : t + window]          # (window, n_in,  H, W) Takes window through timesteps for all variables.
                x_win = x_win.transpose(1, 0, 2, 3)        # (n_in,  window, H, W) transpose to put the variables first, as models expect it.
                x_win = x_win.reshape(n_in * window, H, W) # flatten times and variables into channels. Model sees it as seperate channels

                # ── Target: q only, next out_steps ───────────────────────────
                y_win = y_data[r, t + window : t + window + out_steps]  # (out_steps, n_out, H, W) futures steps for traget variables
                y_win = y_win.transpose(1, 0, 2, 3)                      # (n_out, out_steps, H, W) varaibles first again.
                y_win = y_win.reshape(n_out * out_steps, H, W)      #flatten time and variable into channels.

                xs.append(x_win) #append flatten (channels, H, W) window input and tragets into the list of samples.
                ys.append(y_win)

    #Convert to from np to pytorch tensor and save as attributes. (n_samples, channels, H, W), where for training channels = n_in * window. and for target channels = n_out * out_steps.
    #Compatible for GPU training.
        self.X = torch.tensor(np.array(xs), dtype=torch.float32) 
        self.Y = torch.tensor(np.array(ys), dtype=torch.float32)

        print(f"[OceanDataset] X={self.X.shape}  Y={self.Y.shape}")

    def __len__(self):        #Number of samples, from sliding window.
        return len(self.X)

    def __getitem__(self, idx): #required for pytorch, for indexing dataset.
        return self.X[idx], self.Y[idx]


def get_dataloaders(
    data_path:   str        = DATA_PATH,
    input_vars:  list       = INPUT_VARS,
    target_vars: list       = TARGET_VARS,
    level:       int        = LEVEL,
    window:      int        = WINDOW_SIZE,
    out_steps:   int        = OUT_STEPS,
    batch_size:  int        = 16,
    num_workers: int        = 0,
):
    x_tr, x_va, x_te, y_tr, y_va, y_te, norm_stats = load_and_split(          #load and split data
        data_path, input_vars, target_vars, level
    )


        #Pytroch datasets
    train_ds = OceanDataset(x_tr, y_tr, window, out_steps)  #raw data input and output, input and output steps.
    val_ds   = OceanDataset(x_va, y_va, window, out_steps)
    test_ds  = OceanDataset(x_te, y_te, window, out_steps)


#Create dataloaders for each split. 
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                                #No shuffle to keep order for evaluation on ordered sequences, not randoms.
                              shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)



    in_channels  = len(input_vars)  * window     #Channels in and out of the model
    out_channels = len(target_vars) * out_steps  

    return train_loader, val_loader, test_loader, in_channels, out_channels, norm_stats
