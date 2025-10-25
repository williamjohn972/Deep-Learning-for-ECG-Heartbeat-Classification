import torch 
from torch import nn
from torch.utils.data import Dataset

import pandas as pd
import numpy as np

import os

class ECG_Dataset(Dataset):
    
    def __init__(self, data: pd.DataFrame):
        
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        target_row = self.data.iloc[index]

        # for future reference
        # CNN --> (batch, channels, length)
        # RNN --> (batch, seq_len, features per sequence)

        X = torch.tensor(target_row.iloc[:-1].values).type(torch.float)
        y = torch.tensor(target_row.iloc[-1]).type(torch.long)

        return {
            "x": X,
            "y": y
        }
    
# Early Stopping Class
class EarlyStopping():

    def __init__(self, 
                patience=10, 
                delta=0,
                checkpoint_path='best_model.pt'):
    
        self.patience = patience
        self.delta = delta
        self.checkpoint_path = checkpoint_path
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, 
                val_loss, 
                model: nn.Module,
                optimizer: torch.optim.Optimizer,
                epoch: int):
        
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(
                val_loss, 
                model,
                optimizer,
                epoch)

        elif score < self.best_score + self.delta:
            self.counter += 1
            
            print(f"Early Stopping Counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.counter = 0
            self.best_score = score
            self.save_checkpoint(
                val_loss, 
                model,
                optimizer,
                epoch)

    def save_checkpoint(self, 
                        val_loss, 
                        model: nn.Module,
                        optimizer: torch.optim.Optimizer,
                        epoch: int):
        
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)

        print(f"Validation loss decreased ({self.val_loss_min: .4f} --> {val_loss: .4f})")

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss_min': val_loss,

            'patience': self.patience,
            'delta': self.delta
        }

        torch.save(checkpoint, self.checkpoint_path)
        self.val_loss_min = val_loss
