from tqdm.auto import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader

from utils.torch_classes import EarlyStopping

import numpy as np
import matplotlib.pyplot as plt

def avg_loss(losses):
    return sum(losses) / len(losses)

def train_loop(model: nn.Module, 
                train_dataloader: DataLoader,
                loss_fn,
                optimizer: torch.optim.Optimizer,
                device
                ):
    
    model.train()
    
    train_losses = []
    y_pred_train = []
    y_true_train = []

    for batch in tqdm(train_dataloader, desc="Training", leave=False):
        # if batch_index == 5: break
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        optimizer.zero_grad()

        y_pred = model(x)
        loss = loss_fn(y_pred, y)

        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        y_pred_train.append(y_pred.argmax(dim=1))
        y_true_train.append(y)

        # if (batch_index + 1) % 100 == 0:  # every 100 batches
        #     print(f"Batch [{batch_index + 1}/{len(trainloader)}], Loss: {loss.item():.4f}")
    return {
        "losses": train_losses,
        "y_pred": y_pred_train,
        "y_true": y_true_train
    }
        
def val_loop(model: nn.Module,
             val_dataloader: DataLoader,
             loss_fn,
             device):
    
    model.eval()
    with torch.inference_mode():

        val_losses = []
        y_pred_val = []
        y_true_val = []

        for batch in tqdm(val_dataloader, desc="Validation", leave=False):
            x = batch["x"].to(device)
            y = batch["y"].to(device)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)

            val_losses.append(loss.item())

            y_pred_val.append(y_pred.argmax(dim=1))
            y_true_val.append(y)

    
    return {
        "losses": val_losses,
        "y_pred": y_pred_val,
        "y_true": y_true_val
    }
        
def test_loop(model: nn.Module,
             test_dataloader: DataLoader,
             device):
    
    model.eval()
    with torch.inference_mode():

        y_pred_test = []
        y_true_test = []

        for batch in tqdm(test_dataloader, desc="Testing", leave=False):
            x = batch["x"].to(device)
            y = batch["y"].to(device)

            y_pred = model(x)

            y_pred_test.append(y_pred.argmax(dim=1))
            y_true_test.append(y)


        final_y_pred_test = torch.cat(y_pred_test).cpu().numpy()
        final_y_true_test = torch.cat(y_true_test).cpu().numpy()

    
    return {
        "y_pred": final_y_pred_test,
        "y_true": final_y_true_test
    }
        
# MAIN
def train_and_eval_model(
        model: nn.Module,
        loss_fn,
        optimizer: torch.optim.Optimizer,
        
        train_dataloader: DataLoader, 
        val_dataloader: DataLoader,
        
        epochs: int,
        device: str,
        
        early_stopper: EarlyStopping = None,
        debug=True,
        verbose=True):

    history = {
        "train_loss": [], "val_loss": [],
        "train_pred": [], "val_pred": [],
        "train_true": [], "val_true": []
    }

    for epoch in range(epochs):

        # -------------- Training Data Per Epoch -----------
        train_data = train_loop(model=model,
                loss_fn=loss_fn,
                train_dataloader=train_dataloader,
                optimizer=optimizer,
                device=device)
        
        train_loss = avg_loss(train_data["losses"])
        train_pred = torch.cat(train_data["y_pred"])
        train_true = torch.cat(train_data["y_true"])

        if debug:
            print(f"Classes Predicted: {train_pred.unique()}")

         # -------------- Validating Data Per Epoch -----------        
        val_data = val_loop(model=model,
                            loss_fn=loss_fn,
                            val_dataloader=val_dataloader,
                            device=device)
        
        val_loss = avg_loss(val_data["losses"])
        val_pred = torch.cat(val_data["y_pred"])
        val_true = torch.cat(val_data["y_true"])
        
        # Early Stopper 
        if early_stopper is not None:
            early_stopper(
                val_loss=val_loss, 
                model=model,
                optimizer = optimizer,
                epoch=epoch+1
                )

        # store metrics in history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_pred"].append(train_pred.cpu())
        history["train_true"].append(train_true.cpu())
        history["val_pred"].append(val_pred.cpu())
        history["val_true"].append(val_true.cpu())

        if verbose:
            # Print Logs
            print(f"Epoch {epoch + 1} / {epochs}")
            print(f"Train Loss: {train_loss: .3f} | Val Loss: {val_loss: .3f}")

            if early_stopper is not None:
                print(f"Best Val Loss: {early_stopper.val_loss_min: .3f}")

            print("-------------------------------------------------")

            # Early Stopping
            if early_stopper is not None and early_stopper.early_stop:
                print(f"Stopping Early at epoch {epoch} ! Val Loss has not improved for {early_stopper.patience} epochs")
                print("-------------------------------------------------")
                break

    return history


# PLOTTING
def plot_training_history(history: dict, best_epoch):

    TRAIN_COLOUR = "blue"
    VAL_COLOUR = "red"
    AXVLINE_COLOUR = "green"
    AXVLINE_STYLE = "--"

    epochs = np.arange(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(15,5))

    # Plot loss curves
    plt.subplot(1,2,1)
    plt.plot(epochs, history['train_loss'], label="Train Loss", color=TRAIN_COLOUR)
    plt.plot(epochs, history['val_loss'], label="Validation Loss", color=VAL_COLOUR)
    plt.axvline(x=best_epoch, color=AXVLINE_COLOUR, linestyle=AXVLINE_STYLE, label=f"Best Loss ({best_epoch})")

    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Plot acc curves
    plt.subplot(1,2,2)
    plt.plot(epochs, history["train_acc"], color=TRAIN_COLOUR)
    plt.plot(epochs, history["val_acc"], color=VAL_COLOUR)
    plt.axvline(x=best_epoch, color=AXVLINE_COLOUR, linestyle=AXVLINE_STYLE, label=f"Best Acc ({best_epoch})")

    



        