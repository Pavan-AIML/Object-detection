import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from tqdm import tqdm

from exercise_code.model.hog import HoG


def train(
    dataloader: DataLoader,
    model: nn.Module,
    criterion,
    optimizer: optim.Optimizer,
    num_epochs: int,
    device: torch.device,
):
    for epoch in range(1, num_epochs + 1):
        losses = []
        correct = 0.0
        total = 0
        TP,TN,P,N,P_pred = 0,0,0,0,0

        with tqdm(dataloader, unit=" batches") as tepoch:
            for batch in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                patches = batch[1]["patches"].to(device)
                hog_features = HoG(
                    transforms.functional.rgb_to_grayscale(patches.flatten(0, 1)).squeeze(-3),
                    8,
                    9,
                ) # B, F
                y = batch[1]["patch_labels"].flatten(0, 1).to(device) # B

                optimizer.zero_grad()

                y_pred = model(hog_features)

                loss = criterion(y_pred, y)  # calculate the loss with the network predictions and ground Truth
                loss.backward()  # Perform a backward pass to calculate the gradients
                optimizer.step()

                losses.append(loss.item())

                len_running_loss = 10
                running_loss = losses[-len_running_loss if len(losses) > len_running_loss else 0 :]
                correct += (y_pred > 0.5).eq(y).sum().item()
                TP += (((y_pred > 0.5).eq(y))*y).sum().item()
                TN += (((y_pred > 0.5).eq(y))*(1-y)).sum().item()
                P += y.sum().item()
                N += (1-y).sum().item()
                P_pred += (y_pred > 0.5).sum().item()
                total += y.size(0)

                tepoch.set_postfix(
                    {
                        "running loss": sum(running_loss) / len(running_loss),
                        "train accuracy": 100.0 * (correct) / total,
                        "train precision": 100.0 * (TP) / P,
                        "train recall": 100.0 * (TP) / P_pred,
                    }
                )


def evaluate(
    dataloader: DataLoader,
    model: nn.Module,
    device: torch.device,
):
    model.eval()
    with torch.no_grad():
        correct = 0.0
        total = 0
        TP,TN,P,N,P_pred = 0,0,0,0,0

        with tqdm(dataloader, unit=" batches") as tepoch:
            for batch_idx, batch in enumerate(tepoch):
                patches = batch[1]["patches"].to(device)
                hog_features = HoG(
                    transforms.functional.rgb_to_grayscale(patches.flatten(0, 1)).squeeze(-3),
                    8,
                    9,
                )
                y = batch[1]["patch_labels"].flatten(0, 1).to(device)

                y_pred = model(hog_features)

                correct += (y_pred > 0.5).eq(y).sum().item()
                TP += (((y_pred > 0.5).eq(y))*y).sum().item()
                TN += (((y_pred > 0.5).eq(y))*(1-y)).sum().item()
                P += y.sum().item()
                N += (1-y).sum().item()
                P_pred += y_pred.sum().item()
                total += y.size(0)

                if batch_idx >= len(tepoch) - 1:
                    tepoch.set_postfix(
                        {
                            "test accuracy": 100.0 * (correct) / total,
                            "test precision": 100.0 * (TP) / P,
                            "test recall": 100.0 * (TP) / P_pred,
                        }
                    )
