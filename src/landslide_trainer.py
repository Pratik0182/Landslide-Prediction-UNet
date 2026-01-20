import torch
import sys
import os

# Add the project root to sys.path to import from utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.metrics import calculate_metrics

#training function for one epoch
def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    total_prec, total_rec, total_f1 = 0, 0, 0
    for images, masks in dataloader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        prec, rec, f1 = calculate_metrics(outputs, masks)
        total_prec += prec
        total_rec += rec
        total_f1 += f1
        
    n = len(dataloader)
    return total_loss / n, total_prec / n, total_rec / n, total_f1 / n

#validation function for one epoch
def validate_one_epoch(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    total_prec, total_rec, total_f1 = 0, 0, 0
    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            total_loss += loss.item()
            
            prec, rec, f1 = calculate_metrics(outputs, masks)
            total_prec += prec
            total_rec += rec
            total_f1 += f1
            
    n = len(dataloader)
    return total_loss / n, total_prec / n, total_rec / n, total_f1 / n
