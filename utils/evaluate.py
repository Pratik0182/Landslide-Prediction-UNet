import os
import argparse
import h5py
import numpy as np
import torch
import sys

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.landslide_dataloader import LandslideDataset
from src.landslide_model import UNet
from utils.metrics import calculate_metrics

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    test_img_path = os.path.join(args.data_path, "TrainData", "img") # Using TrainData/img for validation split evaluation
    test_mask_path = os.path.join(args.data_path, "TrainData", "mask")
    
    # Load validation filenames
    val_filenames = os.path.join(args.filenames_dir, "val.txt")
    
    dataset = LandslideDataset(test_img_path, test_mask_path, val_filenames)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # load the model
    model = UNet(in_channels=14, out_channels=1).to(device)
    if not os.path.exists(args.model_path):
        print(f"error: model file {args.model_path} not found.")
        return
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    print(f"evaluating model on {len(dataset)} validation samples...")

    total_prec, total_rec, total_f1 = 0, 0, 0
    
    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            
            prec, rec, f1 = calculate_metrics(outputs, masks)
            total_prec += prec
            total_rec += rec
            total_f1 += f1

    n = len(dataloader)
    print("\n--- Evaluation Results ---")
    print(f"Average Precision: {total_prec/n:.4f}")
    print(f"Average Recall:    {total_rec/n:.4f}")
    print(f"Average F1-Score:  {total_f1/n:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Landslide Model Evaluation Script")
    parser.add_argument("--data_path", type=str, required=True, help="path to the dataset root")
    parser.add_argument("--filenames_dir", type=str, default="utils/filenames", help="directory containing val.txt")
    parser.add_argument("--model_path", type=str, default="models/unet_landslide.pth", help="path to the model weights")

    args = parser.parse_args()
    evaluate(args)
