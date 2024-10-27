import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard
import time
from datetime import datetime
from sklearn.metrics import fbeta_score

from data.DufercoDataset import DufercoDataset
from data.transforms import train_transforms, test_transforms
from models.EfficientNet import EfficientNetBinaryClassifier
from models.trainer import train_model


def train_dataloaders(args):
    train_dataset = DufercoDataset(
        args.data_config_path, 
        split='train',
        transform=train_transforms,
    )
    val_dataset = DufercoDataset(
        args.data_config_path,
        split='val',
        transform=test_transforms
    )

    sample_weights = train_dataset.get_sample_weights()
    sampler = WeightedRandomSampler(
        weights=sample_weights, 
        num_samples=len(sample_weights), 
        replacement=True
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        # shuffle=True, 
        sampler=sampler
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False
    )
    return train_loader, val_loader


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    # Data
    train_loader, val_loader = train_dataloaders(args)

    # Model for Binary Classification
    model = EfficientNetBinaryClassifier()
    model = nn.DataParallel(model)  # Wrap model for multi-GPU support
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    # TensorBoard writer and checkpoints
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f'runs/experiment_{timestamp}'
    writer = SummaryWriter(log_dir=log_dir)
    checkpoint_path = os.path.join(args.checkpoint_path, timestamp)
    os.makedirs(checkpoint_path, exist_ok=True)
    
    # Train the model    
    train_model(
        model, 
        train_loader, 
        val_loader, 
        criterion, 
        optimizer, 
        args.num_epochs,
        writer, 
        device, 
        checkpoint_path
    )
    writer.close()  


def argument_parser():
    parser = argparse.ArgumentParser(description="Train EfficientNet on a Duferco dataset")
    parser.add_argument('--data_config_path', 
                        type=str, 
                        required=True, 
                        help='Path to dataset JSON')
    parser.add_argument('--batch_size',
                        type=int,
                        default=16,
                        help='Batch size')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=20,
                        help='Number of epochs')
    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.001,
                        help='Number of epochs')
    parser.add_argument('--checkpoint_path',
                        type=str,
                        required=True,
                        help='Checkpoint path')
    args = parser.parse_args()
    return args


if __name__ == '__main__':    
    args = argument_parser()
    main(args)
