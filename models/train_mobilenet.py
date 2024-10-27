import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard
import time
from datetime import datetime
from sklearn.metrics import fbeta_score

from data.DufercoDataset import DufercoDataset
from data.transforms import train_transforms, test_transforms
from models.trainer import train_model


def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    # Setup dataset
    train_dataset = DufercoDataset(
        args.data_dir, 
        split='train',
        transform=train_transforms,
    )
    
    print("Trained data loaded") 

    val_dataset = DufercoDataset(
        args.data_dir,
        split='val',
        transform=test_transforms
    )
    print("Validation data loaded")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Model Setup: MobileNetV2 and Modify for Binary Classification
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)  # Update output layer 
    model = model.to(device)

    # Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Initialize TensorBoard writer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f'runs/experiment_{timestamp}'
    writer = SummaryWriter(log_dir=log_dir)
    checkpoint_path = os.path.join(args.checkpoint_path, timestamp)

    # Train the model    
    train_model(model, train_loader, val_loader, 
                criterion, optimizer, args.num_epochs, 
                writer, device, checkpoint_path)

    writer.close()  


def argument_parser():
    parser = argparse.ArgumentParser(description="Train MobileNet on a Duferco dataset")
    parser.add_argument('--data_dir', 
                        type=str, 
                        required=True, 
                        help='Path to dataset directory')
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='Batch size')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=10,
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
