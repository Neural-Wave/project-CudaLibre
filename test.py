import sys
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.DufercoDataset import DufercoDataset
from data.transforms import test_transforms
from models.EfficientNet import EfficientNetBinaryClassifier

def load_dataloaders(args):
    test_dataset = DufercoDataset(
        args.data_config_path,
        split='test',
        transform=test_transforms
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    return test_loader

def evaluate_model(model, test_loader, device, beta=0.5):
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images, labels = images.to(device), labels.to(device)
            labels = labels.unsqueeze(1).to(torch.float32)

            outputs = model(images)
            predicted = (outputs > 0.5).float()

            all_labels.extend(labels.cpu().detach().numpy().astype(int))
            all_predictions.extend(predicted.cpu().detach().numpy().astype(int))

    # Convert lists to numpy arrays for metric calculations
    all_labels = np.array(all_labels).flatten()
    all_predictions = np.array(all_predictions).flatten()

    # Calculate evaluation metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    fbeta = fbeta_score(all_labels, all_predictions, beta=beta)
    confusion = confusion_matrix(all_labels, all_predictions)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"F-beta Score (beta={beta}): {fbeta:.4f}")
    print(f"Confusion Matrix:\n{confusion}")

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on: {device}")

    # Load test data
    test_loader = load_dataloaders(args)

    # Load model
    model = EfficientNetBinaryClassifier()
    model = nn.DataParallel(model)  # Wrap for multi-GPU support if necessary
    model.load_state_dict(torch.load(args.model_path)["model_state_dict"])
    model = model.to(device)

    # Evaluate model
    evaluate_model(model, test_loader, device, beta=args.beta)

def argument_parser():
    parser = argparse.ArgumentParser(description="Evaluate EfficientNet on Duferco test dataset")
    parser.add_argument('--data_config_path', 
                        type=str, 
                        required=True, 
                        help='Path to dataset JSON')
    parser.add_argument('--batch_size',
                        type=int,
                        default=16,
                        help='Batch size')
    parser.add_argument('--model_path',
                        type=str,
                        required=True,
                        help='Path to the trained model checkpoint')
    parser.add_argument('--beta',
                        type=float,
                        default=0.5,
                        help='Beta value for F-beta score')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = argument_parser()
    main(args)
