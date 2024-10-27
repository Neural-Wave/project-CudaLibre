import sys
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import time

from data.transforms import test_transforms
from models.EfficientNet import EfficientNetBinaryClassifier


class InferenceDataset(datasets.VisionDataset):
    def __init__(self, root_dir, transform=None):
        super().__init__(root_dir, transform=transform)
        self.root_dir = root_dir
        self.transform = transform

        self.image_paths = []
        for label in ['aligned', 'not_aligned']:
            folder_path = os.path.join(root_dir, label)
            self.image_paths.extend([(os.path.join(folder_path, img), label) 
                                     for img in os.listdir(folder_path) if img.endswith(('png', 'jpg', 'jpeg'))])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path, label = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = 1 if label == "aligned" else 0

        if self.transform:
            image = self.transform(image)
            
        return image, label 


def evaluate_model(model, test_loader, device, beta=0.5):
    model.eval()
    all_labels = []
    all_predictions = []
    inference_times = []  # List to store inference times

    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images, labels = images.to(device), labels.to(device)
            labels = labels.unsqueeze(1).to(torch.float32)
            
            # Start time for inference
            start_time = time.time()
            
            outputs = model(images)
            predicted = (outputs > 0.5).float()
            
            # End time and calculate duration
            end_time = time.time()
            inference_time = end_time - start_time
            inference_times.append(inference_time)  # Store the duration

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

    # Calculate statistics on inference times
    inference_times = np.array(inference_times)
    mean_time = np.mean(inference_times)
    quantiles = np.quantile(inference_times, [0.25, 0.5, 0.75])

    print(f"\nInference Time Statistics:")
    print(f"Mean Time: {mean_time:.4f} seconds")
    print(f"25th Percentile: {quantiles[0]:.4f} seconds")
    print(f"Median (50th Percentile): {quantiles[1]:.4f} seconds")
    print(f"75th Percentile: {quantiles[2]:.4f} seconds")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on: {device}")

    # Load test data
    test_dataset = InferenceDataset(root_dir=args.s, 
                                   transform=test_transforms)
    test_loader = DataLoader(test_dataset, 
                             batch_size=args.batch_size,
                             shuffle=False)

    # Load model
    model = EfficientNetBinaryClassifier()
    model = nn.DataParallel(model)  
    model.load_state_dict(torch.load(args.model_path)["model_state_dict"])
    model = model.to(device)

    # Evaluate model
    evaluate_model(model, test_loader, device, beta=0.5)


def argument_parser():
    parser = argparse.ArgumentParser(description="Evaluate EfficientNet on Duferco test dataset")
    parser.add_argument('--s', 
                        type=str, 
                        required=True, 
                        help='Path to dataset')
    parser.add_argument('--batch_size',
                        type=int,
                        default=16,
                        help='Batch size')
    parser.add_argument('--model_path',
                        type=str,
                        default='checkpoints/efficient_net/20241027_083453/model_epoch_5.pt',
                        help='Path to the trained model checkpoint')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = argument_parser()
    main(args)
