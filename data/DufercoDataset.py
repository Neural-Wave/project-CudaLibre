import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class DufercoDataset(Dataset):
    def __init__(self, data_config_path, split, transform=None):
        """
        Args:
            data_config_path (str): Path to the dataset JSON.
            split (str): One of 'train', 'val', or 'test'.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.data_config_path = data_config_path
        self.split = split
        self.transform = transform

        # Load the JSON file containing paths and splits
        with open(data_config_path, 'r') as f:
            self.data_config = json.load(f)

        self.dataset = self.data_config[split]
        self.image_paths = list(self.dataset.keys())

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Get image path and label
        img_path = self.image_paths[idx]
        label = 1 if self.dataset[img_path] == "aligned" else 0

        # Open the image
        image = Image.open(img_path).convert("RGB")
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)

        return image, label

    def get_sample_weights(self):       
        # Count samples in each class
        class_counts = [0, 0]
        for img_path in self.image_paths:
            label = 1 if self.dataset[img_path] == "aligned" else 0
            class_counts[label] += 1

        # Calculate class weights (inverse of class frequency)
        class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
        sample_weights = [class_weights[1 if self.dataset[img_path] == "aligned" else 0] 
                          for img_path in self.image_paths]
        return sample_weights
    