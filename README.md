# CUDA_Libre Steel Bar Alignment Detection

This repository contains the code for the "CUDA_Libre" team's solution to the Neural Wave Hackathon 2024. The goal of this project is to automate the verification of steel bar alignment in a rolling mill using AI, enhancing efficiency and reducing human error in industrial settings.

## Overview

The project addresses the problem of detecting whether steel bars are aligned with a mechanical stopper. The solution involves:
1. **Labeling Workflow**: Semi-automated labeling of the dataset using DINO v2 for embedding and K-Nearest Neighbors (KNN) with majority voting.
2. **Model Training**: Training an EfficientNet model for binary classification (aligned or not aligned).
3. **Inference**: Running the model on new images to predict alignment status.

## File Structure

- `checkpoints/`: Directory to save model checkpoints.
- `data/`: Directory for auxiliary data files (e.g., configuration).
- `dataset/`: Directory containing training and test datasets.
- `dino/`: Directory for DINO v2 embeddings used in labeling.
- `labeling_workflow/`: Code and scripts related to the semi-automated labeling process.
- `models/`: Directory for storing the EfficientNet model and related files.
- `train.py`: Script to train the model.
- `test.py`: Script to test the model.
- `inference.py`: Script for running inference on new data.
- `run.sh`: Shell script to streamline training and testing.
- `README.md`: This file.

## Dependencies

The project relies on the following dependencies:
- `torch`
- `torchvision`
- `numpy`
- `scikit-learn`
- `opencv-python`

To install all dependencies, you can use:
```bash
pip install -r requirements.txt
```
## Running the Code

### Training the Model
To train the model, use the `run.sh` script. This script will start training with specified parameters and save the best model to the checkpoints directory.

```bash
bash run.sh
```

Alternatively, you can run the training script directly:
```bash
python train.py --data_config_path "dataset/processed_augmented_split.json" --batch_size 32 --num_epochs 30 --learning_rate 0.0001 --checkpoint_path "checkpoints/efficient_net"
```

## Testing the Model

The testing script evaluates the trained model on a validation dataset. You can run the test script as follows:
```bash
python test.py --data_config_path "dataset/split.json" --batch_size 16 --model_path "checkpoints/efficient_net/20241027_083453/model_epoch_10.pt"
```
