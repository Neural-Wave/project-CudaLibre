import argparse
import os
import json
from collections import Counter
from tqdm import tqdm
import faiss
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, AutoModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from PIL import Image


def read_json(input_path):
    with open(input_path, 'r') as f:
        input_file = json.load(f)
    return input_file


def build_training_index(dataset_config, imgpath_to_idx):
    # "Training" index: Training + Validation
    faiss_train_index = faiss.IndexFlatL2(1536)

    new_idx = 0
    new_idx_to_imgpath = {}

    for split in ['train', 'val']:
        for img_path in dataset_config[split]:
            idx = imgpath_to_idx[img_path]
            vector_path = f'dino_features/{int(idx):05d}.npy'
            if not os.path.exists(vector_path):
                continue
                
            vector = np.load(vector_path)
            faiss.normalize_L2(vector)
            faiss_train_index.add(vector)

            new_idx_to_imgpath[new_idx] = img_path
            new_idx += 1

    return faiss_train_index, new_idx_to_imgpath


def check_performance_on_test_data(dataset_config, 
                                   imgpath_to_idx,
                                   new_idx_to_imgpath, 
                                   faiss_train_index,
                                   K):
    true_labels = []
    predicted_labels = []
    for img_path in dataset_config['test']:
        idx = imgpath_to_idx[img_path]
        vector_path = f'dino_features/{int(idx):05d}.npy'
        if not os.path.exists(vector_path):
            continue

        vector = np.load(vector_path)
        faiss.normalize_L2(vector)

        # Perform search of top-K images
        dists, closest_neighbor_idxs = faiss_train_index.search(vector, K)
        print('distances:', dists, 'indexes:', closest_neighbor_idxs)

        # Access the closest image using the indices from the index search
        closest_neighbor_idxs = closest_neighbor_idxs[0]

        true_label = dataset_config['test'][img_path][1]
        k_predicted_labels = []
        for closest_idx in closest_neighbor_idxs:
            img_path_train = new_idx_to_imgpath[closest_idx]
            if img_path_train in dataset_config['train']:
                k_predicted_label = dataset_config['train'][img_path_train][1]
            else:
                k_predicted_label = dataset_config['val'][img_path_train][1]
            k_predicted_labels.append(k_predicted_label)

        # Majority vote to determine the final predicted label
        predicted_label = Counter(k_predicted_labels).most_common(1)[0][0]
        
        true_labels.append(true_label)
        predicted_labels.append(predicted_label)

    # Check the unique labels
    unique_labels = set(true_labels + predicted_labels)
    print("Unique labels:", unique_labels)

    # Check if labels need to be mapped
    label_mapping = {"l": "aligned", "o": "not_aligned"}
    true_labels = [label_mapping.get(label, label) for label in true_labels]
    predicted_labels = [label_mapping.get(label, label) for label in predicted_labels]

    # Compute all the relevant metrics to evaluate the performance of the binary classifier
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, pos_label="aligned", average='binary')
    recall = recall_score(true_labels, predicted_labels, pos_label="aligned", average='binary')
    f1 = f1_score(true_labels, predicted_labels, pos_label="aligned", average='binary')
    conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=["aligned", "not_aligned"])

    # Print out the metrics
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision (aligned): {precision:.4f}')
    print(f'Recall (aligned): {recall:.4f}')
    print(f'F1 Score (aligned): {f1:.4f}')
    print(f'Confusion Matrix:\n{conf_matrix}')


def main():
    K = 3
    dataset_config = read_json('/teamspace/studios/this_studio/dataset/split.json')
    idx_to_imgpath = read_json('mappings/idx_to_imgpath.json')
    imgpath_to_idx = read_json('mappings/imgpath_to_idx.json')
    
    # Build the trained index
    faiss_train_index, new_idx_to_imgpath = build_training_index(dataset_config, imgpath_to_idx)

    # Label the whole dataset
    augmented_labels = {}
    for idx in idx_to_imgpath:

        vector_path = f'dino_features/{int(idx):05d}.npy'
        if not os.path.exists(vector_path):
                continue        
        vector = np.load(vector_path)
        faiss.normalize_L2(vector)

        dists, closest_neighbor_idxs = faiss_train_index.search(vector, K)
        closest_neighbor_idxs = closest_neighbor_idxs[0]

        k_predicted_labels = []
        for closest_idx in closest_neighbor_idxs:
            img_path_train = new_idx_to_imgpath[closest_idx]
            if img_path_train in dataset_config['train']:
                k_predicted_label = dataset_config['train'][img_path_train][1]
            else:
                k_predicted_label = dataset_config['val'][img_path_train][1]
            k_predicted_labels.append(k_predicted_label)

        # Majority vote to determine the final predicted label
        predicted_label = Counter(k_predicted_labels).most_common(1)[0][0]

        image_path = idx_to_imgpath[idx]
        augmented_labels[image_path] = predicted_label

    label_mapping = {"l": "aligned", "o": "not_aligned"}
    for key in augmented_labels:
        label = augmented_labels[key]
        augmented_labels[key] = label_mapping[label]
        
    # Save split to JSON
    with open('/teamspace/studios/this_studio/dataset/augmented_split.json', 'w') as json_file:
        json.dump(augmented_labels, json_file, indent=4)


    # Check performance of the method on a test set
    check_performance_on_test_data(
        dataset_config,
        imgpath_to_idx,
        new_idx_to_imgpath, 
        faiss_train_index,
        K,
    )
        

if __name__ == '__main__':
    
    main()
    