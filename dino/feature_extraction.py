import argparse
import json
import os
from tqdm import tqdm
from PIL import Image
import faiss
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel


def read_json(input_path):
    with open(input_path, 'r') as f:
        input_file = json.load(f)
    return input_file


def save_vector(idx, vector):
    idx = int(idx)
    save_path = f"./dino_features/{idx:05d}.npy"
    np.save(save_path, vector)


def main(args):
    # Dataset
    data_config = read_json(args.data_config_path) 

    # Load the DINOv2 model and processor
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    print(f'Running on: {device}')
    processor = AutoImageProcessor.from_pretrained(f'facebook/{args.dino_model}')
    model = AutoModel.from_pretrained(f'facebook/{args.dino_model}').to(device)

    idx_to_vector = {}

    # Faiss index
    faiss_index = faiss.IndexFlatL2(args.embedding_size) 
    
    for idx in data_config:
        img_path = data_config[idx]
        img = Image.open(img_path).convert('RGB')
        print(f"Image: {idx}")

        with torch.no_grad():
            inputs = processor(images=img, return_tensors="pt").to(device)
            outputs = model(**inputs)
        
        features = outputs.last_hidden_state
        
        # Normalizes embeddings and add them to the index
        embedding = features.mean(dim=1)
        vector = embedding.detach().cpu().numpy()
        vector = np.float32(vector)

        idx_to_vector[idx] = vector
        save_vector(idx, vector)

        faiss.normalize_L2(vector)
        faiss_index.add(vector)

    faiss.write_index(faiss_index, args.index_path)
 

def argument_parser():
    parser = argparse.ArgumentParser(description='Parser for DINO features')
    parser.add_argument('--data_config_path', 
                        type=str, 
                        help='Directory containing the JSON dataset', 
                        required=True)
    parser.add_argument('--index_path', 
                        type=str, 
                        help='Index output path', 
                        required=True)
    parser.add_argument('--dino_model', 
                        type=str, 
                        help='DINOv2 model dimension', 
                        default='dinov2-giant')
    parser.add_argument('--embedding_size',
                        type=int,
                        help='Embedding size',
                        default=1536)
    args = parser.parse_args()
    return args


if __name__ == '__main__':    
    args = argument_parser()
    main(args)
