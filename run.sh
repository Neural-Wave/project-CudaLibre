#!/bin/bash

python train.py \
    --data_config_path "dataset/processed_augmented_split.json" \
    --batch_size 32 \
    --num_epochs 30 \
    --learning_rate 0.0001 \
    --checkpoint_path "/teamspace/studios/this_studio/checkpoints/efficient_net"

python test.py \
    --data_config_path "dataset/split.json" \
    --batch_size 16 \
    --model_path "checkpoints/efficient_net/20241027_083453/model_epoch_10.pt"