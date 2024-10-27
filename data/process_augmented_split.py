import random
import json

def add_and_split_new_data(old_data, new_data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):

    new_train, new_val, new_test = {}, {}, {}
    
    # 1. Find overlapping data and add them to the same split
    for image_path, label in new_data.items():
        if image_path in old_data["train"]:
            continue
        elif image_path in old_data["val"]:
            continue
        elif image_path in old_data["test"]:
            continue
        else:
            # If not overlapping, add to a new list to split later
            if image_path not in new_train and image_path not in new_val and image_path not in new_test:
                new_train[image_path] = label

    # 2. Split the non-overlapping data into train, val, and test
    non_overlapping_data = list(new_train.items())  # Convert to list for random shuffling
    random.shuffle(non_overlapping_data)  # Shuffle to ensure random split

    # Determine the number of images for each split
    total_new = len(non_overlapping_data)
    train_end = int(total_new * train_ratio)
    val_end = train_end + int(total_new * val_ratio)

    # Assign to train, val, and test
    for idx, (image_path, label) in enumerate(non_overlapping_data):
        if idx < train_end:
            old_data["train"][image_path] = label
        elif idx < val_end:
            old_data["val"][image_path] = label
        else:
            old_data["test"][image_path] = label

    return old_data


def read_json(input_path):
    with open(input_path, 'r') as f:
        input_file = json.load(f)
    return input_file


augmented_split = read_json('/teamspace/studios/this_studio/dataset/augmented_split.json')
original_slit = read_json('/teamspace/studios/this_studio/dataset/split.json')

augmented_processed_split = add_and_split_new_data(original_slit, augmented_split)

# Save split to JSON
with open('/teamspace/studios/this_studio/dataset/processed_augmented_split.json', 'w') as json_file:
    json.dump(augmented_processed_split, json_file, indent=4)
