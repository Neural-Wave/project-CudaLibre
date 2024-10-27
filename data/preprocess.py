import os
import json
import random 


def read_json(input_path):
    with open(input_path, 'r') as f:
        input_file = json.load(f)
    return input_file


def get_img_paths(img_dir):
    img_paths = [os.path.join(img_dir, img_path) for img_path in os.listdir(img_dir)]
    return img_paths 


def get_labeled_subset(data_dir):    
    labeled_data_dir = os.path.join(data_dir, 'example_set')

    aligned_img_dir = os.path.join(labeled_data_dir, 'aligned')
    aligned_img_paths = get_img_paths(aligned_img_dir)

    not_aligned_img_dir = os.path.join(labeled_data_dir, 'not_aligned')
    not_aligned_img_paths = get_img_paths(not_aligned_img_dir)

    images_paths = aligned_img_paths + not_aligned_img_paths

    images_dict = {}
    for image_path in images_paths:
        images_dict[image_path] = os.path.dirname(image_path).split('/')[-1]

    return images_dict


def get_labeled_trainset(labeled_dict, unlabeled_data_dir):
    images_dict = {}
    for img_name in labeled_dict:
        images_dict[os.path.join(unlabeled_data_dir, img_name)] = labeled_dict[img_name][1]

    return images_dict


def stratified_split(data, train_ratio, val_ratio, test_ratio):
    # Separate images by their labels
    aligned = {k: v for k, v in data.items() if v == "aligned"}
    not_aligned = {k: v for k, v in data.items() if v == "not_aligned"}

    # Helper function to split a dictionary by a given ratio
    def split_dict(data_dict, train_ratio, val_ratio):
        total_items = len(data_dict)
        train_size = int(total_items * train_ratio)
        val_size = int(total_items * val_ratio)
        keys = list(data_dict.keys())
        random.shuffle(keys)
        
        train_keys = keys[:train_size]
        val_keys = keys[train_size:train_size + val_size]
        test_keys = keys[train_size + val_size:]

        return (
            {k: data_dict[k] for k in train_keys},
            {k: data_dict[k] for k in val_keys},
            {k: data_dict[k] for k in test_keys}
        )

    # Perform the stratified split for each label
    train_aligned, val_aligned, test_aligned = split_dict(aligned, train_ratio, val_ratio)
    train_not_aligned, val_not_aligned, test_not_aligned = split_dict(not_aligned, train_ratio, val_ratio)

    # Combine the splits to form the final train, val, and test sets
    train_set = {**train_aligned, **train_not_aligned}
    val_set = {**val_aligned, **val_not_aligned}
    test_set = {**test_aligned, **test_not_aligned}

    return train_set, val_set, test_set


if __name__ == '__main__':

    data_dir = '/teamspace/s3_connections/dtp-sbm-segmentation-video-tasks-bars-stopper-alignment-images-hackaton-usi'

    # Labeled dataset subset
    labeled_imgs = get_labeled_subset(data_dir)

    # Manually labeled dataset
    unlabeled_data_dir = os.path.join(data_dir, 'train_set')

    good_light_labeled_dict = read_json('dataset/label_extraction/good_light_image_labels.json')
    bad_light_labeled_dict = read_json('dataset/label_extraction/bad_light_image_label.json')
    
    good_light_imgs = get_labeled_trainset(good_light_labeled_dict, unlabeled_data_dir)
    bad_light_imgs = get_labeled_trainset(bad_light_labeled_dict, unlabeled_data_dir)
    
    print("---------------------------------------")
    print(len(labeled_imgs))
    print(len(good_light_imgs))
    print(len(bad_light_imgs))
    print("---------------------------------------")

    img = {**labeled_imgs, **good_light_imgs, **bad_light_imgs} 

    # Split into train, val, and test with stratified sampling
    train_set, val_set, test_set = stratified_split(img, 
                                                    train_ratio=0.70, 
                                                    val_ratio=0.15, 
                                                    test_ratio=0.15)
    
    # Output split
    out = {
        'train': train_set,
        'val': val_set,
        'test': test_set
    }

    # Save split to JSON
    with open('dataset/split.json', 'w') as json_file:
        json.dump(out, json_file, indent=4)
