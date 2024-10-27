import argparse
import os
import json


def save_mapping(path, map): 
    with open(path, 'w') as json_file:
        json.dump(map, json_file, indent=4)


def get_img_paths(img_dir):
    img_paths = [os.path.join(img_dir, img_path) for img_path in os.listdir(img_dir)]
    return img_paths 


def get_example_set(args):
    example_set_dir = os.path.join(args.data_dir, 'example_set')
    aligned_img_dir = os.path.join(example_set_dir, 'aligned')
    not_aligned_img_dir = os.path.join(example_set_dir, 'not_aligned')

    aligned_img_paths = get_img_paths(aligned_img_dir)
    not_aligned_img_paths = get_img_paths(not_aligned_img_dir)
    img_paths = aligned_img_paths + not_aligned_img_paths
    return img_paths


def get_train_set(args):
    train_set_dir = os.path.join(args.data_dir, 'train_set')
    good_light_dir = os.path.join(train_set_dir, 'good_light')
    bad_light_dir = os.path.join(train_set_dir, 'bad_light')

    good_light_img_paths = get_img_paths(good_light_dir)
    bad_light_img_paths = get_img_paths(bad_light_dir)
    img_paths = good_light_img_paths + bad_light_img_paths
    return img_paths


def main(args):
    example_set_img_paths = get_example_set(args)
    train_set_img_paths = get_train_set(args)

    img_paths = example_set_img_paths + train_set_img_paths
    
    idx_to_imgpath = {}
    for i, img_path in enumerate(img_paths):
        idx_to_imgpath[i] = img_path

    imgpath_to_idx = {}
    for idx in idx_to_imgpath:
        imgpath = idx_to_imgpath[idx]
        imgpath_to_idx[imgpath] = idx

    save_mapping('mappings/idx_to_imgpath.json', idx_to_imgpath)
    save_mapping('mappings/imgpath_to_idx.json', imgpath_to_idx)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build mapping between paths and indexes")
    parser.add_argument('--data_dir', 
                        type=str, 
                        required=True, 
                        help='Dataset directory')
    args = parser.parse_args()
    main(args)
    