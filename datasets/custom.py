"""
Custom dataset.

Mostly copy-paste from coco.py
"""
from pathlib import Path

from .coco import CocoDetection, make_coco_transforms

def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided path {root} to custom dataset does not exist'
    training_json_file = 'molina_train.json'
    validation_json_file = 'molina_validate.json'
    PATHS = {
        "train": ('/content/Capstone-Project/annotations_and_images/images/training_images', '/content/Capstone-Project/annotations_and_images/annotations/molina_train/molina_train.json'),
        "val": ('/content/Capstone-Project/annotations_and_images/images/validation_images', '/content/Capstone-Project/annotations_and_images/annotations/molina_validate/molina_validate.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=args.masks)
    return dataset
