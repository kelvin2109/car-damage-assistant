import albumentations as A
import numpy as np
import torch

# Define bounding box parameters (adjust format to your data)
bbox_params = A.BboxParams(format='pascal_voc', label_fields=['labels'])

# Common base augmentations (e.g., Affine)
common = [
    A.Affine(
        scale=(0.8, 1.2),
        translate_percent=(-0.1, 0.1),
        rotate=(-30, 30),
        shear=(-10, 10),
        p=0.5
    ),
]

# Majority transform (your original, with lower probabilities)
majority_transform = A.Compose(
    common + [
        A.OneOf([
            A.RandomRain(p=0.5),
            A.RandomSnow(p=0.5)
        ], p=0.4),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1),
            A.RandomGamma(gamma_limit=(90, 110))
        ], p=0.2)
    ],
    bbox_params=bbox_params,
    mask_params='mask'           # assuming single mask for segmentation
)

# Minority transform (stronger / higher probabilities)
minority_transform = A.Compose(
    common + [
        A.OneOf([
            A.RandomRain(p=0.7),    # higher probability
            A.RandomSnow(p=0.7)
        ], p=0.8),                  # higher overall p
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.RandomGamma(gamma_limit=(80, 120))
        ], p=0.5),
        A.HorizontalFlip(p=0.5)     # extra augmentation
    ],
    bbox_params=bbox_params,
    mask_params='mask'
)

class BalancedDataset(torch.utils.data.Dataset):
    def __init__(self, images, bboxes, labels, masks, minority_classes=[2,5]):
        self.images = images
        self.bboxes = bboxes
        self.labels = labels
        self.masks = masks
        self.minority_classes = minority_classes
        self.majority_transform = majority_transform
        self.minority_transform = minority_transform

    def __getitem__(self, idx):
        image = self.images[idx]
        bboxes = self.bboxes[idx]          # list of boxes for this image
        labels = self.labels[idx]          # list of class IDs
        mask = self.masks[idx]             # (H, W) mask if needed

        # Check if this image contains any minority class
        contains_minority = any(label in self.minority_classes for label in labels)

        transform = self.minority_transform if contains_minority else self.majority_transform

        augmented = transform(image=image, bboxes=bboxes, labels=labels, mask=mask)
        return (augmented['image'], augmented['bboxes'], 
                augmented['labels'], augmented['mask'])

    def __len__(self):
        return len(self.images)