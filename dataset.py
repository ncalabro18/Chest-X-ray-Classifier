"""
© 2026 Nicholas J. Calabro. All rights reserved.

Dataset and Augmentation
- CXR8Dataset: PyTorch Dataset class for NIH CXR8
- make_train_tf: Creates the training augmentation pipeline
- make_value_tf: Creates the validation/test augmentation pipeline
- init_split: Splits the dataset into training and validation sets,
        ensuring patient-level separation and stratification
    

"""

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
import torch
from torch.utils.data import Dataset
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from classes import ALL_CLASSES


### Augmentation / Preprocessing Constants ###

HORIZONTAL_FLIP_PROB = 0.5

ROTATION_DEGREES = 2.8
ROTATION_PROB = 0.5

JITTER_PROB       = 0.5
JITTER_BRIGHTNESS = 0.08
JITTER_CONTRAST   = 0.08

# Contrast Limited Adaptive Histogram Equalization
CLAHE_CLIP_LIMIT = 1.5
CLAHE_TILE_GRID_SIZE = 4
CLAHE_PROB = 1.0 # making consistant for now to increase stability

# ElasticTransform
ELASTIC_ALPHA = 1.0
ELASTIC_SIGMA = 6.0
ELASTIC_INTERPOLATION = cv2.INTER_LINEAR
ELASTIC_FILL  = 0
ELASTIC_PROB  = 0.3

### Calculated Constants
# not used but keeping for possible future testing
# IMAGENET_MEAN = [0.485, 0.456, 0.406]
# IMAGENET_STD  = [0.229, 0.224, 0.225]

NIH_CXR8_CUSTOM_MEAN = [0.5249, 0.5249, 0.5249]
NIH_CXR8_CUSTOM_STD  = [0.2622, 0.2622, 0.2622]




# Removes global brightness variation
class PerImageStandardize(object):
    def __call__(self, x):
        mean = x.mean()
        std = x.std()
        return (x - mean) / (std + 1e-6)
    


### Transformer Creation ###
def make_value_tf(size):
    return A.Compose([
        A.Resize(size, size),
        A.CLAHE(
            clip_limit=CLAHE_CLIP_LIMIT,
            tile_grid_size=(CLAHE_TILE_GRID_SIZE, CLAHE_TILE_GRID_SIZE),
            p=CLAHE_PROB
        ),
        ToTensorV2(),
    ])

# Flip, rotate, elastic transform,
# grid distortion, color jitter, coarse dropout
def make_train_tf(size):
    return A.Compose([
        A.Resize(size, size),
        A.HorizontalFlip(p=HORIZONTAL_FLIP_PROB),
        A.Rotate(
            limit=ROTATION_DEGREES,
            p=ROTATION_PROB,
            interpolation=cv2.INTER_LINEAR),
        A.CLAHE(
            clip_limit=CLAHE_CLIP_LIMIT,
            tile_grid_size=(CLAHE_TILE_GRID_SIZE, CLAHE_TILE_GRID_SIZE),
            p=CLAHE_PROB
        ),
        A.ElasticTransform(
            alpha=ELASTIC_ALPHA,
            sigma=ELASTIC_SIGMA,
            interpolation=ELASTIC_INTERPOLATION,
            fill=ELASTIC_FILL,
            p=ELASTIC_PROB
        ),
        A.GridDistortion(num_steps=5, distort_limit=0.05, p=0.3),
        A.ColorJitter(
            brightness=JITTER_BRIGHTNESS,
            contrast=JITTER_CONTRAST,
            p=JITTER_PROB
        ),
        
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(4, 16),
            hole_width_range=(4, 16),
            p=0.2
        ),
        A.GaussNoise(std_range=(0.01, 0.02), p=0.3),
        A.RandomGamma(gamma_limit=(80, 120), p=0.4),
        ToTensorV2(),
    ])


### Dataset ###
class CXR8Dataset(Dataset):
    def __init__(self, df, labels, idx_array, transform, lookup,
                 verify_label_alignment=False):
        self.df        = df.iloc[idx_array].reset_index(drop=True)
        self.labels    = labels[idx_array]
        self.transform = transform
        self.lookup    = lookup

         # After building train_ds, verify alignment:
        img, lbl, view = self[0]
        expected_label = labels[idx_array[0]]
        assert np.array_equal(lbl.numpy(), expected_label), "Label mismatch!"


    def __len__(self):
        return len(self.df)
    def __getitem__(self, i):
        fname = self.df.loc[i, "Image Index"]
        path = self.lookup[fname]
        img = Image.open(path).convert('RGB')
        img = np.array(img)
        img = self.transform(image=img)["image"]
        img = img.float() / 255.0
        img = PerImageStandardize()(img)
        lbl = torch.tensor(self.labels[i], dtype=torch.float32)
        view_id = torch.tensor(self.df.loc[i, "view_id"], dtype=torch.long)
        return img, lbl, view_id


# Split at patient level so value set doesn't see patients from training set
def init_split(df, label_matrix):
    patient_ids = df["Patient ID"].unique()

    patient_label_matrix = np.zeros((len(patient_ids), len(ALL_CLASSES)), dtype=int)
    patient_id_to_idx = {pid: i for i, pid in enumerate(patient_ids)}
    for img_idx, row in df.iterrows():
        p = patient_id_to_idx[row["Patient ID"]]
        patient_label_matrix[p] |= label_matrix[img_idx]

    combo_strings = ["_".join(map(str, row)) for row in patient_label_matrix]
    from collections import Counter
    counts = Counter(combo_strings)
    MIN_COMBO_COUNT = 2
    strat_labels = [c if counts[c] >= MIN_COMBO_COUNT else "__other__" for c in combo_strings]

    # Three-way split: thresh 7%, then val 15% of remainder
    sss_thresh = StratifiedShuffleSplit(n_splits=1, test_size=0.07, random_state=99)
    remaining_idx, thresh_patient_idx = next(sss_thresh.split(patient_ids, strat_labels))

    remaining_patients = patient_ids[remaining_idx]
    remaining_strat    = [strat_labels[i] for i in remaining_idx]

    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_patient_idx, val_patient_idx = next(
        sss_val.split(remaining_patients, remaining_strat)
    )

    train_patients  = set(remaining_patients[train_patient_idx])
    value_patients  = set(remaining_patients[val_patient_idx])
    thresh_patients = set(patient_ids[thresh_patient_idx])

    train_idx  = df[df["Patient ID"].isin(train_patients)].index.to_numpy()
    value_idx  = df[df["Patient ID"].isin(value_patients)].index.to_numpy()
    thresh_idx = df[df["Patient ID"].isin(thresh_patients)].index.to_numpy()

    for split_name, idx in [("train", train_idx), ("val", value_idx), ("thresh", thresh_idx)]:
        n_hernia = label_matrix[idx, ALL_CLASSES.index("Hernia")].sum()
        print(f"{split_name} Hernia positives: {n_hernia}")

    return train_idx, value_idx, thresh_idx



def print_dataset_parameters():
    print("Dataset Parameters:")
    print("  CLAHE_CLIP_LIMIT", CLAHE_CLIP_LIMIT)
    print("  CLAHE_TILE_GRID_SIZE", CLAHE_TILE_GRID_SIZE)
    print("  CLAHE_PROB", CLAHE_PROB)
    print("  HORIZONTAL_FLIP_PROB", HORIZONTAL_FLIP_PROB)
    print("  ROTATION_DEGREES", ROTATION_DEGREES)
    print("  ROTATION_PROB", ROTATION_PROB)
    print("  JITTER_PROB", JITTER_PROB)
    print("  JITTER_BRIGHTNESS", JITTER_BRIGHTNESS)
    print("  JITTER_CONTRAST", JITTER_CONTRAST)
    print("  NIH_CXR8_CUSTOM_MEAN", NIH_CXR8_CUSTOM_MEAN)
    print("  NIH_CXR8_CUSTOM_STD", NIH_CXR8_CUSTOM_STD)


