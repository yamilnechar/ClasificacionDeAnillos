import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import lightning.pytorch as pl
from astropy.io import fits
from sklearn.model_selection import train_test_split
from torchvision import transforms


def get_augmentation_transforms(num_channels=3):
    """
    Get standard data augmentation transforms for image classification.
    
    Args:
        num_channels: Number of image channels (1 for grayscale, 3 for RGB)
    
    Returns:
        Augmentation transform composition
    """
    return transforms.Compose([
        # Geometric invariances (galaxies are approximately rotation/flip invariant)
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(
            degrees=180,
            interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),
            scale=(0.9, 1.1),
            interpolation=transforms.InterpolationMode.BILINEAR
        ),
        # Mild photometric jitter to improve robustness to exposure / background variation
        transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.05
        )
        # Optional small erasing to simulate artifacts (low prob and scale)
        # transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
    ])

class FitsDataset(Dataset):
    """
    Dataset for FITS galaxy images with multilabel ring classification.
    
    Labels are converted from ring_class (0-3) to multilabel format:
    - 0 → [0, 0]: No rings
    - 1 → [1, 0]: Inner ring only
    - 2 → [0, 1]: Outer ring only
    - 3 → [1, 1]: Both rings (inner + outer)
    """
    
    # Multilabel mapping: ring_class → [inner_ring, outer_ring]
    RING_CLASS_TO_MULTILABEL = {
        0: [0, 0],  # No rings
        1: [1, 0],  # Inner ring only
        2: [0, 1],  # Outer ring only
        3: [1, 1]   # Both rings
    }
    
    def __init__(self, file_paths, labels, transform=None, augmentation_transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        self.augmentation_transform = augmentation_transform

    def _preprocess_fits_bands(self, data):
        # Handle NaNs and apply arcsinh scaling
        data = np.nan_to_num(data)
        # Using a standard scaling factor; adjust based on your data intensity
        data = np.arcsinh(data) 
        
        d_min, d_max = data.min(), data.max()
        if d_max > d_min:
            data = (data - d_min) / (d_max - d_min)
        return data.astype(np.float32)
    # def _preprocess_fits_bands(self, data):
    #     """Preprocess individual bands before Lupton RGB conversion."""
    #     # Handle NaNs
    #     data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    #     # Clip negative values
    #     data = np.clip(data, 0, None)
    #     return data

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        
        with fits.open(file_path) as hdul:
            data = hdul[0].data

        data = np.asarray(data, dtype=np.float32)
        image_tensor = torch.from_numpy(data)

        # Apply preprocessing transforms if provided (e.g. full astrophysical pipeline
        # defined in the notebook: asinh/sky subtraction/unsharp mask/Lupton RGB/etc.).
        if self.transform is not None:
            image_tensor = self.transform(image_tensor)

        # Apply augmentation transforms if provided; otherwise, optionally fall back
        # to a lightweight geometric + normalization pipeline when no preprocessing
        # pipeline has been supplied.
        if self.augmentation_transform is not None:
            image_tensor = self.augmentation_transform(image_tensor)
        elif self.transform is None:
            default_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
            image_tensor = default_transform(image_tensor)
        
        # Convert ring_class (0-3) to binary multilabel format [inner_ring, outer_ring]
        ring_class = self.labels[idx]
        binary_label = self.RING_CLASS_TO_MULTILABEL[ring_class]
        label = torch.tensor(binary_label, dtype=torch.float32)
        
        return {
            'image': image_tensor,
            'ring_class': label  # Shape: [2] → [inner_ring, outer_ring]
        }

class ZoobotFitsDataModule(pl.LightningDataModule):
    def __init__(self, csv_path, img_dir=None, batch_size=32, num_workers=4, transform_pipeline = None, use_augmentation=False, sampler_exponent=1.0):
        """
        Args:
            csv_path (str): Path to the catalog CSV.
            img_dir (str, optional): Root directory if file_loc is relative.
            batch_size (int): Batch size for dataloaders.
            num_workers (int): Number of worker processes.
            transform_pipeline (torchvision.transforms, optional): Preprocessing transforms to apply to all datasets.
            use_augmentation (bool): Whether to apply data augmentation to training set.
            sampler_exponent (float): Exponent for inverse-frequency sampler weights.
                1.0 = pure inverse-frequency (v31), 0.75 = moderate, 0.5 = sqrt.
        """
        super().__init__()
        self.csv_path = csv_path
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_augmentation = use_augmentation
        self.sampler_exponent = sampler_exponent

        self.transform = transform_pipeline if transform_pipeline else transforms.Compose([transforms.Resize((224, 224), antialias=True), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        self.augmentation_transform = get_augmentation_transforms(num_channels=3) if use_augmentation else None

    def setup(self, stage=None):
        """
        Setup datasets with class balancing.
        
        Multilabel Mapping:
        - ring_class 0 → [inner_ring=0, outer_ring=0] → No rings
        - ring_class 1 → [inner_ring=1, outer_ring=0] → Inner ring only
        - ring_class 2 → [inner_ring=0, outer_ring=1] → Outer ring only
        - ring_class 3 → [inner_ring=1, outer_ring=1] → Both rings
        """
        # Multilabel class mapping
        RING_CLASS_TO_MULTILABEL = {
            0: [0, 0],  # No rings
            1: [1, 0],  # Inner ring only
            2: [0, 1],  # Outer ring only
            3: [1, 1]   # Both rings
        }
        
        df = pd.read_csv(self.csv_path)
        
        paths = df['file_loc'].tolist()
        labels = df['ring_class'].values

        # Split: Train 80%, Val 10%, Test 10%
        # Stratify by ring_class to maintain class distribution across splits
        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            paths, labels, test_size=0.2, random_state=42, stratify=labels
        )

        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
        )

        # Compute per-class counts on the training split for downstream
        # balancing strategies (e.g. WeightedRandomSampler, loss pos_weights).
        unique_classes, counts = np.unique(train_labels, return_counts=True)
        class_counts = dict(zip(unique_classes.tolist(), counts.tolist()))
        print("Training class distribution (ring_class → count):", class_counts)

        # Create datasets for ALL stages
        # FitsDataset will convert ring_class to multilabel format [inner_ring, outer_ring]
        self.train_ds = FitsDataset(
            train_paths,
            train_labels,
            self.transform,
            augmentation_transform=self.augmentation_transform,
        )
        self.val_ds = FitsDataset(val_paths, val_labels, self.transform)
        self.test_ds = FitsDataset(test_paths, test_labels, self.transform)
        self.predict_ds = FitsDataset(test_paths, test_labels, self.transform)

        # Store train labels for pos_weight computation (ring_class 0-3)
        self._train_labels = train_labels

    def get_pos_weight(self):
        """
        Compute per-label positive weights for BCEWithLogitsLoss.
        Inner ring positive = ring_class in {1, 3}; outer ring positive = ring_class in {2, 3}.
        Returns list [w_inner, w_outer] with w_c = (N - N_pos_c) / max(N_pos_c, 1).
        """
        labels = np.asarray(self._train_labels)
        n = len(labels)
        n_inner = np.sum((labels == 1) | (labels == 3))
        n_outer = np.sum((labels == 2) | (labels == 3))
        w_inner = (n - n_inner) / max(n_inner, 1)
        w_outer = (n - n_outer) / max(n_outer, 1)
        return [float(w_inner), float(w_outer)]

    def train_dataloader(self):
        # Use a WeightedRandomSampler to mitigate class imbalance without
        # constructing an explicitly oversampled training set.
        train_labels = np.array(self.train_ds.labels)
        unique_classes, counts = np.unique(train_labels, return_counts=True)
        class_counts = dict(zip(unique_classes.tolist(), counts.tolist()))

        # Inverse-frequency weighting per class
        class_weights = {
            cls: (1/count)**self.sampler_exponent for cls, count in class_counts.items() if count > 0
        }
        sample_weights = np.array([class_weights[int(lbl)] for lbl in train_labels], dtype=np.float32)

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
    
    def predict_dataloader(self):
        return DataLoader(
            self.predict_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )