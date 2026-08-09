import torch
import pandas as pd
import torchvision.transforms as T
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from typing import Tuple, Callable, Optional, Dict



class GalaxyZooDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_dir: Path, transform=None):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        galaxy_id = self.df.index[idx]
        image_path = self.image_dir / f"{galaxy_id}.jpg"
        
        # Load the image
        image = Image.open(image_path).convert('RGB')
        
        # Apply transformations if any
        if self.transform:
            image = self.transform(image)
        
        row = self.df.iloc[idx]
        raw_labels = row.values        
        labels_tensor = torch.tensor(raw_labels, dtype=torch.float32)

        return image, labels_tensor

def get_transforms(crop_size: int = 256, resize_size: int = 224) -> Dict[str, T.Compose]:
    """
    Constructs the image transformation pipelines for training and evaluation.
    
    Returns:
        A dictionary with two keys: 'train' and 'val'. Each contains a 
        torchvision.transforms.Compose object with the respective pipeline.
    """
    
    # 1. Define the training pipeline (Includes Random Augmentations)

    train_transform = T.Compose([
        T.CenterCrop(crop_size),

        T.Resize((resize_size, resize_size)),

        T.RandomHorizontalFlip(p=0.5),

        T.RandomVerticalFlip(p=0.5),

        T.RandomRotation(degrees=180),

        T.ToTensor()
    ])
    
    # 2. Define the validation pipeline (Strictly deterministic, NO augmentations)

    val_transform = T.Compose([
        T.CenterCrop(crop_size),
        T.Resize((resize_size, resize_size)),
        T.ToTensor()
    ])
    
    return {
        "train": train_transform,
        "val": val_transform
    }
