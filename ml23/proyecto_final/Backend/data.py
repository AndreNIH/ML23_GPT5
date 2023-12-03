import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.transforms import transforms
import torchvision
import matplotlib.pyplot as plt
import os
import pandas as pd
from torchvision.io import read_image
from pathlib import Path
import glob

file_path = Path(__file__).parent.absolute()
dataset_path = file_path / 'data' / 'Dataset'

ESTADO_MAP = {
    0: "Closed",
    1: "Open"
}

global_transforms = [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Grayscale(),
        torchvision.transforms.Resize((48, 48))
    ]
    
mean, std = 0.5, 0.5

train_transforms = transforms.Compose([
    *global_transforms,
    #Data augmentation
    torchvision.transforms.Normalize((mean,), (std,))
])

test_transforms = transforms.Compose([
    *global_transforms,
    torchvision.transforms.Normalize((mean,), (std,))
])

class EyeDataset(Dataset):
    def __init__(self, root_dir, transform = None):
        self.root_dir = root_dir
        self.transform = transform
        self.dataset_files = glob.glob(os.path.join(self.root_dir, "*"))

    def __getitem__(self, idx):
        image = read_image(self.dataset_files[idx])
        label = self.dataset_files[idx].split('_')[4]
        label = int(label)

        image = transforms.ToPILImage()(image)
        if self.transform:
            image = self.transform(image)

        estado = ESTADO_MAP[label]

        return {"transformed": image,
                "label": label,
                "estado": estado}

    def __len__(self):
        return len(os.listdir(self.root_dir))

def get_loader(batch_size, shuffle=True, num_workers=0):

    train_dataset = EyeDataset(root_dir = dataset_path, transform=train_transforms)
    val_dataset = EyeDataset(root_dir = dataset_path, transform=test_transforms)

    total_len = len(train_dataset)
    
    indices = torch.randperm(len(train_dataset)) # Sortea los indices
    val_size = total_len//3 # Un tercio del total aplicado piso
    train_dataset = torch.utils.data.Subset(train_dataset, indices[:-val_size]) # Selecciona todos los indices menos los ultimos val_size indices
    val_dataset = torch.utils.data.Subset(val_dataset, indices[-val_size:]) # Selecciona solo los ultimos val_size indices

    train_dataloader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
    )
    val_dataloader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
    )

    return train_dataset, train_dataloader, val_dataset, val_dataloader


