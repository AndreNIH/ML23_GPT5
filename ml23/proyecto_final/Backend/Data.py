import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.transforms import transforms
import torchvision
import matplotlib.pyplot as plt
import os
import pandas as pd
from torchvision.io import read_image

dataset_path = 'ml23/proyecto_final/Backend/data/Dataset'

global_transforms = [
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Grayscale(),
    torchvision.transforms.Resize((48, 48))
]

train_transforms = transforms.Compose([
    *global_transforms,
    torchvision.transforms.Normalize((mean,), (std,))
])

