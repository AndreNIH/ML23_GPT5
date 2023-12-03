import pathlib
import glob
import torch
import os
import math
from typing import Any, Callable, Optional

from torch.utils.data import Dataset
from torchvision.io import read_image

'''
class EyeDataset(Dataset):
    def __init__(self,  img_dir):
        super().__init__()
        self.img_dir = img_dir
        self.dataset_files = glob.glob(os.path.join(img_dir, "*"))
                
    def __len__(self):
        return len(self.dataset_files)
    
    def __geitem__(self, idx):
        label = self.dataset_files[idx].split('_')[4]
        img = read_image(self.dataset_files[idx])
        return img, label
    

def get_loader(split, batch_size, shuffle=True, num_workers=0):
    dataset = 
'''

file_path = pathlib.Path(__file__).parent.absolute()

class EyeDataset(Dataset):
    def __init__(self, 
                 root : str,
                 split : str,
                 target_transform: Optional[Callable] = None):
        self.img_size = 48
        self.split = split
        self.root  = root
        
        _read_data()


    def _read_data(self):
        base_folder = pathlib.Path(self.root) / "data"
        files = glob.glob(base_folder, "*")
        trainingSetSize =  math.ceil(len(files) * 0.7)
        validationSetSize = len(files) - trainingSetSize
        targets = targets = files[0: trainingSetSize] if self.split == "train" or "val" else files[:validationSetSize]