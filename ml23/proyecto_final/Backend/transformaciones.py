import numpy as np
import os
import json
import pathlib
import cv2
import torch
import torch.nn as nn
import torchvision

file_path = pathlib.Path(__file__).parent.absolute()

def to_torch(array: np.ndarray, roll_dims=True):
    '''
    Convert tensor to numpy array
    args:
        - array (np.ndarray): array to convert
            size: (H, W, C)
    returns:
        - array (np.ndarray): converted tensor
            size: (C, H, W)
    '''
    if roll_dims:
        if len(array.shape) <= 2:
            array = np.expand_dims(array, axis=2) # (H, W) -> (H, W, 1)
        array = array.transpose((2, 0, 1)) # (H, W, C) -> (C, H, W)
    tensor = torch.tensor(array)
    return tensor

def to_numpy(tensor: torch.tensor, roll_dims = True):
    '''
    Convert tensor to numpy array
    args:
        - tensor (torch.tensor): tensor to convert
            size: (C, H, W)
    returns:
        - array (np.ndarray): converted array
            size: (H, W, C)
    '''
    if roll_dims:
        if len(tensor.shape) > 3:
            tensor = tensor.squeeze(0) # (1, C, H, W) -> (C, H, W)
        tensor = tensor.permute(1, 2, 0) # (C, H, W) -> (H, W, C)
    array = tensor.detach().cpu().numpy()
    return array

def add_img_text(img: np.ndarray, text_label: str):
    '''
    Add text to image
    args:
        - img (np.ndarray): image to add text to
            - size: (C, H, W)
        - text (str): text to add to image
    '''
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    fontColor = (255, 0, 0)
    thickness = 2

    # For the text background
    # Finds space required by the text so that we can put a background with that amount of width.
    (text_w, text_h), _ = cv2.getTextSize(text_label, font, fontScale, thickness)

    # Center text
    x1, y1 = 0, text_h  # Top left corner
    img = cv2.rectangle(img,
                        (x1, y1 - 20),
                        (x1 + text_w, y1),
                        (255, 255, 255),
                        -1)
    if img.shape[-1] == 1 or len(img.shape) == 2: # Grayscale image
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = cv2.putText(img, text_label,
                      (x1, y1),
                      font, fontScale, fontColor, thickness)
    return img
