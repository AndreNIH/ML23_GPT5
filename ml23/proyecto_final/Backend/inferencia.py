import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from network import Network
import torch
from transformaciones import *
from data import ESTADO_MAP, test_transforms, visualizar_transforms
import pathlib

file_path = pathlib.Path(__file__).parent.absolute()

def load_img(path):
    assert os.path.isfile(path), f"El archivo {path} no existe"
    img = cv2.imread(path)
    tensor_img = test_transforms(img)
    denormalized = visualizar_transforms(img)
    return img, tensor_img, denormalized

def predict(img_title_paths):
    modelo = Network(48)
    modelo.load_model("modelo_val_10.pt")
    for path in img_title_paths:
        im_file = (file_path / path).as_posix()
        original, transformed, denormalized = load_img(im_file)

        if len(transformed.shape) ==3:
            transformed = transformed.unsqueeze(1)
        proba = modelo.predict(transformed)
        print(proba)
        pred = torch.where(proba >= 0.5, torch.tensor(1), torch.tensor(0))
        pred = pred.item()
        pred = int(pred)
        pred_label = ESTADO_MAP[pred]
        print(ESTADO_MAP[pred])
        print(path)


        h, w = original.shape[:2]
        resize_value = 300
        img = cv2.resize(original, (w * resize_value // h, resize_value))
        img = add_img_text(img, f"Pred: {pred_label}")

        denormalized = to_numpy(denormalized)
        denormalized = cv2.resize(denormalized, (resize_value, resize_value))
        cv2.imshow("Predicción - original", img)
        cv2.imshow("Predicción - transformed", denormalized)
        cv2.waitKey(0)

if __name__=="__main__":
    img_paths = [
        "./test_imgs/User1.jpg",
        "./test_imgs/User2.jpg",
        "./test_imgs/User3.jpg",
        "./test_imgs/User4.jpg",
        "./test_imgs/User5.jpg",
        "./test_imgs/User6.jpg",
        "./test_imgs/User7.jpg",
        "./test_imgs/User8.jpg",
        "./test_imgs/User9.jpg"
    ]
    predict(img_paths)