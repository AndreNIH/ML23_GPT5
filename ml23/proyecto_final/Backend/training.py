from data import EyeDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import torch.optim as optim
import torch
import torch.nn as nn
from tqdm import tqdm
from data import get_loader
from network import Network
from plot_losses import PlotLosses

def validation_step(val_loader, net, cost_function):
    val_loss = 0.0
    correct = 0
    total = 0
    for i, batch in enumerate(val_loader, 0):
        batch_imgs = batch['transformed']
        batch_labels = batch['label']
        if torch.cuda.is_available():
            batch_labels = batch_labels.cuda()
        with torch.inference_mode():
            proba = net(batch_imgs)
            proba = proba.squeeze()

            loss = cost_function(proba.to(torch.float32), batch_labels.to(torch.float32))
            val_loss += loss.item()

            preds = torch.where(proba >= 0.5, torch.tensor(1), torch.tensor(0))

            total += batch_labels.size(0)
            cor = (preds == batch_labels).sum().item()
            correct += cor

    accuracy = 100 * float(correct) / total
    print(f"Accuracy: {accuracy}")
    return val_loss/len(val_loader)

def train():
    learning_rate = 1e-5
    n_epochs= 5
    batch_size = 256

    train_dataset, train_loader, _, _ = \
        get_loader(batch_size=batch_size,
                    shuffle=True)
    _, _, val_dataset, val_loader = \
        get_loader(batch_size=batch_size,
                    shuffle=False)

    print(f"Cargando datasets --> entrenamiento: {len(train_dataset)}, validacion: {len(val_dataset)}")

    plotter = PlotLosses()
    modelo = Network(input_dim = 48)

    cost_function = nn.BCELoss()

    optimizer = optim.Adam(modelo.parameters(), learning_rate, weight_decay=0.01)
    best_epoch_loss = np.inf
    best_epoch_loss_train = np.inf
    for epoch in range(n_epochs):
        train_loss = 0
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch: {epoch}")):
            batch_imgs = batch['transformed']
            batch_labels = batch['label']
            if torch.cuda.is_available():
                batch_labels = batch_labels.cuda()
            optimizer.zero_grad()
            preds = modelo(batch_imgs)
            preds = preds.squeeze()
            loss = cost_function(preds.to(torch.float32), batch_labels.to(torch.float32))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss = train_loss/len(train_loader)
        val_loss = validation_step(val_loader, modelo, cost_function)
        tqdm.write(f"Epoch: {epoch}, train_loss: {train_loss:.2f}, val_loss: {val_loss:.2f}")

        if(val_loss<best_epoch_loss):
            modelo.save_model("modelo_val_1.pt")
            best_epoch_loss=val_loss
        if(train_loss<best_epoch_loss_train):
            modelo.save_model("modelo_ent_1.pt")
            best_epoch_loss=train_loss
        plotter.on_epoch_end(epoch, train_loss, val_loss)
    modelo.save_model("modelo_fin_1.pt")
    plotter.on_train_end()

if __name__=="__main__":
    train()