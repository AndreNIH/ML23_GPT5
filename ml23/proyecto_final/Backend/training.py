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
from sklearn.metrics import precision_score, recall_score, accuracy_score

def validation_step(val_loader, net, cost_function):
    val_loss = 0.0
    predictions = []
    true_labels = []

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

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(batch_labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}")

    return val_loss/len(val_loader)

def train():
    learning_rate = 1e-5
    n_epochs = 30
    batch_size = 128

    train_dataset, train_loader, val_dataset, val_loader = get_loader(batch_size=batch_size)

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
            modelo.save_model("modelo_val.pt")
            best_epoch_loss=val_loss
        if(train_loss<best_epoch_loss_train):
            modelo.save_model("modelo_ent.pt")
            best_epoch_loss=train_loss
        plotter.on_epoch_end(epoch, train_loss, val_loss)
    modelo.save_model("modelo_fin.pt")
    plotter.on_train_end()

if __name__=="__main__":
    train()