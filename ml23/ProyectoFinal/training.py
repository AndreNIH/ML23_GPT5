import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import torch.optim as optim
from tqdm import tqdm
from network import Network
import datetime as dt
import pathlib
import matplotlib
import matplotlib.pyplot as plt
from dataset import get_loader


file_path = Path.cwd()

class PlotLosses():
    def __init__(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, train_loss, val_loss):        
        self.x.append(self.i)
        self.losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.i += 1
        plt.cla()
        plt.plot(self.x, self.losses, label="Costo de entrenamiento promedio")
        plt.plot(self.x, self.val_losses, label="Costo de validación promedio")
        plt.xlabel('epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show(block=False)
        plt.pause(5)

    def on_train_end(self):
        plt.show()
        today = dt.datetime.now().strftime("%Y-%m-%d")
        losses_file = file_path/ f'figures/losses_{today}.png'
        plt.savefig(losses_file)

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
            outputs,proba = net(batch_imgs)
            loss = cost_function(outputs, batch_labels)
            val_loss += loss.item()

            predictions = torch.argmax(proba,dim=1)
            total += batch_labels.size(0)
            cor = (predictions == batch_labels).sum().item()
            correct += cor

    accuracy = 100 * float(correct) / total
    print(f"Accuracy: {accuracy}")
    return val_loss/len(val_loader)

def train():
    learning_rate = 1e-5
    n_epochs= 50
    batch_size = 32

    train_dataset, train_loader, _, _ = \
        get_loader(batch_size=batch_size,
                    shuffle=True)
    _, _, val_dataset, val_loader = \
        get_loader(batch_size=batch_size,
                    shuffle=False)

    print(f"Cargando datasets --> entrenamiento: {len(train_dataset)}, validacion: {len(val_dataset)}")

    plotter = PlotLosses()
    modelo = Network(input_dim = 48,
                    n_classes = 7)

    cost_function = nn.CrossEntropyLoss()

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
            preds,_ = modelo(batch_imgs)
            loss = cost_function(preds, batch_labels)
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


if __name__ == '__main__':
    train()