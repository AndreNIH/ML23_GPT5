import sys
import os
import torch
import pathlib

import subprocess
package = "PyQt5"
subprocess.check_call([sys.executable, "-m", "pip", "install", package])

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import Qt

full_dir = os.getcwd()
Backend_path = os.path.join(full_dir, 'ml23/proyecto_final/Backend')
sys.path.insert(0, Backend_path)
from inferencia import load_img
from network import Network

file_path = pathlib.Path(__file__).parent.absolute()

def get_eye_state(path):
    modelo = Network(48)
    modelo.load_model("modelo_val_1.pt")
    im_file = (file_path / path).as_posix()
    original, transformed, denormalized = load_img(im_file)

    if len(transformed.shape) ==3:
        transformed = transformed.unsqueeze(1)

    proba = modelo.predict(transformed)
    print(proba)
    pred = torch.where(proba >= 0.5, torch.tensor(1), torch.tensor(0))
    pred = pred.item()
    return int(pred)

def on_button_clicked(label):
    open_path = os.path.join(full_dir, 'ml23/proyecto_final/Frontend/Assets/Open_Door')
    open_pixmap = QPixmap(open_path)
    open_pixmap = open_pixmap.scaled(360, 640, Qt.KeepAspectRatio)
    if get_eye_state("./Camera/sampleClosed.png") == 1:
        label.setPixmap(open_pixmap)
    else:
        QMessageBox.information(None, "Message", "Mire el anuncio para desbloquear DOOR")

def window():
    app = QApplication(sys.argv)
    win = QMainWindow()
    win.setGeometry(1200, 300, 700, 780)
    win.setWindowTitle("DOOR")
    gpt_path = os.path.join(full_dir, 'ml23/proyecto_final/Frontend/Assets/GPT-5_Logo.png')
    win.setWindowIcon(QIcon(gpt_path))
    win.setToolTip("DOOR v2")

    label = QLabel(win)
    closed_path = os.path.join(full_dir, 'ml23/proyecto_final/Frontend/Assets/Closed_Door')
    closed_pixmap = QPixmap(closed_path)
    closed_pixmap = closed_pixmap.scaled(360, 640, Qt.KeepAspectRatio)
    label.setPixmap(closed_pixmap)
    label.setScaledContents(True)
    label.setAlignment(Qt.AlignCenter)

    button = QPushButton('Abrir', win)
    button.clicked.connect(lambda: on_button_clicked(label))

    label.setGeometry(170, 30, 360, 640)
    button.setGeometry(250, 680, 200, 50)

    win.show()
    sys.exit(app.exec_())

window()