import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pathlib


class Network(nn.Module):
    def __init__(self, input_dim: int, n_classes: int) -> None:
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        out_dim =  self.calc_out_dim(input_dim,5, padding=2)

        self.conv1 = nn.Conv2d(1,16,kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(16,32,kernel_size=5)
        self.lineal1 = nn.Linear(32*self.calc_out_dim(out_dim,5)*self.calc_out_dim(out_dim,5),1024)
        self.lineal2 = nn.Linear(1024,n_classes)
        
        self.to(self.device)
 
    def calc_out_dim(self, in_dim, kernel_size, stride=1, padding=0):
        out_dim = math.floor((in_dim - kernel_size + 2*padding)/stride) + 1
        return out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.cuda.is_available():
            x = x.cuda()
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = torch.flatten(x, start_dim=1)
        x = self.lineal1(x)
        x = F.relu(x)
        logits = self.lineal2(x)
        proba = F.softmax(logits, dim=1)

        return logits, proba

    def predict(self, x):
        with torch.inference_mode():
            return self.forward(x)

    def save_model(self, model_name: str):
        models_path = file_path / 'models' / model_name
        torch.save(self.state_dict(),models_path)

    def load_model(self, model_name: str):
        self.load_state_dict(torch.load(file_path / 'models' / model_name))
