import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from typing import List

class CustomFashionModel(nn.Module):
    def __init__(self) -> None:
        super(CustomFashionModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def train_one_epoch(self, train_loader: DataLoader, criterion: nn.Module, 
                       optimizer: torch.optim.Optimizer, device: torch.device) -> tuple[float, float]:
        self.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = self(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * data.size(0)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy
    
    def test_one_epoch(self, test_loader: DataLoader, criterion: nn.Module, 
                      device: torch.device) -> tuple[float, float]:
        self.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = self(data)
                loss = criterion(output, target)
                
                total_loss += loss.item() * data.size(0)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy
    
    def get_model_parameters(self) -> List[np.ndarray]:
        return [param.cpu().detach().numpy() for param in self.parameters()]
    
    def set_model_parameters(self, parameters: List[np.ndarray]) -> None:
        for param, new_param in zip(self.parameters(), parameters):
            param.data = torch.from_numpy(new_param).to(param.device)