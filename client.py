import flwr as fl
from flwr.common import (
    GetPropertiesIns, GetPropertiesRes, GetParametersIns, GetParametersRes,
    Parameters, FitIns, FitRes, EvaluateIns, EvaluateRes, Code, Status,
    ndarrays_to_parameters, parameters_to_ndarrays
)
import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, Optional, Union, List, Tuple
from config import FEDPROX_MU, LEARNING_RATE
from model import CustomFashionModel

class CustomClient(fl.client.Client):
    def __init__(self, model: CustomFashionModel, train_loader: DataLoader, 
                 test_loader: DataLoader, device: torch.device) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=LEARNING_RATE)
        self.global_params = None  # For FedProx
        self.mu = FEDPROX_MU  # Proximal term coefficient
        self.local_control = None  # For SCAFFOLD
        self.global_control = None  # For SCAFFOLD
        self.initial_params = None  # For SCAFFOLD
    
    def fit(self, ins: FitIns) -> FitRes:
        parameters = parameters_to_ndarrays(ins.parameters)
        self.model.set_model_parameters(parameters)
        
        # Store initial parameters for SCAFFOLD
        self.initial_params = [torch.from_numpy(p).to(self.device) for p in parameters]
        
        # For FedProx
        self.global_params = [torch.from_numpy(p).to(self.device) for p in parameters]
        
        # For SCAFFOLD
        if "global_control_variate" in ins.config:
            self.global_control = [torch.from_numpy(arr).to(self.device) 
                                 for arr in ins.config["global_control_variate"]]
            
            # Initialize local control if not exists
            if self.local_control is None:
                self.local_control = [torch.zeros_like(p).to(self.device) 
                                    for p in self.model.parameters()]
        
        # Train based on algorithm
        if self.global_control is not None:  # SCAFFOLD
            loss, accuracy = self._train_one_epoch_scaffold()
        else:  # FedProx or FedAvg
            loss, accuracy = self._train_one_epoch_fedprox()
        
        updated_parameters = ndarrays_to_parameters(self.model.get_model_parameters())
        
        # Prepare metrics and control variates for SCAFFOLD
        metrics = {"accuracy": accuracy}
        if self.global_control is not None:
            # Update local control variate for SCAFFOLD
            self._update_local_control_variate()
            metrics["control_variates"] = [cv.cpu().numpy() for cv in self.local_control]
        
        return FitRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=updated_parameters,
            num_examples=len(self.train_loader.dataset),
            metrics=metrics
        )
    
    def _train_one_epoch_fedprox(self) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for data, target in self.train_loader:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            
            # Standard cross-entropy loss
            loss = self.criterion(output, target)
            
            # Add proximal term for FedProx
            if self.global_params is not None and self.mu > 0:
                proximal_term = 0.0
                for local_param, global_param in zip(self.model.parameters(), self.global_params):
                    proximal_term += (local_param - global_param).norm(2)
                loss += (self.mu / 2) * proximal_term
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * data.size(0)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy
    
    def _train_one_epoch_scaffold(self) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for data, target in self.train_loader:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            
            # Apply SCAFFOLD correction to gradients
            with torch.no_grad():
                for param, local_c, global_c in zip(self.model.parameters(), 
                                                  self.local_control, 
                                                  self.global_control):
                    if param.grad is not None:
                        param.grad += global_c - local_c
            
            self.optimizer.step()
            
            total_loss += loss.item() * data.size(0)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        avg_loss = total_loss / total
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy
    
    def _update_local_control_variate(self) -> None:
        """Update local control variate according to SCAFFOLD formula."""
        if self.initial_params is None or self.global_control is None:
            return
            
        current_params = list(self.model.parameters())
        num_batches = len(self.train_loader)
        
        with torch.no_grad():
            for local_c, global_c, init_p, curr_p in zip(
                self.local_control, self.global_control, 
                self.initial_params, current_params
            ):
                # c_k = c_k - c + (w_t - w_{t+1})/(eta * T)
                local_c.data = (local_c - global_c + 
                               (init_p - curr_p) / (LEARNING_RATE * num_batches))
    
    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        parameters = parameters_to_ndarrays(ins.parameters)
        self.model.set_model_parameters(parameters)
        
        loss, accuracy = self.model.test_one_epoch(self.test_loader, self.criterion, self.device)
        
        return EvaluateRes(
            status=Status(code=Code.OK, message="Success"),
            loss=float(loss),
            num_examples=len(self.test_loader.dataset),
            metrics={"accuracy": accuracy}
        )