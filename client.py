import flwr as fl
from flwr.common import (
    GetPropertiesIns, GetPropertiesRes, GetParametersIns, GetParametersRes,
    Parameters, FitIns, FitRes, EvaluateIns, EvaluateRes, Code, Status,
    ndarrays_to_parameters, parameters_to_ndarrays
)
import torch
from torch.utils.data import DataLoader
from typing import Dict, Optional, Union
from model import CustomFashionModel

class CustomClient(fl.client.Client):
    def __init__(self, model: torch.nn.Module, train_loader: DataLoader, 
                 test_loader: DataLoader, device: torch.device) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
    
    def get_properties(self, ins: GetPropertiesIns) -> GetPropertiesRes:
        return GetPropertiesRes(
            status=Status(code=Code.OK, message="Success"),
            properties={}
        )
    
    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        parameters = ndarrays_to_parameters(self.model.get_model_parameters())
        return GetParametersRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=parameters
        )
    
    def fit(self, ins: FitIns) -> FitRes:
        parameters = parameters_to_ndarrays(ins.parameters)
        self.model.set_model_parameters(parameters)
        
        loss, accuracy = self.model.train_one_epoch(
            self.train_loader, self.criterion, self.optimizer, self.device
        )
        
        updated_parameters = ndarrays_to_parameters(self.model.get_model_parameters())
        
        return FitRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=updated_parameters,
            num_examples=len(self.train_loader.dataset),
            metrics={"accuracy": accuracy}
        )
    
    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        parameters = parameters_to_ndarrays(ins.parameters)
        self.model.set_model_parameters(parameters)
        
        loss, accuracy = self.model.test_one_epoch(
            self.test_loader, self.criterion, self.device
        )
        
        return EvaluateRes(
            status=Status(code=Code.OK, message="Success"),
            loss=loss,
            num_examples=len(self.test_loader.dataset),
            metrics={"accuracy": accuracy}
        )
    
    def to_client(self) -> 'CustomClient':
        return self