import flwr as fl
from flwr.server.client_manager import ClientManager, ClientProxy
from flwr.server.strategy import Strategy
from flwr.common import Parameters, FitIns, EvaluateIns, Scalar, FitRes, EvaluateRes
from typing import List, Tuple, Dict, Optional
import threading
import numpy as np
import torch
from model import CustomFashionModel
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

class CustomClientManager(ClientManager):
    def __init__(self):
        self.clients: Dict[str, ClientProxy] = {}
        self.lock = threading.Lock()
    
    def num_available(self) -> int:
        with self.lock:
            return len(self.clients)
    
    def register(self, client: ClientProxy) -> bool:
        with self.lock:
            if client.cid in self.clients:
                return False
            self.clients[client.cid] = client
            return True
    
    def unregister(self, client: ClientProxy) -> None:
        with self.lock:
            self.clients.pop(client.cid, None)
    
    def all(self) -> Dict[str, ClientProxy]:
        with self.lock:
            return self.clients.copy()
    
    def wait_for(self, num_clients: int, timeout: int) -> bool:
        import time
        start_time = time.time()
        while time.time() - start_time < timeout:
            with self.lock:
                if len(self.clients) >= num_clients:
                    return True
            time.sleep(1)
        return False
    
    def sample(self, num_clients: int, min_num_clients: Optional[int] = None, 
               criterion: Optional[object] = None) -> List[ClientProxy]:
        min_num_clients = min_num_clients or num_clients
        if not self.wait_for(min_num_clients, timeout=60):
            return []
        with self.lock:
            import random
            clients = list(self.clients.values())
            return random.sample(clients, min(num_clients, len(clients)))

class FedAvgStrategy(Strategy):
    def __init__(self):
        self.model = CustomFashionModel()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        parameters = ndarrays_to_parameters(self.model.get_model_parameters())
        return parameters
    
    def configure_fit(self, server_round: int, parameters: Parameters, 
                     client_manager: ClientManager) -> List[Tuple[ClientProxy, FitIns]]:
        clients = client_manager.sample(num_clients=10, min_num_clients=10)
        fit_ins = FitIns(parameters, {})
        return [(client, fit_ins) for client in clients]
    
    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], 
                     failures: List[BaseException]) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}
        
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        aggregated_weights = self._aggregate(weights_results)
        parameters = ndarrays_to_parameters(aggregated_weights)
        
        accuracies = [res.metrics["accuracy"] * res.num_examples for _, res in results]
        total_examples = sum(res.num_examples for _, res in results)
        aggregated_accuracy = sum(accuracies) / total_examples if total_examples > 0 else 0
        
        return parameters, {"accuracy": aggregated_accuracy}
    
    def configure_evaluate(self, server_round: int, parameters: Parameters, 
                         client_manager: ClientManager) -> List[Tuple[ClientProxy, EvaluateIns]]:
        clients = client_manager.sample(num_clients=10, min_num_clients=10)
        evaluate_ins = EvaluateIns(parameters, {})
        return [(client, evaluate_ins) for client in clients]
    
    def aggregate_evaluate(self, server_round: int, results: List[Tuple[ClientProxy, EvaluateRes]], 
                          failures: List[BaseException]) -> Tuple[Optional[float], Dict[str, Scalar]]:
        if not results:
            return None, {}
        
        total_loss = sum(res.loss * res.num_examples for _, res in results)
        total_examples = sum(res.num_examples for _, res in results)
        aggregated_loss = total_loss / total_examples if total_examples > 0 else 0
        
        accuracies = [res.metrics["accuracy"] * res.num_examples for _, res in results]
        aggregated_accuracy = sum(accuracies) / total_examples if total_examples > 0 else 0
        
        return aggregated_loss, {"accuracy": aggregated_accuracy}
    
    def evaluate(self, server_round: int, parameters: Parameters) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        # No centralized evaluation required for TP1
        return None
    
    def _aggregate(self, weights_results: List[Tuple[List[np.ndarray], int]]) -> List[np.ndarray]:
        total_samples = sum(num_examples for _, num_examples in weights_results)
        aggregated_weights = [
            np.zeros_like(w) for w in weights_results[0][0]
        ]
        for weights, num_examples in weights_results:
            for layer_idx, weight in enumerate(weights):
                aggregated_weights[layer_idx] += weight * num_examples
        for layer_idx in range(len(aggregated_weights)):
            aggregated_weights[layer_idx] /= total_samples
        return aggregated_weights