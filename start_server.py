import flwr as fl
import json
from server import CustomClientManager, FedAvgStrategy

def main():
    client_manager = CustomClientManager()
    strategy = FedAvgStrategy()
    
    history = fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=30),
        strategy=strategy,
        client_manager=client_manager
    )
    
    results = {
        "losses_distributed": history.losses_distributed,
        "metrics_distributed_fit": history.metrics_distributed_fit,
        "metrics_distributed": history.metrics_distributed
    }
    with open("fl_results.json", "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    main()