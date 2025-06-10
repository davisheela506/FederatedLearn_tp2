import flwr as fl
import json
from server import CustomClientManager, FedAvgStrategy, FedProxStrategy, SCAFFOLDStrategy
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", type=str, default="fedavg",
                       choices=["fedavg", "fedprox", "scaffold"],
                       help="FL algorithm to use")
    parser.add_argument("--mu", type=float, default=0.1,
                       help="Proximal term coefficient for FedProx")
    args = parser.parse_args()
    
    client_manager = CustomClientManager()
    
    # Select strategy based on algorithm
    if args.algorithm == "fedavg":
        strategy = FedAvgStrategy()
    elif args.algorithm == "fedprox":
        strategy = FedProxStrategy(mu=args.mu)
    elif args.algorithm == "scaffold":
        strategy = SCAFFOLDStrategy()
    
    history = fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=50),
        strategy=strategy,
        client_manager=client_manager
    )
    
    results = {
        "losses_distributed": history.losses_distributed,
        "metrics_distributed_fit": history.metrics_distributed_fit,
        "metrics_distributed": history.metrics_distributed
    }
    with open(f"results_{args.algorithm}.json", "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    main()