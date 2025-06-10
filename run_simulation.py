import subprocess
import time
from data_utils import generate_distributed_data
import argparse
import os
import time

def run_simulation(alpha: float, algorithm: str = "fedavg", mu: float = 0.1):
    # Hyperparameters
    NUM_CLIENTS = 10
    BATCH_SIZE = 64
    NUM_ROUNDS = 50
    SEED = 42
    LEARNING_RATE = 0.01
    
    # Generate distributed data
    print(f"Generating distributed datasets with alpha={alpha}...")
    generate_distributed_data(NUM_CLIENTS, alpha, "./client_data")
    
    # Start server with appropriate algorithm
    print(f"Starting server with {algorithm.upper()}...")
    server_cmd = ["python", "start_server.py", "--algorithm", algorithm]
    if algorithm == "fedprox":
        server_cmd.extend(["--mu", str(mu)])
    server_process = subprocess.Popen(server_cmd)
    
    # Wait for server to start
    time.sleep(5)
    
    # Start clients
    print("Starting clients...")
    client_processes = []
    for cid in range(NUM_CLIENTS):
        process = subprocess.Popen(["python", "run_client.py", "--cid", str(cid)])

    # Wait for server to finish
    server_process.wait()
    
    # Terminate clients
    for process in client_processes:
        process.terminate()
    
    # Rename results file to include alpha value
    original_file = f"results_{algorithm}.json"
    new_file = f"results_{algorithm}_alpha{alpha}.json"
    if os.path.exists(original_file):
        os.rename(original_file, new_file)
    
    # Analyze results
    print("Analyzing results...")
    subprocess.run(["python", "analyze_results.py", new_file])

def main():
    parser = argparse.ArgumentParser(description="Run federated learning simulations")
    parser.add_argument("--algorithm", type=str, default="fedavg", 
                       choices=["fedavg", "fedprox", "scaffold"],
                       help="FL algorithm to use")
    parser.add_argument("--alpha", type=float, default=1.0,
                       help="Dirichlet distribution parameter for data heterogeneity")
    parser.add_argument("--mu", type=float, default=0.1,
                       help="Proximal term coefficient for FedProx")
    args = parser.parse_args()
    
    run_simulation(args.alpha, args.algorithm, args.mu)

if __name__ == "__main__":
    main()