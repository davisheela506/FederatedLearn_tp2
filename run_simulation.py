import subprocess
import time
from data_utils import generate_distributed_data

def main():
    # Hyperparameters
    NUM_CLIENTS = 10
    ALPHA = 1.0
    BATCH_SIZE = 32
    NUM_ROUNDS = 30
    
    # Generate distributed data
    print("Generating distributed datasets...")
    generate_distributed_data(NUM_CLIENTS, ALPHA, "./client_data")
    
    # Start server
    print("Starting server...")
    server_process = subprocess.Popen(["python", "start_server.py"])
    
    # Wait for server to start
    time.sleep(5)
    
    # Start clients
    print("Starting clients...")
    client_processes = []
    for cid in range(NUM_CLIENTS):
        process = subprocess.Popen(["python", "run_client.py", "--cid", str(cid)])
        client_processes.append(process)
    
    # Wait for server to finish
    server_process.wait()
    
    # Terminate clients
    for process in client_processes:
        process.terminate()
    
    # Analyze results
    print("Analyzing results...")
    subprocess.run(["python", "analyze_results.py"])

if __name__ == "__main__":
    main()