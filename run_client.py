import argparse
import flwr as fl
import torch
from torch.utils.data import DataLoader
from model import CustomFashionModel
from client import CustomClient
from data_utils import load_client_data

def main():
    parser = argparse.ArgumentParser(description="Run a federated learning client")
    parser.add_argument("--cid", type=int, required=True, help="Client ID")
    args = parser.parse_args()
    
    data_dir = "./client_data"
    batch_size = 32
    train_loader, val_loader = load_client_data(args.cid, data_dir, batch_size)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomFashionModel().to(device)
    client = CustomClient(model, train_loader, val_loader, device)
    
    fl.client.start_client(
        server_address="localhost:8080",
        client=client
    )

if __name__ == "__main__":
    main()