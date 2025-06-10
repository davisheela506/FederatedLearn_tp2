import json
import matplotlib.pyplot as plt
import os
from pathlib import Path
from prettytable import PrettyTable
import glob

class ResultsVisualizer:
    def __init__(self):
        self.results = None
    
    def load_simulation_results(self, file_path: str) -> None:
        try:
            with open(file_path, 'r') as f:
                self.results = json.load(f)
        except FileNotFoundError:
            print(f"Error: Results file '{file_path}' not found. Please ensure the simulation ran successfully.")
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in '{file_path}'.")
    
    def print_results_table(self) -> None:
        if not self.results or not self.results.get("losses_distributed"):
            print("No results to display.")
            return
        
        table = PrettyTable()
        table.field_names = ["Round", "Loss", "Accuracy"]
        
        for round_num, loss in self.results["losses_distributed"]:
            accuracy = next((acc for r, acc in self.results["metrics_distributed"].get("accuracy", []) if r == round_num), 0.0)
            table.add_row([round_num, f"{loss:.4f}", f"{accuracy:.4f}"])
        
        print(table)
    
    def plot_results(self, output_dir: str = "./figures") -> None:
        if not self.results or not self.results.get("losses_distributed"):
            print("No results to plot.")
            return
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Extract filename for title
        file_name = os.path.basename(self.file_path) if hasattr(self, 'file_path') else "results"
        algorithm = file_name.split('_')[1]
        alpha = file_name.split('alpha')[-1].split('.')[0]
        
        # Plot loss
        rounds, losses = zip(*self.results["losses_distributed"])
        plt.figure()
        plt.plot(rounds, losses, label="Loss")
        plt.xlabel("Round")
        plt.ylabel("Loss")
        plt.title(f"{algorithm.upper()} Loss Over Rounds (α={alpha})")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"{algorithm}_loss_alpha{alpha}.png"))
        plt.close()
        
        # Plot accuracy
        if self.results.get("metrics_distributed") and "accuracy" in self.results["metrics_distributed"]:
            rounds, accuracies = zip(*self.results["metrics_distributed"]["accuracy"])
            plt.figure()
            plt.plot(rounds, accuracies, label="Accuracy", color="orange")
            plt.xlabel("Round")
            plt.ylabel("Accuracy")
            plt.title(f"{algorithm.upper()} Accuracy Over Rounds (α={alpha})")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, f"{algorithm}_accuracy_alpha{alpha}.png"))
            plt.close()
        else:
            print("No accuracy data to plot.")

def compare_algorithms(alpha: float = 0.1):
    """Compare different algorithms for a given alpha value."""
    algorithms = ["fedavg", "fedprox", "scaffold"]
    results = {}
    
    # Load all results for the given alpha
    for algo in algorithms:
        file_name = f"results_{algo}_alpha{alpha}.json"
        if os.path.exists(file_name):
            with open(file_name, 'r') as f:
                results[algo] = json.load(f)
    
    if not results:
        print(f"No results found for alpha={alpha}")
        return
    
    # Plot comparison of losses
    plt.figure()
    for algo, data in results.items():
        if "losses_distributed" in data:
            rounds, losses = zip(*data["losses_distributed"])
            plt.plot(rounds, losses, label=algo.upper())
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.title(f"Loss Comparison (α={alpha})")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"./figures/compare_loss_alpha{alpha}.png")
    plt.close()
    
    # Plot comparison of accuracies
    plt.figure()
    for algo, data in results.items():
        if "metrics_distributed" in data and "accuracy" in data["metrics_distributed"]:
            rounds, accuracies = zip(*data["metrics_distributed"]["accuracy"])
            plt.plot(rounds, accuracies, label=algo.upper())
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy Comparison (α={alpha})")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"./figures/compare_accuracy_alpha{alpha}.png")
    plt.close()

def main():
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        alpha = sys.argv[2] if len(sys.argv) > 2 else "0.1"
        compare_algorithms(float(alpha))
    else:
        file_name = sys.argv[1] if len(sys.argv) > 1 else "results_fedavg_alpha0.1.json"
        visualizer = ResultsVisualizer()
        visualizer.file_path = file_name
        visualizer.load_simulation_results(file_name)
        if visualizer.results:
            visualizer.print_results_table()
            visualizer.plot_results()
        else:
            print("Analysis skipped due to missing or invalid results.")

if __name__ == "__main__":
    main()