import json
import matplotlib.pyplot as plt
import os
from pathlib import Path
from prettytable import PrettyTable

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
        
        # Plot loss
        rounds, losses = zip(*self.results["losses_distributed"])
        plt.figure()
        plt.plot(rounds, losses, label="Loss")
        plt.xlabel("Round")
        plt.ylabel("Loss")
        plt.title("Distributed Loss Over Rounds")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "loss_plot.png"))
        plt.close()
        
        # Plot accuracy
        if self.results.get("metrics_distributed") and "accuracy" in self.results["metrics_distributed"]:
            rounds, accuracies = zip(*self.results["metrics_distributed"]["accuracy"])
            plt.figure()
            plt.plot(rounds, accuracies, label="Accuracy", color="orange")
            plt.xlabel("Round")
            plt.ylabel("Accuracy")
            plt.title("Distributed Accuracy Over Rounds")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, "accuracy_plot.png"))
            plt.close()
        else:
            print("No accuracy data to plot.")

def main():
    import sys
    file_name = sys.argv[1] if len(sys.argv) > 1 else "fl_results.json"
    visualizer = ResultsVisualizer()
    visualizer.load_simulation_results(file_name)
    if visualizer.results:
        visualizer.print_results_table()
        visualizer.plot_results()
    else:
        print("Analysis skipped due to missing or invalid results.")

if __name__ == "__main__":
    main()
