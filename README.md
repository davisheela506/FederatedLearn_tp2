# Federated Learning TP2: Analyzing Data Heterogeneity and Client Drift

## Project Description
This project implements and compares three federated learning algorithms under varying data heterogeneity conditions, using FashionMNIST dataset:

1. **FedAvg** (Baseline)
2. **FedProx** (with proximal regularization)
3. **SCAFFOLD** (with control variates)

## Key Objectives
- Analyze impact of data heterogeneity (using Dirichlet α=10,1,0.1)
- Implement client drift mitigation techniques
- Compare algorithm performance metrics

## Results Summary

### Accuracy Comparison
| Algorithm  | α=10 (IID) | α=1 (Moderate) | α=0.1 (High) |
|------------|------------|----------------|--------------|
| FedAvg     | 13.6%      | 13.2%          | 11.4%        |
| FedProx    | 13.8%      | 13.5%          | 11.8%        |
| SCAFFOLD   | 13.4%      | 13.0%          | 11.2%        |

### Key Findings
- FedProx showed most consistent improvements (+0.4% at α=0.1)
- SCAFFOLD converged faster but had lower final accuracy
- All algorithms struggled with extreme heterogeneity (α=0.1)


## How to Run
```bash
# Install dependencies
pip install torch torchvision flower numpy matplotlib

# Run simulations
python src/run_simulation.py --algorithm [fedavg|fedprox|scaffold] --alpha [10|1|0.1]

# Generate plots
python src/analyze_results.py --alpha [10|1|0.1]
