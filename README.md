<invoke name="artifacts"><parameter name="command">create</parameter><parameter name="type">text/markdown</parameter><parameter name="id">fl_readme</parameter><parameter name="title">Federated Learning Repository README</parameter><parameter name="content"># Federated Learning with Flower Framework
A comprehensive implementation of various federated learning strategies using the Flower framework on CIFAR-10 dataset.
📋 Table of Contents

Overview
Repository Structure
Implemented Strategies
Installation
Quick Start
Configuration
Data Distributions
Running Experiments
Results and Metrics
Branch Organization
Troubleshooting

🔍 Overview
This repository contains implementations of multiple federated learning strategies with support for both IID (Independent and Identically Distributed) and non-IID data distributions. Each strategy is implemented in a separate branch for easy comparison and experimentation.
Key Features:

Multiple federated learning strategies (FedAvg, FedProx, FedAdagrad)
Support for IID and non-IID data distributions (Dirichlet, Pathological, Exponential)
Comprehensive metrics tracking (loss, accuracy, training time)
CSV logging for all experiments
Configurable hyperparameters via pyproject.toml

📁 Repository Structure
flower-rise/
├── flower_rise/
│   ├── __init__.py
│   ├── task.py                    # Data loading and model definition
│   ├── task_fedprox.py           # FedProx-specific task file
│   ├── task_fedadagrad.py        # FedAdagrad-specific task file
│   ├── client_app.py             # Client application
│   ├── server_app.py             # Server application
│   ├── custom_strategy.py        # Custom strategy implementations
│   └── ...
├── Fed_AVG_Records/              # FedAvg experiment results
├── Fed_Prox_Records/             # FedProx experiment results
├── Fed_Adagrad_Records/          # FedAdagrad experiment results
├── pyproject.toml                # Flower configuration
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🎯 Implemented Strategies

### 1. **FedAvg (Federated Averaging)**
The baseline federated learning algorithm that averages model weights from all participating clients.

**Branch:** `main` or `fedavg`

**Key Parameters:**
- `fraction-train`: Fraction of clients participating in each round
- `local-epochs`: Number of local training epochs per client
- `lr`: Learning rate

### 2. **FedProx (Federated Proximal)**
Adds a proximal term to the loss function to prevent client drift from the global model, especially useful for heterogeneous data.

**Branch:** `fedprox`

**Key Parameters:**
- `mu`: Proximal term coefficient (0.01 - 1.0)
  - Higher values = stronger regularization toward global model
  - Recommended: 0.01 (light), 0.1 (moderate), 1.0 (strong)

**Loss Function:**
```
L_FedProx = L(w) + (μ/2) * ||w - w_global||²
```

### 3. **FedAdagrad (Federated Adagrad)**
Uses adaptive learning rates by accumulating squared gradients on the server side, providing better convergence in non-IID settings.

**Branch:** `fedadagrad`

**Key Parameters:**
- `server-lr`: Server-side learning rate (η)
- `tau`: Initial accumulator value for numerical stability (1e-9)

**Update Rule:**
```
v_t = v_{t-1} + Δ²
w_new = w_old - η * Δ / sqrt(v_t)
