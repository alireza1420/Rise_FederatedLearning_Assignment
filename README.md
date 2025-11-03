# Rise FederatedLearning Assignment  
*This repository contains the files and branches related to Rise Federated learning Assignment on Task 1*


## Project Overview  
This project implements two main variants of model training on the CIFAR-10 dataset:  
- A standard **centralized** convolutional neural network (CNN) model trained in the typical way.  
- A **federated learning** setup in which multiple client nodes train on local splits and aggregate into a global model using the Flower framework.

The goal is to analyse the results under different scenarios and hyperparameters

## ⚠️ Important Notes on Branches and Strategies

The **main branch** currently includes the **FedProx** strategy as part of the latest updates.  
While the main branch remains backward-compatible with other implemented strategies, we strongly recommend exploring the repository branches in the following order to understand the progressive development process — from the baseline implementation to advanced strategies:

| Branch | Description |
|:-------|:-------------|
| `1-cent_eval_fl` | Centralized evaluation and baseline setup. |
| `2-ray_tune_base` | Hyperparameter tuning using Ray Tune on the centralized model. |
| `3-Records_FedAvg` | Implementation and recorded results for the **FedAvg** strategy. |
| `4-Records_FedProx` | Implementation and recorded results for the **FedProx** strategy (main branch currently reflects this). |
| `5-Records_FedAdagrad` | Implementation and logs for the **FedAdagrad** optimization-based strategy. |

Additionally, a **Non-IID data partitioning** implementation is available in the relevant branch, allowing the evaluation of performance under heterogeneous data conditions.

**Recommendation:**  
Start by checking out the earliest branches (e.g.,`0-initial_setup` `1-cent_eval_fl`, `3-Records_FedAvg`) to understand the incremental changes in both the model logic and configuration. The main branch should be used when testing or benchmarking **FedProx**, or when comparing across all strategies.


## Repository Structure  
Here’s how the repository is laid out (modify based on your actual files):  
