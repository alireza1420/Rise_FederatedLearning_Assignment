# Rise FederatedLearning Assignment  
*A simple Federated Learning setup and a centralized CNN model using CIFAR-10*

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Motivation](#motivation)  
3. [Repository Structure](#repository-structure)  
4. [Requirements & Setup](#requirements--setup)  
5. [Usage](#usage)  
6. [How It Works](#how-it-works)  
   - Centralized CNN baseline  
   - Federated Learning setup (with Flower)  
7. [Experiments & Results](#experiments--results)  
8. [How to Extend / Close-to-Production Considerations](#how-to-extend--close-to-production-considerations)  
9. [Limitations](#limitations)  
10. [Contributing](#contributing)  
11. [License](#license)  

## Project Overview  
This project implements two main variants of model training on the CIFAR-10 dataset:  
- A standard **centralized** convolutional neural network (CNN) model trained in the typical way.  
- A **federated learning** setup in which multiple client nodes train on local splits and aggregate into a global model using the Flower framework.

The goal is to compare performance, communication overhead, resource usage (CPU/GPU) — and in your extended work, maybe implement algorithms like FedProx or FedCS (as you’re already aiming for) and monitor CPU/GPU usage across clients.

## Motivation  
Why this project? A few reasons:  
- Federated learning is increasingly relevant for privacy-preserving distributed training (e.g., mobile devices, edge).  
- Implementing a baseline (centralized) gives a meaningful comparison point.  
- Your interest in resource monitoring (CPU/GPU) means this project can demonstrate more than just accuracy: also scalability, heterogeneity, client speed differences, algorithmic robustness.  
- It’s a good assignment to build foundations and experiment while you later implement FedProx/FedCS etc.

## Repository Structure  
Here’s how the repository is laid out (modify based on your actual files):  
