# Rise FederatedLearning Assignment  
*This repository contains the files and branches related to Rise Federated learning Assignment on Task 1*

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

The goal is to analyse the results under different scenarios and hyperparameters

## ! Important  
As some of the strategies required changes to the files, the main branch is updated with FedProx. While it can still run other strategies, implemented and mentioned in the document, we encourage you
to follow the branches inorder to first see the implementation process from base, FedAvg, FedProx, Non-iid Implementation, and FedAdadagrad.
1-cent_eval_fl
2-ray_tune_base
3-Records_FedAvg
4-Records_FedProx
5-Records_FedAdagrad

## Repository Structure  
Here’s how the repository is laid out (modify based on your actual files):  
