import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from ray import tune, train
from ray.tune.schedulers import ASHAScheduler
import os


class Net(nn.Module):
    def __init__(self, conv1_size=6, conv2_size=16, fc1_size=120, fc2_size=84):
        super().__init__()
        self.conv1 = nn.Conv2d(3, conv1_size, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(conv1_size, conv2_size, 5)
        self.fc1 = nn.Linear(conv2_size * 5 * 5, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_cifar(config):
    try:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Create network with hyperparameters
        net = Net(
            conv1_size=config["conv1_size"],
            conv2_size=config["conv2_size"],
            fc1_size=config["fc1_size"],
            fc2_size=config["fc2_size"]
        )
        net = net.to(device)
        
        # Data loading with num_workers=0 for Windows compatibility
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform
        )
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=int(config["batch_size"]),
            shuffle=True, num_workers=0, pin_memory=False
        )
        
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=int(config["batch_size"]),
            shuffle=False, num_workers=0, pin_memory=False
        )
        
        # Optimizer with hyperparameters
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            net.parameters(),
            lr=config["lr"],
            momentum=config["momentum"]
        )
        
        # Training loop
        for epoch in range(10):  # Fixed to 10 epochs
            running_loss = 0.0
            net.train()
            
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            # Validation
            net.eval()
            correct = 0
            total = 0
            val_loss = 0.0
            
            with torch.no_grad():
                for data in testloader:
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)
                    
                    outputs = net(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            accuracy = correct / total
            avg_val_loss = val_loss / len(testloader)
            
            # Report metrics to Ray Tune using train.report (new API)
            train.report({"loss": avg_val_loss, "accuracy": accuracy})
    except Exception as e:
        print(f"Error in trial: {e}")
        raise


if __name__ == '__main__':
    # Define search space
    config = {
        "conv1_size": tune.choice([6, 12, 16]),
        "conv2_size": tune.choice([16, 32, 64]),
        "fc1_size": tune.choice([84, 120, 256]),
        "fc2_size": tune.choice([64, 84, 128]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "momentum": tune.uniform(0.8, 0.99),
        "batch_size": tune.choice([16, 32, 64, 128])
    }
    
    # Configure ASHA scheduler for early stopping
    scheduler = ASHAScheduler(
        metric="accuracy",
        mode="max",
        max_t=10,
        grace_period=3,
        reduction_factor=2
    )
    
    # Custom trial name creator to shorten paths
    def trial_name_creator(trial):
        return f"trial_{trial.trial_id}"
    
    # Run hyperparameter search
    result = tune.run(
        train_cifar,
        name="cifar_tune",  # Shorter experiment name
        storage_path="file:///content/ray_results",
        resources_per_trial={"cpu": 2, "gpu": 0.25 if torch.cuda.is_available() else 0},
        config=config,
        num_samples=20,  # Number of different hyperparameter combinations to try
        scheduler=scheduler,
        trial_dirname_creator=trial_name_creator,  # Use short trial names
        max_concurrent_trials=4,  # Limit concurrent trials
        raise_on_failed_trial=False,  # Continue even if some trials fail
        progress_reporter=tune.CLIReporter(
            metric_columns=["loss", "accuracy", "training_iteration"]
        ),
        verbose=1
    )
    
    # Get best trial
    best_trial = result.get_best_trial("accuracy", "max", "last")
    print(f"\nBest trial config: {best_trial.config}")
    print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']:.4f}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']:.4f}")
    
    # Train final model with best hyperparameters
    print("\n" + "="*50)
    print("Training final model with best hyperparameters...")
    print("="*50)
    
    best_config = best_trial.config
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    best_net = Net(
        conv1_size=best_config["conv1_size"],
        conv2_size=best_config["conv2_size"],
        fc1_size=best_config["fc1_size"],
        fc2_size=best_config["fc2_size"]
    )
    best_net = best_net.to(device)
    
    # Save the best model
    PATH = './cifar_net_best.pth'
    torch.save(best_net.state_dict(), PATH)
    print(f"Best model saved to {PATH}")