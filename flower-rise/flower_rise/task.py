"""app-pytorch: A Flower / PyTorch app."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
import time
import copy


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16,64, 5)
        self.fc1 = nn.Linear(64 * 5 * 5, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

#multiple train - iid, non-iid
fds_cache = {}

pytorch_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch


def load_data_iid(partition_id: int, num_partitions: int):
    cache_key = "iid"

    if cache_key not in fds_cache:
        partitioner= IidPartitioner(num_partitions=num_partitions)
        fds_cache[cache_key]=FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},

        )

    fds = fds_cache[cache_key]

    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    # Construct dataloaders
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=32)
    return trainloader, testloader

def load_data_dirichlet(partition_id: int, num_partitions: int, alpha: float = 0.5):

    cache_key = f"dirichlet_{alpha}"
    
    if cache_key not in fds_cache:
        partitioner = DirichletPartitioner(
            num_partitions=num_partitions,
            partition_by="label",
            alpha=alpha,
            min_partition_size=10,
            self_balancing=True,
        )
        fds_cache[cache_key] = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )
    
    fds = fds_cache[cache_key]
    partition = fds.load_partition(partition_id)
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    
    trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=32)
    
    label_counts = analyze_class_distribution(trainloader)
    print(f"[Dirichlet α={alpha}] Client {partition_id}: {len(trainloader.dataset)} train, {len(testloader.dataset)} test")
    print(f"  Class distribution: {dict(sorted(label_counts.items()))}")
    
    return trainloader, testloader

def analyze_class_distribution(dataloader):
    """counts how items are distributed."""
    label_counts = {}
    for batch in dataloader:
        labels = batch["label"]
        for label in labels:
            label_item = label.item()
            label_counts[label_item] = label_counts.get(label_item, 0) + 1
    return label_counts

def load_centralized_dataset():
    """Load test set and return dataloader."""
    # Load entire test set
    test_dataset = load_dataset("uoft-cs/cifar10", split="test")
    dataset = test_dataset.with_format("torch").with_transform(apply_transforms)
    return DataLoader(dataset, batch_size=32)


def train(net, trainloader, epochs, lr,momentum, device):
    """Train the model on the training set."""
    train_start_time = time.time()
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr,momentum=momentum)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    avg_trainloss = running_loss / len(trainloader)
    train_end_time = time.time()
    total_duration = train_end_time - train_start_time
    print(f"Total wall-clock time: {total_duration:.2f} seconds")

    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set.
    used on centralized acc """
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy