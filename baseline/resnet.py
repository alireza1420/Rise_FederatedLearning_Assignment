import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import csv
import torch.backends.cudnn as cudnn


num_epochs = 20


def imshow(img):
    img = img / 2 + 0.5     
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion      
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


if __name__ == '__main__':
    # Check for GPU availability
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_on_gpu = torch.cuda.is_available()
    print(f'Using device: {device}')
    print(f'Train on GPU: {train_on_gpu}')

    # Create model
    ResNet18 = ResNet(BasicBlock, [2, 2, 2, 2])
    print(ResNet18)

    if train_on_gpu:
        ResNet18 = torch.nn.DataParallel(ResNet18)
        cudnn.benchmark = True
    
    ResNet18.to(device)

    train_losses = []
    test_accuracies = []

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    batch_size = 16

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ResNet18.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
    
    # Initialize minimum validation loss
    valid_loss_min = float('inf')

    # Start keeping record
    with open(f"res_net_metrics_{num_epochs}_epochs.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train_Loss", "Test_Loss", "Test_Accuracy"])

    # Training loop
    for epoch in range(1, num_epochs + 1):
        train_loss = 0.0
        valid_loss = 0.0
        
        # train
        ResNet18.train()
        for batch_idx, (data, target) in enumerate(trainloader):
            # Move data to GPU
            if train_on_gpu:
                data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            

            outputs = ResNet18(data)
            loss = criterion(outputs, target)
      
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * data.size(0)

        # Eval
        ResNet18.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(testloader):
                # Move data to GPU
                if train_on_gpu:
                    data, target = data.to(device), target.to(device)
                
          
                output = ResNet18(data)
                
            #calculate loss
                loss = criterion(output, target)
                valid_loss += loss.item() * data.size(0)
                
                # Calculate accuracy
                _, predicted = torch.max(output, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        # Calculate average losses
        train_loss = train_loss / len(trainloader.dataset)
        valid_loss = valid_loss / len(testloader.dataset)
        test_accuracy = 100 * correct / total
        
        # Print statistics
        print(f'Epoch: {epoch} \tTraining Loss: {train_loss:.6f} \tTest Loss: {valid_loss:.6f} \tTest Accuracy: {test_accuracy:.2f}%')
        
        # Save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print(f'Test loss decreased ({valid_loss_min:.6f} --> {valid_loss:.6f}). Saving model...')
            torch.save(ResNet18.state_dict(), 'ResNet18.pt')
            valid_loss_min = valid_loss

        # csv save
        with open(f"res_net_metrics_{num_epochs}_epochs.csv", "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, valid_loss, test_accuracy])
        
        train_losses.append(train_loss)
        test_accuracies.append(test_accuracy)

    print('Training complete!')
    print(f'Best test loss: {valid_loss_min:.6f}')