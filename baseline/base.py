import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import csv
import time

num_epochs=10

def imshow(img):
    img = img / 2 + 0.5     
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class Net(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1=nn.Conv2d(3,16,5)
        self.pool=nn.MaxPool2d(2,2)
        self.conv2=nn.Conv2d(16,64,5)
        self.fc1=nn.Linear(64*5*5,256)
        self.fc2=nn.Linear(256,128)
        self.fc3=nn.Linear(128,10)

    def forward(self,x):
        x=self.pool(F.relu(self.conv1(x)))
        x=self.pool(F.relu(self.conv2(x))) 
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    train_losses = []
    test_accuracies = []

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 32

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
    
    net= Net()
    net = net.to(device)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.00655453, momentum=0.838864)
    #start keeping record
    with open(f"./Base_Records/metrics_{num_epochs}_epochs.csv", "a+", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Loss", "Accuracy","Record_time_eval_and_train"])

    for epoch in range(num_epochs):
        total_start_time = time.time()

        # Training phase
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = inputs.to(device)  
            labels = labels.to(device)  

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            if i % 2000 == 1999:    # print every 2000 mini batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
        
        # Calculate average loss for the epoch
        avg_epoch_loss = running_loss / (len(trainloader) % 2000)  
        train_losses.append(avg_epoch_loss)
        
        # Evaluation phase - AFTER the epoch
        correct = 0
        total = 0


        net.eval() 

        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)

                outputs = net(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        total_end_time = time.time()
        total_duration = total_end_time - total_start_time
        print(f"Total wall-clock time: {total_duration:.2f} seconds")


        epoch_accuracy = 100 * correct / total
        test_accuracies.append(epoch_accuracy)
        net.train()


    

        with open(f"./Base_Records/metrics_{num_epochs}_epochs.csv", "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_epoch_loss, epoch_accuracy,total_duration])

    print("Training done here!")

    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)

    dataiter = iter(testloader)
    images, labels = next(dataiter)

    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

    net = Net()
    net.load_state_dict(torch.load(PATH, weights_only=True))
    net = net.to(device)
    print("Err in 101")
    images = images.to(device)
    outputs = net(images)

    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                                  for j in range(4)))

    correct = 0
    total = 0

    #accuracy on test data
    with torch.no_grad():
        for data in testloader:
            images, labels = data

            images=images.to(device)
            labels=labels.to(device)

            # calculate outputs by running images through the network
            outputs = net(images)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the {len(testloader)} test images: {100 * correct / total} %')

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

#accuracy of each class
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images=images.to(device)
            labels=labels.to(device)
            
            outputs = net(images)
          

            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1


    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

    plt.figure(figsize=(12, 5))

    # Plot 1: Training Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss', color='blue', marker='o')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot 2: Test Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies, label='Test Accuracy', color='orange', marker='o')
    plt.title('Test Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    