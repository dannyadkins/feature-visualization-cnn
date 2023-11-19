import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import dataloader

from cnn import CNN

def train(model: CNN, train_loader: dataloader, num_epochs=5, lr=0.001, hyperparams=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % hyperparams['log_interval'] == hyperparams['log_interval']-1:
                print('[%d, %5d] loss: %.3f' %
                    (epoch+1, i+1, running_loss / hyperparams['log_interval']))
                running_loss = 0.0
    print('Finished Training')

def test(model: CNN, test_loader: dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
    
def main():
    # small toy model 
    hyperparams = {
        'batch_size': 64,
        'test_batch_size': 1000,
        'epochs': 1,
        'lr': 0.01,
        'momentum': 0.5,
        'no_cuda': True,
        'seed': 1,
        'log_interval': 100,
        'save_model': True
    }

    # load data
    train_loader = dataloader.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,)) # mean, std
                        ])),
        batch_size=hyperparams['batch_size'], shuffle=True)

    test_loader = dataloader.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
        batch_size=hyperparams['test_batch_size'], shuffle=True)

    model = CNN()
    train(model, train_loader, num_epochs=hyperparams['epochs'], lr=hyperparams['lr'], hyperparams=hyperparams)
    test(model, test_loader)

if __name__ == "__main__":
    main()
