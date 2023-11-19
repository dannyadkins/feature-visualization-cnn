import argparse
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
    
def main(train_model: bool, test_model: bool, load_model: bool, save_model):
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
    if (load_model):
        model.load_state_dict(torch.load("mnist_cnn.pt"))

    if (train_model):
        train(model, train_loader, num_epochs=hyperparams['epochs'], lr=hyperparams['lr'], hyperparams=hyperparams)
    
    if (test_model):
        test(model, test_loader)
    
    if (save_model):
        torch.save(model.state_dict(), "mnist_cnn.pt")

if __name__ == "__main__":

    # collect args via argparse: load_model, train, test -L -T -t

    argparser = argparse.ArgumentParser(description='Train CNN on MNIST dataset')
    argparser.add_argument('-L', '--load_model', help='load model from file', action='store_true')
    argparser.add_argument('-T', '--train', help='train model', action='store_true')
    argparser.add_argument('-t', '--test', help='test model', action='store_true')
    argparser.add_argument('-s', '--save', help='save model', action='store_true')

    args = argparser.parse_args()

    train_model = args.train
    test_model = args.test
    load_model = args.load_model
    save_model = args.save

    main(train_model, test_model, load_model, save_model)
