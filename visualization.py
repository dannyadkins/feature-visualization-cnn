import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from cnn import CNN
from torch.utils.data import dataloader
from torchvision import datasets, transforms
import argparse

def visualize_for_neuron(model: CNN, layer_name: str, neuron_idx: int, train_loader, save: bool = False, num_iterations: int = 5000, lr = 0.01, regularization_weight = 0.05):
    # Load the pre-trained model
    model.eval()

    # Get one image from train loader
    img, label = next(iter(train_loader))
    img.requires_grad = True

    # Define optimizer
    optimizer = torch.optim.Adam([img], lr=lr)

    for i in range(num_iterations):
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass of the model
        output = model(img)

        # Get the activation of the specified layer for the specified neuron
        activation = output[0, neuron_idx]

        # Compute the loss as the negative activation
        loss = -activation

        # Compute the total variation regularization term
        tv_loss = regularization_weight * (
            torch.sum(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])) +
            torch.sum(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]))
        )

        # Add the regularization term to the loss
        loss += tv_loss

        # Backward pass
        loss.backward()

        # Update the image
        optimizer.step()
        
        if (i % 200 == 0):
            print(f'Iteration: {i}, Loss: {loss.item()}')
    # Convert the optimized image to a numpy array
    img_pil = transforms.ToPILImage()(img.squeeze())

    # Visualize the optimized image
    plt.imshow(img_pil)
    plt.axis('off')
    plt.show()

    # Save the optimized image if desired
    if save:
        img_pil.save(f'./outputs/{neuron_idx}.jpg')

if __name__ == '__main__':
    # Load the pre-trained model
    # argparse neuron id or range of neuron ids:
    
    argparser = argparse.ArgumentParser(description='Visualize a specific neuron optimized input')

    model = CNN()
    model.load_state_dict(torch.load('mnist_cnn.pt'))

    train_loader = dataloader.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,)) # mean, std
                        ])),
        batch_size=1, shuffle=True)

    # Visualize the first neuron in the first convolutional layer
    for i in range(10):
        visualize_for_neuron(model, 'conv1', 1, train_loader, save=True)