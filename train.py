import math
import matplotlib as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_loader import CelebAMask
from model import UNet

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train = CelebAMask(mode='train')
    BATCH_SIZE = 100
    data_loader_train = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # Instantiate the model and loss function
    model = UNet()
    model = model.to(device)
    loss_func = nn.CrossEntropyLoss()

    NUM_EPOCHS = 10
    LEARNING_RATE = 0.1
    MOMENTUM = 0.9
    optim = torch.optim.SGD(model.parameters(), lr = LEARNING_RATE, weight_decay = 0.05, momentum = MOMENTUM)

    # Initialise variables for computing and tracking stats
    iterations_per_epoch = math.ceil(len(train)/BATCH_SIZE)
    training_losses = []
    training_accuracies = []

    for epoch in range(NUM_EPOCHS):

        total_loss = 0
        correct = 0
        total = 0

        for i, (inputs, labels) in enumerate(data_loader_train):
            optim.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)

            labels[:, 0, :, :] = labels[:, 0, :, :] * 255.0
            labels = labels[:, 0, :, :]

            loss = loss_func(outputs, labels.long())
            loss.backward()
            optim.step()

            # Accuracy
            pred_y = torch.argmax(outputs, 1)
            correct += (pred_y == labels).sum()
            total += float(labels.size(0))
            total_loss += loss*inputs.shape[0]

            if (i+1) % 25 == 0:
                print('Epoch [{}/{}], Iteration [{}/{}], Loss: {:.4f}'.format(epoch + 1, NUM_EPOCHS, i + 1, iterations_per_epoch, loss.item()))
        
        total_loss /= len(train)
        training_losses.append(total_loss.item())
        training_accuracies.append(correct/total)
        print('Train accuracy over epoch {}: {:.4f}'.format(epoch+1,training_accuracies[-1]))

    plt.title('Training Accuracy')
    plt.plot(range(len(training_accuracies)), training_accuracies, 'g')
    plt.xlabel('Epoch')
    plt.ylabel('Classification accuracy')

    plt.title('Training Loss')
    plt.plot(range(len(training_losses)), training_losses, 'r')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')