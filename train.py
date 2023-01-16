import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_loader import CelebAMask
from model import UNet

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train = CelebAMask(mode='train')
    BATCH_SIZE = 10
    data_loader_train = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # Instantiate the model and loss function
    model = UNet()
    model = model.to(device)
    loss_func = nn.CrossEntropyLoss()

    NUM_EPOCHS = 1
    LEARNING_RATE = 0.01
    MOMENTUM = 0.8
    optim = torch.optim.SGD(model.parameters(), lr = LEARNING_RATE, weight_decay = 0.05, momentum = MOMENTUM)

    # Initialise variables for computing and tracking stats
    iterations_per_epoch = math.ceil(len(train)/BATCH_SIZE)

    for epoch in range(NUM_EPOCHS):

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

            if (i+1) % 25 == 0:
                print('Epoch [{}/{}], Iteration [{}/{}], Loss: {:.4f}'.format(epoch + 1, NUM_EPOCHS, i + 1, iterations_per_epoch, loss.item()))