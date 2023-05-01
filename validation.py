import math
import os
import matplotlib as plt
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from data_loader import CelebAMask
from model import UNet

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    print('Device name:', device)
    
    BATCH_SIZE = 32
    val= CelebAMask(mode='val')
    data_loader_val = DataLoader(val, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # Load the model
    model = UNet()
    model.load_state_dict(torch.load('checkpoint.pt'))
    model = model.to(device)
    model.eval()
    loss_func = nn.CrossEntropyLoss()

    # Initialise variables for computing and tracking stats
    correct= 0
    total = 0
    total_loss = 0
    f1 = 0

    with torch.no_grad():
        for inputs, labels in data_loader_val:
            inputs = inputs.to(device)
            labels = labels.to(device)

            labels[:, 0, :, :] = labels[:, 0, :, :] * 255.0
            labels = labels[:, 0, :, :]

            outputs = model(inputs)
            loss = loss_func(outputs, labels.long())

            pred_y = torch.argmax(outputs, 1)
            correct += float((pred_y == labels).sum())
            total += float(labels.size(0) * labels.size(1) * labels.size(2))
            total_loss += float(loss*inputs.shape[0])
            temp = f1_score(labels[0].cpu().flatten(), pred_y[0].cpu().flatten(), labels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], average=None, zero_division=1)
            f1 += temp

    print('Validation f1_score: {:.4f}'.format(f1/len(data_loader_val)))
    print('Validation accuracy: {:.4f}'.format((correct/total) * 100))
    print('Validation loss: {:.4f}'.format(total_loss/len(val)))