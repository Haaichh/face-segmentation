import math
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
    train = CelebAMask(mode='train')
    data_loader_train = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    test = CelebAMask(mode='test')
    data_loader_test = DataLoader(test, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # Instantiate the model and loss function
    model = UNet()
    model = model.to(device)
    loss_func = nn.CrossEntropyLoss()

    NUM_EPOCHS = 100
    LEARNING_RATE = 0.01
    optim = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

    # Initialise variables for computing and tracking stats
    iterations_per_epoch = math.ceil(len(train)/BATCH_SIZE)
    training_losses = []
    training_accuracies = []
    testing_losses = []
    testing_accuracies = []
    testing_f1_scores = []

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
            outputs = torch.softmax(outputs, 1)
            pred_y = torch.argmax(outputs, 1)
            correct += float((pred_y == labels).sum())
            total += float(labels.size(0) * labels.size(1) * labels.size(2))
            total_loss += float(loss*inputs.shape[0])

            transform = T.ToPILImage()
            pred_img = transform(pred_y[0] / 255.0)
            pred_img.show()

            true_img = transform(labels[0] / 255.0)
            true_img.show()

            #if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Iteration [{}/{}], Loss: {:.4f}'.format(epoch + 1, NUM_EPOCHS, i + 1, iterations_per_epoch, loss.item()))
        
        total_loss /= len(train)
        training_losses.append(total_loss)
        training_accuracies.append((correct/total) * 100)
        print('Train accuracy over epoch {}: {:.4f}'.format(epoch + 1, training_accuracies[-1]))

        # Evaluation on the test set
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        f1 = 0
        with torch.no_grad():
            for inputs, labels in data_loader_test:
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
                f1 += f1_score(labels[0].cpu().flatten(), pred_y[0].cpu().flatten(), average='macro')
        
        total_loss /= len(test)
        testing_losses.append(total_loss)
        testing_accuracies.append((correct/total) * 100)
        testing_f1_scores.append(f1/len(test))
        model.train()

        print('Test f1_score at epoch {}: {:.4f}'.format(epoch + 1, f1/len(test)))
        print('Test accuracy at epoch {}: {:.4f}\n'.format(epoch + 1, testing_accuracies[-1]))


    plt.title('Training Accuracy')
    plt.plot(range(len(training_accuracies)), training_accuracies, 'g')
    plt.xlabel('Epoch')
    plt.ylabel('Classification accuracy')

    plt.title('Training Loss')
    plt.plot(range(len(training_losses)), training_losses, 'r')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')


    plt.title('Testing Accuracy')
    plt.plot(range(len(testing_accuracies)), testing_accuracies, 'g')
    plt.xlabel('Epoch')
    plt.ylabel('Classification accuracy')

    plt.title('Testing Loss')
    plt.plot(range(len(testing_losses)), testing_losses, 'r')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
