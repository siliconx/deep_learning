import sys
import pdb
import time
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch import optim
from torchvision import models
from torchvision import transforms
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

is_cuda = False
device = None
if torch.cuda.is_available():
    is_cuda = True
    count = torch.cuda.device_count()
    device = torch.device(count - 1)  # last gpu

# show image
def imshow(inp):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.show()

# Load data into tensor
simple_transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
train = ImageFolder('./dogsandcats/train/', simple_transform)
valid = ImageFolder('./dogsandcats/valid/', simple_transform)
print('train.class_to_idx:', train.class_to_idx)
print('train.classes:', train.classes)

# imshow(train[0][0])

# Create data generators
train_data_gen = DataLoader(train, shuffle=True, batch_size=64, num_workers=3)
valid_data_gen = DataLoader(valid, shuffle=True, batch_size=64, num_workers=3)
dataset_sizes = {
        'train': len(train_data_gen.dataset),
        'valid': len(valid_data_gen.dataset)
    }
print('dataset_sizes:', dataset_sizes)
dataloaders = {
        'train': train_data_gen,
        'valid': valid_data_gen
    }

# Create network
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
print('resnet18 model:', model)
if is_cuda:
    model = model.to(device)

# Loss & Optimizer
learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer,
        step_size=7, gamma=0.1)  # dynamically modify lr

# Train
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    start = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train(True)  # set model to training mode
            else:
                model.train(False)  # set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over batch data
            for i, data in enumerate(dataloaders[phase]):
                # progress bar
                print('\rTraining({})'.format(i), end='')
                print('{:20s}'.format('.' * (i % 20)), end='', flush=True)

                # get inputs
                inputs, labels = data
                if is_cuda:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward & optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                # statistics
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels).item()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            print('\n{} -- Loss: {:.4f}, Acc: {:.4f}\n'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.4f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best accuracy: {:.4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

best_model = train_model(model, criterion, optimizer, exp_lr_scheduler,
        num_epochs=2)

