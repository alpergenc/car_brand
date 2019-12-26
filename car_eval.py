
from __future__ import print_function, division
#from funcs import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import cv2  as cv
import copy
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
#from utils import *
#import utils
#import uts
import os, sys, time, datetime, random
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.patches as patches
from PIL import Image
import datetime


'''

Parçalar üzerinde eðitim yapmak için kullanýlýyor. Foldername yerine daha önce klasörü açýlmýþ olan eðitim konusunun
adý yazýlýr. Model her epoch döndüðünde kendini kayýt eder. Kayýt ismi klasör_ismi.pth þeklinde.

vgg kullanýldý. 


'''


batch_size_c=4
folder_name="car_data"           #car_brand, sson, car_data_basic, flowers-recognition

start=datetime.datetime.now()
if __name__ == '__main__':

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            #transforms.RandomResizedCrop(100),
            #transforms.RandomHorizontalFlip(),
            #transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(524),
            transforms.CenterCrop(524),
            #transforms.RandomResizedCrop(100),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    data_dir = 'D:/brand/car_data/'+folder_name                                ############            FOLDER

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size_c,
                                                 shuffle=True, num_workers=4)
                  for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    print(class_names)
    data_names = image_datasets['train'].imgs
    print("Dataset Loaded")

    device = torch.device("cuda:0")
    # Get a batch of training data
    inputs, classes = next(iter(dataloaders['train']))

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)

    def train_model(model, criterion, optimizer, scheduler, num_epochs=24):

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            ft=time.time()
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            for phase in ['train', 'val']:
                model.to(device)
                if phase == 'train':
                    scheduler.step()
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        csx, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print('{} Loss: {:.9f} Acc: {:.9f}'.format(
                    phase, epoch_loss, epoch_acc))
                torch.save(model_ft, "D:/load/" + folder_name + ".pth")
                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                et=time.time()-ft
                if(epoch==0):
                    print("Epoch time:",et)
                    print("Estimated total minutes of process:", 60*et*num_epochs/(60*24))
                    print("Start Time:", start)



        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model


    #model_ft = models.resnet18(pretrained=True)
    #os.environ['TORCH_MODEL_ZOO'] = 'C:/'  # setting the environment variable
    model_ft = models.vgg16(pretrained=True)
    model_ft.classifier[6]=nn.Linear(4096,len(class_names))
    model_ft=nn.Sequential(model_ft,nn.Softmax(dim=1))

    #num_ftrs = model_ft.fc.in_features
    #model_ft.fc = nn.Linear(num_ftrs,len(class_names))                                       ####################################################       OUTPUT LAYER
    #model_ft.fc = nn.Sequential(nn.Linear(num_ftrs,len(class_names)),nn.Softmax(dim=1))                                       ####################################################       OUTPUT LAYER

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized

    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,               ####################################################       EPOCH
                           num_epochs=199)

    torch.save(model_ft,"D:/load/xy"+folder_name+".pth")
    print("saved")

    plt.ioff()
    plt.show()