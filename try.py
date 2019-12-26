import cv2 as cv
from math import *
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import random
import shutil
import seaborn as sns
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import glob
bolen = 5  # was 5

if __name__ == '__main__':

    debug = 0
    saving = 1

    font = cv.FONT_HERSHEY_SIMPLEX

    folder_name = "brand"  # car_brand, sson, car_data_basic, flowers-recognition
    version = "brand_vgg"

    data_transforms = transforms.Compose([

        transforms.Resize(224),
        # transforms.CenterCrop(224),
        # transforms.RandomRotation(150),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    s_path = "D:/"+ str(folder_name) +"/car_data/car_data/" + "sonuc/"
    path = "D:/"+ str(folder_name) +"/car_data/car_data/test/"  ############################
    pathx = "D:/"+ str(folder_name) +"/car_data/car_data/"
    print("Dataset Loaded")

    device = torch.device("cuda:0")

    def image_loader(image):
        """load image, returns cuda tensor"""
        image = Image.fromarray(image)
        image = data_transforms(image)
        image = Variable(image, requires_grad=True)
        image = image.unsqueeze(0)  # this is for VGG, may not be needed for ResNet

        return image.cuda()  # assumes that you're using GPU

    model_ft = torch.load("D:/load/" + str(version) + ".pth")

    model_ft.cuda()
    model_ft.eval()

    t = 0
    print("Start")
    try:
        shutil.rmtree(s_path)
    except:
        pass
    time.sleep(1)
    os.mkdir(s_path)
    bg = 0
    gg = 0
    rep = 0
    tlist=os.listdir(pathx+"/train/")
    dossx=[]

    while(1):
        dossx = list(filter(os.path.isfile, glob.glob(path+"*")))
        tmp=len(dossx)
        y=0
        dossx.sort(key=lambda x: os.path.getatime(x))

        if(t<len(dossx)):
            while (t < len(dossx)):  # 32
                #pol = cv.imread(path + dossi[t])
                pol = cv.imread(dossx[t])

                if pol is None:
                    break
                act = dossx[t][:]
                act=act.split("test\\")
                act=act[1]
                #act=act.split(" ")
                #act=act[0]

                img = pol * 1

                print("Number: ", t)

                if (debug):
                    img1 = img * 1
                    img2 = img * 1
                    cv.imwrite(pathx + "circle_full.png", img1)

                if (saving):
                    imgx = img * 1
                img = image_loader(img)

                img = img.to(device)
                prd = model_ft(img)
                xs = (torch.max(prd, 1)[0].cpu().detach().numpy())
                preds = int(torch.max(prd, 1)[1].cpu().detach().numpy())

                pred=tlist[preds]
                print("Resim: "+act)
                print("Tahmin: "+pred)

                if (pred == act):
                    gg = gg + 1
                    if (saving):
                        cv.imwrite(s_path + "good_" + str(t) + ".jpg", imgx)


                elif (pred != act) :
                    bg = bg + 1
                    if (saving):
                        cv.imwrite(s_path + "act_" + (act) +"_pred_"+pred+ ".jpg", imgx)

                #acc = (gg) / (gg +bg)
                #print("Current Accuracy: %", acc * 100)
                window_name = str(pred)
                cv.imshow(window_name,pol)
                cv.waitKey(0)

                t = t + 1
                print("*"*50)