


import os

from MTA_LOSS_038 import *
import torch.utils.data as Data
import torch.optim as optim
from torch.autograd import Variable
from mlp_mixer_pytorch_mta import MLPMixer as MLPMixer
from mlp_mixer_pytorch_mta import weight_init
import torch
import pandas as pd
from torchsummary import summary

from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ExponentialLR

import torch
import torch.nn as nn
import numpy as np
from torchsummary import summary
from torch.nn import Conv2d
from torch.nn import ReLU
from einops.layers.torch import Rearrange, Reduce
from tensorboardX import SummaryWriter




target_size = 427
batch_size = 3

epochs = 1000000


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

myNet = MLPMixer(in_channels=1, dim=427, num_classes=427*427, image_size=427, depth=1, token_dim=427,
                     channel_dim=427).to(device)



myNet.to(device)




def save_result(allOutPut,all_files, save_dir = './MTA/result_13'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    allOutPut = allOutPut.reshape(-1,1,target_size, target_size)
    for index, each_day in tqdm(enumerate(allOutPut)):
        for index_one, each in enumerate(allOutPut[index]):

            each = np.round(each)
            data = pd.DataFrame(data=each)
            data.to_csv(f'{save_dir}/{all_files[index]}')

    print('Finished predict')




train_data, label_data, all_files = load_data_for_more_loss(predict=True)

trainData = train_data
labelData = label_data.clone()


train_data = Data.TensorDataset(trainData, labelData)


train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)




optimizer = torch.optim.Adam(myNet.parameters(), lr=10)
lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=300, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=10, min_lr=0, eps=1e-8)

earlyStop = 5000
earlyStep = 0
best_loss = float('inf')
needSave = False
save_path = r'C:/Users/remote/Desktop/lan-resnet-fina/MTA/result_13/makabaka.pth' 




start_epoch = 0
RESUME =  False     
if RESUME:
    path_checkpoint = r'C:/Users/remote/Desktop/lan-resnet-fina/MTA/result_13/models/checkpoint/ckpt_best.pth'  
    checkpoint = torch.load(path_checkpoint, map_location=torch.device('cpu')) 

    myNet.load_state_dict(checkpoint['net']) 

    optimizer.load_state_dict(checkpoint['optimizer'])  
    start_epoch = checkpoint['epoch'] +1  





max_norm = None
for epoch in range(start_epoch, epochs):
    myNet.train()
    earlyStep += 1
    running_loss = 0.0
    train_bar = tqdm(train_loader)
    allOutPut = None
    for step, (images,labels) in enumerate(train_bar):
        images.to(device)
        labels.to(device)
        logits = myNet(images.to(device))

        loss = computelossforSix(logits, labels.to(device), device=device, targetShape=target_size)
        optimizer.zero_grad()

        '''
        torch.autograd.set_detect_anomaly(True)
        with torch.autograd.detect_anomaly():
            loss.backward()
        '''


        loss.backward()

        if max_norm:
            nn.utils.clip_grad_norm(myNet.parameters(), max_norm=1)
        optimizer.step()
        running_loss += loss.item()
        train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,epochs,loss)
        save_data = logits.cpu().detach().numpy()


        en_zeros_index = torch.where(labels[:, :, :, 0] == 0)
        ex_zeros_index = torch.where(labels[:, :, :, 1] == 0)
        save_data[en_zeros_index[0], en_zeros_index[2], :] = 0
        save_data[ex_zeros_index[0], :, ex_zeros_index[2]] = 0

        if step == 0:
            allOutPut = save_data

        else:
            allOutPut = np.vstack((allOutPut, save_data))

    now_loss = running_loss/len(train_loader)
    lr_scheduler.step(now_loss)

    if now_loss < best_loss:
        earlyStep = 0
        best_loss = now_loss

        checkpoint = {
            "net": myNet.state_dict(),
            'optimizer':optimizer.state_dict(),
            "epoch": epoch,
            'lr_scheduler': lr_scheduler.state_dict()
        }
        if not os.path.isdir("C:/Users/remote/Desktop/lan-resnet-fina/MTA/result_13/models/checkpoint"):
          os.makedirs("C:/Users/remote/Desktop/lan-resnet-fina/MTA/result_13/models/checkpoint")
        torch.save(checkpoint, "C:/Users/remote/Desktop/lan-resnet-fina/MTA/result_13/models/checkpoint/ckpt_best.pth")

        if needSave:
            torch.save(myNet.state_dict(), save_path)
        print(f"save files with loss {best_loss}")
        save_result(allOutPut, all_files)

    if earlyStep > earlyStop:
        break

print('Finished Training')





