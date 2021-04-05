import os
from sklearn.model_selection import StratifiedKFold
import cv2
import pdb
import time
import warnings
import random
import numpy as np
import pandas as pd
from tqdm import tqdm as tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset, sampler
from matplotlib import pyplot as plt
from .metric import Meter, epoch_log
from .dataloader import provider_cv, provider_trai_test_split,provider_mt_cv
import sys
from .losses import BCEDiceLoss, FocalLoss, JaccardLoss, DiceLoss
from .lovasz_losses import LovaszLoss, LovaszLossSymmetric

sys.path.append('..')
from configs.train_params import *
from .optimizers import RAdam, Over9000, Adam


# coding:utf-8
import os, torchvision
import torch.nn as nn
import numpy as np
import imageio
import matplotlib.pyplot as plt
from PIL import Image
import torch


def tensor2im(input_image, imtype=np.uint8):
    """"将tensor的数据类型转成numpy类型，并反归一化.

    Parameters:
        input_image (tensor) --  输入的图像tensor数组
        imtype (type)        --  转换后的numpy的数据类型
    """
    mean = [0.485,0.456,0.406] #自己设置的
    std = [0.229,0.224,0.225]  #自己设置的
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        for i in range(len(mean)):
            image_numpy[i] = image_numpy[i] * std[i] + mean[i]
        image_numpy = image_numpy * 255
        image_numpy = np.transpose(image_numpy, (1, 2, 0))  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)

def save_img(im, path, size):
    """im可是没经过任何处理的tensor类型的数据,将数据存储到path中

    Parameters:
        im (tensor) --  输入的图像tensor数组
        path (str)  --  图像保存的路径
        size (int)  --  一行有size张图,最好是2的倍数
    """
    im_grid = torchvision.utils.make_grid(im, size) #将batchsize的图合成一张图
    im_numpy = tensor2im(im_grid) #转成numpy类型并反归一化
    im_array = Image.fromarray(im_numpy)
    im_array.save(path)

class Trainer_cv(object):
    '''This class takes care of training and validation of our model'''

    def __init__(self, model, num_epochs, current_fold=0, batch_size={"train": 4, "val": 4}, optimizer_state=None):
        self.current_fold = current_fold
        self.total_folds = TOTAL_FOLDS
        self.num_workers = 4
        self.batch_size = batch_size
        self.accumulation_steps = 32 // self.batch_size['train']
        self.lr = LEARNING_RATE
        self.num_epochs = num_epochs
        self.best_metric = INITIAL_MINIMUM_DICE  # float("inf")
        self.phases = ["train", "val"]
        self.device = torch.device("cuda:0")
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        self.net = model  # torch.nn.BCEWithLogitsLoss()
        self.criterion = BCEDiceLoss()  # JaccardLoss()#LovaszLossSymmetric(per_image=True, classes=[0,1,2,3])
        # BCEDiceLoss()  # BCEDiceLoss()#FocalLoss(num_class=4)  # BCEDiceLoss()  # torch.nn.BCEWithLogitsLoss()
        self.optimizer = RAdam([
            {'params': self.net.decoder.parameters(), 'lr': self.lr},
            {'params': self.net.encoder.parameters(), 'lr': self.lr},
        ])  # optim.Adam(self.net.parameters(), lr=self.lr)

        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)
        self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.9, mode="min", patience=3, verbose=True)
        self.net = self.net.to(self.device)
        cudnn.benchmark = True
        # self.dataloaders = {
        #     phase: provider_cv(
        #         fold=self.current_fold,
        #         total_folds=self.total_folds,
        #         data_folder=data_folder,
        #         df_path=train_df_path,
        #         phase=phase,
        #         mean=(0.485, 0.456, 0.406),
        #         std=(0.229, 0.224, 0.225),
        #         batch_size=self.batch_size[phase],
        #         num_workers=self.num_workers,
        #     )
        #     for phase in self.phases
        # }

        self.dataloaders = {
            phase:provider_mt_cv(train=phase=='train') for phase in self.phases
        }
        self.losses = {phase: [] for phase in self.phases}
        # self.iou_scores = {phase: [] for phase in self.phases}
        self.dice_scores = {phase: [] for phase in self.phases}

    def forward(self, images, targets):
        images = images.to(self.device)
        masks = targets.to(self.device)
        outputs = self.net(images)
        # pdb.set_trace()
        loss = self.criterion(outputs, masks)
        return loss, outputs

    def iterate(self, epoch, phase):
        meter = Meter(phase, epoch)
        start = time.strftime("%H:%M:%S")
        print(f"Starting epoch: {epoch} | phase: {phase} | ⏰: {start}")
        batch_size = self.batch_size[phase]
        self.net.train(phase == "train")
        dataloader = self.dataloaders[phase]
        running_loss = 0.0
        total_batches = len(dataloader)
        tk0 = tqdm(dataloader, total=total_batches)
        self.optimizer.zero_grad()
        # print('A')
        
        # pdb.set_trace()
        for itr, batch in enumerate(dataloader):
            # if epoch == 2:
            #     print(itr)
            images, targets, mask_img = batch
            pdb.set_trace()
            loss, outputs = self.forward(images, targets)
            loss = loss / self.accumulation_steps
            # if epoch == 2:
            #     print('B')
            if phase == "train":
                loss.backward()
                if (itr + 1) % self.accumulation_steps == 0:
                    # if epoch == 2:
                    #     print('C')
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            # if epoch == 2:
            #     print('D')
            running_loss += loss.item()
            outputs = outputs.detach().cpu()
            meter.update(targets, outputs)
            tk0.update(1)
            tk0.set_postfix(loss=(running_loss / (itr + 1)))
        tk0.close()
        epoch_loss = (running_loss * self.accumulation_steps) / total_batches
        dice = epoch_log(phase, epoch, epoch_loss, meter, start)
        self.losses[phase].append(epoch_loss)
        self.dice_scores[phase].append(dice)
        # self.iou_scores[phase].append(iou)
        torch.cuda.empty_cache()
        return epoch_loss, dice

    def start(self):
        epoch_wo_improve_score = 0
        for epoch in range(self.num_epochs):
            if EARLY_STOPING is not None and epoch_wo_improve_score >= EARLY_STOPING:
                print('Early stopping {}'.format(EARLY_STOPING))
                # torch.save(state, "./model_weights/model_{}_fold_{}_last_epoch_{}_dice_{}.pth".format(
                #     unet_encoder, self.current_fold, epoch, val_dice))
                torch.save(state, "./model_weights_mt/model_{}_fold_{}_last_epoch_{}_dice_{}.pth".format(
                    unet_encoder, self.current_fold, epoch, val_dice))
                break
            self.iterate(epoch, "train")
            state = {
                "epoch": epoch,
                "best_metric": self.best_metric,
                "state_dict": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict()
            }
            val_loss, val_dice = self.iterate(epoch, "val")
            self.scheduler.step(val_loss)
            if val_dice > self.best_metric:
                print("******** New optimal found, saving state ********")
                state["best_metric"] = self.best_metric = val_dice
                # torch.save(state, "./model_weights/model_{}_fold_{}_epoch_{}_dice_{}.pth".format(
                #     unet_encoder, self.current_fold, epoch, val_dice))
                torch.save(state, "./model_weights_mt/model_{}_fold_{}_epoch_{}_dice_{}.pth".format(
                    unet_encoder, self.current_fold, epoch, val_dice))
                epoch_wo_improve_score = 0
            else:
                epoch_wo_improve_score += 1
            print()
        if num_epochs > 1:
            # torch.save(state, "./model_weights/model_{}_fold_{}_last_epoch_{}_dice_{}.pth".format(
            #     unet_encoder, self.current_fold, epoch, val_dice))
            torch.save(state, "./model_weights_mt/model_{}_fold_{}_last_epoch_{}_dice_{}.pth".format(
                unet_encoder, self.current_fold, epoch, val_dice))

    def generate_pseudo_mask(self):
        self.pretrained_net = load_model_fpn(self.load_path)
        images = images.to(self.device)
        masks = targets.to(self.device)
        outputs = self.net(images)
        # pdb.set_trace()
        return outputs

    def evaluate(self):

        color_0 = torch.zeros((3,256,1600),device='cpu')
        color_0[0,:,:] = 255
        color_0[1,:,:] = 0
        color_0[2,:,:] = 0

        color_1 = torch.zeros((3,256,1600),device='cpu')
        color_1[0,:,:] = 0
        color_1[1,:,:] = 255
        color_1[2,:,:] = 0

        color_2 = torch.zeros((3,256,1600),device='cpu')
        color_2[0,:,:] = 0
        color_2[1,:,:] = 0
        color_2[2,:,:] = 255

        color_3 = torch.zeros((3,256,1600),device='cpu')
        color_3[0,:,:] = 0
        color_3[1,:,:] = 255
        color_3[2,:,:] = 255

        color_list = []
        color_list.append(color_0)
        color_list.append(color_1)
        color_list.append(color_2)
        color_list.append(color_3)


        phase = 'val'
        epoch = 1
        # ##########
        path = './model_weights/model_se_resnext50_32x4d_fold_1_last_epoch_30_dice_0.8624916076660156.pth'
        self.net.load_state_dict(torch.load(path)["state_dict"])


        visual_saved_path = ''

        meter = Meter(phase, epoch)
        # start = time.strftime("%H:%M:%S")
        batch_size = self.batch_size[phase]
        self.net.train(phase == "train")
        dataloader = self.dataloaders[phase]
        running_loss = 0.0
        total_batches = len(dataloader)
        tk0 = tqdm(dataloader, total=total_batches)

        for itr, batch in enumerate(dataloader):
            images, targets = batch
            loss, outputs = self.forward(images, targets)
            loss = loss / self.accumulation_steps

            running_loss += loss.item()
            outputs = torch.nn.functional.sigmoid(outputs)

            # pdb.set_trace()
            
            outputs = outputs.detach().cpu()
            mask = outputs[0] > 0.5
            gt_mask = targets[0]
            for i in range(4):
                # seg_image = torch.zeros((3,256,1600))
                this_mask = mask[i].unsqueeze(dim=0)
                seg_image = this_mask * color_list[i]
    
                seg_image = np.transpose(seg_image.numpy(), (1, 2, 0))
                seg_image = seg_image.astype(np.uint8)
                # pdb.set_trace()
                seg_image = Image.fromarray(seg_image)
                seg_image.save('./visualization/'+str(itr)+'_pred_'+str(i)+'.png')


                gt_this_mask = gt_mask[i].unsqueeze(dim=0)
                gt_seg_image = gt_this_mask * color_list[i]
                gt_seg_image = np.transpose(gt_seg_image.numpy(), (1,2,0))
                gt_seg_image = gt_seg_image.astype(np.uint8)
                gt_seg_image = Image.fromarray(gt_seg_image)
                gt_seg_image.save('./visualization/'+str(itr)+'_gt_'+str(i)+'.png')

            im_numpy = tensor2im(images.squeeze())
            im_array = Image.fromarray(im_numpy)
            im_array.save('./visualization/'+str(itr)+'.png')
            
            # pdb.set_trace()
            meter.update(targets, outputs)
            tk0.update(1)
            tk0.set_postfix(loss=(running_loss / (itr + 1)))
        tk0.close()
        # epoch_loss = (running_loss * self.accumulation_steps) / total_batches
        # dice = epoch_log(phase, epoch, epoch_loss, meter, start)
        print(meter.get_metrics())
        # self.iou_scores[phase].append(iou)
        # torch.cuda.empty_cache()



    def evaluate_mt(self):

        color_0 = torch.zeros((3,224,224),device='cpu')
        color_0[0,:,:] = 255
        color_0[1,:,:] = 0
        color_0[2,:,:] = 0

        color_1 = torch.zeros((3,224,224),device='cpu')
        color_1[0,:,:] = 0
        color_1[1,:,:] = 255
        color_1[2,:,:] = 0

        color_2 = torch.zeros((3,224,224),device='cpu')
        color_2[0,:,:] = 0
        color_2[1,:,:] = 0
        color_2[2,:,:] = 255

        color_3 = torch.zeros((3,224,224),device='cpu')
        color_3[0,:,:] = 0
        color_3[1,:,:] = 255
        color_3[2,:,:] = 255

        color_4 = torch.zeros((3,224,224),device='cpu')
        color_4[0,:,:] = 255
        color_4[1,:,:] = 255
        color_4[2,:,:] = 0

        color_list = []
        color_list.append(color_0)
        color_list.append(color_1)
        color_list.append(color_2)
        color_list.append(color_3)
        color_list.append(color_4)


        phase = 'val'
        epoch = 1
        path = './model_weights_mt/model_se_resnext50_32x4d_fold_0_epoch_96_dice_0.9301743507385254.pth'
        self.net.load_state_dict(torch.load(path)["state_dict"])


        visual_saved_path = ''

        meter = Meter(phase, epoch)
        # start = time.strftime("%H:%M:%S")
        batch_size = self.batch_size[phase]
        self.net.train(phase == "train")
        dataloader = self.dataloaders[phase]
        running_loss = 0.0
        total_batches = len(dataloader)
        tk0 = tqdm(dataloader, total=total_batches)

        for itr, batch in enumerate(dataloader):
            images, targets, mask_img_gt = batch
            loss, outputs = self.forward(images, targets)
            loss = loss / self.accumulation_steps

            running_loss += loss.item()
            

            # pdb.set_trace()
            
            outputs = outputs.detach().cpu()
            meter.update(targets, outputs)

            # pdb.set_trace()

            
            mask = torch.nn.functional.sigmoid(outputs)
            # mask = (mask[0] > 0.5)
            mask = mask[0]
            gt_mask = targets[0]
            print(mask.sum())
            print(gt_mask.sum())
            print('---------------')
            pdb.set_trace()
            for i in range(5):
                # seg_image = torch.zeros((3,256,1600))
                this_mask = mask[i].unsqueeze(dim=0)
                seg_image = this_mask * color_list[i]

                # pdb.set_trace()
    
                seg_image = np.transpose(seg_image.numpy(), (1, 2, 0))
                seg_image = seg_image.astype(np.uint8)
                # pdb.set_trace()
                seg_image = Image.fromarray(seg_image)
                
                seg_image.save('./visualization_mt/'+str(itr)+'_pred_'+str(i)+'.png')
                # seg_image.save('./'+str(itr)+'_pred_'+str(i)+'.png')

            # pdb.set_trace()
            im_numpy = tensor2im(images.squeeze())
            im_array = Image.fromarray(im_numpy)
            im_array.save('./visualization_mt/'+str(itr)+'.jpg')

            im_numpy = tensor2im(mask_img_gt.squeeze())
            im_array = Image.fromarray(im_numpy)
            im_array.save('./visualization_mt/'+str(itr)+'.png')
            
            
            # pdb.set_trace()
            
            tk0.update(1)
            tk0.set_postfix(loss=(running_loss / (itr + 1)))
        tk0.close()
        # epoch_loss = (running_loss * self.accumulation_steps) / total_batches
        # dice = epoch_log(phase, epoch, epoch_loss, meter, start)
        print(meter.get_metrics())
        # self.iou_scores[phase].append(iou)
        # torch.cuda.empty_cache()


""" WARNING DEPRECATED
class Trainer_split(object):
    '''This class takes care of training and validation of our model'''

    def __init__(self, model):
        self.num_workers = 6
        self.batch_size = {"train": 4, "val": 4}
        self.accumulation_steps = 32 // self.batch_size['train']
        self.lr = 5e-4
        self.num_epochs = 20
        self.best_loss = float("inf")
        self.phases = ["train", "val"]
        self.device = torch.device("cuda:0")
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        self.net = model
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", patience=3, verbose=True)
        self.net = self.net.to(self.device)
        cudnn.benchmark = True
        self.dataloaders = {
            phase: provider_trai_test_split(
                data_folder=data_folder,
                df_path=train_df_path,
                phase=phase,
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                batch_size=self.batch_size[phase],
                num_workers=self.num_workers,
            )
            for phase in self.phases
        }
        self.losses = {phase: [] for phase in self.phases}
        # self.iou_scores = {phase: [] for phase in self.phases}
        self.dice_scores = {phase: [] for phase in self.phases}

    def forward(self, images, targets):
        images = images.to(self.device)
        masks = targets.to(self.device)
        outputs = self.net(images)
        loss = self.criterion(outputs, masks)
        return loss, outputs

    def iterate(self, epoch, phase):
        meter = Meter(phase, epoch)
        start = time.strftime("%H:%M:%S")
        print(f"Starting epoch: {epoch} | phase: {phase} | ⏰: {start}")
        batch_size = self.batch_size[phase]
        self.net.train(phase == "train")
        dataloader = self.dataloaders[phase]
        running_loss = 0.0
        total_batches = len(dataloader)
        #         tk0 = tqdm(dataloader, total=total_batches)
        self.optimizer.zero_grad()
        for itr, batch in enumerate(dataloader):  # replace `dataloader` with `tk0` for tqdm
            images, targets = batch
            loss, outputs = self.forward(images, targets)
            loss = loss / self.accumulation_steps
            if phase == "train":
                loss.backward()
                if (itr + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            running_loss += loss.item()
            outputs = outputs.detach().cpu()
            meter.update(targets, outputs)
        #             tk0.set_postfix(loss=(running_loss / ((itr + 1))))
        epoch_loss = (running_loss * self.accumulation_steps) / total_batches
        dice = epoch_log(phase, epoch, epoch_loss, meter, start)
        self.losses[phase].append(epoch_loss)
        self.dice_scores[phase].append(dice)
        # self.iou_scores[phase].append(iou)
        torch.cuda.empty_cache()
        return epoch_loss

    def start(self):
        for epoch in range(self.num_epochs):
            self.iterate(epoch, "train")
            state = {
                "epoch": epoch,
                "best_metric": self.best_loss,
                "state_dict": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            with torch.no_grad():
                val_loss = self.iterate(epoch, "val")
                self.scheduler.step(val_loss)
            if val_loss < self.best_loss:
                # TODO save weights on last epoch too
                print("******** New optimal found, saving state ********")
                state["best_metric"] = self.best_loss = val_loss
                torch.save(state, "./model.pth")
            print()
"""
