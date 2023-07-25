#!/usr/bin/env python
# coding: utf-8

# In[8]:


import os
import cv2
import pandas as pd
import numpy as np
import time
import datetime as dt
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold, KFold

from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

import albumentations as A
import albumentations.pytorch
import wandb
from typing import List, Union
from joblib import Parallel, delayed


# In[9]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device


# ##### Hyper Parameters

# In[10]:


CFG = {
    'fold_num': 5, ## train:valid : 5 :1
    'seed': 42,
    'img_size': 224,
    'model': 'Unet',
    'epochs': 200,
    'train_bs':32,
    'valid_bs':32,
    'lr': 1e-4, ## learning rate
    'num_workers': 8,
    'verbose_step': 1,
    'patience' : 5,
    'device': 'cuda:0',
    'freezing': False,
    'model_path': './models'
}


# ###### WANDB Init & Model save name

# In[11]:


category = 'satellite'
time_now = dt.datetime.now()
run_id = time_now.strftime("%Y%m%d%H%M")
project_name = category
user = 'hojunking'
run_name = project_name + '_' + run_id


# In[12]:


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


# In[13]:


# RLE 디코딩 함수
def rle_decode(mask_rle, shape):
    if isinstance(mask_rle, int):
        s = mask_rle
    else:
        s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s [0:][::2], s [1:][::2])]
    ends = starts + lengths
    img = np.zeros(shape [0]*shape [1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img [lo:hi] = 1
    return img.reshape(shape)

# RLE 인코딩 함수
def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels [1:]!= pixels [:-1])[0] + 1
    runs [1::2] -= runs [::2]
    return ' '. join(str(x) for x in runs)


# ##### Aumentation

# In[14]:


transform_train = A.Compose(    [   
    A.RandomResizedCrop(p=1, height=CFG['img_size'] ,width=CFG['img_size'], scale=(0.25, 0.35),ratio=(0.90, 1.10)),
    A.ColorJitter(always_apply=True, p=0.5, contrast=0.2, saturation=0.3, hue=0.2),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=True, p=1.0),
     A.pytorch.transforms.ToTensorV2()
])
transform_test = A.Compose([
    A.Resize(height = CFG['img_size'], width = CFG['img_size']),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=True, p=1.0),
     A.pytorch.transforms.ToTensorV2()
])


# In[15]:


train_df = pd.read_csv('../Data/satellite/train.csv')
img_path = train_df.iloc[0]['img_path']


# In[16]:


image = cv2.imread('../Data/satellite/'+ img_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image.shape[1]


# ##### Dataset

# In[17]:


class SatelliteDataset(Dataset):
    def __init__(self, df, transform=None, infer=False):
        super(SatelliteDataset,self).__init__()
        self.df = df.reset_index(drop=True).copy()
        self.transform = transform
        self.infer = infer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['img_path']
        image = cv2.imread('../Data/satellite/'+ img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.infer:
            if self.transform:
                image = self.transform(image=image)['image']
            return image

        mask_rle = self.df.iloc[idx]['mask_rle']
        mask = rle_decode(mask_rle, (image.shape[0], image.shape[1]))

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask


# In[18]:


def prepare_dataloader(df, trn_idx, val_idx):
    
    train_ = df.loc[trn_idx,:].reset_index(drop=True)
    valid_ = df.loc[val_idx,:].reset_index(drop=True)
        
    train_ds = SatelliteDataset(train_, transform=transform_train, infer=False)
    valid_ds = SatelliteDataset(valid_, transform=transform_test, infer=False)
    
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=CFG['train_bs'],
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        num_workers=CFG['num_workers']
    )
    val_loader = torch.utils.data.DataLoader(
        valid_ds, 
        batch_size=CFG['valid_bs'],
        num_workers=CFG['num_workers'],
        shuffle=False,
        pin_memory=True,
    )
    return train_loader, val_loader


# ##### Model

# In[20]:


# U-Net의 기본 구성 요소인 Double Convolution Block을 정의합니다.
def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )

# 간단한 U-Net 모델 정의
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        

        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)

        self.conv_last = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   

        x = self.dconv_down4(x)

        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   

        x = self.dconv_up1(x)

        out = self.conv_last(x)

        return out


# ##### Train

# In[21]:


def train_one_epoch(epoch, model, loss_fn, optimizer, train_loader, device, scheduler=None):
    t = time.time()
    
    # SET MODEL TRAINING MODE
    model.train()
    
    running_loss = None
    loss_sum = 0
    image_preds_all = []
    image_targets_all = []
    
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for step, (imgs, masks) in pbar:
        imgs = imgs.to(device).float()
        masks = masks.to(device).float()
        optimizer.zero_grad()
        
        # MODEL PREDICTION
        with torch.cuda.amp.autocast():
            image_preds = model(imgs)   #output = model(input)
            loss = loss_fn(image_preds, masks.unsqueeze(1)) # CRITERION
            loss_sum+=loss.detach()
            
            # BACKPROPAGATION
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
            if running_loss is None:
                running_loss = loss.item()
            else:
                running_loss = running_loss * .99 + loss.item() * .01    
        
            # TQDM VERBOSE_STEP TRACKING
            if ((step + 1) % CFG['verbose_step'] == 0) or ((step + 1) == len(train_loader)):
                description = f'epoch {epoch} loss: {running_loss:.4f}'
                pbar.set_description(description)
        
        image_preds_all += [torch.argmax(image_preds, 1).detach().cpu().numpy()]
        image_targets_all += [masks.detach().cpu().numpy()]
        
    if scheduler is not None:
        scheduler.step()
    
    image_preds_all = np.concatenate(image_preds_all)
    image_targets_all = np.concatenate(image_targets_all)
    
    trn_loss = loss_sum/len(train_loader)
    
    return image_preds_all, trn_loss


# ##### Valid

# In[22]:


def valid_one_epoch(epoch, model, loss_fn, val_loader, device, scheduler=None):
    t = time.time()
    
    # SET MODEL VALID MODE
    model.eval()
    
    loss_sum = 0
    sample_num = 0
    avg_loss = 0
    image_preds_all = []
    image_targets_all = []
    
    pbar = tqdm(enumerate(val_loader), total=len(val_loader))
    for step, (imgs, masks) in pbar:
        imgs = imgs.to(device).float()
        masks = masks.to(device).float()
        
        # MODEL PREDICTION
        image_preds = model(imgs)
        
        image_preds_all += [torch.argmax(image_preds, 1).detach().cpu().numpy()]
        image_targets_all += [masks.detach().cpu().numpy()]
        
        loss = loss_fn(image_preds, masks.unsqueeze(1))
        
        avg_loss += loss.item()
        loss_sum += loss.item()*masks.shape[0]
        sample_num += masks.shape[0]
        
        # TQDM
        description = f'epoch {epoch} loss: {loss_sum/sample_num:.4f}'
        pbar.set_description(description)
    
    image_preds_all = np.concatenate(image_preds_all)
    image_targets_all = np.concatenate(image_targets_all)
    val_loss = avg_loss/len(val_loader)
    
    return image_preds_all, val_loss


# In[23]:


class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, score):
        print(f' present score: {score}')
        if self.best_score is None:
            self.best_score = score
        elif score >= self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            print(f'Best loss from now: {self.best_score :.5f}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        
        return self.early_stop


# In[24]:


if __name__ == '__main__':
    seed_everything(CFG['seed'])
    
    # WANDB TRACKER INIT
    wandb.init(project=project_name, entity=user)
    wandb.config.update(CFG)
    wandb.run.name = run_name
    wandb.define_metric("Train Loss", step_metric="epoch")
    wandb.define_metric("Valid Loss", step_metric="epoch")
    wandb.define_metric("Train-Valid Coefficient", step_metric="epoch")
    
    model_dir = CFG['model_path'] + '/{}'.format(run_name)
    #train_dir = train.dir.values
    best_fold = 0
    best_loss = 1
    print('Model: {}'.format(CFG['model']))
    # MAKE MODEL DIR
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    train_df = pd.read_csv('../Data/satellite/train.csv')
    # STRATIFIED K-FOLD DEFINITION
    folds = KFold(n_splits=CFG['fold_num'], shuffle=True, random_state=CFG['seed']).split(np.arange(train_df.shape[0]))
    
    # TEST PROCESS FOLD BREAK
    for fold, (trn_idx, val_idx) in enumerate(folds):
    
        print(f'Start with train :{len(trn_idx)}, valid :{len(val_idx)}')
        print(f'Training start with fold: {fold} epoch: {CFG["epochs"]} \n')
    
        # EARLY STOPPING DEFINITION
        early_stopping = EarlyStopping(patience=CFG["patience"], verbose=True)
    
        # DATALOADER DEFINITION
        train_loader, val_loader = prepare_dataloader(train_df, trn_idx, val_idx)
    
        # MODEL & DEVICE DEFINITION 
        device = torch.device(CFG['device'])
        model =UNet()
        
        # # MODEL FREEZING
        # #model.freezing(freeze = CFG['freezing'], trainable_layer = CFG['trainable_layer'])
        # if CFG['freezing'] ==True:
        #     for name, param in model.named_parameters():
        #         if param.requires_grad == True:
        #             print(f"{name}: {param.requires_grad}")
    
        model.to(device)
        # MODEL DATA PARALLEL
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    
        scaler = torch.cuda.amp.GradScaler()   
        optimizer = torch.optim.Adam(model.parameters(), lr=CFG['lr'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.5, step_size=5)
    
        # CRITERION (LOSS FUNCTION)
        loss_tr = torch.nn.BCEWithLogitsLoss()
        loss_fn = torch.nn.BCEWithLogitsLoss()
    
        wandb.watch(model, loss_tr, log='all')
    
        start = time.time()
        print(f'Fold: {fold}')
        for epoch in range(CFG['epochs']):
            print('Epoch {}/{}'.format(epoch, CFG['epochs'] - 1))
    
            # TRAINIG
            train_preds_all, train_loss = train_one_epoch(epoch, model, loss_tr, optimizer, train_loader, device, scheduler=scheduler)
            wandb.log({'Train Loss' : train_loss, 'epoch' : epoch})
    
            # VALIDATION
            with torch.no_grad():
                valid_preds_all, valid_loss = valid_one_epoch(epoch, model, loss_fn, val_loader, device, scheduler=None)
                wandb.log({'Valid Loss' : valid_loss ,'epoch' : epoch})
            print(f'Epoch [{epoch}], Train Loss : [{train_loss :.5f}] Val Loss : [{valid_loss :.5f}]')
            
            # SAVE ALL RESULTS
            valid_loss_list = []
    
            # MODEL SAVE (THE BEST MODEL OF ALL OF FOLD PROCESS)
            if valid_loss > best_loss:
                best_loss = valid_loss
                best_epoch = epoch
                # SAVE WITH DATAPARARELLEL WRAPPER
                #torch.save(model.state_dict(), (model_dir+'/{}.pth').format(CFG['model']))
                # SAVE WITHOUT DATAPARARELLEL WRAPPER
                torch.save(model.module.state_dict(), (model_dir+'/{}.pth').format(CFG['model']))
    
            # EARLY STOPPING
            stop = early_stopping(valid_loss)
            if stop:
                print("stop called")   
                break
    
        end = time.time() - start
        time_ = str(datetime.timedelta(seconds=end)).split(".")[0]
        print("time :", time_)
    
        # PRINT BEST F1 SCORE MODEL OF FOLD
        best_index = valid_loss_list.index(max(valid_loss_list))
        print(f'fold: {fold}, Best Epoch : {best_index}/ {len(valid_loss_list)}')
        print(f'Best valid_loss : {valid_loss_list[best_index]:.5f}')
        print('-----------------------------------------------------------------------')

    # K-FOLD END
    if valid_loss_list[best_index] < best_fold:
        best_fold = valid_loss_list[best_index]
        top_fold = fold
    print(f'Best valid_loss: {best_fold} Top fold : {top_fold}')


# In[ ]:


test_df =pd.read_csv('../Data/satellite/test.csv')


# In[ ]:


test_dataset = SatelliteDataset(test_df, transform=transform_test, infer=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=8)


# In[ ]:


with torch.no_grad():
    model.eval()
    result = []
    for images in tqdm(test_dataloader):
        images = images.float().to(device)
        
        outputs = model(images)
        masks = torch.sigmoid(outputs).cpu().numpy()
        masks = np.squeeze(masks, axis=1)
        masks = (masks > 0.35).astype(np.uint8) # Threshold = 0.35
        
        for i in range(len(images)):
            mask_rle = rle_encode(masks[i])
            if mask_rle == '': # 예측된 건물 픽셀이 아예 없는 경우 -1
                result.append(-1)
            else:
                result.append(mask_rle)


# In[ ]:




