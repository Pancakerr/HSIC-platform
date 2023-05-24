
import os
import time
from typing import Tuple
import spectral
import torch
import random
import torch.nn as nn
import pandas as pd
import numpy as np
from scipy import io as sio
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,recall_score,cohen_kappa_score,accuracy_score
from sklearn.preprocessing import minmax_scale
from scipy.io import loadmat
from visdom import Visdom
from models import SSRNet, HybridSN, S3KAIResNet,ViT, fucontnet,UNet, MLWBDN, LCA_FCN, ghostnet, MobileNetV1,SSUN
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler


def write_log(text, win, viz, title):
    """Show text in visdom.
    Args:
        text: string need to write
        win: window name in visdom
        viz: visdom env name
        title: win title
    Returns:
        win
    """
    append = False if win is None else True
    win=viz.text(text.replace('\n', '<br/>'), win = win, append = append,opts={'title':title})
    return win


def loadData(name:str): ## customize data and return data label and class_name
    """load dataset

    Args:
        name (str): name of dataset e.g IP UP SA
    Returns:
        data: ndarray (M,N,C)
        labels: ndarray (M,N)
        class_name: list
        rgb_band: [R,G,B]
    """
    
    data_path = os.path.join(os.getcwd(),'dataset')
    if name == 'IP':
        data = loadmat(os.path.join(data_path, 'IndianPines\\Indian_pines_corrected.mat'))['indian_pines_corrected']
        labels = loadmat(os.path.join(data_path, 'IndianPines\\Indian_pines_gt.mat'))['indian_pines_gt']
        class_name = [ "Alfalfa", "Corn-notill", "Corn-mintill","Corn", "Grass-pasture", 
                       "Grass-trees","Grass-pasture-mowed", "Hay-windrowed", "Oats","Soybean-notill", "Soybean-mintill", "Soybean-clean","Wheat", "Woods", "Buildings-Grass-Trees-Drives","Stone-Steel-Towers"]
        rgb_band = [36,18,8]

    elif name == 'SA':
        data = loadmat(os.path.join(data_path, 'Salinas\\Salinas_corrected.mat'))['salinas_corrected']
        labels = loadmat(os.path.join(data_path, 'Salinas\\Salinas_gt.mat'))['salinas_gt']
        class_name = ['Brocoli_green_weeds_1','Brocoli_green_weeds_2','Fallow',
                        'Fallow_rough_plow','Fallow_smooth','Stubble','Celery','Grapes_untrained','Soil_vinyard_develop','Corn_senesced_green','Lettuce_romaine_4wk','Lettuce_romaine_5wk','Lettuce_romaine_6wk','Lettuce_romaine_7wk','Vinyard_untrained','Vinyard_vertical']
        rgb_band = [36,18,8]
    elif name == 'UP':
        data = loadmat(os.path.join(data_path, 'PaviaU\\PaviaU.mat'))['paviaU']
        labels = loadmat(os.path.join(data_path, 'PaviaU\\PaviaU_gt.mat'))['paviaU_gt']
        class_name = ['Asphalt', 'Meadows', 'Gravel', 'Trees','Painted metal sheets', 'Bare Soil', 
                      'Bitumen','Self-Blocking Bricks', 'Shadows']
        rgb_band = [55,41,12]
    
    elif name == 'UH':
        data = loadmat(os.path.join(data_path, 'GRSS2013\\HoustonU.mat'))['ans']
        labels = loadmat(os.path.join(data_path, 'GRSS2013\\HoustonU_gt.mat'))['name']
        class_name = ['Healthy grass','Stressed grass','Synthetic grass',
                        'Trees','Soil','Water','Residential','Commercial','Road','Highway','Railway','Parking Lot 1','Parking Lot 2','Tennis Court','Running Track']
        rgb_band = [59, 70, 23]


    return data, labels, class_name, rgb_band

def img_display(data = None, rgb_band = None, classes = None,title = '',viz = None, savepath = None) -> None:
    """
    display false color image of HSI data or colorful label image in visdom and save label mat
    """
    
    if data is not None:
        im_rgb = np.zeros_like(data[:,:,0:3])
        im_rgb = data[:,:,rgb_band]
        im_rgb = im_rgb/(np.max(np.max(im_rgb,axis = 1),axis = 0))*255
        im_rgb = np.asarray(im_rgb,np.uint8)
        viz.images([np.transpose(im_rgb, (2, 0, 1))],
                    opts={'caption': title})
    elif classes is not None:
        palette = spectral.spy_colors
        rgb_class = np.zeros((classes.shape[0],classes.shape[1],3))
        if savepath != None: sio.savemat(os.path.join(savepath,f'{title}.mat') ,{'label':classes})
        for i in np.unique(classes):
            rgb_class[classes==i]=palette[i]
        rgb_class = np.asarray(rgb_class, np.uint8)
        viz.images([np.transpose(rgb_class, (2, 0, 1))],
                    opts={'caption': title})
        
def applyPCA(X: np.ndarray, numComponents: int = 15, norm: bool = True) -> Tuple[np.ndarray,int]:
    """
    Args:
        X (np.ndarray): input data
        numComponents (int, optional): number of reserved components. Defaults to 15.
        norm (bool, optional): normalization or not. Defaults to True.

    Returns:
        Tuple[np.ndarray,int]: processed data and spectral dimension of output data
    """
    
    if numComponents == 0:
        newX = np.reshape(X, (-1, X.shape[2]))
    else:
        newX = np.reshape(X, (-1, X.shape[2]))
        pca = PCA(n_components=numComponents)   ##PCA and normalization
        newX = pca.fit_transform(newX)
    if norm:
        newX = minmax_scale(newX, axis=0)
    newX = np.reshape(newX, (X.shape[0],X.shape[1], -1))
    return newX, newX.shape[2]

def sample_gt(gt, train_rate,seed = 100):
    """ generate training gt for training dataset
    Args:
        gt (ndarray): full classmap
        train_rate (float): ratio of training dataset
    Returns:
        train_gt(ndarray): classmap of training data
        test_gt(ndarray): classmap of test data
    """
    indices = np.nonzero(gt)  ##([x1,x2,...],[y1,y2,...])
    X = list(zip(*indices))  ## X=[(x1,y1),(x2,y2),...] location of pixels
    y = gt[indices].ravel()
    train_gt = np.zeros_like(gt)
    test_gt = np.zeros_like(gt)
    if train_rate > 1:
        train_rate = int(train_rate)
        train_indices, test_indices = [], []
        for c in np.unique(gt):
            if c == 0:
                continue
            indices = np.nonzero(gt == c) 
            X = list(zip(*indices)) # x,y features 
            label_num = (gt == c).sum().sum()
            if  label_num <= train_rate*2: 
                train, test = train_test_split(X, train_size=label_num//2,random_state=seed) 
            else:
                train, test = train_test_split(X, train_size=train_rate,random_state=seed)
            train_indices += train
            test_indices += test
    else:
        train_indices, test_indices = train_test_split(X, train_size=train_rate, stratify=y,random_state=seed)
    train_indices = [t for t in zip(*train_indices)]   ##[[x1,x2,...],[y1,y2,...]]
    test_indices = [t for t in zip(*test_indices)]
    train_gt[tuple(train_indices)] = gt[tuple(train_indices)]
    test_gt[tuple(test_indices)] = gt[tuple(test_indices)]
    return train_gt, test_gt

def sample_info(label,train_gt,val_gt,test_gt,class_name):
    sample_report = f"{'class': ^25}{'train_num':^10}{'val_num': ^10}{'test_num': ^10}{'total': ^10}\n"
    for i in np.unique(label):
        if i == 0: continue
        sample_report += f"{class_name[i-1]: ^25}{(train_gt==i).sum(): ^10}{(val_gt==i).sum(): ^10}{(test_gt==i).sum(): ^10}{(label==i).sum(): ^10}\n"
    sample_report += f"{'total': ^25}{np.count_nonzero(train_gt): ^10}{np.count_nonzero(val_gt): ^10}{np.count_nonzero(test_gt): ^10}{np.count_nonzero(label): ^10}"
    return sample_report
    
    
def bulid_dataloader(data,data_origin,train_gt,val_gt,test_gt,**config):
    
    if config['MODEL'] == 'SSUN':
        train_data = DualSet(data,data_origin, train_gt, time_step=config['TIME_STEP'],is_pred=False,**config)
    else: train_data = ComPositionSet(data, train_gt, is_pred=False, **config)
    if config['SAMPLE_MODE'] == 'PWS':
        if config['MODEL'] == 'SSUN':
            val_data = DualSet(data,data_origin,val_gt, time_step=config['TIME_STEP'],is_pred=False,**config)
        else: val_data = ComPositionSet(data, val_gt, is_pred=False,**config)
    else:
        val_data = ComPositionSet(data, val_gt, is_pred=True,**config)
    if config['MODEL'] == 'SSUN':
        test_data =  DualSet(data,data_origin, test_gt, time_step=config['TIME_STEP'],is_pred=True,**config)
    else: test_data = ComPositionSet(data, test_gt, is_pred=True,**config)

    if config['SAMPLE_MODE'] == 'FIS': 
        train_loader = DataLoader(train_data,batch_size=config['BATCH_SIZE'],shuffle= False, num_workers=0)
        val_loader = DataLoader(val_data,batch_size=1,shuffle= False, num_workers=0)
        test_loader = DataLoader(test_data,batch_size=1,shuffle= False, num_workers=0)
    elif config['SAMPLE_MODE'] == 'PWS':
        train_loader = DataLoader(train_data,config['BATCH_SIZE'],shuffle= True, num_workers=0)
        val_loader = DataLoader(val_data,config['BATCH_SIZE'],shuffle= False, num_workers=0)
        test_loader = DataLoader(test_data,config['BATCH_SIZE'],shuffle= False, num_workers=0)
    elif config['SAMPLE_MODE'] in ['PPTR','SLS']:
        train_loader = DataLoader(train_data,config['BATCH_SIZE'],shuffle= True, num_workers=0)
        val_loader = DataLoader(val_data,1,shuffle= False, num_workers=0)
        test_loader = DataLoader(test_data,1,shuffle= False, num_workers=0)
    return train_loader, val_loader, test_loader

def get_model(**config):
    lr_scheduler = None
    if config['N_PCA'] == 0:
        config['N_PCA'] = config['DATA_BAND']
    ##SSRNet
    if config['MODEL'] == 'SSRNet':
        config['NORM'] = True
        if config['SAMPLE_MODE'] == None :  config['SAMPLE_MODE'] = 'PPTR'
        if config['LR'] == None :
            if config['SAMPLE_MODE'] == 'PPTR':
                config['LR'] = 0.01 if config['DATASET'] == 'IP' else 0.02 if config['DATASET'] == 'UP' else 0.04
            elif config['SAMPLE_MODE'] in ['PWS','SLS','FIS']: config['LR'] = 0.02
        if config['EPOCH'] == None :  
            if config['SAMPLE_MODE'] == 'PPTR':
                config['EPOCH'] = 80 if config['DATASET'] =='IP' else 60
            elif config['SAMPLE_MODE'] == 'FIS': config['EPOCH'] = 400
            elif config['SAMPLE_MODE'] in ['PWS','SLS']: config['EPOCH'] = 200
        if config['WEIGHT_DECAY'] == None:
            config['WEIGHT_DECAY'] = 1e-5
        if config['BATCH_SIZE'] == None :
            if config['SAMPLE_MODE'] == 'PPTR':
                config['BATCH_SIZE'] = 256 if config['DATASET'] =='IP' else 512 if config['DATASET'] == 'UP' else 1024
            elif config['SAMPLE_MODE'] == 'FIS': config['BATCH_SIZE'] = 1
            elif config['SAMPLE_MODE'] in ['PWS','SLS']: config['BATCH_SIZE'] = 256
        if config['PATCH_SIZE'] == None :
            config['PATCH_SIZE'] = 13 if config['DATASET'] in ['IP'] else 15
        if config['MODEL_MODE'] == None :  config['MODEL_MODE'] =  3 
        if config['PPTR_RATE'] == None : config['PPTR_RATE'] = 0.1 if config['DATASET'] in ['IP','UP'] else 0.2
        
        model = SSRNet(config['N_PCA'],config['PATCH_SIZE'],n_classes=config['NUM_CLASS'], 
                       ratio = 3, hid_layer=2,hid_num=69,aspp_rate=4,mode = config['MODEL_MODE'],act='leaky_relu',att_mode=config['ATT_MODE'])
        optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, model.parameters()),
                                     config['LR'],weight_decay=config['WEIGHT_DECAY'])
        # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode= 'min',factor=0.5,patience=config['EPOCH']//4,verbose=True)
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=config['EPOCH']//4,eta_min=config['LR']/4,verbose = True)
        lr_scheduler = 'Poly'
        criterion = FocalLoss(alpha=1, gamma=2, num_classes=config['NUM_CLASS'], ignore_index=-1,
                              size_average=True)
    
    elif config['MODEL'] == 'A2S2KResNet':
        if config['PATCH_SIZE'] == None :
            config['PATCH_SIZE'] = 7 if config['DATASET'] in ['IP','SA'] else 11
        if config['EPOCH'] == None :  config['EPOCH'] = 200
        if config['LR'] == None :  config['LR'] = 0.001
        if config['BATCH_SIZE'] == None :  config['BATCH_SIZE'] = 32
        if config['SAMPLE_MODE'] == None :  config['SAMPLE_MODE'] = 'PWS'
        if config['WEIGHT_DECAY'] == None :  config['WEIGHT_DECAY'] = 0
        config['NORM'] = True
        config['MODEL_MODE'] = 0
        model = S3KAIResNet(config['N_PCA'], config['NUM_CLASS'], 2)
        optimizer = torch.optim.Adam(model.parameters(),lr=config['LR'],betas=(0.9, 0.999),eps=1e-8,weight_decay=config['WEIGHT_DECAY'])
        criterion = nn.CrossEntropyLoss()
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 15, eta_min=0.0, last_epoch=-1)
    
    elif config['MODEL'] == 'HybridSN':
        if config['PATCH_SIZE'] == None :  config['PATCH_SIZE'] = 25
        if config['EPOCH'] == None :  config['EPOCH'] = 200
        if config['BATCH_SIZE'] == None :  config['BATCH_SIZE'] = 256
        if config['LR'] == None :  config['LR'] = 0.001
        if config['SAMPLE_MODE'] == None :  config['SAMPLE_MODE'] = 'PWS'
        if config['WEIGHT_DECAY'] == None :  config['WEIGHT_DECAY'] = 1e-6
        config['NORM'] = True
        config['MODEL_MODE'] = 0
        config['N_PCA'] = 30 if config['DATASET'] == 'IP' else 15
        model = HybridSN(config['N_PCA'],config['PATCH_SIZE'], config['NUM_CLASS'])
        optimizer = torch.optim.Adam(model.parameters(), config['LR'], weight_decay=config['WEIGHT_DECAY'])
        criterion = nn.CrossEntropyLoss()
    
    elif config['MODEL'] == 'SFormer_px':
        if config['PATCH_SIZE'] == None :  config['PATCH_SIZE'] = 1
        if config['EPOCH'] == None :  
            config['EPOCH'] = 290 if config['DATASET'] == 'IP' else 500 if config['DATASET'] == 'UP' else 520
        if config['BATCH_SIZE'] == None :  config['BATCH_SIZE'] = 64
        if config['LR'] == None :  config['LR'] = 5e-4
        if config['SAMPLE_MODE'] == None :  config['SAMPLE_MODE'] = 'PWS'
        if config['WEIGHT_DECAY'] == None :  
            config['WEIGHT_DECAY'] = 0 if config['DATASET'] == 'IP' else 5e-3
        config['NORM'] = True
        config['MODEL_MODE'] = 0
        config['BAND_PATCH'] = 3
        model = ViT(image_size = config['PATCH_SIZE'],near_band = config['BAND_PATCH'], 
                    num_patches = config['N_PCA'], num_classes = config['NUM_CLASS'], dim = 64,
                    depth = 5,heads = 4,mlp_dim = 8,dropout = 0.1,emb_dropout = 0.1,mode ='CAF')
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config['LR'], weight_decay=config['WEIGHT_DECAY'])
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size= config['EPOCH']//10, gamma=0.9)
    
    elif config['MODEL'] == 'SFormer_pt':
        if config['PATCH_SIZE'] == None :  config['PATCH_SIZE'] = 7
        if config['EPOCH'] == None :  
            config['EPOCH'] = 300 if config['DATASET'] == 'IP' else 480 if config['DATASET'] == 'UP' else 600
        if config['BATCH_SIZE'] == None :  config['BATCH_SIZE'] = 64
        if config['LR'] == None :  config['LR'] = 5e-4
        if config['SAMPLE_MODE'] == None :  config['SAMPLE_MODE'] = 'PWS'
        if config['WEIGHT_DECAY'] == None :  config['WEIGHT_DECAY'] = 5e-3
        config['NORM'] = True
        config['MODEL_MODE'] = 0
        config['BAND_PATCH'] = 7 if config['DATASET'] == 'UP' else 3
        model = ViT(image_size = config['PATCH_SIZE'],near_band = config['BAND_PATCH'], 
                    num_patches = config['N_PCA'], num_classes = config['NUM_CLASS'], dim = 64,
                    depth = 5,heads = 4,mlp_dim = 8,dropout = 0.1,emb_dropout = 0.1,mode ='CAF')
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config['LR'], weight_decay=config['WEIGHT_DECAY'])
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size= config['EPOCH']//10, gamma=0.9)    
    
    elif config['MODEL'] == 'VIT':
        if config['PATCH_SIZE'] == None :  config['PATCH_SIZE'] = 1
        if config['EPOCH'] == None :  
            config['EPOCH'] = 1400 if config['DATASET'] == 'IP' else 1000 if config['DATASET'] == 'UP' else 900
        if config['BATCH_SIZE'] == None :  config['BATCH_SIZE'] = 64
        if config['LR'] == None :  config['LR'] = 5e-4
        if config['SAMPLE_MODE'] == None :  config['SAMPLE_MODE'] = 'PWS'
        if config['WEIGHT_DECAY'] == None :  config['WEIGHT_DECAY'] = 0
        config['NORM'] = True
        config['MODEL_MODE'] = 0
        config['BAND_PATCH'] = 1
        model = ViT(image_size = config['PATCH_SIZE'],near_band = config['BAND_PATCH'], 
                    num_patches = config['N_PCA'], num_classes = config['NUM_CLASS'], dim = 64,
                    depth = 5,heads = 4,mlp_dim = 8,dropout = 0.1,emb_dropout = 0.1,mode ='CAF')
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config['LR'], weight_decay=config['WEIGHT_DECAY'])
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size= config['EPOCH']//10, gamma=0.9)    
    
    elif config['MODEL'] == 'FullyContNet':
        config['NORM'] = True
        config['MODEL_MODE'] = 0
        if config['LR'] == None : config['LR'] = 0.01
        if config['WEIGHT_DECAY'] == None : config['WEIGHT_DECAY'] = 0.0001
        if config['BATCH_SIZE'] == None : config['BATCH_SIZE'] = 1
        if config['EPOCH'] == None :  config['EPOCH'] = 1000
        if config['SAMPLE_MODE'] == None :  config['SAMPLE_MODE'] = 'FIS'
        class tmp(): pass
        args = tmp()
        args.network = 'FContNet'
        args.head = 'aspp'
        args.mode = 'p_c_s'
        args.input_size = config['DATA_SIZE'][:-1]
        args.network = 'FContNet'
        model = fucontnet(args,config['N_PCA'],config['NUM_CLASS'])
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
        optimizer = torch.optim.SGD(model.parameters(), momentum=0.9,
                                    lr=config['LR'], weight_decay=config['WEIGHT_DECAY'])
        lr_scheduler = 'Poly'

    elif config['MODEL'] == 'UNet':
        config['NORM'] = True
        config['N_PCA'] = 3
        config['MODEL_MODE'] = 0
        if config['LR'] == None : config['LR'] = 0.01
        if config['WEIGHT_DECAY'] == None : config['WEIGHT_DECAY'] = 0.0001
        if config['BATCH_SIZE'] == None : config['BATCH_SIZE'] = 1
        if config['EPOCH'] == None :  config['EPOCH'] = 400
        if config['SAMPLE_MODE'] == None :  config['SAMPLE_MODE'] = 'FIS'
        model = UNet(config['N_PCA'],config['NUM_CLASS'])
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
        optimizer = torch.optim.SGD(model.parameters(), momentum=0.9,
                                    lr=config['LR'], weight_decay=config['WEIGHT_DECAY'])
        lr_scheduler = 'Poly'
    
    elif config['MODEL'] in ['MLWBDN-L1','MLWBDN-L2','MLWBDN-L3']:
        if config['EPOCH'] == None :  config['EPOCH'] = 300
        if config['BATCH_SIZE'] == None :  config['BATCH_SIZE'] = 104 if config['DATASET'] == 'IP' else 225 if config['DATASET'] == 'UP' else 106
        if config['LR'] == None :  config['LR'] = 0.005 if config['DATASET'] == 'IP' else 0.01
        if config['SAMPLE_MODE'] == None :  config['SAMPLE_MODE'] = 'PWS'
        if config['WEIGHT_DECAY'] == None :  config['WEIGHT_DECAY'] = 1e-5
        if config['MODEL'] == 'MLWBDN-L1': 
            if config['PATCH_SIZE'] == None: config['PATCH_SIZE'] = 10
            block_num = 1
        elif config['MODEL'] == 'MLWBDN-L2': 
            if config['PATCH_SIZE'] == None: config['PATCH_SIZE'] = 16
            block_num = 2
        elif config['MODEL'] == 'MLWBDN-L3': 
            if config['PATCH_SIZE'] == None: config['PATCH_SIZE'] = 24
            block_num = 3
        cmp_rate = 0.1 if config['DATASET'] == 'UP' or 'IP' else 0.2
        config['TEST_BATCH'] = 512
        config['NORM'] = True
        config['MODEL_MODE'] = 0
        config['N_PCA'] = 15
        model = MLWBDN(config['N_PCA'],config['NUM_CLASS'],block_num=block_num,cmp_rate= cmp_rate,growth_rate=36)
        optimizer = torch.optim.Adam(model.parameters(), config['LR'], weight_decay=config['WEIGHT_DECAY'])
        criterion = nn.CrossEntropyLoss()
        # lr_scheduler = 'Poly'
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode= 'min',factor=0.5,patience=config['EPOCH']//4,verbose=True)

    elif config['MODEL'] == 'LCA-FCN':
        config['NORM'] = True
        config['MODEL_MODE'] = 0
        if config['LR'] == None : config['LR'] = 0.01
        if config['WEIGHT_DECAY'] == None : config['WEIGHT_DECAY'] = 1e-5
        if config['BATCH_SIZE'] == None : config['BATCH_SIZE'] = 1
        if config['EPOCH'] == None :  config['EPOCH'] = 300
        if config['SAMPLE_MODE'] == None :  config['SAMPLE_MODE'] = 'FIS'
        if config['PATCH_SIZE'] == None :
            config['PATCH_SIZE'] = 13 if config['DATASET'] in ['IP'] else 15
        model = LCA_FCN(config['N_PCA'],config['PATCH_SIZE'],config['NUM_CLASS'],ratio = 3, hid_layer=3,hid_num=69,act='leaky_relu',att_mode=config['ATT_MODE'])
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
        optimizer = torch.optim.SGD(model.parameters(), lr=config['LR'], momentum=0.9,weight_decay=config['WEIGHT_DECAY'])
        # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode= 'min',factor=0.5,patience=config['EPOCH']//4,verbose=True)

    elif config['MODEL'] == 'GhostNet':
        config['NORM'] = True
        config['MODEL_MODE'] = 0
        if config['PATCH_SIZE'] == None: config['PATCH_SIZE'] = 25
        if config['LR'] == None : config['LR'] = 0.01
        if config['WEIGHT_DECAY'] == None : config['WEIGHT_DECAY'] =1e-6
        if config['BATCH_SIZE'] == None : config['BATCH_SIZE'] = 256
        if config['EPOCH'] == None :  config['EPOCH'] = 200
        if config['SAMPLE_MODE'] == None :  config['SAMPLE_MODE'] = 'PWS'
        model = ghostnet(in_chs = config['N_PCA'],num_classes = config['NUM_CLASS'])
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
        optimizer = torch.optim.Adam(model.parameters(),lr=config['LR'],weight_decay=config['WEIGHT_DECAY'])
        # lr_scheduler = 'Poly'
    
    elif config['MODEL'] == 'MobileNet':
        config['NORM'] = True
        config['MODEL_MODE'] = 0
        if config['PATCH_SIZE'] == None: config['PATCH_SIZE'] = 25
        if config['LR'] == None : config['LR'] = 0.01
        if config['WEIGHT_DECAY'] == None : config['WEIGHT_DECAY'] =1e-5
        if config['BATCH_SIZE'] == None : config['BATCH_SIZE'] = 256
        if config['EPOCH'] == None :  config['EPOCH'] = 200
        if config['SAMPLE_MODE'] == None :  config['SAMPLE_MODE'] = 'PWS'
        model = MobileNetV1(in_chs = config['N_PCA'],num_classes = config['NUM_CLASS'])
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
        optimizer = torch.optim.Adam(model.parameters(),lr=config['LR'],weight_decay=config['WEIGHT_DECAY'])
        # lr_scheduler = 'Poly'


    elif config['MODEL'] == 'M3DCNN':
        config['NORM'] = True
        config['MODEL_MODE'] = 0
        if config['PATCH_SIZE'] == None: config['PATCH_SIZE'] = 7
        if config['LR'] == None : config['LR'] = 0.01
        if config['WEIGHT_DECAY'] == None : config['WEIGHT_DECAY'] =0.01
        if config['BATCH_SIZE'] == None : config['BATCH_SIZE'] = 40
        if config['EPOCH'] == None :  config['EPOCH'] = 200
        if config['SAMPLE_MODE'] == None :  config['SAMPLE_MODE'] = 'PWS'
        model = MobileNetV1(in_chs = config['N_PCA'],num_classes = config['NUM_CLASS'])
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
        optimizer = torch.optim.Adagrad(model.parameters(),lr=config['LR'],weight_decay=config['WEIGHT_DECAY'])

    elif config['MODEL'] == 'SSUN':
        config['TIME_STEP'] = 3
        config['N_PCA'] = 4
        if config['PATCH_SIZE'] == None :
            config['PATCH_SIZE'] = 28
        if config['EPOCH'] == None :  config['EPOCH'] = 500
        if config['LR'] == None :  config['LR'] = 0.001
        if config['BATCH_SIZE'] == None :  config['BATCH_SIZE'] = 64
        if config['SAMPLE_MODE'] == None :  config['SAMPLE_MODE'] = 'PWS'
        config['NORM'] = True
        config['MODEL_MODE'] = 2
        model = SSUN(config['TIME_STEP'],int(config['DATA_BAND']/config['TIME_STEP']),config['N_PCA'],config['PATCH_SIZE'],config['PATCH_SIZE'], config['NUM_CLASS'])
        optimizer = torch.optim.Adam(model.parameters(),lr=config['LR'])
        criterion = nn.CrossEntropyLoss()
    

    if config['PATCH_SIZE'] == None :  config['PATCH_SIZE'] = 0
    if config['TEST_BATCH'] == None: config['TEST_BATCH'] = config['BATCH_SIZE']
    return model, optimizer, lr_scheduler, criterion, config

## get best model path and del other models
def get_best_model(acc_list, epoch_list, val_loss_list, save_path, del_others = True,loss_first = False) -> str:
    """get best model path by valuation list
    Args:
        acc_list (list): list of valuation accuracy
        epoch_list (list): list of valuation epoch
        val_loss_list(list): list of valuation loss
        save_path (str): path of save dir
        del_others (bool): whether delete all ckpt except best model
        loss_first (bool): determine the best model by the lowest val loss
    Returns:
        best_model_path: path of best model
    """
    acc_list = np.array(acc_list)
    epoch_list = np.array(epoch_list)
    acc_list[:len(acc_list)//3] = 0
    if loss_first:
        val_loss_list = np.array(val_loss_list)
        best_index = np.argwhere(val_loss_list==np.min(val_loss_list)).flatten()[-1]
    else:
        best_index = np.argwhere(acc_list==np.max(acc_list)).flatten()
        if best_index.size == 1:
            best_index = best_index.item()
        else:
            best_index = val_loss_list.index(min([val_loss_list[i] for i in best_index]))
    best_epoch = epoch_list[best_index]
    best_acc = acc_list[best_index]
    file_name = f"epoch_{best_epoch}_acc_{best_acc:.4f}.pth"
    best_model_path=os.path.join(save_path, file_name)
    print(f"best model:{file_name}")
    ##del save model except best model
    if del_others:
        for f in os.listdir(save_path):
            if f[-3:]=='pth' and os.path.join(save_path,f)!=best_model_path:
                os.remove(os.path.join(save_path,f))
    return best_model_path,file_name


def val_one_epoch(model,criterion, val_loader, device, **config):
    model.to(device)
    model.eval()
    val_num = val_loader.dataset.__len__()
    OA = [ 0 for i in range(config['MODEL_MODE']+1)]
    val_loss = [ 0 for i in range(config['MODEL_MODE']+1)]
    label_num = 0
    with torch.no_grad() :
        for idx,(data_t, target) in enumerate(val_loader):
            if config['MODEL'] == 'SSRNet' and config['SAMPLE_MODE'] == 'PWS':
                    tmp = torch.zeros((data_t.shape[0],data_t.shape[2],data_t.shape[3]),dtype=torch.int64)
                    tmp[:,config['PATCH_SIZE']//2,config['PATCH_SIZE']//2] = target
                    target = tmp.to(device)
                    data_t = data_t.to(device)
            elif config['MODEL'] == 'SSUN':
                    data_t[0],data_t[1] = data_t[0].to(device),data_t[1].to(device)
                    target = target.to(device)
            else: data_t,target = data_t.to(device),target.to(device)
            val_t = np.asarray((target).squeeze(0).detach().cpu(),dtype=np.uint8)
            label_num += (val_t!=0).sum()
            target -= 1
            if config['MODEL'] == 'SSUN': out = model(*data_t)
            else: out = model(data_t)
            if not isinstance(out,list):out = [out]
            for i in range(len(out)):
                val_loss[i] += criterion(out[i],target).item()
                _, pred_map = torch.max(out[i],dim=1)
                pred_map = np.asarray((pred_map+1).squeeze(0).detach().cpu(),dtype=np.uint8)
                OA[i] += (pred_map[val_t!=0] == val_t[val_t!=0]).sum()
    return [oa / label_num for oa in OA], [loss / val_num for loss in val_loss]


def train(model, train_loader, val_loader, criterion, optimizer, lr_scheduler = None, viz = None, config = None):
    device = torch.device(config['DEVICE'] if config['DEVICE']>=0 and torch.cuda.is_available() else 'cpu')
    loss_list = [[] for i in range(config['MODEL_MODE']+1)]
    acc_list = [[] for i in range(config['MODEL_MODE']+1)]
    val_acc_list = [[] for i in range(config['MODEL_MODE']+1)]
    val_loss_list = [[] for i in range(config['MODEL_MODE']+2)]
    val_epoch_list = []
    epoch_start = 0
    model.to(device)
    train_num = train_loader.dataset.__len__()
    # loss_win, acc_win, batch_win = 'loss_win', 'loss_win', 'loss_win'
    if config['AMP']:
        scaler = GradScaler()
    
    if config['CHECK_POINT'] is not None:
        model_ckpt = torch.load(config['CHECK_POINT'],map_location = device)
        model.load_state_dict(model_ckpt['state_dict'])
        epoch_start = model_ckpt['epoch']
        optimizer.load_state_dict(model_ckpt['optimizer'])
        loss_list = model_ckpt['loss_list']
        acc_list = model_ckpt['acc_list']
        val_acc_list = model_ckpt['val_acc_list']
        val_epoch_list = model_ckpt['val_epoch_list']
    # torch.cuda.synchronize()
    train_st = time.time()
    e = 0
    try:
        for e in tqdm(range(epoch_start,config['EPOCH']), desc="Training"):
            model.train()
            epoch_st = time.time()
            avg_loss = [ 0 for i in range(config['MODEL_MODE']+1)]
            train_acc = [ 0 for i in range(config['MODEL_MODE']+1)]
            label_sum = 0
            for batch_idx, (data, target) in tqdm(enumerate(train_loader),desc = 'Batch', total= len(train_loader)):
                # if len(train_loader)!= 1 and data.shape[0] < config['BATCH_SIZE']*0.8: continue
                if config['MODEL'] == 'SSRNet' and config['SAMPLE_MODE'] == 'PWS':
                    tmp = torch.zeros((data.shape[0],data.shape[2],data.shape[3]),dtype=torch.int64)
                    tmp[:,config['PATCH_SIZE']//2,config['PATCH_SIZE']//2] = target
                    target = tmp.to(device)
                    data = data.to(device)
                elif config['MODEL'] == 'SSUN':
                    data[0],data[1] = data[0].to(device),data[1].to(device)
                    target = target.to(device)
                else: data,target = data.to(device),target.to(device)
                optimizer.zero_grad()
                if config['AMP']:
                    with autocast():
                        if config['MODEL'] == 'SSUN': out = model(*data)
                        else: out = model(data)
                        index = target!=0
                        label_sum += index.sum().sum().item()
                        target -= 1
                        losses = []
                        loss = 0
                        if not isinstance(out,list): out = [out]
                        for i in range(len(out)):
                            losses.append(criterion(out[i],target))
                            avg_loss[i] += losses[i].item()
                            train_acc[i] += (torch.max(out[i],dim=1)[-1][index] == target[index]).sum().item()
                        for i in range(len(losses)):
                            loss += losses[i]
                        viz.line( Y = [loss.item()],X = [e*len(train_loader)+batch_idx],win = 'batch_win', name = 'train_loss', opts = dict(title = 'loss_per_batch', xlabel = 'iters',ylabel = 'loss', showlegend = True),update = 'append')
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if config['MODEL'] == 'SSUN': out = model(*data)
                    else: out = model(data)
                    index = target!=0
                    label_sum += index.sum().sum().item()
                    target -= 1
                    losses = []
                    loss = 0
                    if not isinstance(out,list): out = [out]
                    for i in range(len(out)):
                        losses.append(criterion(out[i],target))
                        avg_loss[i] += losses[i].item()
                        train_acc[i] += (torch.max(out[i],dim=1)[-1][index] == target[index]).sum().item()
                    for i in range(len(losses)):
                        loss += losses[i]
                    viz.line( Y = [loss.item()],X = [e*len(train_loader)+batch_idx],win = 'batch_win', name = 'train_loss', opts = dict(title = 'loss_per_batch', xlabel = 'iters',ylabel = 'loss', showlegend = True),update = 'append')
                    loss.backward()
                    optimizer.step()

            if isinstance(lr_scheduler,torch.optim.lr_scheduler.ReduceLROnPlateau):
                pass
            elif lr_scheduler == 'Poly':
                for param_group in optimizer.param_groups:
                    param_group['lr'] = config['LR']*(1-(e+1)/config['EPOCH'])**0.9
            elif lr_scheduler!= None: 
                lr_scheduler.step()
                
            for i in range(config['MODEL_MODE']+1):
                loss_list[i].append(avg_loss[i]/train_num)
                acc_list[i].append(train_acc[i]/label_sum)
                if viz!=None:
                    viz.line(Y = [loss_list[i][e]], X = [e+1], win = 'loss_win', name = 'train_loss%d'%i, opts = dict(title = 'loss_per_epoch', xlabel = 'epoch',ylabel = 'loss', showlegend = True), update =  'append')
                    if e == 0:
                        viz.line(Y = [0,acc_list[i][e]], X = [0,1], win = 'acc_win', name = 'train_acc%d'%i, opts = dict(title = 'acc_per_epoch', xlabel = 'epoch',ylabel = 'acc', showlegend = True), update = 'append')
                        viz.line(Y = [0], X = [0], win = 'acc_win', name = 'val_acc_%d'%i,opts = dict(title = 'acc_per_epoch', xlabel = 'epoch',ylabel = 'acc', showlegend = True),  update ='append')
                    else:
                        viz.line(Y = [acc_list[i][e]], X = [e+1], win = 'acc_win', name = 'train_acc%d'%i, update = 'append')
            ep_t = time.time() - epoch_st
            log_str = [f"training {e}/{config['EPOCH']}:\n"]
            log_str += ['{:<18}'.format(f"Acc_{i}:{acc_list[i][e]:.4f}") for i in range(config['MODEL_MODE']+1)]
            log_str += ['{:<18}'.format(f"Loss_{i}:{loss_list[i][e]:.2e}") for i in range(config['MODEL_MODE']+1)]
            log_str += ['{:<18}'.format(f"LR:{optimizer.param_groups[0]['lr']:.4e}")]
            log_str += ['{:<18}'.format(f"Time:{ep_t:.2f}")]
            print(''.join(log_str))
            ## valuation
            if (e+1)%config['VAL_EPOCH'] == 0 or (e+1)==config['EPOCH']:
                val_oa, val_loss = val_one_epoch(model, criterion, val_loader, device, **config)
                torch.cuda.empty_cache()
                log_str = [f"val {e}/{config['EPOCH']}:\n"]
                val_epoch_list.append(e)
                if isinstance(lr_scheduler,torch.optim.lr_scheduler.ReduceLROnPlateau):
                    metric = -1*val_oa[-1]
                    lr_scheduler.step(metric)
                tmp = 0
                for i in range(len(val_oa)):
                    val_acc_list[i].append(val_oa[i])
                    val_loss_list[i].append(val_loss[i])
                    tmp += val_loss[i]
                val_loss_list[-1].append(tmp)
                if viz!=None:
                    viz.line(Y = [val_acc_list[i][-1]], X = [e+1], win = 'acc_win', name = 'val_acc_%d'%i,  update ='append')
                    viz.line(Y = [val_loss_list[-2][-1]], X = [e+1],  win = 'loss_win', name = 'val_loss', update = 'append')
                    viz.line(Y = [val_loss_list[-1][-1]], X = [e+1],  win = 'loss_win', name = 'sum_val_loss', update = 'append')
                log_str += ['{:<20}'.format(f"ValAcc_{i}:{val_acc_list[i][-1]:.4f}") for i in range(config['MODEL_MODE']+1)]
                log_str += ['{:<20}'.format(f"ValLoss_{i}:{val_loss_list[i][-1]:.2e}") for i in range(config['MODEL_MODE']+1)]
                log_str += f"ValLoss_sum:{val_loss_list[-1][-1]:.2e}"
                print(''.join(log_str))
                infer_acc = []
                for i in range(len(val_acc_list[0])):
                    tmp = 0
                    for j in range(len(val_acc_list)):
                        tmp += val_acc_list[j][i]
                    infer_acc.append(tmp/len(val_acc_list))
                save_name = os.path.join(config['SAVE_PATH'], f"epoch_{e}_acc_{val_acc_list[-1][-1]:.4f}.pth")
                # save_name = os.path.join(config['SAVE_PATH'], f"epoch_{e}_acc_{infer_acc[-1]:.4f}.pth")
                save_dict = {'state_dict':model.state_dict(), 'epoch':e+1, 'optimizer': optimizer.state_dict(),
                            'loss_list':loss_list, 'acc_list': acc_list, 'val_acc_list': val_acc_list, 'val_epoch_list':val_epoch_list}
                torch.save(save_dict,save_name)
                
    except KeyboardInterrupt:
        print(f'KeyboardInterrupt in epoch {e}')
        e -= 1
    
    finally: 
        print(f'Stop in epoch {e}')
    train_time = time.time()-train_st
    config['train_time'] = train_time
    print(f"training time: {train_time}")
    ##display loss and acc
    fig1 = plt.figure()
    fig2 = plt.figure()
    ax1 = fig1.add_subplot(1,1,1)
    ax2 = fig2.add_subplot(1,1,1)
    for i in range(config['MODEL_MODE']+1):
        ax1.plot(np.arange(e+1),loss_list[i][:e+1],label = f'loss_{i}')
        ax1.plot(val_epoch_list,val_loss_list[i][:e+1],label = f'val_loss_{i}')
        ax2.plot(np.arange(e+1),acc_list[i][:e+1],label = f'train_acc_{i}')
        ax2.plot(val_epoch_list,val_acc_list[i][:e+1],label = f'val_acc_{i}')
    ax1.set_title(f'loss')
    ax1.set_xlabel('epoch')
    ax1.legend()
    ax2.set_title(f'acc')
    ax2.set_xlabel('epoch')
    ax2.legend()
    fig1.savefig(os.path.join(config['SAVE_PATH'],'loss.png'))
    fig2.savefig(os.path.join(config['SAVE_PATH'],'acc.png'))
    
    best_model_path,model_file_name = get_best_model(val_acc_list[-1],val_epoch_list,val_loss_list[-1],config['SAVE_PATH'],loss_first=False)
    # best_model_path,model_file_name = get_best_model(infer_acc,val_epoch_list,val_loss_list[-2],config['SAVE_PATH'],loss_first=False)
    if viz!= None:
        _ = write_log(f"best model:{model_file_name}",None,viz,'log')
    return best_model_path


def test(model: nn.Module, out_num: int, label: np.ndarray, test_gt: np.ndarray, test_loader: DataLoader, viz: Visdom, **config):
    
    device = torch.device(config['DEVICE'] if config['DEVICE']>=0 and torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    infer_st = time.time()
    with torch.no_grad():
        if config['SAMPLE_MODE'] != 'PWS':
            idx, (data_t, target) = next(enumerate(test_loader))
            data_t = data_t.to(device)
            out = model(data_t)
            if not isinstance(out,list): out = [out]
            _, pred = torch.max(out[out_num], dim = 1)
            pred = (pred+1).squeeze(0).detach().cpu()
            pred_map = np.asarray(pred,dtype=np.uint8)
        else:
            pred_map = []
            for idx,(data_t,_) in tqdm(enumerate(test_loader),desc = 'Batch', total= len(test_loader)):
                if config['MODEL'] == 'SSUN':
                    data_t[0],data_t[1] = data_t[0].to(device),data_t[1].to(device)
                    out = model(*data_t)
                else:
                    data_t = data_t.to(device)
                    out = model(data_t)
                if not isinstance(out,list): out = [out]
                _, pred = torch.max(out[out_num], dim = 1)
                if config['MODEL'] == 'SSRNet':
                    pred_map += [np.array(pred[:,config['PATCH_SIZE']//2,config['PATCH_SIZE']//2].detach().cpu() + 1)]
                else:
                    pred_map += [np.array(pred.detach().cpu() + 1)]
            pred_map = np.asarray(np.hstack(pred_map),dtype=np.uint8).reshape(label.shape[0],label.shape[1])
    infer_time = time.time() - infer_st
    
    ## classfication report
    test_pred = pred_map[test_gt!=0]
    test_true = test_gt[test_gt!=0]
    OA = accuracy_score(test_true,test_pred)
    AA = recall_score(test_true,test_pred,average='macro')
    kappa = cohen_kappa_score(test_true,test_pred)
    report_log = F"OA: {OA}\nAA: {AA}\nKappa: {kappa}\nInfer_time:{infer_time}\n"
    
    if viz!= None:
        img_display(classes=pred_map,title=f'prediction_{out_num}_{OA:.4f}',viz = viz, savepath= config['SAVE_PATH'])
        img_display(classes=pred_map*(label!=0),title=f'prediction_masked_{out_num}_{OA:.4f}',viz = viz,savepath= config['SAVE_PATH'])
    spectral.save_rgb(os.path.join(config['SAVE_PATH'],f"prediction_{out_num}_{OA:.4f}.png"),pred_map,colors = spectral.spy_colors)
    spectral.save_rgb(os.path.join(config['SAVE_PATH'],f"prediction_masked_{out_num}_{OA:.4f}.png"),pred_map*(label!=0),colors = spectral.spy_colors)
    
    report_log += classification_report(test_true,test_pred,target_names=config['CLASS_NAME'],digits=4)
    print(report_log)
    
    fp = open(os.path.join(config['SAVE_PATH'],'classfication_report.txt'),'a+')
    fp.writelines(report_log)
    fp.close()

    cr = classification_report(test_true,test_pred,target_names=config['CLASS_NAME'],digits=4,output_dict=True)
    res = []
    idx_name = []
    for cn in config['CLASS_NAME']:
        res.append(cr[cn]['recall'])
        idx_name += [cn]
    res += [accuracy_score(test_true,test_pred),recall_score(test_true,test_pred,average='macro'),
        cohen_kappa_score(test_true,test_pred),config['train_time'],infer_time]
    idx_name += ['OA','AA','KAPPA','train_time','infer_time']
    return pd.Series(res,index=idx_name), report_log


def gain_neighborhood_band(x_train, band_patch, patch=7): ## [C P P]
    nn = band_patch // 2
    pp = (patch*patch) // 2
    band = x_train.shape[0]
    x_train = np.array(x_train)
    x_train_reshape = x_train.transpose((1,2,0)).reshape(patch*patch, band)
    x_train_band = np.zeros((patch*patch*band_patch, band),dtype=np.float32)
    # 中心区域
    x_train_band[nn*patch*patch:(nn+1)*patch*patch,:] = x_train_reshape
    #左边镜像
    for i in range(nn):
        if pp > 0:
            x_train_band[i*patch*patch:(i+1)*patch*patch,:i+1] = x_train_reshape[:,band-i-1:]
            x_train_band[i*patch*patch:(i+1)*patch*patch,i+1:] = x_train_reshape[:,:band-i-1]
        else:
            x_train_band[i:(i+1),:(nn-i)] = x_train_reshape[0:1,(band-nn+i):]
            x_train_band[i:(i+1),(nn-i):] = x_train_reshape[0:1,:(band-nn+i)]
    #右边镜像
    for i in range(nn):
        if pp > 0:
            x_train_band[(nn+i+1)*patch*patch:(nn+i+2)*patch*patch,:band-i-1] = x_train_reshape[:,i+1:]
            x_train_band[(nn+i+1)*patch*patch:(nn+i+2)*patch*patch,band-i-1:] = x_train_reshape[:,:i+1]
        else:
            x_train_band[(nn+1+i):(nn+2+i),(band-i-1):] = x_train_reshape[0:1,:(i+1)]
            x_train_band[(nn+1+i):(nn+2+i),:(band-i-1)] = x_train_reshape[0:1,(i+1):]
    return torch.from_numpy(x_train_band.transpose((1,0))).type(torch.FloatTensor)   

class ComPositionSet(Dataset):
    def __init__(self, data, sample_gt, is_pred = False, **config):
        super(ComPositionSet, self).__init__()
        self.sample_mode = config['SAMPLE_MODE']
        self.is_pred = is_pred
        self.is_bandpatch = False
        self.patch_size = config['PATCH_SIZE']
        self.pptr_rate = config['PPTR_RATE']
        if config['MODEL'] in ['SFormer_px','SFormer_pt','VIT']:
            self.p = self.patch_size // 2
            self.is_bandpatch = True
            self.band_patch = config['BAND_PATCH']
        self.sample_gt = np.asarray(sample_gt,dtype='int64')
        mask = np.zeros_like(sample_gt)
        if (self.sample_mode == 'FIS') or ((self.sample_mode in ['PPTR','SLS']) and is_pred):
            h,w,_ = data.shape
            # if half_img:
            #     self.data = [torch.from_numpy(np.asarray(data[:h//2+1,:w//2+1,:], dtype='float32').transpose((2, 0, 1))),torch.from_numpy(np.asarray(data[h//2:2*(h//2)+1,w//2:2*(w//2)+1,:], dtype='float32').transpose((2, 0, 1)))]
            #     self.sample_gt = [torch.from_numpy(self.sample_gt[:h//2+1,:w//2+1]),torch.from_numpy(self.sample_gt[h//2:2*(h//2)+1,w//2:2*(w//2)+1])]
            
            self.data = [torch.from_numpy(np.asarray(data, dtype='float32').transpose((2, 0, 1)))]
            self.sample_gt = [torch.from_numpy(self.sample_gt)]
        elif self.sample_mode in ['PWS','SLS']:
            self.p = self.patch_size // 2
            self.data = np.pad(data,((self.p,self.p),(self.p,self.p),(0,0)),'symmetric')
            self.pad_sample_gt = np.pad(self.sample_gt,(self.p,self.p),'constant',constant_values = 0)
            if is_pred: 
                x_pos, y_pos = np.nonzero(np.ones_like(sample_gt))
                x_pos, y_pos = x_pos + self.p, y_pos + self.p
            else:x_pos, y_pos = np.nonzero(self.pad_sample_gt)   ##indices after padding
            self.indices = np.array([(x,y) for x,y in zip(x_pos, y_pos)])
            if not is_pred:
                np.random.shuffle(self.indices)
            self.data = torch.from_numpy(np.asarray(self.data, dtype='float32').transpose((2, 0, 1)))
            self.sample_gt = torch.from_numpy(self.sample_gt)
            self.pad_sample_gt = torch.from_numpy(np.asarray(self.pad_sample_gt,dtype='int64'))
            
        elif self.sample_mode == 'PPTR':
            self.p = self.patch_size // 2
            self.data = torch.from_numpy(np.asarray(data, dtype='float32').transpose((2, 0, 1)))
            self.sample_gt = torch.from_numpy(self.sample_gt)
            h,w = self.sample_gt.shape
            # mask[(patch_size-1):(1-patch_size),(patch_size-1):(1-patch_size)] = 1
            x_pos, y_pos = np.nonzero(sample_gt)
            indice = []
            for x,y in zip(x_pos, y_pos):
                indice += self.get_center_shift_list(x,y,h,w)
            # indice += self.get_border_list()
            self.indices = np.array(list(set(indice)))
            np.random.shuffle(self.indices)

    def __len__(self):
        if self.sample_mode == 'FIS' or (self.sample_mode in ['PPTR','SLS'] and self.is_pred):
            return len(self.data)
        return len(self.indices)
    
    def get_center_shift_list(self, x, y, x_max, y_max):
        
        if x-self.p < self.p: x_low = self.p
        else:x_low = x-self.p
        if x+self.p > x_max-1-self.p: x_high = x_max-1-self.p
        else:x_high = x+self.p
        if y-self.p < self.p: y_low = self.p
        else:y_low = y-self.p
        if y+self.p > y_max-1-self.p: y_high = y_max-1-self.p
        else:y_high = y+self.p
        
        x_ = np.arange(x_low,x_high+1,1)
        y_ = np.arange(y_low,y_high+1,1)
        xv,yv = np.meshgrid(x_,y_)
        ps = [i for i in zip(xv.flat,yv.flat)]
        n = round(len(ps)*self.pptr_rate)
        return random.sample(ps,n if n>0 else 1)
    
    # def get_border_list(self):
    #     sg = np.array(self.sample_gt)
    #     mask_ = np.zeros_like(sg)
    #     mask_[(self.p):(-1*self.p),(self.p):(-1*self.p)] = 1  
    #     mask_[(self.patch_size-1+self.p):(1-self.patch_size-self.p),(self.patch_size-1+self.p):(1-self.patch_size-self.p)] = 0  ##border center position
    #     x_, y_ = np.nonzero(mask_)
    #     res = []
    #     for x, y in zip(x_,y_):
    #         tmp = sg[(x-self.p):(x+self.p+1),(y-self.p):(y+self.p+1)]
    #         if np.count_nonzero(tmp)>0:
    #             res.append((x,y))
    #     return res
        # return random.sample(res,round(len(res)*self.pptr_rate))
    
    def __getitem__(self, i):
        if self.sample_mode == 'FIS' or (self.sample_mode in ['PPTR','SLS'] and self.is_pred):
            return self.data[i], self.sample_gt[i]
        else:
            x, y = self.indices[i]
            x1, y1 = x - self.p, y - self.p
            x2, y2 = x1 + self.patch_size, y1 + self.patch_size
            if self.sample_mode == 'PWS':
                if self.is_bandpatch:
                    return gain_neighborhood_band(self.data[:, x1:x2, y1:y2],self.band_patch,self.patch_size),self.pad_sample_gt[x,y]
                if self.is_pred: return self.data[:, x1:x2, y1:y2], self.sample_gt
                return self.data[:, x1:x2, y1:y2], self.pad_sample_gt[x,y]
            elif self.sample_mode == 'SLS':
                return self.data[:, x1:x2, y1:y2], self.pad_sample_gt[x1:x2, y1:y2]
            elif self.sample_mode == 'PPTR':
                return self.data[:, x1:x2, y1:y2], self.sample_gt[x1:x2, y1:y2]

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, num_classes = 9, size_average=False, ignore_index = -1):
        """ Focal_Loss= -1*alpha*(1-pt)^gamma*log(pt)
        Args:
            alpha (int, optional): weight for balance of sample. Defaults to 1.
            gamma (int, optional): weight for balance of difficulty. Defaults to 2.
            num_classes (int, optional): number of classes. Defaults to 9.
            size_average (bool, optional): whether return mean otherwise sum of loss. Defaults to False.
            ignore_index (int, optional): class label need to ignore. Defaults to -1.
        """
        super(FocalLoss,self).__init__()
        self.size_average = size_average
        self.ign_idx = ignore_index
        if isinstance(alpha,list):
            assert len(alpha)==num_classes
            self.alpha = torch.Tensor(alpha)
        else:
            self.alpha = torch.zeros(num_classes)
            self.alpha += alpha  # [ α, α, α, α, α, ...] 
        self.gamma = gamma

    def forward(self, preds, labels):
        """ 
        Args:
            preds (Tensor): model prediction (B,C,H,W)
            labels (Tensor): (B,H,W)
        """
        # assert preds.dim() != 4 or labels.dim() != 3, "please make sure that input:(B,C,H,W) and label:(B,H,W)"
        ## B,C,H,W -> N,C & B,H,W -> N
        index = labels!=self.ign_idx  ##
        labels = labels[index] ## [N] 
        preds = preds.permute(1,0,2,3).contiguous()
        preds = preds[:,index].permute(1,0).contiguous()  
        preds = preds.view(-1,preds.shape[-1]) ## [N,C]
        
        ## calculate focal loss 
        alpha = self.alpha.to(preds.device) ## [C]
        log_softmax = F.log_softmax(preds, dim=1) # log_softmax [N,C]
        pt = torch.exp(log_softmax).gather(1,labels.view(-1,1))   # pt [N,1]
        logpt = log_softmax.gather(1,labels.view(-1,1))   # nll_loss [N,1]
        celoss = -1*logpt  ## Cross entropy loss
        alpha = alpha.gather(0,labels.view(-1)) ##[N]
        loss = alpha.view(alpha.shape[0],1)*torch.pow((1-pt), self.gamma)*celoss  # -1*alpha*(1-pt)^gamma*log(pt) [N,1]
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

class DualSet(Dataset):
    def __init__(self,data,data_origin,sample_gt,time_step=3,is_pred = False,**config):
        super(DualSet).__init__()
        self.sample_mode = config['SAMPLE_MODE']
        self.is_pred = is_pred
        self.patch_size = config['PATCH_SIZE']
        self.p = self.patch_size // 2
        self.input_dim = int(data_origin.shape[-1]/time_step)
        self.time_step = time_step
        self.sample_gt = np.asarray(sample_gt,dtype='int64')
        self.p = self.patch_size // 2
        self.data_pca = np.pad(data,((self.p,self.p),(self.p,self.p),(0,0)),'symmetric')
        self.data_origin = np.pad(data_origin,((self.p,self.p),(self.p,self.p),(0,0)),'symmetric')
        self.pad_sample_gt = np.pad(self.sample_gt,(self.p,self.p),'constant',constant_values = 0)
        if is_pred: 
            x_pos, y_pos = np.nonzero(np.ones_like(sample_gt))
            x_pos, y_pos = x_pos + self.p, y_pos + self.p
        else:x_pos, y_pos = np.nonzero(self.pad_sample_gt)   ##indices after padding
        self.indices = np.array([(x,y) for x,y in zip(x_pos, y_pos)])
        if not is_pred:
            np.random.shuffle(self.indices)
        self.data_pca = torch.from_numpy(np.asarray(self.data_pca, dtype='float32').transpose((2, 0, 1)))
        self.data_origin = torch.from_numpy(np.asarray(self.data_origin, dtype='float32').transpose((2, 0, 1)))
        self.sample_gt = torch.from_numpy(self.sample_gt)
        self.pad_sample_gt = torch.from_numpy(np.asarray(self.pad_sample_gt,dtype='int64'))
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, i):
        x, y = self.indices[i]
        x1, y1 = x - self.p, y - self.p
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size
        x_lstm = torch.zeros((self.time_step,self.input_dim))
        for i in range(0,self.time_step):
            x_lstm[i,:] = self.data_origin[i:i+(self.input_dim-1)*self.time_step+1:self.time_step,x,y]
        if self.sample_mode == 'PWS':
            if self.is_pred: 
                return (x_lstm, self.data_pca[:, x1:x2, y1:y2]), self.sample_gt
            return (x_lstm, self.data_pca[:, x1:x2, y1:y2]), self.pad_sample_gt[x,y]
    
   