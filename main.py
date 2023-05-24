import os
import random
import shutil
import spectral
import torch
import copy
import numpy as np
import pandas as pd
from torchinfo import summary
from param import get_parser
from utils import loadData, write_log, img_display, applyPCA, sample_gt,\
    sample_info, bulid_dataloader, get_model, train, test
import visdom

def run(**config):
    ## load data
    data_origin,label,config['CLASS_NAME'],config['RGB_BAND'] = loadData(config['DATASET'])
    config['NUM_CLASS'] = label.max()
    config['DATA_SIZE'],config['DATA_BAND'] = data_origin.shape,data_origin.shape[-1]
    spectral.save_rgb(os.path.join('results','RGB_origin_{}.png'.format(config['DATASET'])),data_origin,config['RGB_BAND'])
    spectral.save_rgb(os.path.join('results','gt_{}.png'.format(config['DATASET'])),label,colors = spectral.spy_colors)
    
    
    ##complete config
    _, _, _, _, config =  get_model(**config)
    
    ## preprocess
    if config['N_PCA'] == config['DATA_BAND']:
        data,config['N_PCA'] = applyPCA(data_origin, 0, config['NORM'])
    else:
        data,config['N_PCA'] = applyPCA(data_origin, config['N_PCA'], config['NORM'])
    data_origin,_ = applyPCA(data_origin, 0, config['NORM'])
    res_all_run = [pd.DataFrame() for i in range(config['MODEL_MODE']+1)]
    for irun in range(config['RUNS']):
        # # Set random seed
        # seed = 666
        # random.seed(seed)
        # torch.manual_seed(seed)
        # torch.cuda.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)
        # np.random.seed(seed)
        # # torch.backends.cudnn.benchmark = True
        # torch.backends.cudnn.deterministic = True
        seed = random.randint(0,1000)
        ## sampling
        if config['TRAIN_RATE']<1 and config['VAL_RATE']<1:
            train_gt, test_gt = sample_gt(label,config['TRAIN_RATE']+config['VAL_RATE'],seed = seed)
            val_gt,train_gt = sample_gt(train_gt,config['VAL_RATE']/(config['TRAIN_RATE']+config['VAL_RATE']),seed = seed)
        else:
            train_gt, test_gt = sample_gt(label,config['TRAIN_RATE']+config['VAL_RATE'],seed = seed)
            val_gt,train_gt = sample_gt(train_gt,config['VAL_RATE'],seed = seed)
        ## display sampling info
        sample_report = sample_info(label,train_gt,val_gt,test_gt,config['CLASS_NAME'])
        print(sample_report)
        
        ## create dataloader
        model, optimizer, lr_scheduler, criterion, config =  get_model(**config)
        train_loader,val_loader,test_loader = bulid_dataloader(data,data_origin,train_gt,val_gt,test_gt,**config)
        
        env_name = "{}_{}_{}_{}_run{}".format(config['ENV_NAME'],config['MODEL'],config['DATASET'],config['SAMPLE_MODE'],irun)
        config['SAVE_PATH'] = "results\\%s"%env_name

        if os.path.isdir(config['SAVE_PATH']):
            shutil.rmtree(config['SAVE_PATH'])
        os.makedirs(config['SAVE_PATH'])
        
        if config['VISDOM'] == 1:
            viz = visdom.Visdom(env = env_name)
            if not viz.check_connection:
                print("Visdom is not connected. Did you run 'python -m visdom.server' ?")
            viz.close()
        else: viz = None
        logwin = None
        logtext = f'train samples:{train_loader.dataset.__len__()}  val samples:{val_loader.dataset.__len__()}  test samples:{test_loader.dataset.__len__()}'
        print(logtext)
        if viz!=None:
            logwin = write_log(logtext,logwin,viz,'log')
            logwin = write_log(sample_report,logwin,viz,'log')
            logwin = write_log(str(config),logwin,viz,'log')
        with torch.no_grad():
            if config['MODEL'] == 'SSUN':
                in_data = (iter(train_loader).next()[0][0][0].unsqueeze(0),iter(train_loader).next()[0][1][0].unsqueeze(0))
            else: in_data = iter(train_loader).next()[0][0].unsqueeze(0)
            logtext = summary(model, input_data=in_data, col_names=['num_params','kernel_size','mult_adds','input_size','output_size'],col_width=10, row_settings=['var_names'], depth=10, device = 'cpu')
            if viz!=None:
                logwin = write_log(str(logtext),logwin,viz,'log')
            del in_data
        # img_display(classes=train_gt,title='train_gt',viz = viz)
        # img_display(classes=val_gt,title='val_gt',viz = viz)
        # img_display(classes=test_gt,title='test_gt',viz = viz)
        
        ## train
        best_model = copy.deepcopy(model)
        bestpath = train(model, train_loader, val_loader, criterion, optimizer, lr_scheduler, viz, config)
        torch.cuda.empty_cache() 
        best_model.load_state_dict(torch.load(bestpath)['state_dict'])
        
        ## test
        predwin = None
        for i in range(config['MODEL_MODE']+1):
            torch.cuda.empty_cache()
            if not config['OUT_MORE']:
                if i!=config['MODEL_MODE']:continue
            res_all_run[i][f'run_{irun}'] ,report_log= test(best_model,i,label,test_gt,test_loader,viz,**config)
            if viz!= None:
                predwin = write_log(report_log+'\n',predwin,viz,'classification results')
        
        if viz!= None:
            viz.save(viz.get_env_list())
        del train_loader,val_loader,test_loader,model,best_model,train_gt, test_gt,val_gt
        torch.cuda.empty_cache()
    ## result summary
    reswin = None
    with pd.ExcelWriter(os.path.join(config['SAVE_PATH'],'results.xlsx')) as writer:
        for idx, df in enumerate(res_all_run):
            df['mean'] = df.mean(axis = 1)
            df['std'] = df.std(axis = 1)
            if viz!= None:
                reswin = write_log(str(df)+'\n',reswin,viz,'results')
            df.to_excel(writer,sheet_name='results', startcol = idx*(config['RUNS']+4))
   
if __name__ == '__main__':
    config = get_parser()
    run(**config)
    
