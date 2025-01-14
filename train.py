import sys
import warnings
import time
from tqdm import tqdm, trange
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
import torch.optim as optim
from torch.optim import SGD, Adam, AdamW, RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data.dataloader import get_dataloaders
from utils.checkpoint import save, restore
from utils.running import train, evaluate
from utils.setup_network import setup_network
from utils.helper import epoch_time, store_folder
from models.losses.arcface import ArcFaceLoss

warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device: ", device)

if torch.cuda.is_available():
    torch.cuda.manual_seed(123)
    torch.cuda.synchronize() 
else:
    torch.manual_seed(123)
np.random.seed(123)

# CUDA_VISIBLE_DEVICES=1 python train_lr_opt.py --bs 64 --target_size 48 --num_epochs 300 --lr 0.01 
# --optimizer AdamW --network finalv2 --data_name fer_plus --training_mode new > fer_plus.txt

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, required=True, default="fer2013")
    parser.add_argument("--weights", type=str, default="best.pth")
    parser.add_argument("--bs", type=int, required=True, default=64,
                        help="Batch size of model")
    parser.add_argument("--target_size", type=int, required=True, default=48,
                        help="Image target size to resize")
    parser.add_argument("--num_epochs", type=int, required=True, default=300,
                        help="Number of epochs")
    parser.add_argument("--lr", type=float, required=True, default=0.1,
                        help="Learning rate")
    parser.add_argument("--optimizer", type=str, required=True, default="SGD",
                        help="Optimizer to update weights")
    parser.add_argument("--network", type=str, required=True, default="cbam_resmob",
                        help="Name of network")
    parser.add_argument("--model_path", type=str, default="./models",
                        help="Path to models folder that contains many models")
    parser.add_argument("--dataset_path", type=str, default="/home/aime/hoangdh/emotions/dataset")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--es_min_delta", type=float, default=0.0)
    parser.add_argument("--es_patience", type=int, default=5)
    args = vars(parser.parse_args())
    
    return args

def run(net, 
        logger, 
        model_weight,
        model_loss,
        dataset_path, 
        data_name, 
        batch_size, 
        target_size, 
        optimizer, 
        learning_rate, 
        num_epochs, 
        model_save_dir,
        target_names,
        save_freq,
        start_epoch=0,
        ):
    
    print('batch_size: ', batch_size)
    print('learning_rate: ', learning_rate)
    print('optimizer: ', optimizer)
    
    # Scaler
    scaler = GradScaler()

    # Optimizer
    optimizer = getattr(optim, optimizer)(net.parameters(), lr= learning_rate)
    if optimizer == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, nesterov=True, weight_decay=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.75, patience=5, verbose=True)
    
    # Loss function
    if model_loss is None:
        criterion = nn.CrossEntropyLoss() 
    elif model_loss == 'arcface':
        criterion = ArcFaceLoss()
    
    if model_weight is not None:
        ckpt_info = restore(
            model_weight,
            net, logger,
            optimizer
        )
        start_epoch = ckpt_info['epoch'] + 1
        if start_epoch > 0:
            criterion = ckpt_info['loss']
            optimizer = ckpt_info['optimizer']
            net = ckpt_info['net']
            logger = ckpt_info['logger']
            scheduler = ckpt_info['scheduler']
            print(f'Continual Training from Epoch {start_epoch} =>')
            

    data_path = os.path.join(dataset_path, data_name)
    phases = os.listdir(data_path)
    data_loaders = get_dataloaders(data_path,
                                   phases=phases,
                                   target_size=target_size, 
                                   batch_size=batch_size)
    if 'train' not in data_loaders:
        raise ValueError('train data not found')
    if len(data_loaders) == 1:
        raise ValueError('please add val or test data')
    train_dataloader = data_loaders['train']
    if 'val' in phases:
        val_dataloader = data_loaders['val']
    if 'test' in phases:
        test_dataloader = data_loaders['test']
    net = net.to(device)
    
    best_train_acc = -1
    best_train_loss = 9999
    
    best_val_acc = -1
    best_val_loss = 9999
    
    best_test_acc = -1
    best_test_loss = 9999
    
    for epoch in trange(num_epochs, desc="Epochs"):
        start_time = time.monotonic()
        loss_train, acc_train, f1_train = train(net, train_dataloader, criterion, optimizer, scaler)
        logger.loss_train.append(loss_train)
        logger.acc_train.append(acc_train)
        epoch_mins = ''
        epoch_secs = ''
        if 'val' in phases:
            loss_val, acc_val, f1_val = evaluate(net, val_dataloader, criterion, target_names)
            logger.loss_val.append(loss_val)
            logger.acc_val.append(acc_val)

            # if learning_rate >= 0.01:
            scheduler.step(acc_val)
            
            end_time = time.monotonic()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            
            # save best checkpoint
            if acc_val > best_val_acc:
                best_val_acc = acc_val
                save(net.model, logger, model_save_dir, optimizer, scheduler, epoch, criterion, "best")
            logger.save_plt(model_save_dir)
        if 'test' in phases:
            loss_test, acc_test, f1_test = evaluate(net, test_dataloader, criterion, target_names)    

        # save checkpoint (frequency)
        if epoch > 0 and epoch % save_freq == 0:
            save(net.model, logger, model_save_dir, optimizer, scheduler, epoch, criterion, "last" + str(save_freq))
        logger.save_plt(model_save_dir)
            
        print(f'epoch: {epoch+1:02} | epoch time: {epoch_mins}m {epoch_secs}s')
        print(f'\t train loss: {loss_train:.3f} | train acc: {acc_train*100:.2f}% | train F1: {f1_train*100:.2f}%')
        if 'val' in phases:
            print(f'\t val loss: {loss_val:.3f} |  val acc: {acc_val*100:.2f}% | val F1: {f1_val*100:.2f}%')
        if 'test' in phases:
            print(f'\t test loss: {loss_test:.3f} | test acc: {acc_test*100:.2f}% | test F1: {f1_test*100:.2f}%')
            
        best_train_acc = max(best_train_acc, acc_train)
        best_train_loss = min(best_train_loss, loss_train)
        
        best_val_acc = max(best_val_acc, acc_val)
        best_val_loss = min(best_val_loss, loss_val)
        
        best_test_acc = max(best_test_acc, acc_test)
        best_test_loss = min(best_test_loss, loss_test)

    if 'test' in phases:       
        _ = evaluate(net, test_dataloader, criterion, "Testing")
    print('=============**********=============')
    print('=============EVALUATION=============')
    print("train")
    print(f'\t acc: {best_train_acc} | loss: {best_train_loss}')
    
    print("val")
    print(f'\t acc: {best_val_acc} | loss: {best_val_loss}')
    
    print("test")
    print(f'\t acc: {best_test_acc} | loss: {best_test_loss}')
    # return best_acc
  
if __name__ == "__main__":
    import yaml
    config_path = '/kaggle/working/fgw-pretrained-image-master/pretrained_config.yaml'

    ### LOAD CONFIG ###
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        file.close()

    ### MODEL CONFIG ###
    model_config = config['model']
    model_root_path = model_config['root']
    model_weight = model_config['weight']
    model_name = model_config['name']
    model_pretrained = model_config['pretrained']
    model_loss = model_config['loss']
    freeze_layers = model_config['freeze_layers']

    ### DATASET CONFIG ###
    dataset_config = config['dataset']
    data_root_path = dataset_config['root']
    data_name = dataset_config['name']
    target_names = sorted(os.listdir(os.path.join(data_root_path, data_name, 'train')))
    num_classes = len(target_names)

    ### HYPER-PARAMETERS ###
    hyperparams = config['hyperparams']
    bs = int(hyperparams['bs'])
    lr = float(hyperparams['lr'])
    epochs = int(hyperparams['epochs'])
    optimizer = hyperparams['optimizer']
    target_size = int(hyperparams['target_size'])
    image_channels = int(hyperparams['image_channels'])
    scheduler = hyperparams['scheduler']
    save_freq = int(hyperparams['save_freq'])

    #
    available_nets = set(filename.split(".")[0] for filename in os.listdir(model_root_path))
    model_save_dir = store_folder(
        data_name,
        model_name,
        bs,
        optimizer,
        lr,
        available_nets,
        model_loss,
        model_pretrained
    )

    logger, net = setup_network(model_name, 
                                image_channels, 
                                model_weight, 
                                num_classes,
                                freeze_layers='all',
                                weight_vggface2='/kaggle/input/restnet50/resnet50_ft_weight.pkl',
                                pt='vggface2',
                                feature_extractor_name='layer2.0.conv1',
                                ft_name='linear')
    # exit()
    run(
        net=net,
        logger=logger,
        model_weight=model_weight,
        model_loss=model_loss,
        dataset_path=data_root_path,
        data_name=data_name,
        batch_size=bs,
        target_size=target_size,
        optimizer=optimizer,
        learning_rate=lr,
        num_epochs=epochs,
        target_names=target_names,
        save_freq=save_freq,
        model_save_dir=model_save_dir
    )
