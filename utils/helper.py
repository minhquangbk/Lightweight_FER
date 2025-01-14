import torch
import torch.nn as nn
import os


def store_folder(dataset_name, 
                 network, 
                 batch_size, 
                 optimizer, 
                 learning_rate, 
                 available_nets, 
                 loss,
                 pretrained=True):
    
    # Invalid network check
    if network not in available_nets:
        raise ValueError("Invalid network.\nPossible ones include:\n - " + '\n - '.join(available_nets))

    # create checkpoint directory
    model_save_dir = os.path.join(os.getcwd(), 'checkpoints', network)
    model_save_dir = model_save_dir + "_" + dataset_name + "_" + str(batch_size) + "_" + optimizer + "_" + str(learning_rate) + "_" + str(loss)
    if pretrained == False:
        model_save_dir = model_save_dir + "_finetune"
    
    # cbam_base_64_sgd_0.01
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    return model_save_dir

def init_weights(layer):
    if type(layer) == nn.Conv2d:
        nn.init.kaiming_normal_(layer.weight, mode='fan_out')
    if type(layer) == nn.Linear:
        nn.init.normal_(layer.weight, std=1e-3)
    if type(layer) == nn.BatchNorm2d:
        nn.init.constant_(layer.weight, 1)
        nn.init.constant_(layer.bias, 0)

def load_model(model_path, model, device):
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def calculate_accuracy(y_pred, y):
    # print('sssss ', y_pred.shape)
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return top_pred, acc

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def count_parameters(model): 
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class EarlyStopping(object):
    def __init__(self, min_delta = 0.0, patience = 7):
        
        self.min_delta = min_delta
        self.patience = patience
        self.counter = 0
        self.early_stop = False
        
    def __call__(self, curr_val_loss, prev_val_loss):
        if curr_val_loss - prev_val_loss > self.min_delta:
            self.counter += 1
            if self.counter == self.patience:
                self.early_stop = True
        else:
            self.counter = 0