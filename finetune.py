import yaml
from code_base.models.hub.fgw import Model
from utils.helper import count_parameters, load_model
from train import *

import torch
import warnings
import numpy as np

warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device: ", device)

if torch.cuda.is_available():
    torch.cuda.manual_seed(123)
    torch.cuda.synchronize() 
else:
    torch.manual_seed(123)
np.random.seed(123)

def load_pretrained_model(path):
    model = Model(1, 8)
    model = load_model(path,
                       model,
                       device)
    return model

# model = pretrained_model('./checkpoints/model_fer_plus_2_AdamW_0.01/best.pth')
# print('before trainable params: ', count_parameters(model))

def get_finetune_layers(model):


    ## 1. freeze all layers

    ## 2. unfrezze block4, last_conv,  layers to tune by name
    block4 = []
    last_conv = []
    for name, param in model.named_parameters():
        if 'block4' in name:
            block4.append(param)
        elif 'last_conv' in name:
            last_conv.append(param)
        else:
            param.requires_grad = False

    return block4, last_conv


# print(block4)

# print()
# for name, param in model.named_parameters():
#     if param.requires_grad == False:
#         print(name)


if __name__ == '__main__':
    path = './checkpoints/model_fer_plus_2_AdamW_0.01/best.pth'
    pretrained_model = load_pretrained_model(path)
    print('before trainable params: ', count_parameters(pretrained_model))

    block4_params, last_conv_params = get_finetune_layers(pretrained_model)
    print('after trainable params: ', count_parameters(pretrained_model))
    # # exit()
    # print(block4_params)
    # print(last_conv_params)

    args = get_args()
    available_nets = set(filename.split(".")[0] for filename in os.listdir(args["model_path"]))
    available_datasets = set(filename for filename in os.listdir(args["dataset_path"]))
    in_channels = 1
    if "FERG" in args["data_name"]:
        in_channels = 3
    model_save_dir = store_folder(args["data_name"], args["network"], args["bs"], args["optimizer"], args["lr"], available_nets)
    logger, _ = setup_network(args["network"], in_channels)
    best_acc = run(pretrained_model, logger, args["dataset_path"], args["data_name"], args["bs"], args["target_size"], args["optimizer"], 
                    args["lr"], args["num_epochs"], model_save_dir, 
                    parameters_config=[
                        {'params': block4_params, 'lr': 1e-4},
                        {'params': last_conv_params, 'lr': 1e-4}
                    ])