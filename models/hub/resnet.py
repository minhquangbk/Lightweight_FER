from torchvision.models import resnet101, ResNet101_Weights, resnet50, ResNet50_Weights
import torch
import torch.nn as nn
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

from .fgw import Block, conv3x3
from .mobilenetv2 import MobileNetV2
from .resnet50 import resnet50, load_state_dict

class Model(nn.Module):
    def __init__(self, 
                 in_channels=3, 
                 num_classes=7,
                 *args, **kwargs):
        
        super(Model, self).__init__()
        weight_pt = kwargs['weight_vggface2']
        if kwargs['pt'] == 'imagenet' or weight_pt == None:
          resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        elif kwargs['pt'] == 'vggface2':
          resnet = resnet50(num_classes=8631, include_top=True)
          load_state_dict(resnet, weight_pt)
        if in_channels != resnet.conv1.in_channels:
            resnet.conv1 = nn.Conv2d(
                in_channels, 64, 
                kernel_size=(7, 7), 
                stride=(2, 2), 
                padding=(3, 3), 
                bias=False
            )
        ## freeze all layers
        if kwargs['freeze_layers'] == 'all':
            for param in resnet.parameters():
                param.requires_grad = False

        ## get feature extractor
        self.extract_layer_name = kwargs['feature_extractor_name']
        if self.extract_layer_name:
            print(1123)
            if kwargs['ft_name'] == 'linear':
                return_nodes = {'avgpool': 'feature_extractor'}
                print("last_layer: ", 'avg')
            else:
                return_nodes = {self.extract_layer_name: 'feature_extractor'}
            self.feature_extractor = create_feature_extractor(resnet, return_nodes=return_nodes)
            last_layer = list(self.feature_extractor.modules())[-1]
            if isinstance(last_layer, nn.Conv2d):
                self.last_out_channels = last_layer.out_channels # 256
            elif isinstance(last_layer, nn.AdaptiveAvgPool2d) or isinstance(last_layer, nn.AvgPool2d):
                self.last_out_channels = 2048
            
        if kwargs['ft_name'] == 'fgw':
            self.model = nn.Sequential(
                # fgw module
                Block(self.last_out_channels, self.last_out_channels*2, keep_dim=False), # 256x28x28
                Block(self.last_out_channels*2, self.last_out_channels*4, keep_dim=False), # 512x14x14
                Block(self.last_out_channels*4, self.last_out_channels*4, keep_dim=False), # 512x7x7

                # glb module
                conv3x3(512, num_classes),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
                
                # # fcn module
                # nn.Flatten(),
                # nn.Linear(in_features=512*7*7, out_features=512),
                # nn.ReLU(inplace=True),
                # nn.Dropout(0.5),
                # nn.Linear(512, num_classes, bias=True)
            )
        elif kwargs['ft_name'] == 'mobilenetv2':
            self.model = MobileNetV2(self.last_out_channels,
                                     num_classes=num_classes)
        elif kwargs['ft_name'] == 'linear':
            self.model = nn.Sequential(
                nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(self.last_out_channels, 1024),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.4),
                    
                    nn.Linear(1024, 512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.4),
                    
                    nn.Linear(512, num_classes, bias=True)
                )
            )
        total_feature_extractor_params, trainable_feature_extractor_params = self.count_parameters(self.feature_extractor)
        # print(f'feature extractor params: {trainable_feature_extractor_params}/{total_feature_extractor_params}')
        total_model_params, trainable_model_params = self.count_parameters(self.model)
        # print(f'model params: {trainable_model_params}/{total_model_params}')
        print('total parameters: ', total_feature_extractor_params + total_model_params)
        print('trainable parameters: ', trainable_feature_extractor_params + trainable_model_params)
        
    def count_parameters(self, model): 
        total_params = 0
        trainable_params = 0
        for p in model.parameters():
            if p.requires_grad:
                trainable_params += p.numel()
            total_params += p.numel()
        return total_params, trainable_params
        # return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def forward(self, x):
        with torch.no_grad():
            out = self.feature_extractor(x)['feature_extractor']
        #print(out.shape)
        out = self.model(out)
        return out
    
if __name__ == '__main__':
    x = torch.randn(1, 3, 128, 128)
    model = Model(3, 7, freeze_layers='linear')
    print(model(x).shape)
