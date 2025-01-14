from torchvision.models import vit_l_32, ViT_L_32_Weights, vit_l_16, ViT_L_16_Weights
import torch
import torch.nn as nn
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names


class Model(nn.Module):
    def __init__(self, 
                 in_channels=3, 
                 num_classes=7,
                 *args, **kwargs):
        
        super(Model, self).__init__()
        vit = vit_l_16(weights=ViT_L_16_Weights.DEFAULT)
        if in_channels != vit.conv_proj.in_channels:
            vit.conv_proj = nn.Conv2d(
                in_channels, 1024, 
                kernel_size=(32, 32), 
                stride=(32, 32)
            )
        if kwargs['freeze_layers'] == 'all':
            for param in vit.parameters():
                param.requires_grad = False
        self.extract_layer_name = kwargs['feature_extractor_name']
        if self.extract_layer_name:
            if kwargs['ft_name'] == 'linear':
                return_nodes = {'encoder.ln': 'feature_extractor'}
            else:
                pass
            self.feature_extractor = create_feature_extractor(vit, return_nodes=return_nodes)
            self.last_out_channels = 201728
        
        if kwargs['ft_name'] == 'linear':
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
        
    def forward(self, x):
        with torch.no_grad():
            out = self.feature_extractor(x)['feature_extractor']
        out = self.model(out)
        return out

if __name__ == '__main__':
    x = torch.randn(1, 3, 224, 224)
    model = Model(3)
    print(model(x).shape)
