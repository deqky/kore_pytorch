import torch
from torch import nn


class CNet(nn.Module):
    def __init__(self, model, name, mode, num_classes, metrics, class_names, **kwargs):
        super().__init__()
        self.name = name
        self.model = model(weights = None, num_classes = num_classes)
        self.mode = mode

        self.name += self.mode
        num_filters = kwargs.get("num_filters", 0)
        try:
            self.custom_layer = kwargs.pop("custom_layer")
            self.name += self.custom_layer.name
            if num_filters:
                l_name = self.custom_layer.name
            
                self.filter = nn.Sequential(
                                nn.Conv2d(1 if l_name == " Saturation" else 3, num_filters, kernel_size = 7, padding = 3), 
                                nn.ReLU(inplace = True))
            else: self.filter = nn.Identity()
        except: pass
            
        self.metrics = metrics(self.name, class_names)
        
        match self.mode:
            case " To Classifier":
                if isinstance(self.model.classifier, nn.Conv2d):
                    model_out_channels = self.model.classifier.in_channels
                    self.model.classifier = nn.Identity()
                else: 
                    model_out_channels = self.model.classifier[4].in_channels
                    self.model.classifier[4] = nn.Identity()
                    
                self.out = nn.Conv2d(model_out_channels + num_filters, num_classes, kernel_size = 1) 
                
            case " As Input":
                if isinstance(self.model.classifier, nn.Conv2d):
                    self.model.up_convs[0] = self.model.double_conv(3 + num_filters, 64)
                else:
                    self.model.backbone.conv1 = nn.Conv2d(3 + num_filters, 64, kernel_size=7, stride=2, padding=3, bias=False)
            
            case " ":
                pass
            case _:
                raise "No such mode"


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        match self.mode:
            case " As Input" | " To Classifier":
                custom_layer_output = self.custom_layer(x)
                custom_layer_output = self.filter(custom_layer_output)
                
                if self.mode == " To Classifier":
                    model_output = self.model(x)['out']
                    x = torch.cat((model_output, custom_layer_output), dim = 1)
                    return self.out(x)
                elif self.mode == " As Input":
                    x = torch.cat([x, custom_layer_output], dim = 1)
                    return self.model(x)['out']
                    
            case " ":
                return self.model(x)['out']