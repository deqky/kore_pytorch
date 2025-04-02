import torch
from torch import nn


class CNet(nn.Module):
    def __init__(self, model, name, mode, num_classes, metrics = None, class_names = None, **kwargs):
        super().__init__()
        self.name = name
        self.model = model(weights = None, num_classes = num_classes)
        self.mode = mode
        self.name += self.mode

        match self.mode:
            case " To Classifier" | " As input":

                num_filters = kwargs.get("num_filters", 0)
                if num_filters:

                    self.custom_layer = kwargs.pop("custom_layer")
                    self.name += self.custom_layer.name
                    l_name = self.custom_layer.name
                
                    self.filter = nn.Sequential(
                                    nn.Conv2d(1 if l_name == " Saturation" else 3, num_filters, kernel_size = 5, padding = 3), 
                                    nn.ReLU(inplace = True))
                else: self.filter = nn.Identity()

                if self.mode == " To Classifier":
                    if isinstance(self.model.classifier, nn.Conv2d):
                        model_out_channels = self.model.classifier.in_channels
                        self.model.classifier = nn.Identity()
                    else: 
                        model_out_channels = self.model.classifier[4].in_channels
                        self.model.classifier[4] = nn.Identity()
                        
                    self.out = nn.Conv2d(model_out_channels + num_filters, num_classes, kernel_size = 1) 
                elif self.mode == " As input":
                    if isinstance(self.model.classifier, nn.Conv2d):
                        self.model.up_convs[0] = self.model.double_conv(3 + num_filters, 64)
                    else:
                        self.model.backbone.conv1 = nn.Conv2d(3 + num_filters, 64, kernel_size=7, stride=2, padding=3, bias=False)
            
            case " ":
                pass
            case _:
                raise "No such mode"
        if metrics: self.metrics = metrics(self.name, class_names)


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
            

if __name__ == "__main__":
    import torchvision.models as models
    input_tensor = torch.randint(0, 255, (1, 3, 256, 256)).type(torch.float32)
    deeplab = models.segmentation.deeplabv3_resnet50
    model = CNet(model = deeplab, name = "deeplab", mode = " ", num_classes = 6)
    model.eval()
    with torch.no_grad():
        y = model(input_tensor)
    print(y.shape)
