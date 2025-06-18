import torch
from torch import nn

from UNet import UNet

class CNet(nn.Module):
    AS_INPUT = " As Input"
    TO_CLASSIFIER = " To Classifier"
    STACK = " Stack"
    ATTENTION = " Attention"
    REGULAR = " "
    
    def __init__(self, model, name, mode, num_classes, metrics = None, class_names = None, custom_layer = None, custom_filter = None, **kwargs):
        super().__init__()
        self.name = name
        self.mode = mode
        self.name += mode
        self.model = model(weights = None, num_classes = num_classes)
        self.filter = custom_filter

        if mode != self.REGULAR:
            self.custom_layer = custom_layer
            self.name += self.custom_layer.name

        if mode == self.TO_CLASSIFIER:
            if isinstance(self.model, UNet):
                model_out_channels = self.model.classifier.in_channels
                self.model.classifier = nn.Identity()
            else: 
                model_out_channels = self.model.classifier[4].in_channels
                self.model.classifier[4] = nn.Identity()

            num_filters = custom_filter[0].out_channels if custom_filter else custom_layer.out_channels
            self.out = nn.Conv2d(model_out_channels + num_filters, num_classes, kernel_size = 1) 

        elif mode == self.AS_INPUT:
            if isinstance(self.model, UNet):
                self.model.up_convs[0] = self.model.double_conv(3 + custom_layer.out_channels, 64)
            else:
                self.model.backbone.conv1 = nn.Conv2d(3 + custom_layer.out_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        elif mode == self.STACK:
            num_filters = custom_filter[0].out_channels if custom_filter else custom_layer.out_channels
            self.out = nn.Conv2d(num_classes + num_filters, num_classes, kernel_size=1)
        
        elif mode == self.ATTENTION:
            num_filters = custom_filter[0].out_channels if custom_filter else custom_layer.out_channels
            self.attention = nn.Sequential(
                nn.Conv2d(num_filters, 1, kernel_size=1), 
                nn.Sigmoid()
            )
        elif mode == self.REGULAR: pass
        else: raise "No such mode."
            
        if metrics: self.metrics = metrics(self.name, class_names)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == self.REGULAR: return self.model(x)['out']

        custom_layer_output = self.custom_layer(x)
        
        if self.mode == self.AS_INPUT:
            x = torch.cat([x, custom_layer_output], dim = 1)
            return self.model(x)['out']

        model_output = self.model(x)['out']

        if self.filter:
            custom_layer_output = self.filter(custom_layer_output)
        
        if self.mode == self.TO_CLASSIFIER or self.mode == self.STACK:
            x = torch.cat((model_output, custom_layer_output), dim = 1)
            return self.out(x)

        if self.mode == self.ATTENTION:
            attention_out = self.attention(custom_layer_output)
            return model_output * attention_out

            

if __name__ == "__main__":
    import torchvision.models as models
    input_tensor = torch.randint(0, 255, (1, 3, 256, 256)).type(torch.float32)
    deeplab = models.segmentation.deeplabv3_resnet50
    model = CNet(model = deeplab, name = "deeplab", mode = " ", num_classes = 6)
    model.eval()
    with torch.no_grad():
        y = model(input_tensor)
    print(y.shape)
