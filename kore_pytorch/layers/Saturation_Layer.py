import torch 
from torch import nn


class SaturationLayer(nn.Module):
    scale_dict = {0:(-0.996078431372549, 0.1715728751454531), 1:(0, 1.9921875), 2:(0, 1), 3:(0, 2)}
    name = " Saturation"
    
    def __init__(self, chroma_mode = 0, return_mode = "default", **kwargs):
        super(SaturationLayer, self).__init__()
        self.chroma_mode = chroma_mode
        self.return_mode = return_mode
        self.name += f" Chroma mode {chroma_mode} Return mode {return_mode}"
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        max_channel, _ = torch.max(x, dim = 1, keepdim = True)
        min_channel, _ = torch.min(x, dim = 1, keepdim = True)
        
        hsv = torch.where(max_channel == 0, 0, 1 - (min_channel / max_channel))
        hls = torch.where(max_channel == min_channel, 0, 
                              (max_channel - min_channel) / (1 - torch.abs(1 - (max_channel + min_channel))))
        
        match self.chroma_mode:            
            case 0: result =  hsv - hls
            case 1: result = torch.where(hls != 0, hsv / hls, 0)
            case 2: result = hsv * hls
            case 3: result = hsv + hls
            case _: raise "No such mode."

        minv, maxv = self.scale_dict[self.chroma_mode]
        result = torch.clip((result - minv)/(maxv - minv), 0 ,1)

        match self.return_mode:
            case "default": return result
            case "mulmax": return result * max_channel
            case "catmax": return torch.cat((result, max_channel), dim = 1)
            case _: raise "No such mode. Available modes are: default, mulmax, catmax"


if __name__ == "__main__":
    import torchvision
    import matplotlib.pyplot as plt
    
    image = torchvision.io.decode_image("kore_pytorch/layers/image13.png", mode = "RGB")
    image = torch.stack((image, image), dim=0) # Make pseudo batch
    image = image.type(torch.float32) / 255.0
    
    plt.figure(figsize=(10,10))
    plt.subplot(3, 3, 1)
    plt.imshow((image * 255)[0].byte().numpy().transpose(1, 2, 0))
    plt.title('Original image')
    plt.axis('off')

    for mode in range(4):
        for idx, rmode in enumerate(("default", "mulmax", "catmax"), 1):
            if rmode == 'catmax':continue
            model = SaturationLayer(chroma_mode = mode, return_mode = rmode)
            y = model(image)
            plt.subplot(3, 3, (mode + 1) * idx + 1)
            plt.imshow((y * 255)[0].byte().numpy().transpose(1, 2, 0), cmap='gray')
            plt.title(f"Chroma mode: {mode}, return mode: {rmode}")
            plt.axis('off')
    
    plt.tight_layout()
    plt.show()
