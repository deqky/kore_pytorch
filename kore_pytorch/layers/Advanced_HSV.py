import torch
from torch import nn 
from typing import Tuple


class AdvancedHSV(nn.Module):
    name = " Advanced HSV"
    
    def __init__(self, *args, **kwargs):
        super(AdvancedHSV, self).__init__()
        self.white_thr = nn.Parameter(torch.tensor([249/255], dtype=torch.float32))
        self.color_thr = nn.Parameter(torch.tensor([30/255], dtype=torch.float32))
        self.gray_thr = nn.Parameter(torch.tensor([5/255], dtype=torch.float32))
    
    def make_ahsv(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # w_thr = torch.sigmoid(self.white_thr)
        # c_thr = torch.sigmoid(self.color_thr)
        # g_thr = torch.sigmoid(self.gray_thr)
        
        max_channel, max_arg = torch.max(image, dim = 1, keepdims= True)
        min_channel, _ = torch.min(image, dim = 1, keepdims= True)
        
        R = image[:, 0, :, :].unsqueeze(1) 
        G = image[:, 1, :, :].unsqueeze(1)  
        B = image[:, 2, :, :].unsqueeze(1)
        C = max_channel - min_channel
        
        gray_c = torch.zeros_like(max_channel)        
        gray_c = torch.where(min_channel > self.white_thr, 1, gray_c)
        gray_c = torch.where((gray_c == 0) & (max_channel > self.color_thr), -1, gray_c)
        gray_c = torch.where((gray_c == -1) & (C < self.gray_thr), (max_channel + min_channel) / 2.0, gray_c)

        output = torch.full(max_channel.size(), -1, device=max_channel.device) 
        output = torch.where((max_arg == 0) & (gray_c == -1),  ((G - B) / C) % 6, output)
        output = torch.where((max_arg == 1) & (gray_c == -1),  ((B - R) / C) + 2, output)
        output = torch.where((max_arg == 2) & (gray_c == -1),  ((R - G) / C) + 4, output)
        output = torch.where(output != -1,  output * 60, output)

        return output.squeeze(1), gray_c
    
    def to_rgb_convert(self, image: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
        rgb = rgb.repeat(1, 3, 1, 1)
        for i, n in enumerate((5, 3, 1)):
            k = (n + image/60) % 6
            m = torch.minimum(k, 4 - k)
            m = torch.minimum(m, torch.tensor(1))
            
            res = 1 - torch.maximum(torch.tensor(0), m)
            rgb[:, i, :, :] = torch.where(rgb[:, i, :, :] == -1, res, rgb[:, i, :, :])
        return rgb
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, gray_c = self.make_ahsv(x)
        x = self.to_rgb_convert(output, gray_c)
        return x


if __name__ == "__main__":
    import torchvision
    import matplotlib.pyplot as plt
    
    model = AdvancedHSV()
    image = torchvision.io.decode_image("kore_pytorch/layers/image13.png", mode = "RGB")
    image = torch.stack((image, image), dim=0)
    image = image.type(torch.float32) / 255.0
    y = model(image)

    plt.figure(figsize=(10,10))
    plt.subplot(1, 2, 1)
    plt.imshow((image * 255)[0].byte().numpy().transpose(1, 2, 0))
    plt.title("original image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow((y * 255)[0].byte().numpy().transpose(1, 2, 0))
    plt.title("HSV mask")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
