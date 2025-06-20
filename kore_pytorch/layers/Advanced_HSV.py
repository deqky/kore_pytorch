import torch
from torch import nn 
from typing import Tuple
from time import time

class AdvancedHSV(nn.Module):
    name = " Advanced HSV"
    
    def __init__(self, *args, **kwargs):
        super(AdvancedHSV, self).__init__()
        self.white_thr = nn.Parameter(torch.tensor([249/255], dtype=torch.float32))
        self.color_thr = nn.Parameter(torch.tensor([30/255], dtype=torch.float32))
        self.gray_thr = nn.Parameter(torch.tensor([5/255], dtype=torch.float32))

    @property
    def out_channels(self):
        return 3
    
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

class AdvancedHSValt(nn.Module):
    name = " Advanced HSV alt"
    temp = 20 
    one = torch.tensor(1)
    zero = torch.tensor(0)

    def __init__(self, *args, **kwargs):
        super(AdvancedHSValt, self).__init__()
        self.thr = nn.Parameter(torch.tensor(5/255, dtype=torch.float32))

    @property
    def out_channels(self):
        return 3

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        max_channel, max_arg = torch.max(image, dim = 1, keepdims= True)
        min_channel, _ = torch.min(image, dim = 1, keepdims= True)
        
        R = image[:, 0, :, :].unsqueeze(1) 
        G = image[:, 1, :, :].unsqueeze(1)  
        B = image[:, 2, :, :].unsqueeze(1)
        C = max_channel - min_channel
        C[C==self.thr] -= 1/255
        C[C==0] += 1/255
    
        hsv = torch.where(max_arg == 0,  ((G - B) / C) % 6, 0)
        hsv = torch.where(max_arg == 1,  ((B - R) / C) + 2, hsv)
        hsv = torch.where(max_arg == 2,  ((R - G) / C) + 4, hsv)

        hsv = hsv.repeat(1, 3, 1, 1)
        for i, n in enumerate((5, 3, 1)):
            k = (n + hsv[:, i, :, :]) % 6
            m = torch.minimum(k, 4 - k)
            m = torch.minimum(m, self.one)
            
            res = 1 - torch.maximum(self.zero, m)
            hsv[:, i, :, :] = res

        gray = (max_channel + min_channel) / 2      
        gray = gray.repeat(1, 3, 1, 1)
        mask = torch.sigmoid((C - self.thr) * 255 * self.temp)
 
        return mask * hsv + (1 - mask) * gray

class AdvancedHSValt(nn.Module):
    name = " Advanced HSV alt"
    epsilon = 1e-9

    def __init__(self, *args, **kwargs):
        super(AdvancedHSValt, self).__init__()
        self.color_thr = nn.Parameter(torch.tensor(0.04, dtype=torch.float32))
        self.black_thr = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        self.white_thr = nn.Parameter(torch.tensor(0.9, dtype=torch.float32))

    @property
    def out_channels(self):
        return 3

    def hard_sigmoid(self, x, slope=1000.0):
        return torch.clamp(x * slope - 0.01, 0.0, 1.0)
    
    def custom_sigmoid(self, x):
        return 1/(1 + torch.exp(-(x-0.5)))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        max_channel, max_arg = torch.max(image, dim = 1, keepdims= True)
        min_channel, _ = torch.min(image, dim = 1, keepdims= True)

        R, G, B = image[:, 0:1], image[:, 1:2], image[:, 2:3]
        C = max_channel - min_channel        
        # st = time()
        # print(time()-st)
        H = torch.where(max_arg == 0,  ((G - B) / (C + self.epsilon)) % 6, 0)
        H = torch.where(max_arg == 1,  ((B - R) / (C + self.epsilon)) + 2, H)
        H = torch.where(max_arg == 2,  ((R - G) / (C + self.epsilon)) + 4, H).repeat(1, 3, 1, 1)
        
        S = torch.where(C != 0, C/(1 - torch.abs(max_channel + min_channel - 1)), 0)
        L = ((max_channel + min_channel) / 2)
        
        for channel_idx, n in enumerate((5, 3, 1)):
            k = (n + H[:, channel_idx, :, :]) % 6
            H[:, channel_idx, :, :] = 1 - torch.minimum(k, 4 - k).clamp(min=0, max=1)

        color_mask = self.hard_sigmoid(S - self.color_thr)
        black_mask = self.hard_sigmoid(self.black_thr - L)
        white_mask = self.hard_sigmoid(L - self.white_thr)
        # color_mask = self.hard_sigmoid(S - self.custom_sigmoid(self.color_thr))
        # black_mask = self.hard_sigmoid(self.custom_sigmoid(self.black_thr) - L)
        # white_mask = self.hard_sigmoid(L - self.custom_sigmoid(self.white_thr))

        return torch.where((color_mask == 1.0) & (black_mask == 0.0) & (white_mask == 0.0), H, L.expand(-1, 3, -1, -1))
        

 

if __name__ == "__main__":
    import torchvision
    import matplotlib.pyplot as plt
    
    model = AdvancedHSValt()
    model.eval()
    image = torchvision.io.decode_image("kore_pytorch/layers/image1.png", mode = "RGB")
    # image = torchvision.io.decode_image("C:/Python/pytorch_semantic/augmented-dubai-dataset-pad-center-256px/images/tile_8_54_90d_flp.jpg", mode = "RGB")
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
