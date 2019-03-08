#%%
import torch
import torch.nn as nn

from modules.models.Blocks import ConvBlock
from modules.dataset.musdb18 import Dataset

class Variation1(nn.Module):
    def __init__(self,use_bn):
        super(Variation1,self).__init__()
        ch = 48
        self.mask_generator = nn.Sequential(
            ConvBlock(2,ch,kernel_size=(1,7),padding="same",stride=1,dilation=(1,1),use_bn=use_bn),
            ConvBlock(ch, ch, kernel_size=(7, 1), padding="same", stride=1, dilation=(1, 1),use_bn=use_bn),
            ConvBlock(ch, ch, kernel_size=(5, 5), padding="same", stride=1, dilation=(1, 1), use_bn=use_bn),
            ConvBlock(ch, ch, kernel_size=(5, 5), padding="same", stride=1, dilation=(2, 1), use_bn=use_bn),
            ConvBlock(ch, ch, kernel_size=(5, 5), padding="same", stride=1, dilation=(4, 1), use_bn=use_bn),
            ConvBlock(ch, ch, kernel_size=(5, 5), padding="same", stride=1, dilation=(8, 1), use_bn=use_bn),
            ConvBlock(ch, ch, kernel_size=(5, 5), padding="same", stride=1, dilation=(16, 1), use_bn=use_bn),
            ConvBlock(ch, ch, kernel_size=(5, 5), padding="same", stride=1, dilation=(32, 1), use_bn=use_bn),
            ConvBlock(ch, ch, kernel_size=(5, 5), padding="same", stride=1, dilation=(1, 1), use_bn=use_bn),
            ConvBlock(ch, ch, kernel_size=(5, 5), padding="same", stride=1, dilation=(2, 2), use_bn=use_bn),
            ConvBlock(ch, ch, kernel_size=(5, 5), padding="same", stride=1, dilation=(4, 4), use_bn=use_bn),
            ConvBlock(ch, ch, kernel_size=(5, 5), padding="same", stride=1, dilation=(8, 8), use_bn=use_bn),
            ConvBlock(ch, ch, kernel_size=(5, 5), padding="same", stride=1, dilation=(16, 16), use_bn=use_bn),
            ConvBlock(ch, ch, kernel_size=(5, 5), padding="same", stride=1, dilation=(32, 32), use_bn=use_bn),
            ConvBlock(ch, 2, kernel_size=(1, 1), padding="same", stride=1, dilation=(1, 1), use_bn=use_bn),
            nn.Sigmoid()
        )

    def forward(self, x):
        mask = self.mask_generator(x)
        x = torch.mul(mask,x)
        return x
    
class Variation2(nn.Module):
    def __init__(self,use_bn):
        super(Variation2,self).__init__()
        ch = 24
        self.mask_generator = nn.Sequential(
            ConvBlock(2,ch,kernel_size=(1,7),padding="same",stride=1,dilation=(1,1),use_bn=use_bn),
            ConvBlock(ch, ch, kernel_size=(7, 1), padding="same", stride=1, dilation=(1, 1),use_bn=use_bn),
            ConvBlock(ch, ch, kernel_size=(5, 5), padding="same", stride=1, dilation=(1, 1), use_bn=use_bn),
            ConvBlock(ch, ch, kernel_size=(5, 5), padding="same", stride=1, dilation=(2, 1), use_bn=use_bn),
            ConvBlock(ch, ch, kernel_size=(5, 5), padding="same", stride=1, dilation=(4, 1), use_bn=use_bn),
            ConvBlock(ch, ch, kernel_size=(5, 5), padding="same", stride=1, dilation=(8, 1), use_bn=use_bn),
            ConvBlock(ch, ch, kernel_size=(5, 5), padding="same", stride=1, dilation=(16, 1), use_bn=use_bn),
            ConvBlock(ch, ch, kernel_size=(5, 5), padding="same", stride=1, dilation=(1, 1), use_bn=use_bn),
            ConvBlock(ch, ch, kernel_size=(5, 5), padding="same", stride=1, dilation=(2, 2), use_bn=use_bn),
            ConvBlock(ch, ch, kernel_size=(5, 5), padding="same", stride=1, dilation=(4, 4), use_bn=use_bn),
            ConvBlock(ch, ch, kernel_size=(5, 5), padding="same", stride=1, dilation=(8, 8), use_bn=use_bn),
            ConvBlock(ch, ch, kernel_size=(5, 5), padding="same", stride=1, dilation=(16, 16), use_bn=use_bn),
            ConvBlock(ch, 2, kernel_size=(1, 1), padding="same", stride=1, dilation=(1, 1), use_bn=use_bn),
            nn.Sigmoid()
        )

    def forward(self, x):
        mask = self.mask_generator(x)
        x = torch.mul(mask,x)
        return x
    
class Variation3(nn.Module):
    def __init__(self,use_bn):
        super(Variation3,self).__init__()
        ch = 24
        self.mask_generator = nn.Sequential(
            ConvBlock(2,ch,kernel_size=(1,7),padding="same",stride=1,dilation=(1,1),use_bn=use_bn),
            ConvBlock(ch, ch, kernel_size=(7, 1), padding="same", stride=1, dilation=(1, 1),use_bn=use_bn),
            ConvBlock(ch, ch, kernel_size=(5, 5), padding="same", stride=1, dilation=(1, 1), use_bn=use_bn),
            ConvBlock(ch, ch, kernel_size=(5, 5), padding="same", stride=1, dilation=(2, 1), use_bn=use_bn),
            ConvBlock(ch, ch, kernel_size=(5, 5), padding="same", stride=1, dilation=(4, 1), use_bn=use_bn),
            ConvBlock(ch, ch, kernel_size=(5, 5), padding="same", stride=1, dilation=(1, 1), use_bn=use_bn),
            ConvBlock(ch, ch, kernel_size=(5, 5), padding="same", stride=1, dilation=(2, 2), use_bn=use_bn),
            ConvBlock(ch, ch, kernel_size=(5, 5), padding="same", stride=1, dilation=(4, 4), use_bn=use_bn),
            ConvBlock(ch, 2, kernel_size=(1, 1), padding="same", stride=1, dilation=(1, 1), use_bn=use_bn),
            nn.Sigmoid()
        )

    def forward(self, x):
        mask = self.mask_generator(x)
        x = torch.mul(mask,x)
        return x
    
class Variation4(nn.Module):
    def __init__(self,use_bn):
        super(Variation4,self).__init__()
        ch = 24
        self.mask_generator = nn.Sequential(
            ConvBlock(1,ch,kernel_size=(1,10),padding="same",stride=1,dilation=(1,1),use_bn=use_bn),
            ConvBlock(ch, ch, kernel_size=(10, 1), padding="same", stride=1, dilation=(1, 1),use_bn=use_bn),
            ConvBlock(ch, ch, kernel_size=(5, 5), padding="same", stride=1, dilation=(1, 1), use_bn=use_bn),
            ConvBlock(ch, ch, kernel_size=(5, 5), padding="same", stride=1, dilation=(1, 1), use_bn=use_bn),
            ConvBlock(ch, ch, kernel_size=(5, 5), padding="same", stride=1, dilation=(2, 1), use_bn=use_bn),
            ConvBlock(ch, ch, kernel_size=(5, 5), padding="same", stride=1, dilation=(4, 1), use_bn=use_bn),
            ConvBlock(ch, ch, kernel_size=(5, 5), padding="same", stride=1, dilation=(1, 1), use_bn=use_bn),
            ConvBlock(ch, ch, kernel_size=(5, 5), padding="same", stride=1, dilation=(2, 2), use_bn=use_bn),
            ConvBlock(ch, ch, kernel_size=(5, 5), padding="same", stride=1, dilation=(4, 4), use_bn=use_bn),
            ConvBlock(ch, 1, kernel_size=(1, 1), padding="same", stride=1, dilation=(1, 1), use_bn=use_bn)
        )

    def forward(self, x):
        mask = self.mask_generator(x[:,:1])
        new_mag = torch.mul(mask,x[:,:1])
        x = torch.cat((new_mag,x[:,1:2]),dim=1)
        return x
#%%
if __name__ == "__main__":
    model = Variation4(use_bn=True)
    dataset = Dataset(r"C:\Users\USER\Desktop\musdb18_npy\musdb_npy_16000\train","train")
    mix, inst = dataset[12]
    mix = mix.unsqueeze(0)
    inst = inst.unsqueeze(0)
    
    output = model(mix)
    
    