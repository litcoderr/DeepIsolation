# -*- coding: utf-8 -*-

from modules.util.stft import STFTConverter

import torch
from torch.utils import data
import os
import numpy as np
import matplotlib.pyplot as plt
try:
    import scipy.io.wavfile as wv
    from scipy.io.wavfile import read
except:
    pass


class Dataset(data.Dataset):
    def __init__(self,root,comp=False,direct=False):
        self.root = root
        self.comp = comp
        self.direct = direct
        
        self.title_list = os.listdir(os.path.join(self.root))
        self.converter = STFTConverter()
        
        data_name = self.title_list[0]
        self.mix = read(os.path.join(self.root,data_name))
        self.mix = np.array(self.mix[1],dtype=float)
        self.mix = self.mix/32767. # normalize
    
    def __len__(self):
        length = int(self.mix.shape[0]/440000)
        return length
    
    def __getitem__(self,index):
        mix = self.mix[index*440000:(index+1)*440000]
        if not self.direct:
            # Convert to stft shape: [2(actual_mag,rad),time,mag]
            mix = self.converter.get_stft(mix,comp=self.comp)
        
        # Convert numpy to torch tensor
        mix = torch.Tensor(mix)
        return mix

#%%
if __name__ == "__main__":
    dataset = Dataset(r"C:\Users\USER\Desktop\project\voice-isolation\source")
    mix = dataset[10]