# -*- coding: utf-8 -*-
from modules.util.stft import STFTConverter

import torch
from torch.utils import data
import os
import numpy as np
import matplotlib.pyplot as plt
try:
    import scipy.io.wavfile as wv
except:
    pass


class Dataset(data.Dataset):
    def __init__(self,root,setting="train",comp=False,direct=False):
        self.root = root
        assert setting=="train" or setting=="test", "[Dataset] invalid setting. Set either 'train' or 'test' !!"
        self.setting = setting
        self.comp = comp
        self.direct = direct
        
        self.title_list = os.listdir(os.path.join(self.root,self.setting,"voc"))
        self.converter = STFTConverter()
    
    def __len__(self):
        length = len(self.title_list)  
        return length
    
    def __getitem__(self,index):
        data_name = self.title_list[index]
        
        inst = np.load(os.path.join(self.root,self.setting,"acc",data_name))
        voc = np.load(os.path.join(self.root,self.setting,"voc",data_name))
        mix = inst + voc
        print(np.max(mix))
        print(mix)
        
#        print("mix shape: ",mix.shape)
#        wv.write(os.path.join(self.root,"mix.wav"),44000,mix)
        
        if not self.direct:
            # Convert to stft shape: [2(actual_mag,rad),time,mag]
            inst = self.converter.get_stft(inst,comp=self.comp)
            mix = self.converter.get_stft(mix,comp=self.comp)
        
        # Convert numpy to torch tensor
        inst = torch.Tensor(inst)
        mix = torch.Tensor(mix)
        return mix, inst

#%%
if __name__ == "__main__":
    dataset = Dataset(r"C:\Users\USER\Desktop\musdb18_npy\musdb_npy_16000\train","test")
    mix, inst = dataset[5]
    
    