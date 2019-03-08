# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

## Deep Lab
from modules.models.modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modules.models.modeling.aspp import build_aspp
from modules.models.modeling.decoder import build_decoder
from modules.models.modeling.backbone import build_backbone

## Front and Backend Convolution
from modules.models.Blocks import FFTConv, IFFTConv


class V2_1(nn.Module): # Regular model
    def __init__(self,n_fft,output_length,freeze_bn=False):
        super(V2_1,self).__init__()
        self.fft = FFTConv(n_fft=n_fft) # input: [batch,1,input_length = output_length]
        self.ifft = IFFTConv(n_fft=n_fft,out_shape=[1,output_length]) # input: [batch,2,timestep,frequency]
        self.deeplab = RegularDeepLab(backbone='xception', output_stride=16,freeze_bn=freeze_bn)
        
        self.bound = nn.Tanh()
        
        for param in self.fft.parameters():
            param.requires_grad = False
        
    def forward(self,x): # input: [batch,1,input_length = output_length] output: [batch,1,input_length = output_length]
        # FFT Transformation
        fft_encoded = self.fft(x)
        
        # Mask Generation
        mask = self.deeplab(fft_encoded)
        bounded_mask = self.bound(mask)
        
        masked_input = bounded_mask * fft_encoded
        waveform_output = self.ifft(masked_input)
        return waveform_output
    
class V2_2(nn.Module): # Complex model
    def __init__(self,n_fft,output_length): # output_length = original input length
        super(V2_2,self).__init__()
        self.fft = FFTConv(n_fft=n_fft) # input: [batch,1,input_length = output_length]
        self.ifft = IFFTConv(n_fft=n_fft,out_shape=[1,output_length]) # input: [batch,2,timestep,frequency]
        
    def forward(self,x):
        pass

class V2_3(nn.Module):
    def __init__(self, backbone='xception', output_stride=16, num_classes=1,
                 sync_bn=True, freeze_bn=False):
        super(V2_3,self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)
        
        self.tanh = nn.Tanh()
        
        if freeze_bn:
            self.freeze_bn()
            
    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        
        x = self.tanh(x)
        input = input*x
        return input

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p
    
class RegularDeepLab(nn.Module):
    def __init__(self, backbone='xception', output_stride=16, num_classes=2,
                 sync_bn=True, freeze_bn=False):
        super(RegularDeepLab,self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)

        if freeze_bn:
            self.freeze_bn()
            
    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

class ComplexDeepLab(nn.Module):
    def __init__(self, backbone='xception', output_stride=16, num_classes=2,
                 sync_bn=True, freeze_bn=False):
        super(RegularDeepLab,self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)

        if freeze_bn:
            self.freeze_bn()
            
    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p
    
#%%
if __name__ == "__main__":
    model = V2_3().eval()

    input = torch.rand(1, 1, 100, 100)
    output = model(input)
    print(output.size())





