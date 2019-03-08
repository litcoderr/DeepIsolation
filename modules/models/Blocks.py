import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from scipy import signal
import math

class Conv2dT(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, out_shape, stride, dilation):
        super(Conv2dT,self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.out_shape = np.array(out_shape)

        ## Model components
        self.conv = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, dilation=dilation)
        
    def forward(self, x):
        x = self.conv(x)
        # 3. compute batch normalization if use_bn
        x_shape = np.array(list(x.shape))[2:] # height width
        pad = x_shape - self.out_shape
        left = int(int(pad[1])/2)
        right = int(pad[1])-left
        x = x[:,:,:,left:int(x_shape[1])-right]
        
        return x

        
class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding, stride, dilation, use_bn,conv1d=False,activation=True):
        super(ConvBlock,self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.kernel_size = np.array(kernel_size)
        self.dilation = np.array(dilation)
        self.stride = np.array(stride)
        self.use_bn = use_bn
        self.activation = activation
        self.padding = padding
        self.conv1d = conv1d

        ## Changes as new input comes in
        if not conv1d:
            self.input_size = np.zeros((2))
        else:
            self.input_size = np.zeros((1))
        self.padding_size = None

        ## Model components
        if not conv1d:
            self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, dilation=dilation)
        else:
            self.conv = nn.Conv1d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 1. compute dynamic padding for flexible image size
        if self.padding == "same":
            x = self.padding_same(x)
        # 2. compute conv
        x = self.conv(x)
        # 3. compute batch normalization if use_bn
        if self.use_bn:
            x = self.bn(x)
        # 4. compute relu
        if self.activation:
            x = self.relu(x)

        return x

    def padding_same(self,x):
        input_size = np.array(list(x.size()))[2:] # only width and heigth (height,width)
        if not np.array_equal(input_size,self.input_size): # if different input size than compute padding size
            self.stride = 1
            new_padding = ((self.stride-1)*input_size-self.stride+self.kernel_size+(self.kernel_size-1)*(self.dilation-1))/2

            self.padding_size = new_padding # (height, width)
            self.input_size = input_size
        # compute padding
        if not self.conv1d:
            padder = nn.ZeroPad2d((int(np.ceil(self.padding_size[1])), int(np.floor(self.padding_size[1])), int(np.ceil(self.padding_size[0])), int(np.floor(self.padding_size[0]))))
            x = padder(x)
        else:
            right = int(np.ceil(self.padding_size[0]))
            left = int(np.floor(self.padding_size[0]))
            pad_size = (left,right)
            x = nn.functional.pad(x,pad_size,'constant',0)
        return x
    
class ComplexConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(ComplexConv,self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.padding = padding

        ## Model components
        self.conv_re = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.conv_im = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
    def forward(self, x): # shpae of x : [batch,2,channel,axis1,axis2]
        real = self.conv_re(x[:,0]) - self.conv_im(x[:,1])
        imaginary = self.conv_re(x[:,1]) + self.conv_im(x[:,0])
        output = torch.stack((real,imaginary),dim=1)
        return output

class FFTConv(nn.Module):
    def __init__(self,n_fft,debug=False):
        super(FFTConv,self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        I = np.eye(n_fft) # I matrix
        W = np.fft.fft(I) # (2048,2048)
        Real = np.real(W[:,:int(n_fft/2)]) # currently: [filter_size,output_channel=int(n_fft/2)]
        Imag = np.imag(W[:,:int(n_fft/2)]) # currently: [filter_size,output_channel=int(n_fft/2)]
        Real *= np.expand_dims(signal.get_window('hanning', n_fft), axis=1)
        Imag *= np.expand_dims(signal.get_window('hanning', n_fft), axis=1)
        
        # [filtersize=n_fft, input_size=1, output_size=n_fft/2]
        Real = np.expand_dims(Real,axis=1) # Untrainable Parameters
        Imag = np.expand_dims(Imag,axis=1) # Untrainable Parameters
        Real = Variable(torch.Tensor(Real)).to(device) # Make to variable
        Imag = Variable(torch.Tensor(Imag)).to(device) # Make to variable
        
        if debug == True:
            print("Real Filter shape : ",Real.shape)
            print("Imag Filter shape : ",Imag.shape)
            
        self.real_conv1d = ConvBlock(1,int(n_fft/2),n_fft,padding="same",stride=int(n_fft/4),dilation=1,use_bn=False,conv1d=True,activation=False)
        self.imag_conv1d = ConvBlock(1,int(n_fft/2),n_fft,padding="same",stride=int(n_fft/4),dilation=1,use_bn=False,conv1d=True,activation=False)
        
        # reassign weights
        self.real_conv1d.conv.weight.data = Real.permute(2,1,0)
        self.imag_conv1d.conv.weight.data = Imag.permute(2,1,0)
        
    def forward(self,x): # (batch, 1, input_size)
        real = self.real_conv1d(x)
        imag = self.imag_conv1d(x)
        result = torch.cat((real.unsqueeze(1),imag.unsqueeze(1)),1)
        return result.permute(0,1,3,2) # [batch, 2, timestep = frame, freq = n_fft/2]
    
class IFFTConv(nn.Module):
    def __init__(self,n_fft,out_shape,debug=False):
        super(IFFTConv,self).__init__()
        self.debug = debug
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        ## Get Filter and make Hermitian Matrix
        I = np.eye(n_fft)
        W = np.fft.fft(I)
        real = np.real(W[:,:int(n_fft/2)]) # filtersize=nfft,freq=nfft/2
        imag = np.imag(W[:,:int(n_fft/2)]) # filtersize=nfft,freq=nfft/2
        
        # split
        real_dc = real[:,0:1]/np.sqrt(n_fft)
        real_rest = real[:,1:]/np.sqrt(n_fft)
        imag_dc = imag[:,0:1]/np.sqrt(n_fft)
        imag_rest = imag[:,1:]/np.sqrt(n_fft)
        
        # to [N sample = filtersize, in_channel=1, freq = out_channel]
        real_dc = np.expand_dims(real_dc, axis=1)
        real_rest = np.expand_dims(real_rest, axis=1)
        imag_dc = np.expand_dims(imag_dc, axis=1)
        imag_rest = np.expand_dims(imag_rest, axis=1)
        
        # to [1, N sample = filtersize, in_channel=1, freq = out_channel]
        C_dc = np.expand_dims(real_dc, axis=0)
        C_rest = np.expand_dims(real_rest, axis=0)
        D_dc = -np.expand_dims(imag_dc, axis=0)
        D_rest = -np.expand_dims(imag_rest, axis=0)
        
        C = np.concatenate(((C_dc)/2,C_rest),axis=3)
        D = np.concatenate(((D_dc)/2,D_rest),axis=3)
        F_real = np.concatenate((C,-D),axis=3) # height, width, output_ch, input_ch
        F_real = Variable(torch.Tensor(F_real)).to(self.device)
        F_real = F_real.permute((3,2,0,1)) # input_ch,output_ch,height,width
        in_ch,out_ch,height,width = list(F_real.shape)
        
        self.real_conv = Conv2dT(in_ch,out_ch,(height,width),out_shape,stride=(1,int(n_fft/4)),dilation=1)

        # Reload pre made weights
        self.real_conv.conv.weight.data = F_real
        
        self.rescale = Rescale(n_fft,out_shape)
    
    def forward(self,x): # [batch,2,time,freq=n_fft/2]
        input_real = x[:,0:1,:,:] # [batch,1,time,freq]
        input_imag = x[:,1:2,:,:]
        x = torch.cat((input_real,input_imag),dim=3) # [batch,height=1,width=time,channel=2*freq=n_fft]
        x = x.permute((0,3,1,2))# [batch,channel,height,width]
        
        real = self.real_conv(x) # batch, channel=1, width=1, height=wav
        real = 2*real.squeeze(1) # batch, width, height
        
        real = self.rescale(real)
        
        return real
    
class Rescale(nn.Module):
    def __init__(self,n_fft,out_shape):
        super(Rescale,self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        stride = int(n_fft/2)
        audio_size = out_shape[1]
        N=n_fft
        padding_size= int((math.ceil(audio_size/stride)-1)*stride + N - audio_size)
        scale_factor = np.zeros([(audio_size + padding_size)], dtype=np.float32)
        hanning_square= np.square(signal.get_window('hanning', N))
        for i in range(int(audio_size / stride)):
            scale_factor[(i * stride): (i*stride +N)] += hanning_square
        scale_factor = scale_factor[int(padding_size / 2): audio_size + int(padding_size / 2)] # audio size
        scale_factor = np.expand_dims(scale_factor,axis=0)
        self.register_buffer("scale_factor",torch.Tensor(np.expand_dims(scale_factor, axis=0)))
        
    def forward(self,x):
        x = x/(self.scale_factor+1e-7)
        return x
    
#%%
if __name__ == "__main__":
    from modules.dataset.musdb18 import Dataset
    import scipy.io.wavfile as wv
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    mix = torch.randn((3,1,480000))
    print(mix.shape)
    fft = FFTConv(n_fft=4096).to(device)
    ifft = IFFTConv(n_fft=4096,out_shape=[1,480000]).to(device)
    
    mix = mix.to(device)
    output = fft(mix)
    print(mix.shape)
    print(output.shape)
    output = ifft(output)
    
    print(output.shape)
    
    mix = mix.cpu().detach().numpy()[0,0,:]
    output = output.cpu().detach().numpy()[0,0,:]
    print(mix.shape)
    
#    wv.write(r"C:\Users\USER\Desktop\project\voice-isolation/mix.wav",44000,mix)
#    wv.write(r"C:\Users\USER\Desktop\project\voice-isolation/gen.wav",44000,output)
    
        
        
        
        
        
        
        
        
