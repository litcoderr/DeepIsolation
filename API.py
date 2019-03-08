import torch
import torch.nn as nn

class Model:
    def __init__(self, n_gpu, path=""):
        self.pretrained_model = path  # pre-trained weight path
        self.n_gpu = n_gpu  # number of GPUs

    def getModel(self):
        # 1. Generate Model Instance

        # 2. return Model Instance as torch.nn.module
        return None

