from API import Model as API

import torch
import torch.nn as nn

# Parameters
n_gpu = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Declare API
api = API(n_gpu)  # [Optional] add path to pre-trained weight

# Load Model
model = api.getModel()

# Use Model


if __name__ == '__main__':
    x = torch.randn((1,1,100,1000)) # batch, 1, x, y

    y = model(x)

    print(y.shape)
