from .API import Model as api

import torch
import torch.nn as nn

# Parameters
n_gpu = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Declare API
API = api(n_gpu)  # [Optional] add path to pre-trained weight

# Load Model
model = api.getModel()
model = model.to(device)

# Use Model
