from .API import Model as api

import torch
import torch.nn as nn

# Parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Declare API
API = api()

# Load Model
model = api.getModel()
model = model.to(device)

# Use Model