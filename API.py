from modules.models.v2 import V2 as MainModel
from modules.util.googledrive import Downloader

import torch
import torch.nn as nn

import os
from collections import OrderedDict

class Model:
    def __init__(self, n_gpu):
        #TODO set pretrained model path
        self.path = self.get_path()  # pre-trained weight path
        self.n_gpu = n_gpu  # number of GPUs

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.pretrained_url_id = "1ZzvYv_KRmf6arhqYUomRgAt3iZBtNyZu"
        self.downloader = Downloader()

    def getModel(self):
        # 1. Make Model Instance
        model = MainModel()
        model = model.to(self.device)
        model = model.eval()

        # ******** Download from google drive ********
        if not os.path.exists(self.path):
            self.downloader.download(self.pretrained_url_id, self.path)

        # 1-1. check if need data parallel
        if self.n_gpu >= 2:
            model = nn.DataParallel(model)

        # 1-2. Load pre-trained model
        state_dict = torch.load(self.path, map_location=self.device)

        # 1-3. check if need to manipulate
        if self.n_gpu < 2:
            temp_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove 'module.'
                temp_dict[name] = v
            state_dict = temp_dict

        # 2. Load pre-trained to module
        model.load_state_dict(state_dict)

        return model

    def get_path(self):
        root = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(root,"pretrained","model")
        return path


if __name__ == '__main__':
    api = Model(1)

    print(api.get_path())

