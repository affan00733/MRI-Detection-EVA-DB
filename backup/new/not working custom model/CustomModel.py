# coding=utf-8
# Copyright 2018-2020 EVA
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List

import pandas as pd
import numpy as np
import os
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from eva.utils.logging_manager import logger

from eva.models.catalog.frame_info import FrameInfo
from eva.models.catalog.properties import ColorSpace
from eva.udfs.abstract.pytorch_abstract_udf import PytorchAbstractClassifierUDF
from torch import Tensor
from torchvision import models
from torchvision.transforms import Compose, ToTensor, Resize
from PIL import Image
import sys

pytorch3dpath = "./EfficientNet-PyTorch-3D"
sys.path.append(pytorch3dpath)
from efficientnet_pytorch_3d import EfficientNet3D

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = EfficientNet3D.from_name("efficientnet-b0", override_params={'num_classes': 2}, in_channels=1)
        n_features = self.net._fc.in_features
        self.net._fc = nn.Linear(in_features=n_features, out_features=1, bias=True)
    
    def forward(self, x):
        out = self.net(x)
        return out

class CustomModel(PytorchAbstractClassifierUDF):

    @property
    def name(self) -> str:
        return "CustomModel"

    def setup(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Model()
        checkpoint = torch.load("FLAIR-e10-loss0.680-auc0.624.pth", map_location=device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        

    @property
    def input_format(self) -> FrameInfo:
        return FrameInfo(-1, -1, 3, ColorSpace.RGB)

    @property
    def labels(self) -> List[str]:
        return [
            'mgmt'
        ]

    def pred(self, image, model):
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        composed = Compose([
            Resize((256, 256)),            
            ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_batch =  composed(image)
        input_batch = input_batch.to('cpu').numpy()
        logger.warning(input_batch.shape)
        input_batch = torch.tensor(input_batch).float() 
        input_batch = input_batch.unsqueeze(0)
        logger.warning(type(input_batch))
        logger.warning(input_batch.shape)
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            model.to('cuda')

        with torch.no_grad():
            output = torch.sigmoid(model(input_batch.to(device))).cpu().numpy().squeeze()
            return output.tolist()

    def forward(self, frames: Tensor):
        outcome = pd.DataFrame()
        
        for frame in frames:
            image = torchvision.transforms.ToPILImage()(frame)
            output = self.pred(image, self.model)
            mask = output.cpu().numpy()[0] > 0.1 
            mask = output.astype(np.uint8)
            outcome = outcome.append({"results": mask}, ignore_index=True)
        return outcome