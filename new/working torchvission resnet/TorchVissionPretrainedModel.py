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


class TorchVissionPretrainedModel(PytorchAbstractClassifierUDF):

    @property
    def name(self) -> str:
        return "TorchVissionPretrainedModel"

    def setup(self):
        
        self.model = torchvision.models.resnet152(pretrained=True)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 2) # binary classification (num_of_class == 2)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_state = torch.load("model.pth", map_location=device)
        self.model.load_state_dict(model_state)
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
        preprocess = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0)

        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            model.to('cuda')

        with torch.no_grad():
            output = model(input_batch)
            _, preds = torch.max(output, 1)
            return preds

    def forward(self, frames: Tensor):
        outcome = pd.DataFrame()
        for frame in frames:
            image = torchvision.transforms.ToPILImage()(frame)
            output = self.pred(image, self.model)
            mask = output.cpu().numpy()[0] 
            mask = mask.astype(np.uint8)
            outcome = outcome.append({"results": mask}, ignore_index=True)
        return outcome