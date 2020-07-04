# -*- coding: utf-8 -*-
"""
Created on Sat May 30 20:51:22 2020

@author: BOLD I.T
"""

import torchvision.models as models
from training import train_model
import build_dataset

mobilenet = models.mobilenet_v2(pretrained=True,progress=True)
model = mobilenet

train_model(model,build_dataset,build_dataset)