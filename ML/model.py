from __future__ import print_function, division
from collections import OrderedDict
import copy
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from torch import nn

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.backends.cudnn as cudnn
import warnings
warnings.filterwarnings('ignore')


cudnn.benchmark = True
plt.ion()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Python code (.py): initialization model
model = torchvision.models.mobilenet_v2(pretrained=True)

class Model:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path

    def load_model(self):
        model = torchvision.models.mobilenet_v2(pretrained=True)
        # model = tf.keras.models.load_model(self.model_path) # tensorflow
        model = model.load_state_dict(
            torch.load("/models/model_Mobilenetv2.pt"))
        return model
