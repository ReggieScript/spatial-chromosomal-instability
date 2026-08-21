import os
import json
import argparse
import pickle
from tqdm import tqdm
import numpy as np
import torch
import torchvision
from torchvision import transforms
import h5py
import timm
from PIL import Image
import pdb
import random

import math
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


class FeatureExtaction():
    def __init__(self,output_path):

        self.output_path = output_path
        np.random.seed(42)
        torch.manual_seed(42)
        random.seed(42)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        
        transforms_val = transforms.Compose([
                        transforms.Resize(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),])
            
        local_dir = model_path
        self.model = timm.create_model("vit_large_patch16_224", img_size=224, patch_size=16, 
                                    init_values=1e-5, num_classes=0, dynamic_img_size=True)
        self.model.load_state_dict(torch.load(os.path.join(local_dir, 
                                    "pytorch_model.bin"), map_location="cpu"), strict=True)
        self.model.to(self.device)
        self.model.eval()

    def download_model(self):


    def resnet_feature_extraction(self, patch_path):

        self.model.eval()

        resnet_output_path = os.path.join(self.output_path, "resnet")

        if not os.path.exists(resnet_output_path):
            os.makedirs(resnet_output_path)

        if os.path.exists(os.path.join):
            pass
