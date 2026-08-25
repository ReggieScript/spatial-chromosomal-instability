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
import random


class FeatureExtaction():
    def __init__(self,output_path):

        self.output_path = output_path
        np.random.seed(42)
        torch.manual_seed(42)
        random.seed(42)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        
        self.transforms_val = transforms.Compose([
                        transforms.Resize(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),])
            
        self.model = timm.create_model(
                "hf-hub:MahmoodLab/uni",
                pretrained=True,
                init_values=1e-5,
                dynamic_img_size=True,
            )
        
        self.model.to(self.device)
        self.model.eval()

    def feature_extraction(self, slide_name, path, max_patch_number = None):

        slide = slide_name

        path_h5 = os.path.join(self.output_path)

        if not os.path.exists(path_h5):
            os.makedirs(path_h5)
        print(path)
        print(path_h5)
        # try:
        with h5py.File(path, 'r') as f_read:
            keys = list(f_read.keys())
            if max_patch_number is not None:
                if len(keys) > max_patch_number: ##TODO: Why do we need the max patch number here??
                    keys = random.sample(keys, max_patch_number) ## Answer: Sanity check
            features_tiles = []
            for key in tqdm(keys):
                image = f_read[key][:]
                image = Image.fromarray(image).convert("RGB")
                image = self.transforms_val(image).to(self.device)
                with torch.no_grad():
                    features = self.model(image[None, :])
                    features_tiles.append(features[0].detach().cpu().numpy())
            features_tiles = np.asarray(features_tiles)
            n_tiles = len(features_tiles)

            f_write = h5py.File(os.path.join(path_h5, slide+'.h5'), "w")
            dset = f_write.create_dataset("uni_features", data = features_tiles)
            f_write.close()

            with open(os.path.join(path_h5, "complete_tile.txt"), 'w') as f_sum:
                f_sum.write(f"Total n patch = {n_tiles}")
        # except Exception as e:
        #     print(f"Feature extraction for {slide} failed: \n {e}")
