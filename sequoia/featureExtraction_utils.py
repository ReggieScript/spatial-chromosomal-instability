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
from sklearn.cluster import KMeans
from multiprocessing import Pool


class PatchDataset(torch.utils.data.Dataset):
    def __init__(self, images, transform):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx]).convert("RGB")
        return self.transform(image)


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

    def _feature_extraction_batch(self, images, batch_size=32, num_workers=4):
        dataset = PatchDataset(images, self.transforms_val)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
        )

        all_features = []
        with torch.no_grad():
            for batch in tqdm(loader):
                batch = batch.to(self.device)
                features = self.model(batch)
                all_features.append(features.detach().cpu().numpy())

        all_features = np.concatenate(all_features, axis=0)
        return all_features, len(all_features)

    def feature_extraction(self, slide_name, patch_path, max_patch_number=None,
                            batch_size=32, num_workers=4):
        slide = slide_name
        path_h5 = self.output_path
        os.makedirs(path_h5, exist_ok=True)

        slide_id = os.path.basename(slide_name).split(".svs")[0]
        out_file = os.path.join(self.output_path, slide_id + '.h5')
        print(out_file)
        if os.path.exists(out_file):
            print(f"Features already exist for {slide_name}. Skipping...")
            return out_file

        with h5py.File(patch_path, 'r') as f_read:
            keys = list(f_read.keys())
            if max_patch_number is not None and len(keys) > max_patch_number:
                keys = random.sample(keys, max_patch_number)

            images = [f_read[key][:] for key in keys]

        features, n_tiles = self._feature_extraction_batch(
            images, batch_size=batch_size, num_workers=num_workers
        )

        with h5py.File(out_file, "w") as f_write:
            f_write.create_dataset("uni_features", data=features)

        with open(os.path.join(path_h5, "complete_tile.txt"), 'w') as f_sum:
            f_sum.write(f"Total n patch = {n_tiles}")

        return out_file



    def k_means(self, feature_file, features_key = "uni_features", num_clusters = 100):
        
        try:
            f = h5py.File(feature_file, "r+")
        except Exception as e:
            print(f"Error - Cannot open file {feature_file} \n {e} ")

        try:
            features = f[features_key]
        except Exception as e:
            print(f"No {features_key} for {feature_file} \n {e}")
            f.close()

        if features.shape[0] < num_clusters:
            print(f"{feature_file}: fewer patches than clusters ({features.shape[0]} < {num_clusters})")
            f.close()
    

        if "cluster_features_uni" in f.keys():
            print(f"Warning: {feature_file}: cluster features already available, skipping")
            

        features_np = np.asarray(features)
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(features)
        clusters = kmeans.labels_

        mean_features = []
        for pos in tqdm(range(num_clusters)):
            indexes = np.where(clusters == pos)
            features_aux = features_np[indexes]
            mean_features.append(np.mean(features_aux, axis=0))

        mean_features = np.asarray(mean_features)

        try:
            f.create_dataset(f"cluster_features_uni", data=mean_features)
            f.close()
        except Exception as e:
            print(f"{feature_file}: Error creating cluster_features_uni")
            print(e)
            f.close()

