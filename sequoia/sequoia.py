
## imports

import os
import sequoia.patch_utils as patch_utils

class Sequoia():

    def __init__(self, main_path, path_to_model):

        self.main_path = main_path
        self.path_to_model = path_to_model

        self.patch_path = os.path.join(self.main_path, "patches")
        if not os.path.isdir(self.patch_path):
            os.makedirs(self.patch_path)

        self.feature_path =  os.path.join(self.main_path, "features")
        if not os.path.isdir(self.feature_path):
            os.makedirs(self.feature_path)
        
        self.mask_path = os.path.join(self.main_path, "masks")
        if not os.path.isdir(self.mask_path):
            os.makedirs(self.mask_path)
        

    def predict(self, file, patch_size=256, max_patches_per_slide = None):
        """
        The function predict recieves a .svs whole slide image file and returns the output prediction.
        """

        ### PATCH EXTRACTION

        patch_utils.patch_extraction(file, patch_size, self.patch_path, self.mask_path, max_patches_per_slide)

        ### 

    def spatial_predict(self, file):

        """
        The function recieves a .svs whole slide image file and returns the output spatial prediction.
        """

    def batch_predict():
        pass