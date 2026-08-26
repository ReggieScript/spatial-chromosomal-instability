
import h5py
import sequoia.mask_utils as mask_utils
import os
import numpy as np
from tqdm import tqdm
from skimage.exposure.exposure import is_low_contrast
from scipy.ndimage import binary_dilation

from openslide import OpenSlide

def get_slide_name(svs_file):
    """
    The function get_slide_name recieves a .svs whole slide image file and returns the name of the slide.
    """
    slide_name = svs_file.split('/')[-1].split('.')[0]
    return slide_name

def patch_extraction(svs_file, patch_size, patch_path, mask_path, max_patches_per_slide = None):
    """
    The function patch_extraction recieves a .svs whole slide image file and extracts patches of size patch_size.
    The extracted patches are saved in the patch_path directory.
    """

    ## ERROR HANDLING

    ##TODO: Not only check that it exists, check also that it is not empty

    if not os.path.exists(patch_path):
        print("The patch_path directory does not exist.  Creating...")
        os.makedirs(patch_path)

    if not os.path.exists(mask_path):
        print("The mask_path directory does not exist.  Creating...")
        os.makedirs(mask_path)

    slide_name = get_slide_name(svs_file)
    path_hdf5 = os.path.join(patch_path, f"{slide_name}_{patch_size}.hdf5")

    if os.path.exists(path_hdf5):
        print(f"Patch extraction for {svs_file} has already been done. Skipping.")
        return path_hdf5



    ### PATCH EXTRACTION

    try:

        hdf = h5py.File(path_hdf5, 'w')

        slide = OpenSlide(svs_file)

        mask, mask_level = mask_utils.get_mask(slide)

        np.save(os.path.join(mask_path, f"{slide_name}_mask.npy"), mask)

        mask_level = len(slide.level_dimensions) - 1


        BACKGROUND_THRESHOLD = 0.2

        PATCH_LEVEL = 0
        ## 0 here (slide.level_dimensions[0]... ) is the PATCH_LEVEL

        ## try

        ratio_x = slide.level_dimensions[PATCH_LEVEL][0] / slide.level_dimensions[mask_level][0]
        ratio_y = slide.level_dimensions[PATCH_LEVEL][1] / slide.level_dimensions[mask_level][1]

        xmax, ymax = slide.level_dimensions[PATCH_LEVEL]

        resize_factor = float(slide.properties.get('aperio.AppMag', 20)) / 20.0
        if not slide.properties.get('aperio.AppMag', 20):
            print(f"Magniications for {slide_name} not found, using default magnification 20x")

        patch_size_resized = (int(resize_factor*patch_size), int(resize_factor * patch_size))
        print(f"Patch size for {slide_name}: {patch_size_resized}")

        i = 0
        indices = [(x,y) for x in range(0, xmax, patch_size_resized[0]) for y in range(0, ymax, patch_size_resized[0])] ### wait is this correct????

        if max_patches_per_slide is None:
            max_patches_per_slide = len(indices)

        np.random.seed(5)
        np.random.shuffle(indices)

        for x, y in tqdm(indices):

            x_mask = int(x/ratio_x)
            y_mask = int(y/ratio_y)

            if mask[x_mask, y_mask] ==1 :
                patch = slide.read_region((x,y), PATCH_LEVEL, patch_size_resized).convert('RGB')
                try:
                    mask_patch = mask_utils.get_mask_image(np.array(patch))
                    mask_patch = binary_dilation(mask_patch, iterations = 3)
                except Exception as e:
                    print(f"Error with slide {slide_name}: \n {e}")

                if (mask_patch.sum() > BACKGROUND_THRESHOLD * mask_patch.size) and not (is_low_contrast(patch)):
                    if resize_factor != 1.0:
                        patch = patch.resize((patch_size, patch_size))
                    patch = np.array(patch)
                    tile_name = f"{x}_{y}"
                    hdf.create_dataset(tile_name, data = patch)
                    i = i + 1
            if i >= max_patches_per_slide:
                break
        print(f"Finished patch extraction for {slide_name}: {i} patches extracted")
        hdf.close()

    except Exception as e:
        print(f"Quitting with error: \n {e}")
        os.remove(path_hdf5)
        print(f"Removed {path_hdf5}")

    ## except:

    return path_hdf5


    ##TODO: original sequoia adds a parallel process function after this...