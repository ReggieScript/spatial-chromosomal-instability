
##imports
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu
from scipy.ndimage import binary_dilation, binary_erosion
import numpy as np

def get_mask(slide, level='max', RGB_min=50):
    #read svs image at a certain level  and compute the otsu mask
    if level == 'max':
        level = len(slide.level_dimensions) - 1
    # note the shape of img_RGB is the transpose of slide.level_dimensions
    img_RGB = np.transpose(np.array(slide.read_region((0, 0),level,slide.level_dimensions[level]).convert('RGB')),
                        axes=[1, 0, 2])

    tissue_mask = get_mask_image(img_RGB, RGB_min)

    tissue_mask = binary_dilation(tissue_mask, iterations=3)
    tissue_mask = binary_erosion(tissue_mask, iterations=3)

    return tissue_mask, level

def get_mask_image(img_RGB, RGB_min=50):
    img_HSV = rgb2hsv(img_RGB)

    background_R = img_RGB[:, :, 0] > threshold_otsu(img_RGB[:, :, 0])
    background_G = img_RGB[:, :, 1] > threshold_otsu(img_RGB[:, :, 1])
    background_B = img_RGB[:, :, 2] > threshold_otsu(img_RGB[:, :, 2])
    tissue_RGB = np.logical_not(background_R & background_G & background_B)
    tissue_S = img_HSV[:, :, 1] > threshold_otsu(img_HSV[:, :, 1])
    min_R = img_RGB[:, :, 0] > RGB_min
    min_G = img_RGB[:, :, 1] > RGB_min
    min_B = img_RGB[:, :, 2] > RGB_min

    mask = tissue_S & tissue_RGB & min_R & min_G & min_B
    return mask

def mask():
    pass

