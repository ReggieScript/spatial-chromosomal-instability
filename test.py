from sequoia.sequoia import Sequoia

import os
import openslide


if __name__ == "__main__":
    experiment = Sequoia("./data", "")
    experiment.predict("/Users/reginacrespo/Documents/Research Assistant/spatial-chromosomal-instability/data/TCGA-BH-A0EA-01Z-00-DX1.85FF2B48-2AF7-4C15-A7E6-FCA68CAB76C7.svs",
                    #max_patches_per_slide= 50 ## TEST 
                    )

