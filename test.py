from sequoia.sequoia import Sequoia



if __name__ == "__main__":
    experiment = Sequoia("./data")
    experiment.predict('/Volumes/ADATA HD710/Regina Crespo/slides/TCGA/TCGA-3C-AALK-01Z-00-DX1.svs',
                    max_patches_per_slide= 100 ## TEST 
                    )

