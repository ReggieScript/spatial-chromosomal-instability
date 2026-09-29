import numpy as np
from .utils import ViS
from tqdm import tqdm

class SequoiaModel():
    def __init__(self, device, cancer):
        self.device = device
        self.cancer = cancer

    def predict(self, image):
        self.model.eval()
        preds = []

        image = image.to(self.device)
        pred = self.model(image)
        preds.append(pred.detach().cpu().numpy())

        preds = np.concatenate(preds, axis = 0)

        return preds


    def fold_prediction(self, features, folds = 5):

        res_preds = []

        print("==========\n")
        print("BEGINNING SEQUOIA PREDICTION...\n")
        print("==========\n")

        if features.dim() == 2:
            features = features.unsqueeze(0)   # (100, dim) -> (1, 100, dim)

        for fold in tqdm(range(folds)):

            # load model from huggingface
            self.model = ViS.from_pretrained(f"gevaertlab/sequoia-{self.cancer}-{fold}")
            self.model.to(self.device)

            preds = self.predict(features)

            res_preds.append(preds)

        print("single fold:", np.asarray(res_preds[0]).shape)
        print("stacked:", np.asarray(res_preds).shape)

        avg_preds = np.mean(res_preds, axis = 0)

        print("\nSEQUOIA PREDICTION FINALIZED...\n")

        print(avg_preds)

        return avg_preds

    

        

        


    
        