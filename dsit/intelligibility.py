from dsit.model import Model
from dsit.preprocessing import Data
import numpy as np
import tensorflow as tf


class Intelligibility:

    def __init__(self, model: Model, data: Data):
        self.model = model
        self.data = data

    def reshape_input(self):

        #get output from CNN
        extract = self.model.get_interpretable_activation_values(self.data)

        #Grouping 1 seacond samples, ingoring the last after decimal
        emb = np.concatenate([extract['layer2'],extract['layer3']],axis=1)
        num_sample = emb.shape[0]//100
        #reshape to the actual input size of tf.Pooling2D
        emb = emb[:(num_sample*100)].reshape(num_sample,1,100,emb.shape[1])
        return emb


    def intel(self) -> float:

        #Initnialize intel model
        
        model_path = "/data/coros2/ProjetPathoLoc/Patho/Work/RUGBI/PHD/" \
                    "TRANSFER_LEARNING/MODELS_embeddings/NO_ATTENTION/" \
                    "intel/EXP6/export/best_exporter/1678073160"

        pred = tf.saved_model.load(model_path).signatures['serving_default']

        # Preparing input
        emb = self.reshape_input()
        predictions= pred(input=tf.convert_to_tensor(emb, dtype=tf.float32))
        #compute average of all second decisions
        intel_score=np.average(predictions['output'].numpy())
        return intel_score


    def sev(self) -> float:

        #Initnialize intel model
        model_path = "/data/coros2/ProjetPathoLoc/Patho/Work/RUGBI/PHD/" \
                    "TRANSFER_LEARNING/MODELS_embeddings/NO_ATTENTION/" \
                    "sev/EXP3/export/best_exporter/1678072575"
        # Preparing input
        emb = self.reshape_input()
        
        pred = tf.saved_model.load(model_path).signatures['serving_default']
        predictions= pred(input=tf.convert_to_tensor(emb, dtype=tf.float32))
        #compute average of all second decisions
        sev_score=np.average(predictions['output'].numpy())
        return sev_score

    def get_score(self) -> float:
        return 10.0


if __name__ == '__main__':
    pass

