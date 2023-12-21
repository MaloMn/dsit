import tensorflow as tf
import numpy as np

from dsit import INTELLIGIBILITY_MODEL_DIR, SEVERITY_MODEL_DIR
from dsit.preprocessing import Data
from dsit.model import Model


class Intelligibility:

    def __init__(self, model: Model, data: Data):
        self.model = model
        self.data = data
        self.embeddings = self.reshape_input()

    def reshape_input(self):
        # Get output from CNN
        extract = self.model.get_interpretable_activation_values(self.data)

        # Grouping 1-second samples, ignoring the last after decimal
        emb = np.concatenate([extract['layer2'], extract['layer3']], axis=1)
        num_sample = emb.shape[0] // 100

        remaining_frames = emb.shape[0]%100
        #Round up the last second if have more than 0.5
        if remaining_frames >=50:
            num_sample += 1
            #Create a artificial second with missing info from previous second
            add_sec = emb[-100:,:]
            emb = np.concatenate((emb[:-remaining_frames,:],add_sec), axis = 0)
        # Reshape to the actual input size of tf.Pooling2D
        return emb[:(num_sample * 100)].reshape(num_sample, 1, 100, emb.shape[1])

    def get_intelligibility_score(self) -> float:
        # Initialize intelligibility model
        pred = tf.saved_model.load(INTELLIGIBILITY_MODEL_DIR).signatures['serving_default']
        predictions = pred(input=tf.convert_to_tensor(self.embeddings, dtype=tf.float32))

        # Compute average of all second decisions
        intel_score = np.average(predictions['output'].numpy())

        return round(float(intel_score), 2)

    def get_severity_score(self) -> float:
        # Initialize severity model
        pred = tf.saved_model.load(SEVERITY_MODEL_DIR).signatures['serving_default']
        predictions = pred(input=tf.convert_to_tensor(self.embeddings, dtype=tf.float32))

        # Compute average of all second decisions
        sev_score = np.average(predictions['output'].numpy())

        return round(float(sev_score), 2)


if __name__ == '__main__':
    pass
