import numpy as np
import tensorflow as tf
from tqdm import tqdm
from abc import ABC, abstractmethod
from sklearn.metrics import balanced_accuracy_score

from dsit.preprocessing import Data
from dsit.utils import get_batch_size


class Model(ABC):

    def __init__(self, model_path):
        self.model_path = model_path

    @abstractmethod
    def load_model(self) -> None:
        pass

    @abstractmethod
    def apply_model(self, *frames) -> int:
        pass


class CNN(Model):

    def __init__(self, model_path):
        super().__init__(model_path)

    def load_model(self) -> bool:
        pass

    def apply_model(self, *frames) -> int:
        Y, Y_pred, acc, nb_true, nb_tot = self.predict(self.model_path, key, data_path, shapes_bs)
        return 0

    def predict(self, data: Data, debug=False):
        shape = shapes_bs[key][0]  # This corresponds to the size or number of files in the dataset?
        batch_size = get_batch_size(shape)
        labels = np.empty(shape, dtype=int)
        y_pred = np.empty(shape, dtype=int)

        predict_fn = tf.saved_model.load(self.model_path).signatures['serving_default']

        iterator = data.get_fbank_labels()

        for i in tqdm(range(shape // batch_size)):
            batch_fbanks, lab = iterator.get_next()
            predictions = predict_fn(Conv1_input=batch_fbanks['Conv1_input'])
            pred = tf.argmax(predictions['dense_3'], axis=1)
            labels[i * batch_size:(i + 1) * batch_size] = lab
            y_pred[i * batch_size:(i + 1) * batch_size] = pred

        count = np.sum(labels == y_pred)
        acc = float(count) / len(y_pred)

        if debug:
            print("Number of correct prediction %d out of %d" % (count, len(y_pred)))
            print("Accuracy is {:.3f}".format(acc))
            print("Balanced accuracy is {:.3f}".format(balanced_accuracy_score(labels, y_pred)))
            print("Balanced Accuracy without silence is {:.3f}".format(
                balanced_accuracy_score(labels[labels != 0], y_pred[labels != 0])))

        return labels, y_pred, acc, float(count), len(y_pred)


if __name__ == '__main__':
    cnn = CNN("models/cnn")
    cnn.predict(Data("path"), debug=True)
