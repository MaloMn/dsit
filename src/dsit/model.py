import numpy as np
from abc import ABC, abstractmethod


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

    def predict(self, debug=False):
        shape = shapes_bs[key][0]  # This corresponds to the size or number of files in the dataset?
        bs = shapes_bs[key][1]
        labels = np.empty(shape, dtype=int)
        y_pred = np.empty(shape, dtype=int)

        predict_fn = tf.saved_model.load(model_path).signatures['serving_default']

        iterator = get_fbank_labels(data_path, bs)

        for i in tqdm(range(int(shape / bs))):
            batch_fbanks, lab = iterator.get_next()
            predictions = predict_fn(Conv1_input=batch_fbanks['Conv1_input'])
            pred = tf.argmax(predictions['dense_3'], axis=1)
            labels[i * bs:(i + 1) * bs] = lab
            y_pred[i * bs:(i + 1) * bs] = pred
        count = np.sum(labels == y_pred)
        acc = float(count) / len(y_pred)

        if debug:
            print("Number of correct prediction %d out of %d" % (count, len(y_pred)))
            print("Accuracy is {:.3f}".format(acc))
            print("Balanced accuracy is {:.3f}".format(balanced_accuracy_score(labels, y_pred)))
            print("Balanced Accuracy without silence is {:.3f}".format(
                balanced_accuracy_score(labels[labels != 0], y_pred[labels != 0])))

        return labels, y_pred, acc, float(count), len(y_pred)