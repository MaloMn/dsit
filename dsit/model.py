import itertools
import unicodedata
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from abc import ABC, abstractmethod
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from typing import Dict

from dsit import PLOTS_DIR
from dsit.preprocessing import Data


class Model(ABC):
    label_names_organized = ['silence', 'aa', 'ai', 'ei', 'ee', 'au', 'ou', 'uu', 'ii', 'an', 'on', 'un', 'ww', 'uy',
                             'yy', 'll', 'rr', 'nn', 'mm', 'gn', 'pp', 'tt', 'kk', 'bb', 'dd', 'gg', 'ff', 'ss', 'ch',
                             'vv', 'zz', 'jj']
    label_names = ['silence', 'aa', 'ai', 'an', 'au', 'bb', 'ch', 'dd', 'ee', 'ei', 'ff', 'gg', 'gn', 'ii', 'jj', 'kk',
                   'll', 'mm', 'nn', 'on', 'ou', 'pp', 'rr', 'ss', 'tt', 'un', 'uu', 'uy', 'vv', 'ww', 'yy', 'zz']
    label_names_organized_phonetic = ['sil', 'a', chr(603), 'e', chr(339), chr(596), 'u', 'y', 'i',
                                      unicodedata.normalize('NFD', '\N{LATIN SMALL LETTER A}\N{COMBINING TILDE}'),
                                      unicodedata.normalize('NFD', '\N{LATIN SMALL LETTER OPEN O}\N{COMBINING TILDE}'),
                                      unicodedata.normalize('NFD', '\N{LATIN SMALL LETTER OPEN E}\N{COMBINING TILDE}'),
                                      'w', chr(613), 'j', 'l', chr(641), 'n', 'm', chr(626), 'p', 't', 'k', 'b', 'd',
                                      'g', 'f', 's', chr(643), 'v', 'z', chr(658)]

    def __init__(self, model_path, debug=False):
        self.model_path = model_path
        self.debug = debug

    @abstractmethod
    def predict(self, data: Data) -> None:
        pass

    def get_confusion_matrix(self) -> Dict:
        pass


class CNN(Model):

    def __init__(self, model_path, debug=False):
        super().__init__(model_path, debug)

        self.model = tf.saved_model.load(self.model_path).signatures['serving_default']
        self.cm = None
        self.data_name = None
        self.labels = None
        self.predictions = None

    def load_model(self) -> bool:
        pass

    def apply_model(self, *frames):
        pass

    def predict(self, data: Data):
        # TODO Fix this batch size handling
        shape = data.get_frames_count()
        batch_size = data.compute_batch_size()

        if self.debug:
            print(f"{shape // batch_size} batch{'es' if shape // batch_size > 1 else ''} of size {batch_size}")

        labels, y_pred = np.empty(shape, dtype=int), np.empty(shape, dtype=int)
        # Loading model and data
        iterator = data.get_fbank_labels()

        for i in tqdm(range(shape // batch_size)):
            batch_fbanks, lab = iterator.get_next()
            predictions = self.model(Conv1_input=batch_fbanks['Conv1_input'])
            pred = tf.argmax(predictions['dense_3'], axis=1)

            # stacks prediction and true labels on top of each other
            labels[i * lab.shape[0]:(i + 1) * lab.shape[0]] = lab
            y_pred[i * pred.shape[0]:(i + 1) * pred.shape[0]] = pred

        count = np.sum(labels == y_pred)
        acc = float(count) / len(y_pred)

        if self.debug:
            print("Number of correct prediction %d out of %d" % (count, len(y_pred)))
            print("Accuracy is {:.3f}".format(acc))
            print("Balanced accuracy is {:.3f}".format(balanced_accuracy_score(labels, y_pred)))
            print("Balanced Accuracy without silence is {:.3f}".format(
                balanced_accuracy_score(labels[labels != 0], y_pred[labels != 0])))

        self.data_name = data.stem
        self.labels, self.predictions = labels, y_pred

    def compute_confusion_matrix(self):
        self.cm = confusion_matrix(self.labels, self.predictions, normalize="true")

        idx_existing = list(set(self.labels) | set(self.predictions))
        label_names_existing = list(np.array(Model.label_names)[idx_existing])

        for i in range(len(Model.label_names_organized)):
            if Model.label_names_organized[i] not in label_names_existing:
                self.cm = np.insert(self.cm, i, 0.0, axis=1)
                self.cm = np.insert(self.cm, i, 0.0, axis=0)

    def plot_confusion_matrix(self, savefig=True, file='', cmap=plt.cm.pink_r):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        self.compute_confusion_matrix()

        fig = plt.figure(dpi=75)
        fig.set_size_inches(15, 12, forward=True)

        plt.imshow(self.cm, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
        plt.title(file, fontsize='20')
        plt.colorbar()

        tick_marks = np.arange(len(Model.label_names_organized_phonetic))
        plt.xticks(tick_marks, Model.label_names_organized_phonetic, fontsize='14')  # ,fontweight='bold'
        plt.yticks(tick_marks, Model.label_names_organized_phonetic, fontsize='14')  # ,fontweight='bold'

        for i, j in itertools.product(range(self.cm.shape[0]), range(self.cm.shape[1])):
            plt.text(j, i, format(self.cm[i, j] * 100, '.1f'),
                     horizontalalignment="center",
                     color="white" if self.cm[i, j] > 0.6 else "black")  # , fontsize='xx-large',fontweight='bold'

        plt.tight_layout(pad=3)
        plt.ylabel('True labels', fontsize='18')
        plt.xlabel('Predicted labels', fontsize='18')

        if savefig:
            plt.savefig(f"{PLOTS_DIR}{self.data_name}.png", bbox_inches='tight', pad_inches=0.5)
            print(f"Confusion matrix was saved at {PLOTS_DIR}{self.data_name}.png")

    def get_confusion_matrix(self):
        self.compute_confusion_matrix()

        return {
            "labels": Model.label_names_organized_phonetic,
            "matrix": self.cm.tolist()
        }


if __name__ == '__main__':
    audios = ["I0MB0843", "I0MB0841", "PME20-TXT-16k_mono", "I0MA0007", "I0MA0008",
              "I0MB0840", "I0MB0842", "I0MB0843", "I0MB0844", "I0MB0845"]

    for audio in audios:
        cnn = CNN("models/cnn", debug=True)
        data = Data(audio)
        data.preprocess()

        cnn.predict(data)
        cnn.plot_confusion_matrix()
