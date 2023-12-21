import itertools
import unicodedata
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from abc import ABC, abstractmethod
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from typing import Dict

from dsit import PLOTS_DIR, DEAD_NEURONS, NORMALIZATION_FACTORS, PHONES_PER_NEURON, EMBEDDINGS
from dsit.preprocessing import Data
from dsit.utils import get_json


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

    @abstractmethod
    def get_confusion_matrix(self) -> Dict:
        pass

    @abstractmethod
    def get_hidden_activation_values(self, data: Data) -> Dict:
        pass

    @abstractmethod
    def get_interpretable_activation_values(self, data: Data) -> Dict:
        pass


class CNN(Model):

    dead_neurons = get_json(DEAD_NEURONS)
    normalization_factors = get_json(NORMALIZATION_FACTORS)
    phonemes_per_neuron = get_json(PHONES_PER_NEURON)

    def __init__(self, model_path, debug=False):
        super().__init__(model_path, debug)

        self.model = tf.saved_model.load(self.model_path).signatures['serving_default']
        self.keras_model = self._build_model()
        self.cm = None
        self.data_name = None
        self.labels = None
        self.predictions = None

        self.layer_identifiers = ['activation_1', 'activation_2']
        self.layer_names = ['layer2', 'layer3']

    def predict(self, data: Data):
        # TODO Check this batch size handling works (also on the Data side!)
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

        # Use the specified labels order to link current cm layout to organized layout
        new_order = [label_names_existing.index(label) for label in Model.label_names_organized if label in label_names_existing]
        self.cm = self.cm[:, new_order][new_order, :]

        # Adding lines for phonemes that were not present in the input audio
        for i in range(len(Model.label_names_organized)):
            if Model.label_names_organized[i] not in label_names_existing:
                self.cm = np.insert(self.cm, i, 0.0, axis=1)
                self.cm = np.insert(self.cm, i, 0.0, axis=0)

        # Set 0 lines to NaN
        for i in range(self.cm.shape[0]):
            if np.count_nonzero(self.cm[i, :]) == 0:
                self.cm[i, :] = np.nan

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

    def get_hidden_activation_values(self, data: Data) -> Dict:
        # TODO Save that in model with variable so that it is not recomputed the second time!
        shape = data.get_frames_count()
        batch_size = data.compute_batch_size()

        # Getting the whole input data
        input_data = np.empty((shape, 11, 120, 1), dtype=float)
        iterator = data.get_fbank_labels()
        for i in tqdm(range(shape // batch_size)):
            batch_fbanks, _ = iterator.get_next()
            input_data[i * batch_size:(i + 1) * batch_size] = batch_fbanks['Conv1_input']

        # Retrieving hidden activations
        output = {}
        for layer, name in zip(self.layer_identifiers, self.layer_names):
            intermediate_layer_model = tf.keras.Model(inputs=self.keras_model.input,
                                                      outputs=self.keras_model.get_layer(layer).output)
            output[name] = intermediate_layer_model(input_data).numpy()

        return self._remove_dead_neurons(output)

    def _remove_dead_neurons(self, activation_values: Dict) -> Dict:
        # TODO Maybe to that in place to use less memory?
        output = {}
        for layer in self.layer_names:
            # Removing dead neurons
            output[layer] = np.delete(activation_values[layer], CNN.dead_neurons[layer], 1)
        return output

    def normalize_activation_values(self, activation_values: Dict) -> Dict:
        # TODO Maybe do that in place to use less memory?
        output = {}
        for layer in self.layer_names:
            # Dividing by normalization factors (they correspond to the maximum reached per neuron on BREF-Int)
            output[layer] = activation_values[layer] / np.array(CNN.normalization_factors[layer])
        return output

    def _build_model(self):
        model = tf.keras.models.Sequential()

        # the input is 11 consecutive frames
        input_shape = (11, 120, 1)
        nb_dense = 3
        dropout_rate = 0.4
        output_shape = 32
        regularizer = tf.keras.regularizers.l2(0.001)

        model.add(
            tf.keras.layers.Conv2D(32, kernel_size=(3, 5), activation='relu', input_shape=input_shape, name='Conv1',
                                   kernel_regularizer=regularizer))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(1, 3), name='Pool1'))
        model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 5), activation='relu', name='Conv2',
                                         kernel_regularizer=regularizer))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(1, 2), name='Pool2'))
        model.add(tf.keras.layers.Flatten())

        for i in range(1, nb_dense + 1):
            model.add(tf.keras.layers.Dense(1024, kernel_regularizer=regularizer))
            model.add(tf.keras.layers.Activation('relu'))
            model.add(tf.keras.layers.Dropout(dropout_rate))

        model.add(tf.keras.layers.Dense(output_shape, activation='softmax'))

        model.load_weights(f"{self.model_path}/variables/variables")

        if self.debug:
            print(model.summary())

        return model

    # TODO Improve the separation of content (ANPS should not be aware of CNN-specific information!
    # TODO Idea, maybe setup CI/CD to make sure same results are obtained, regardless of architecture?
    def get_interpretable_activation_values(self, data: Data) -> Dict:
        activations = self.get_hidden_activation_values(data)
        embeddings = get_json(EMBEDDINGS)

        for layer in self.layer_names:
            # For each layer, only keep rows of interpretable neurons
            # They need to be arranged in a particular order, contained in the EMBEDDINGS json file
            activations[layer] = activations[layer][:, embeddings[layer]]

        return activations


if __name__ == '__main__':
    audios = ["PFG13-TXT-16k_mono", "CCM-002595-01_L01", "I0MB0843", "I0MB0841", "I0MA0007", "I0MA0008",
              "I0MB0840", "I0MB0842", "I0MB0843", "I0MB0844", "I0MB0845"]

    cnn = CNN("dsit/models/cnn", debug=False)
    for audio in audios:
        preprocessed_data = Data(audio)
        preprocessed_data.preprocess()

        cnn.predict(preprocessed_data)
        cnn.plot_confusion_matrix()

        # print(cnn.get_hidden_activation_values(preprocessed_data))
        # print(cnn.get_interpretable_activation_values(preprocessed_data))
        break
