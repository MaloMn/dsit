import tensorflow as tf
from typing import Dict
import numpy as np
import json

from dsit.model import Model
from dsit.preprocessing import Data


class ANPS:

    def __init__(self, model: Model, data: Data):
        self.activations = model.get_hidden_activation_values(data)
        self.normalized_activations = self.get_normalize_activations()

        with open("data/interpretable_neurons_clean.json", "r") as f:
            self.interpretable_neurons = json.load(f)

        with open("data/median_activation_per_neuron_per_phoneme.json", "r") as f:
            self.bref_median_activation_per_neuron_per_phoneme = json.load(f)

        with open("data/numeric_phones.json", "r") as f:
            self.numeric_phones = json.load(f)

        with open("data/phonetic_traits.json", "r") as f:
            self.phonetic_trait_phonemes = json.load(f)

        self.phonemes_list = list(self.bref_median_activation_per_neuron_per_phoneme["1"].keys())

        self.labels = data.labels_num_32
        self.correct_labels_with_30_phonemes()
        self.labels = self.labels[5:len(self.labels) - 5]
        self.current_median_activations = self.get_current_median_activations()

    def get_normalize_activations(self):
        normalized_activations = {}
        for layer in ["layer2", "layer3"]:
            divisor = np.max(self.activations[layer], axis=0)
            divisor[divisor == 0] = 1
            normalized_activations[layer] = self.activations[layer] / divisor

        return normalized_activations

    def correct_labels_with_30_phonemes(self):
        """
        Transforming the numeric phonemes to string phonemes (with 30 phonemes only)
        :return:
        """
        conversion_dict = {v: k for k, v in self.numeric_phones.items() if k in self.phonemes_list}
        conversion_dict[9] = "ai"  # TODO Find a better way of doing this!
        print(conversion_dict)

        for old_value, new_value in conversion_dict.items():
            self.labels[self.labels == old_value] = new_value

        self.labels = np.array(self.labels.Phone.tolist())

    def get_current_median_activations(self):
        """
        :return:
        """
        activations = {}
        for neuron in self.bref_median_activation_per_neuron_per_phoneme.keys():
            activations[str(neuron)] = {}
            for phoneme in self.bref_median_activation_per_neuron_per_phoneme[neuron].keys():
                # Layer 2 <=> neuron < 1024
                if (self.labels == phoneme).sum() <= 0:
                    continue

                neuron = int(neuron)
                if neuron < 1024:
                    activations[str(neuron)][phoneme] = np.median(tf.boolean_mask(self.normalized_activations['layer2'], self.labels == phoneme)[:, neuron])
                else:
                    activations[str(neuron)][phoneme] = np.median(tf.boolean_mask(self.normalized_activations['layer3'], self.labels == phoneme)[:, neuron - 1024])

        return activations

    def get_anps_scores(self) -> Dict[str, Dict[str, float]]:
        output = {}
        for vowel_or_consonant in self.phonetic_trait_phonemes.keys():
            output[vowel_or_consonant] = {}
            for trait in self.phonetic_trait_phonemes[vowel_or_consonant].keys():
                output[vowel_or_consonant][trait] = self._get_anps_score(vowel_or_consonant, trait)

        return output

    def _get_anps_score(self, vowel_or_consonant, phonetic_trait) -> float:
        """
        Compute the local ANPS score for the given phonetic trait.
        TODO The class of the trait can be deduced.
        :param vowel_or_consonant: "vowels" or "consonants"
        :param phonetic_trait: "+voiced" for instance
        :return: ANPS score (positive float)
        """
        phonemes = set(self.phonetic_trait_phonemes[vowel_or_consonant][phonetic_trait])
        # Adding neuro-detectors from layer 2 and 3
        neuro_detectors = self.interpretable_neurons[vowel_or_consonant]["layer2"][phonetic_trait] + \
                          [n + 1024 for n in self.interpretable_neurons[vowel_or_consonant]["layer3"][phonetic_trait]]

        dividend, divisor = 0, 0

        for n in neuro_detectors:
            # Computing for phonemes that appeared in the dataset!
            print(phonemes.intersection(set(self.current_median_activations[str(n)].keys())))
            for k in phonemes.intersection(set(self.current_median_activations[str(n)].keys())):
                print(n, k, self.current_median_activations[str(n)][k])
                dividend += self.current_median_activations[str(n)][k]
                divisor += self.bref_median_activation_per_neuron_per_phoneme[str(n)][k]

        return min(1.0, round(dividend / divisor, 2))

    def get_scores(self):
        """
        @param list_ph: ????
        @param disease_id: Disease identifier (Cereb, dyspho80, Park, Park-AHN, SLA, Temoin_CCM)
        @param list_n_trait_detector:
        @param layer_id: Fully connected layer identifier (1, 2 or 3)
        @return:
        """
        TRAIT_median = {}
        median = {}
        median_bref = {}
        normalized_activations_BREF = np.load(ANPS.normalized_activations_BREF_path.format(layer_id=layer_id))
        lab_list_bref = np.load(f"{ANPS.prefix}PHD/REVUE/Balanced_70_new_version_archiphoneme/lab_list.npy")

        for patient_id in list(ANPS.shapes[disease_id].keys()):
            # Loading normalized activations
            normalized_activations_p = np.load(
                f'{ANPS.prefix}PHD/INTERSPEECH2022/MIXTE_normalized_matrices_archi/{disease_id}/{patient_id}_Layer{layer_id}.npy')

            # Loading true corresponding labels
            true_labels_p = np.load(
                f'{ANPS.prefix}PHD/INTERSPEECH2022/MIXTE_activation_matrices/{disease_id}/{patient_id}_lab_list.npy')
            true_labels_p = ANPS._modify_true_labels_archiphoneme(true_labels_p)

            # eliminate nan due to lack of phones for some patients
            list_ph = list(set(list_ph) & set(true_labels_p))
            median[patient_id] = dict.fromkeys(list_ph, 0)
            median_bref[patient_id] = dict.fromkeys(list_ph, 0)

            for nid in list_n_trait_detector:
                normalized_activations_subset = normalized_activations_p[:, nid]
                normalized_activations_subset_bref = normalized_activations_BREF[:, nid]

                for phoneme in list_ph:
                    median[patient_id][phoneme] += np.median(
                        normalized_activations_subset[true_labels_p == phoneme])
                    median_bref[patient_id][phoneme] += np.median(
                        normalized_activations_subset_bref[lab_list_bref == phoneme])

            TRAIT_median[patient_id] = min(1, round(
                sum(median[patient_id].values()) / sum(median_bref[patient_id].values()), 2))

        return TRAIT_median, median, median_bref


if __name__ == '__main__':
    from dsit.model import CNN

    data = Data("CFS10-TXT-16k_mono")
    data.preprocess()
    print(ANPS(CNN("models/cnn"), data).get_anps_scores())
