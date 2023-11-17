import tensorflow as tf
from typing import Dict, List
from numba import jit
import numpy as np
import json

from dsit.model import Model
from dsit.preprocessing import Data
from dsit.utils import get_json


class ANPS:

    # TODO Put these paths in the __init__
    interpretable_neurons = get_json("data/interpretable_neurons_clean.json")
    bref_median_activation_per_neuron_per_phoneme = get_json("data/median_activation_per_neuron_per_phoneme.json")
    numeric_phones = get_json("data/numeric_phones.json")
    phonetic_trait_phonemes = get_json("data/phonetic_traits.json")
    dead_neurons = get_json("data/dead_neurons.json")
    normalization_factors = get_json("data/normalization_factors.json")
    phonemes_per_neuron = get_json("data/phonemes_per_neuron.json")

    def __init__(self, model: Model, data: Data):
        self.activations = model.get_hidden_activation_values(data)
        self.normalized_activations = self._get_normalized_activations()

        self.phonemes_list = list(ANPS.bref_median_activation_per_neuron_per_phoneme["1"].keys())

        self.labels = data.labels_num_32
        self.correct_labels_with_30_phonemes()
        self.labels = self.labels[5:len(self.labels) - 5]

        self.current_median_activations = self.get_current_median_activations()
        self.bref_sondes = {}

    def _get_normalized_activations(self):
        output = {}
        for layer in ["layer2", "layer3"]:
            # Removing dead neurons
            output[layer] = np.delete(self.activations[layer], ANPS.dead_neurons[layer], 1)
            # Dividing by normalization factors (they correspond to the maximum reached per neuron on BREF-Int)
            output[layer] = output[layer] / np.array(ANPS.normalization_factors[layer])

            print(output[layer].shape)

        return output

    def correct_labels_with_30_phonemes(self):
        """
        Transforming the numeric phonemes to string phonemes (with 30 phonemes only)
        :return:
        """
        conversion_dict = {v: k for k, v in ANPS.numeric_phones.items() if k in self.phonemes_list}
        conversion_dict[9] = "ai"  # TODO Find a better way of doing this!

        for old_value, new_value in conversion_dict.items():
            self.labels[self.labels == old_value] = new_value

        self.labels = np.array(self.labels.Phone.tolist())

    def get_current_median_activations(self):
        activations = {}

        for neuron in ANPS.bref_median_activation_per_neuron_per_phoneme.keys():
            neuron_str = str(neuron)
            activations[neuron_str] = {}

            for phoneme in ANPS.phonemes_per_neuron[neuron]:
                labels_mask = self.labels == phoneme
                if labels_mask.sum() <= 0:
                    continue

                neuron = int(neuron)
                layer_activations = self.normalized_activations['layer2'] if neuron < 1024 else self.normalized_activations['layer3']
                neuron_idx = neuron if neuron < 1024 else neuron - 1024

                activations[neuron_str][phoneme] = np.median(layer_activations[labels_mask, neuron_idx])

        return activations

    def get_anps_scores(self) -> Dict[str, Dict[str, float]]:
        output = {}
        for vowel_or_consonant in ANPS.phonetic_trait_phonemes.keys():
            output[vowel_or_consonant] = {}
            for trait in ANPS.phonetic_trait_phonemes[vowel_or_consonant].keys():
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
        phonemes = set(ANPS.phonetic_trait_phonemes[vowel_or_consonant][phonetic_trait])
        # Adding neuro-detectors from layer 2 and 3
        neuro_detectors = ANPS.interpretable_neurons[vowel_or_consonant]["layer2"][phonetic_trait] + \
                          [n + 1024 for n in ANPS.interpretable_neurons[vowel_or_consonant]["layer3"][phonetic_trait]]

        print(vowel_or_consonant, phonetic_trait, phonemes, neuro_detectors)

        dividend, divisor = 0, 0

        for n in neuro_detectors:
            # Computing for phonemes that appeared in the dataset!
            # print(phonemes.intersection(set(self.current_median_activations[str(n)].keys())))
            for k in phonemes.intersection(ANPS.phonemes_per_neuron[str(n)]):  # set(self.current_median_activations[str(n)].keys())):
                dividend += self.current_median_activations[str(n)][k]
                divisor += ANPS.bref_median_activation_per_neuron_per_phoneme[str(n)][k]

                print(n, k, ANPS.bref_median_activation_per_neuron_per_phoneme[str(n)][k])

        return min(1.0, round(dividend / divisor, 2))

    def get_trait_score_sondes(self, trait, trait_phonemes, layer_id: int, list_n_trait_detector: List[int]) -> float:
        """
        @param list_ph: List of phonemes
        @param list_n_trait_detector: List of neuro detector for the phonetic trait in question
        @param layer_id: Fully connected layer identifier (2 or 3)
        @return:
        """
        # Mapping of neurons
        # list_n_trait_detector = [n for n in list_n_trait_detector if n in self.neuron_mapping[f"Layer{layer_id}"].keys()]

        normalized_activations_BREF = np.load(f"data/bref_hidden_activations/Layer{layer_id}.npy")
        lab_list_bref = np.load(f"data/bref_hidden_activations/lab_list.npy")

        # Loading normalized activations
        normalized_activations_p = self.normalized_activations[f"layer{layer_id}"]
        # normalized_activations_p = np.load(f"data/norm_act_cereb/CCM-002595-01_L01_Layer{layer_id}.npy")

        # Loading true corresponding labels
        # true_labels_p = np.load(f"data/norm_act_cereb/CCM-002595-01_L01_lab_list.npy")
        true_labels_p = self.labels
        # true_labels_p = ANPS._modify_true_labels_archiphoneme(true_labels_p)

        # eliminate nan due to lack of phones for some patients
        list_ph = list(set(true_labels_p) & set(trait_phonemes))
        median = dict.fromkeys(list_ph, 0)
        median_bref = dict.fromkeys(list_ph, 0)

        # print(f"Layer{layer_id}", list_n_trait_detector)
        for nid in list_n_trait_detector:
            normalized_activations_subset = normalized_activations_p[:, nid]
            normalized_activations_subset_bref = normalized_activations_BREF[:, nid]

            for phoneme in list_ph:
                median[phoneme] += np.median(normalized_activations_subset[true_labels_p == phoneme])
                # TODO: Save the np.median(...) inside of an array to reuse it this way!
                # TODO (and check it does not vary from patient to patient)
                self.bref_sondes[trait][phoneme] = float(np.median(normalized_activations_subset_bref[lab_list_bref == phoneme]))
                median_bref[phoneme] += np.median(normalized_activations_subset_bref[lab_list_bref == phoneme])

                print(nid, phoneme, median_bref[phoneme])

        print("END", sum(median.values()), sum(median_bref.values()))
        return min(1, round(sum(median.values()) / sum(median_bref.values()), 2))

    def get_anps_scores_sondes(self) -> Dict[str, Dict[str, float]]:
        output = {}
        for vowel_or_consonant, traits in ANPS.phonetic_trait_phonemes.items():
            output[vowel_or_consonant] = {}
            for trait, trait_phonemes in traits.items():
                print("> trait name:", trait)

                self.bref_sondes[trait] = {}

                layer2 = self.get_trait_score_sondes(trait, trait_phonemes, 2, ANPS.interpretable_neurons[vowel_or_consonant]["layer2"][trait])
                layer3 = self.get_trait_score_sondes(trait, trait_phonemes, 3, ANPS.interpretable_neurons[vowel_or_consonant]["layer3"][trait])
                output[vowel_or_consonant][trait] = (layer2 + layer3) / 2

        with open("logs/CCM-002595.json", "w+") as f:
            json.dump(self.bref_sondes, f)

        return output


if __name__ == '__main__':
    from dsit.model import CNN

    # Clearly calculations are off!

    data = Data("CCM-002595-01_L01")
    data.preprocess()
    anps = ANPS(CNN("models/cnn"), data)

    # print(anps.get_anps_scores_sondes())
    print(anps.get_anps_scores())
