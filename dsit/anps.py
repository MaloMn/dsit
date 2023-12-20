from typing import Dict
import numpy as np

from dsit import INTERPRETABLE_NEURONS, BREF_MEDIAN_ACTIVATIONS, NUMERIC_PHONES, PHONETIC_TRAIT_PHONES, PHONES_PER_NEURON
from dsit.model import Model
from dsit.preprocessing import Data
from dsit.utils import get_json


class ANPS:

    interpretable_neurons = get_json(INTERPRETABLE_NEURONS)
    bref_median_activation_per_neuron_per_phoneme = get_json(BREF_MEDIAN_ACTIVATIONS)
    numeric_phones = get_json(NUMERIC_PHONES)
    phonetic_trait_phonemes = get_json(PHONETIC_TRAIT_PHONES)
    phonemes_per_neuron = get_json(PHONES_PER_NEURON)

    def __init__(self, model: Model, data: Data):
        # Normalization happens in the CNN object
        self.normalized_activations = model.normalize_activation_values(model.get_hidden_activation_values(data))

        self.phonemes_list = list(ANPS.bref_median_activation_per_neuron_per_phoneme["1"].keys())

        self.labels = data.labels_num_32
        self.correct_labels_with_30_phonemes()
        self.labels = self.labels[5:len(self.labels) - 5]

        self.phonemes_in_data = np.unique(self.labels)

        self.current_median_activations = self._get_current_median_activations()

    def correct_labels_with_30_phonemes(self) -> None:
        """
        Transforming the numeric phonemes to string phonemes (with 30 phonemes only)
        """
        conversion_dict = {v: k for k, v in ANPS.numeric_phones.items() if k in self.phonemes_list}
        # TODO Find a cleaner way of doing this!
        conversion_dict[9] = "ai"

        for old_value, new_value in conversion_dict.items():
            self.labels[self.labels == old_value] = new_value

        self.labels = np.array(self.labels.Phone.tolist())

    def _get_current_median_activations(self) -> Dict:
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
        :param vowel_or_consonant: "vowels" or "consonants"
        :param phonetic_trait: "+voiced" for instance
        :return: ANPS score (positive float)
        """
        phonemes = set(ANPS.phonetic_trait_phonemes[vowel_or_consonant][phonetic_trait])
        # Adding neuro-detectors from layer 2 and 3
        neuro_detectors = ANPS.interpretable_neurons[vowel_or_consonant]["layer2"][phonetic_trait] + \
                          [n + 1024 for n in ANPS.interpretable_neurons[vowel_or_consonant]["layer3"][phonetic_trait]]

        dividend, divisor = 0, 0

        for n in neuro_detectors:
            # Computing for phonemes that appeared in the dataset!
            for k in phonemes.intersection(self.phonemes_in_data):
                dividend += self.current_median_activations[str(n)][k]
                divisor += ANPS.bref_median_activation_per_neuron_per_phoneme[str(n)][k]

        return min(1.0, round(dividend / divisor, 2))


if __name__ == '__main__':
    from dsit.model import CNN

    data = Data("CCM-002595-01_L01")
    data.preprocess()
    anps = ANPS(CNN("models/cnn"), data)

    # TODO Fix bug encountered in confusion matrix (rows misordered)
    # TODO Fix pandas warning
    print(anps.get_anps_scores())
