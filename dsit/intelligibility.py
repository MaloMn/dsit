from utils import load_audio, load_phonemes, load_model
from model import CNN
from typing import Dict, List, Union

from dsit.model import Model
from dsit.preprocessing import Data


class Intelligibility:

    def __init__(self, model: Model, data: Data):
        self.model = model
        self.data = data

    def get_score(self) -> float:
        return 0.8

    def get_anps_scores(self) -> Dict[str, float]:
        return {"aa": 0.4, "bb": 0.5, "zz": 0.9}

    def get_results(self) -> Dict:
        return {
            "score": self.get_score(),
            "confusion_matrix": self.get_confusion_matrix(),
            "anps": self.get_anps_scores()
        }


if __name__ == '__main__':
    pass
