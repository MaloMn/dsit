from utils import load_audio, load_phonemes, load_model
from model import CNN
from typing import Dict, List, Union


class Intelligibility:

    def __init__(self, model_path: str, audio_path: str, phonemes_path: str):
        model = CNN(model_path)
        audio = load_audio(audio_path)
        phonemes = load_phonemes(phonemes_path)

    def get_score(self) -> float:
        return 0.8

    def get_confusion_matrix(self) -> Dict[str, Union[List[str], List[List[float]]]]:
        return {
            "labels": ["aa", "bb", "zz"],
            "matrix": [[0.2, 0.4, 0.4], [0.1, 0.8, 0.1], [0.2, 0.05, 0.75]]
        }

    def get_anps_scores(self) -> Dict[str, float]:
        return {"aa": 0.4, "bb": 0.5, "zz": 0.9}

    def get_results(self) -> Dict:
        return {
            "score": self.get_score(),
            "confusion_matrix": self.get_confusion_matrix(),
            "anps": self.get_anps_scores()
        }
