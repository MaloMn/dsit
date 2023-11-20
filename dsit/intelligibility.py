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
        return 10.0


if __name__ == '__main__':
    pass
