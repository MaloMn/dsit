from typing import Dict, Tuple
import argparse

from dsit.anps import ANPS
from dsit.model import CNN, Model
from dsit.preprocessing import Data


def analyse_audio(file_stem: str, model_path="dsit/models/cnn") -> Dict:
    # Load model
    model: Model = CNN(model_path, debug=False)
    data = Data(file_stem)
    data.preprocess()

    # Apply model
    model.predict(data)
    anps = ANPS(model, data)

    return {
        "score": 10,
        "confusion_matrix": model.get_confusion_matrix(),
        "anps": anps.get_anps_scores()
    }


def parse_arguments() -> Tuple[str, str]:
    parser = argparse.ArgumentParser()
    # TODO Add documentation for each parameter
    parser.add_argument("name")
    parser.add_argument("-m", "--model", default="dsit/models/cnn")
    args = parser.parse_args()

    return args.name, args.model


if __name__ == '__main__':
    file_stem_name, phoneme_model_path = parse_arguments()
    print(analyse_audio(file_stem_name, model_path=phoneme_model_path))
