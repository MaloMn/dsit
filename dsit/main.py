from typing import Dict

from dsit.anps import ANPS
from dsit.model import CNN, Model
from dsit.preprocessing import Data


def process_audio(file_stem: str, model_path="models/cnn") -> Dict:
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
        "anps": anps.get_anps_scores_sondes()
    }


if __name__ == '__main__':
    # TODO Add argument parser
    print(process_audio("I0MB0843"))
