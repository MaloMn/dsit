import json
import argparse
from pathlib import Path

from dsit.anps import ANPS
from dsit import OUTPUT_DIR
from dsit.model import CNN, Model
from dsit.preprocessing import Data
from dsit.utils import create_folder_if_not_exists
from dsit.intelligibility import Intelligibility


def analyse_audio(audio: str, transcription: str, model_path="dsit/models/cnn"):
    create_folder_if_not_exists(OUTPUT_DIR)

    # Load model
    model: Model = CNN(model_path, debug=False)
    data = Data(audio, transcription)
    data.preprocess()

    # Apply model
    model.predict(data)
    anps = ANPS(model, data)

    # Get intelligibility
    intelligibility = Intelligibility(model, data)

    output = {
        "intelligibility": intelligibility.get_intelligibility_score(),
        "severity": intelligibility.get_severity_score(),
        "confusion_matrix": model.get_confusion_matrix(),
        "anps": anps.get_anps_scores()
    }

    with open(f"{OUTPUT_DIR}{Path(audio).stem}.json", "w+") as f:
        json.dump(output, f)

    return output


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # TODO Add documentation for each parameter
    parser.add_argument("audio")
    parser.add_argument("transcription")
    parser.add_argument("-m", "--model", default="dsit/models/cnn")
    parser.add_argument("-a", "--action", default="all")
    # TODO Add a method to generate confusion matrix
    # TODO Add a argument to switch between available functions
    # TODO Also make it possible to load audio and phonemes from anywhere!

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    if args.action == "all":
        result = analyse_audio(args.audio, args.transcription, model_path=args.model)
