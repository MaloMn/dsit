import json
import argparse

from dsit.anps import ANPS
from dsit import OUTPUT_DIR
from dsit.cache import Cache
from dsit.model import CNN, Model
from dsit.preprocessing import Data
from dsit.utils import create_folder_if_not_exists


def analyse_audio(file_stem: str, model_path="dsit/models/cnn") -> dict[str, int | dict[str, dict[str, float]] | dict]:
    create_folder_if_not_exists(OUTPUT_DIR)

    # Load model
    model: Model = CNN(model_path, debug=False)
    data = Data(file_stem)
    data.preprocess()

    # Apply model
    model.predict(data)
    anps = ANPS(model, data)

    output = {
            "score": 10,
            "confusion_matrix": model.get_confusion_matrix(),
            "anps": anps.get_anps_scores()
        }

    # TODO implementing caching system here!
    with open(f"{OUTPUT_DIR}{file_stem}.json", "w+") as f:
        json.dump(output, f)

    return output


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # TODO Add documentation for each parameter
    parser.add_argument("name")
    parser.add_argument("-m", "--model", default="dsit/models/cnn")
    parser.add_argument("-a", "--action", default="all")
    parser.add_argument('--no-cache', action='store_true')
    # TODO Add a method to generate confusion matrix, but don't merge it on main
    # TODO Add a argument to switch between available functions

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    # if not args.no_cache and Cache.key_exists(name=args.name, action=args.action, model=args.model):
        # If result has already been computed, then just serve it
        # print(Cache.get_by_key(name=args.name, action=args.action, model=args.model))
    # else:
        # Otherwise, compute it

    if args.action == "all":
        result = analyse_audio(args.name, model_path=args.model)
        # Cache.set_by_key(result, name=args.name, action=args.action, model=args.model)