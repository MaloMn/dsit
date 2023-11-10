from dsit.model import CNN, Model
from dsit.preprocessing import Data


def process_audio(file_stem: str, model_path="models/cnn"):
    # Load model
    model: Model = CNN(model_path, debug=False)
    data = Data(file_stem)
    data.preprocess()

    # Apply model
    model.predict(data)

    return {
        "score": 10,
        "confusion_matrix": model.get_confusion_matrix(),
        "anps": {}
    }


if __name__ == '__main__':
    # TODO Add arhgument parser
    process_audio("asdasd")
