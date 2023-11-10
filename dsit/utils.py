from pathlib import Path
import tensorflow as tf
import pandas as pd
import math
import os


def load_model(path: str):
    pass


def load_audio(path: str):
    pass


def load_phonemes(path: str):
    pass


def parser_test(record):
    """
    This is a parser function. It defines the template for
    interpreting the examples you're feeding in. Basically,
    this function defines what the labels and data look like
    for your labeled data.
    """
    # the 'features' here include your normal data feats along
    # with the label for that data
    features = {
        'fbank': tf.io.FixedLenFeature([11, 120, 1], tf.float32),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }

    parsed = tf.io.parse_single_example(record, features)

    # some conversion and casting to get from bytes to floats and ints
    fbanks = tf.convert_to_tensor(parsed['fbank'], tf.float32)
    label = tf.cast(parsed['label'], tf.int64)

    return {'Conv1_input': fbanks}, label


def check_file_exists(path: Path, extension: str):
    if not (path.exists() and path.suffix.lower() == "." + extension.lower()):
        raise FileNotFoundError(f"`{path}` does not exist, or it isn't a .{extension} file. "
                                "Please check this argument.")


def create_folder_if_not_exists(folder: str):
    if not os.path.exists(folder):
        os.makedirs(folder)


def get_divisors(n):
    divisors = []
    for x in range(1, int(math.sqrt(n)) + 1):
        if n % x == 0:
            divisors.append(x)
            if x != n // x:
                divisors.append(n // x)
    return sorted(divisors)


def get_batch_size(n: int, threshold=30_000):
    divisors = get_divisors(n)

    # If number is prime, then we return n.
    if len(divisors) == 2:
        return n

    # Else, we choose a batch size close to 5,000 but smaller than 30,000
    divisors = divisors[1:]
    filtered = [b for b in divisors if threshold > b > 100]
    filtered.sort(key=lambda z: abs(5000 - z))

    if len(filtered) > 0:
        return filtered[0]

    return divisors[0]


def serialize_example(example):
    """Serialize an item in a dataset
    Arguments:
      example {[list]} -- list of dictionaries with fields "name" , "_type", and "data"

    Returns:
      [type] -- [description]
    """
    dset_item = {}
    for feature in example.keys():
        dset_item[feature] = example[feature]["_type"](example[feature]["data"])
        example_proto = tf.train.Example(features=tf.train.Features(feature=dset_item))
    return example_proto.SerializeToString()


def generate_lbl_from_seg(file_path: Path):
    df = pd.read_csv(file_path, sep=" ", header=None, names=["Patient", "Index", "Start", "Duration", "Phoneme"])
    # Adding end column
    df["End"] = round(df["Start"] + df["Duration"], 2) - 0.01  # lbl files end and next start always have a 0.01 gap.
    # Replacing "pause" by "[pause]"
    df = df.replace({"Phoneme": {"pause": "[pause]"}})
    # Saving lbl file
    df.to_csv(file_path.with_suffix(".lbl"), index=False, sep=" ", header=False, columns=["Start", "End", "Phoneme"])


if __name__ == '__main__':
    generate_lbl_from_seg(Path("audio/I0MB0844.seg"))
