"""
Transforms the input audio into frames linked with their associated phonemes.
Also packs the output in a tensorflow file.
"""
from python_speech_features import logfbank
import scipy.io.wavfile as wav
from pathlib import Path
import tensorflow as tf
from tqdm import tqdm
import pandas as pd
import numpy as np
import json

from dsit.utils import check_file_exists, create_folder_if_not_exists, parser_test, serialize_example, get_batch_size, \
    generate_lbl_from_seg
from dsit import H5_DIR, AUDIO_DIR, PHONES_CSV, TFRECORD_DIR


# TODO Only execute this if audio has changed, and has not been seen before, find a caching technique for this.
# TODO Add better messages while data is being processed.


class Data:
    window = 0.020  # ms
    step = 0.010  # ms

    def __init__(self, file_stem: str, debug=False):
        """
        :param file_stem: Name of the file before the extension (abc.wav => stem = "abc")
        """
        self.debug = debug
        self.stem = file_stem

        # Checking audio file exists
        self.audio_path = Path(f"{AUDIO_DIR}{file_stem}.wav")
        check_file_exists(self.audio_path, "wav")

        # Checking transcription file exists (lbl is created if it doesn't exist)
        try:
            self.phonemes_path = Path(f"{AUDIO_DIR}{file_stem}.lbl")
            check_file_exists(self.phonemes_path, "lbl")
        except FileNotFoundError:
            self.phonemes_path = Path(f"{AUDIO_DIR}{file_stem}.seg")
            check_file_exists(self.phonemes_path, "seg")

            # Generating a lbl file
            generate_lbl_from_seg(self.phonemes_path)

            # Load lbl file
            self.phonemes_path = Path(f"{AUDIO_DIR}{file_stem}.lbl")
            check_file_exists(self.phonemes_path, "lbl")

        # TODO Resample the audio if it isn't 16kHz
        self.framerate, self.audio_signal = wav.read(self.audio_path)

        # TODO Use names that reflect the data inside the hdf5 stores
        self.data_path = f"{H5_DIR}{self.stem}_data.h5"
        self.labels_path = f"{H5_DIR}{self.stem}_labels.h5"
        self.final_path = f"{H5_DIR}{self.stem}_final.h5"
        self.parameters_path = f"{H5_DIR}{self.stem}_parameters.h5"
        self.transcriptions_path = f"{H5_DIR}{self.stem}_transcriptions.h5"
        self.labels_num_path = f"{H5_DIR}{self.stem}_labels_numeric.h5"
        self.data_norm_path = f"{H5_DIR}{self.stem}_data_normalized.h5"
        self.labels_num_32_path = f"{H5_DIR}{self.stem}_labels_numeric_32.h5"

        self.tfrecord_path = f"{TFRECORD_DIR}{self.stem}.tfrecord"
        self.tfrecord_shape = f"{TFRECORD_DIR}{self.stem}_shapes.json"

        self.shape = None

        create_folder_if_not_exists(H5_DIR)
        create_folder_if_not_exists(TFRECORD_DIR)

    def preprocess(self):
        self.extraction_fbank()
        self.prepare_transcription()
        self.merge_fbank_alignment()
        self.split_data_labels()
        self.converting_labels_to_numeric()
        self.converting_labels_to_32()
        self.normalization_fbank_per_segment()
        self.tfrecord_generation()

    def extraction_fbank(self) -> None:
        """
        Fbank feature extraction from the speech signal
        :return: None
        """
        df = pd.DataFrame(np.empty((0, 120)))

        # Compute Mel-Fbanks features (including first and second derivatives)
        fbank_feat = logfbank(self.audio_signal, samplerate=16_000, nfilt=40, winlen=Data.window, winstep=Data.step)
        first_derivative = np.gradient(fbank_feat)
        second_derivative = np.gradient(first_derivative[1])

        frames = pd.DataFrame(np.concatenate((fbank_feat, first_derivative[1], second_derivative[1]), axis=1))
        frames['Segment_name'] = self.stem
        df = pd.concat([df, frames])

        store_data = pd.HDFStore(self.parameters_path, 'w')
        store_data.append("custom", df)
        store_data.close()

    def prepare_transcription(self):
        """Preprocessing phoneme alignment files"""
        dataframe = pd.DataFrame()
        store_data = pd.HDFStore(self.transcriptions_path, 'w')

        with open(self.phonemes_path, "r") as f:
            list_phone = f.readlines()
            list_phone = [line.replace("\n", "") for line in list_phone]

        # Clean raw data
        if list_phone[-1].split(" ")[0] == list_phone[-2].split(" ")[0]:
            list_phone.pop(-1)  # Because of bug in AHN/ALP/LEC file alignments

        data = np.asarray([l.split(" ") for l in list_phone])

        # Replacing phonemes nn+yy by gn
        for prev, curr in zip(data, data[1:]):
            if prev[2] == 'nn' and curr[2] == 'yy':
                prev[2] = curr[2] = 'gn'

        df = pd.DataFrame(data, columns=['Start', 'End', 'Phone'])
        df['Start'] = df['Start'].astype(float)
        df['End'] = df['End'].astype(float)
        df['Phone'] = df['Phone'].astype(str)
        df['Segment_name'] = self.stem

        # *** Rectify th gn duration
        gn_index = list(df[df.Phone == 'gn'].index)
        for i in range(0, len(gn_index), 2):
            df['End'][gn_index[i]] = df['End'][gn_index[i + 1]]

        for i in range(1, len(gn_index), 2):
            assert gn_index[i - 1] + 1 == gn_index[i]
            df = df.drop([gn_index[i]])

        df.reset_index(drop=True, inplace=True)
        ##***

        df = df.drop(df[df.Phone == "[new_sentence]"].index.values)

        df['Nb_frames'] = round((df.End - df.Start) * 100)
        df['Nb_frames'] = df['Nb_frames'].astype(int)
        df = df.drop(['Start', 'End'], axis=1)

        for index, row in df.iterrows():
            if row['Nb_frames'] != 0:
                df = df._append([row] * row['Nb_frames'])  # TODO Maybe use a different array to store this!

        df.reset_index(level=0, inplace=True)
        df.rename(columns={'index': 'Seg'}, inplace=True)
        df1_gb = df.groupby(['Seg'], as_index=False)
        df1_ = df1_gb.apply(lambda x: x.reset_index(drop=True))  # to save each frame unique index
        df1_ = df1_.reset_index(level=1, drop=False)
        df1_.rename(columns={'level_1': 'Frame_rank'}, inplace=True)
        df1_ = df1_.reset_index(level=0, drop=True)
        df1_ = df1_.drop(['Nb_frames'], axis=1)
        df1_ = df1_.rename_axis(index=['Frame'])

        dataframe = pd.concat([dataframe, df1_])

        store_data.append("custom", dataframe)
        store_data.close()

    def merge_fbank_alignment(self):
        """
        Merge the phoneme alignment and the speech signal fbank parameters
        """
        parameters = (pd.read_hdf(self.parameters_path, key="custom", mode='r')
                      .groupby(['Segment_name'], as_index=False)
                      .apply(lambda x: x.reset_index(drop=True))
                      .rename_axis(index=['File', 'Frame']))  # to save each frame unique index

        # TODO Investigate missing part of audio (last second is not present in the fbanks or transcription??)
        transcript = pd.read_hdf(self.transcriptions_path, key="custom", mode='r')

        final_data = pd.merge(parameters, transcript, how='inner', on=['Segment_name', 'Frame'])
        final_data_gb = final_data.groupby(['Segment_name'], as_index=False)
        final_data = final_data_gb.apply(lambda x: x.reset_index(drop=True))

        store_data = pd.HDFStore(self.final_path, 'w')
        store_data.append("custom", final_data)
        store_data.close()

    def split_data_labels(self):
        """Split LABELS  DATA H5DATA"""
        data = (pd.read_hdf(self.final_path, key="custom", mode='r')
                .drop(['Frame_rank', 'Segment_name', 'Seg'], axis=1))

        store1 = pd.HDFStore(self.data_path, 'w')
        store1.append("custom", data.iloc[:, :120])
        store1.close()

        store2 = pd.HDFStore(self.labels_path, 'w')
        store2.append("custom", data.iloc[:, 120])
        store2.close()

    def converting_labels_to_numeric(self):
        """
        Convert labels to numeric
        """
        labels = pd.read_hdf(self.labels_path, key="custom", mode='r')
        lab = pd.DataFrame(labels)

        with open("data/numeric_phones.json", "r") as f:
            dic = json.load(f)

        lab = lab.replace({"Phone": dic})
        lab['Phone'] = lab['Phone'].astype(str)  # added to handle a pandas issue encountered with hdf5

        store_num = pd.HDFStore(self.labels_num_path, 'w')
        store_num.append("custom", lab)
        store_num.close()

    # TODO Remove this, it is now useless
    def converting_labels_to_32(self):
        """Merge [oo&au / ee(oe)&eu] --> 31 classes + silence"""
        labels = pd.read_hdf(self.labels_num_path, key="custom", mode='r')
        labels['Phone'] = labels['Phone'].astype(int)

        store = pd.HDFStore(self.labels_num_32_path, 'w')
        store.append("custom", labels)
        store.close()

    def normalization_fbank_per_segment(self):
        """Mean Variance Normalization of the filterbank speech features per segment"""
        data = pd.DataFrame()
        dir_ = pd.read_hdf(self.data_path, key="custom", mode='r')

        for i in tqdm(range(dir_.index.values[0][0], dir_.index.values[-1][0] + 1)):
            mean = np.mean(dir_.loc[i], axis=0)
            std = np.std(dir_.loc[i], axis=0)  # np.sqrt(np.mean((dir_.loc[i] - mean) ** 2))
            aux = (dir_.loc[i] - mean) / std
            aux['File'] = i
            data = pd.concat([data, aux])

        # parameters = (data.groupby(['File'], as_index=True)
        #               .apply(lambda x: x.reset_index(drop=True))  # to save each frame unique index
        #               .rename_axis(index=['File', 'Frame'])
        #               .drop(['File'], axis=1))

        Data_gb = data.groupby(['File'], as_index=True)
        param = Data_gb.apply(lambda x: x.reset_index(drop=True))  # to save each frame unique index
        param = param.rename_axis(index=['File', 'Frame'])
        param = param.drop(columns=['File'])

        # param is the same as in the notebook.

        store_data = pd.HDFStore(self.data_norm_path, 'w')
        store_data.append("custom", param)
        store_data.close()

    def tfrecord_generation(self):
        """
        TFrecord files generation
        """
        normalized_data = pd.read_hdf(self.data_norm_path, mode="r", key="custom")
        numeric_labels = pd.read_hdf(self.labels_num_32_path, mode="r", key="custom")

        with tf.io.TFRecordWriter(self.tfrecord_path) as writer:
            for j in range(5, normalized_data.loc[0].shape[0] - 5):
                X = (np.array(normalized_data.loc[0].iloc[j - 5:j + 6, :])).reshape(1, 11 * 120)
                X = pd.DataFrame(X)  # X is of size 120 * 11 !
                Y = np.array(numeric_labels.loc[(0, j)])  # [:, None]
                Y = pd.DataFrame(Y)

                # create an item in the dataset converted to the correct formats (float, int, byte)
                example = serialize_example(
                    {
                        "label": {
                            "data": Y.loc[0],
                            "_type": _int64_feature,
                        },
                        "fbank": {
                            "data": X.loc[0],
                            "_type": _float_feature,
                        },
                    }
                )

                # write the defined example into the dataset
                writer.write(example)

            self.shape = normalized_data.loc[0].shape[0] - 10  # -10 because of the padding effect

        # TODO Adapt shape handling when using different sub-corpus in one h5 file!

    # TODO Fix batch size handling when building tfrecords too
    def get_fbank_labels(self) -> tf.data.Iterator:
        dataset = (
            tf.data.Dataset.list_files(self.tfrecord_path, shuffle=False)
            .interleave(tf.data.TFRecordDataset, block_length=1, num_parallel_calls=1, cycle_length=1)
            .map(parser_test, num_parallel_calls=1)
            .batch(self.compute_batch_size())
        )

        return tf.compat.v1.data.make_one_shot_iterator(dataset)

    def get_frames_count(self) -> int:
        # return (len(self.audio_signal) // 16 - 5) // 10 - 10 - 2
        return self.shape

    def compute_batch_size(self) -> int:
        return get_batch_size(self.get_frames_count())


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


if __name__ == '__main__':
    print(Data("I0MB0841").preprocess())
