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

from dsit.utils import check_file_exists, create_folder_if_not_exists, parser_test, serialize_example
from dsit import H5_DIR, AUDIO_DIR, PHONES_CSV, TFRECORD_DIR


# TODO Only execute this if audio has changed, and has not been seen before, find a caching technique for this.
# TODO Add better messages while data is being processed.


class Data:

    def __init__(self, file_stem: str):
        """
        :param file_stem: Name of the file before the extension (abc.wav => stem = "abc")
        """
        self.stem = file_stem
        self.audio_path = Path(f"{AUDIO_DIR}{file_stem}.wav")
        self.phonemes_path = Path(f"{AUDIO_DIR}{file_stem}.phon.seg")

        check_file_exists(self.audio_path, "wav")
        check_file_exists(self.phonemes_path, "seg")

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

        :return:
        """
        df = pd.DataFrame(np.empty((0, 120)))

        # Compute Mel-Fbanks features (including first and second derivatives)
        fbank_feat = logfbank(self.audio_signal, samplerate=16000, nfilt=40, winlen=0.020, winstep=0.010)
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
        list_phone.pop(-1)  # Because of bug in AHN/ALP/LEC file alignements
        data = np.asarray([l.split(" ") for l in list_phone])

        # Replacing phonemes nn+yy by gn
        prev = data[0][2]
        for i in range(1, len(data) - 1):
            transcrip = data[i][2]
            if prev == 'nn' and transcrip == 'yy':
                data[i - 1][2] = 'gn'
                data[i][2] = 'gn'
            prev = data[i][2]

        # TODO End is confusing: it should be duration
        df = pd.DataFrame(data[:, 2:], columns=['Start', 'End', 'Phone'])
        df['Start'] = df['Start'].astype(float)
        df['End'] = df['End'].astype(float)
        df['End'] = df['Start'] + df['End']
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
                df = df._append([row] * row['Nb_frames'])

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
        phones = pd.read_csv(PHONES_CSV, sep=' ')
        phones.columns = ['phone', 'class']
        phones = phones.set_index(phones['class'])
        phones = phones.drop('class', axis=1)
        phones = phones.drop([1], axis=0)
        phones = phones.reset_index(drop=True)
        phones.loc[0] = 'pause'

        labels = pd.read_hdf(self.labels_path, key="custom", mode='r')
        lab = pd.DataFrame(labels)
        dic = phones['phone'].to_dict()
        dic = {y: x for x, y in dic.items()}
        lab = lab.replace({"Phone": dic})
        lab.loc[lab['Phone'] == 'in'] = 27
        lab.loc[lab['Phone'] == 'oe'] = 8
        lab['Phone'] = lab['Phone'].astype(str)  # added to handle a pandas issue encountered with hdf5

        store_num = pd.HDFStore(self.labels_num_path, 'w')
        store_num.append("custom", lab)
        store_num.close()

    def converting_labels_to_32(self):
        """Merge [oo&au / ee(oe)&eu] --> 31 classes + silence"""
        labels = pd.read_hdf(self.labels_num_path, key="custom", mode='r')
        labels['Phone'] = labels['Phone'].astype(int)
        labels.loc[labels['Phone'] == 21] = 4
        labels.loc[labels['Phone'] == 10] = 8
        labels.loc[(10 < labels['Phone']) & (labels['Phone'] < 21)] = labels - 1
        labels.loc[labels['Phone'] > 21] = labels - 2

        store = pd.HDFStore(self.labels_num_32_path, 'w')
        store.append("custom", labels)
        store.close()

    def normalization_fbank_per_segment(self):
        """Mean Variance Normalization of the filterbank speech features per segment"""
        data = pd.DataFrame()
        dir_ = pd.read_hdf(self.data_path, key="custom", mode='r')

        for i in tqdm(range(dir_.index.values[0][0], dir_.index.values[-1][0] + 1)):
            mean = np.mean(dir_.loc[i])
            std = np.sqrt(np.mean((dir_.loc[i] - mean) ** 2))
            aux = (dir_.loc[i] - mean) / std
            aux['File'] = i
            data = pd.concat([data, aux])

        parameters = (data.groupby(['File'], as_index=True)
                      .apply(lambda x: x.reset_index(drop=True))  # to save each frame unique index
                      .rename_axis(index=['File', 'Frame'])
                      .drop(['File'], axis=1))

        store_data = pd.HDFStore(self.data_norm_path, 'w')
        store_data.append("custom", parameters)
        store_data.close()

    def tfrecord_generation(self):
        """TFrecord files generation"""
        mydata = pd.read_hdf(self.data_norm_path, mode="r", key="custom")
        mylab = pd.read_hdf(self.labels_num_32_path, mode="r", key="custom")
        final = pd.read_hdf(self.final_path, mode="r", key="custom")

        path = f"{TFRECORD_DIR}{self.stem}.tfrecord"

        with tf.io.TFRecordWriter(str(path)) as writer:
            for j in range(5, mydata.loc[0].shape[0] - 5):
                X = (np.array(mydata.loc[0].iloc[j - 5:j + 6, :])).reshape(1, 11 * 120)
                X = pd.DataFrame(X)
                Y = np.array(mylab.loc[(0, j)])  # [:, None]
                Y = pd.DataFrame(Y)
                # create an item in the datset converted to the correct formats (float, int, byte)
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

        # The shape of tfrecords files
        dict_shapes = {}
        for l in set(final['Segment_name']):
            id_l = l.split('-')[0]
            dict_shapes[id_l] = [final[final['Segment_name'] == l].shape[0] - 10,
                                 final[final['Segment_name'] == l].shape[0] - 10]

        with open(f"{TFRECORD_DIR}{self.stem}_shapes.json", 'w') as fp:
            json.dump(dict_shapes, fp)

    def get_fbank_labels(self) -> tf.data.Iterator:

        dataset = (
            tf.data.Dataset.list_files(self.data_norm_path, shuffle=False)
            .interleave(tf.data.TFRecordDataset, block_length=1, num_parallel_calls=1, cycle_length=1)
            .map(parser_test, num_parallel_calls=1)
            # .batch(bs)
        )

        return tf.compat.v1.data.make_one_shot_iterator(dataset)

    def get_audio_frames_number(self):
        return (len(self.audio_signal) // 16 - 5) // 10


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


if __name__ == '__main__':
    data = Data("I0MA0007")
    print(data.get_audio_frames_number())

