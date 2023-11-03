"""
Transforms the input audio into frames linked with their associated phonemes.
Also packs the output in a tensorflow file.
"""
from pathlib import Path
import scipy.io.wavfile as wav
from python_speech_features import logfbank
import pandas as pd
import numpy as np
import hashlib

from . import H5_DIR


class PreProcessData:

    def __init__(self, audio_path: str, phonemes_path: str):
        self.audio_path = Path(audio_path)
        if not (self.audio_path.exists() and self.audio_path.suffix.lower() == ".wav"):
            raise Exception("The audio does not exist, or it isn't a WAV file. Please check this argument.")

        # TODO Resample the audio if it isn't 16kHz
        self.framerate, self.audio_signal = wav.read(self.audio_path)

    def extraction_fbank(self):
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
        frames['Segment_name'] = self.audio_path.stem.split('_')[0]
        df = pd.concat([df, frames])

        # TODO To avoid collapse, remove [:6]!
        audio_hash = hashlib.sha256(self.audio_signal).hexdigest()[:6]
        hdf5_path = f"{H5_DIR}parameters_{audio_hash}.h5"
        store_data = pd.HDFStore(hdf5_path, 'a')

        store_data.append("custom", df)
        store_data.close()

    def prepare_transcription(self, lbl_dir):
        '''Preprocessing phoneme alignment files'''
        hdf5_path = h5 + 'Transcription_' + corpus + '.h5'
        DATA = pd.DataFrame()
        j = 0
        store_data = pd.HDFStore(hdf5_path, 'a')
        for filename in os.listdir(lbl_dir):
            j += 1
            print(j, filename)
            file = open(lbl_dir + filename, "r")
            file_name = os.path.basename(file.name).split('.')[0]
            list_phone = file.read().splitlines()
            del (list_phone[-1])  # ajouter à coz du bug dans les fichiers align AHN/ALP/LEC
            lol = [l.split(" ") for l in list_phone]
            data = np.asarray(lol)
            # *** si jamais on veut un phonème gn au lieu de nn+yy
            prev = data[0][2]
            for i in range(1, len(data) - 1):
                transcrip = data[i][2]
                if prev == 'nn' and transcrip == 'yy':
                    data[i - 1][2] = 'gn'
                    data[i][2] = 'gn'
                prev = data[i][2]
            # ***
            df = pd.DataFrame(data, columns=['Start', 'End', 'Phone'])
            df['Start'] = df['Start'].astype(float)
            df['End'] = df['End'].astype(float)
            df['Phone'] = df['Phone'].astype(str)
            df['Segment_name'] = file_name.split('_')[0]
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
                if (row['Nb_frames'] != 0):
                    df = df.append([row] * row['Nb_frames'])
            df.reset_index(level=0, inplace=True)
            df.rename(columns={'index': 'Seg'}, inplace=True)
            df1_gb = df.groupby(['Seg'], as_index=False)
            df1_ = df1_gb.apply(lambda x: x.reset_index(drop=True))  # to save each frame unique index
            df1_ = df1_.reset_index(level=1, drop=False)
            df1_.rename(columns={'level_1': 'Frame_rank'}, inplace=True)
            df1_ = df1_.reset_index(level=0, drop=True)
            df1_ = df1_.drop(['Nb_frames'], axis=1)
            df1_ = df1_.rename_axis(index=['Frame'])
            DATA = pd.concat([DATA, df1_])
        if j > 0:
            store_data.append(subcorpus, DATA)
            store_data.close()
            print("Preprocessed phoneme alignment saved")