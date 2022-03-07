import numpy as np
import glob
import librosa
import librosa.display
import os
import sys

SAMPLING_FREQUENCY = 44100
FREQ_NUM = 128

data_type = sys.argv[1]
data_name = sys.argv[2]
input_dir = "./datasets/{}/wave/{}".format(data_type, data_name)
output_dir = "./datasets/{}/spectrogram/{}".format(data_type, data_name)


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def savepath_create(dir, data):
    new_dir = os.path.join(dir, str(data))
    mkdir(new_dir)
    return new_dir


def CreateSpectrogram(wavs, output_path):
    for wav in wavs:
        sound, fs = librosa.load(wav, sr=SAMPLING_FREQUENCY, mono=False)
        sound_len = len(sound)
        if data_type == "experiment":
            spec = librosa.power_to_db(librosa.feature.melspectrogram(y=sound[11025:33075],
                                                                      sr=SAMPLING_FREQUENCY,
                                                                      n_mels=FREQ_NUM,
                                                                      hop_length=SAMPLING_FREQUENCY // (
                                                                                  2 * FREQ_NUM) + 1))
            np.save(os.path.join(output_path, os.path.splitext(os.path.basename(wav))[0]), spec)
        else:
            for i, n in enumerate(range(0, sound_len, 22050)):
                spec = librosa.power_to_db(librosa.feature.melspectrogram(y=sound[n:n+22050],
                                                                          sr=SAMPLING_FREQUENCY,
                                                                          n_mels=FREQ_NUM,
                                                                          hop_length=SAMPLING_FREQUENCY//(2*FREQ_NUM)+1))
                np.save(os.path.join(output_path, "{}_{}".format(os.path.splitext(os.path.basename(wav))[0], str(i))), spec)


mkdir(output_dir)
labels1 = os.listdir(input_dir)
for label1 in labels1:
    input_l1_dir = os.path.join(input_dir, label1)
    output_l1_dir = savepath_create(output_dir, label1)
    labels2 = os.listdir(input_l1_dir)
    for label2 in labels2:
        input_l2_dir = os.path.join(input_l1_dir, label2)
        output_l2_dir = savepath_create(output_l1_dir, label2)
        wav_paths = glob.glob(os.path.join(input_l2_dir, "*"))
        CreateSpectrogram(wav_paths, output_l2_dir)
