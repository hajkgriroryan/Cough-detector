import os
from tqdm import *
from glob import glob

import librosa
import soundfile as sf

import numpy as np
import random
import pickle
from random import shuffle

# Default data augmentation
def padding(pad):
    def f(sound):
        return np.pad(sound, pad, 'constant')

    return f


# For strong data augmentation
def random_scale(max_scale, interpolate='Linear'):
    def f(sound):
        scale = np.power(max_scale, random.uniform(-1, 1))
        output_size = int(len(sound) * scale)
        ref = np.arange(output_size) / scale
        if interpolate == 'Linear':
            ref1 = ref.astype(np.int32)
            ref2 = np.minimum(ref1 + 1, len(sound) - 1)
            r = ref - ref1
            scaled_sound = sound[ref1] * (1 - r) + sound[ref2] * r
        elif interpolate == 'Nearest':
            scaled_sound = sound[ref.astype(np.int32)]
        else:
            raise Exception('Invalid interpolation mode {}'.format(interpolate))

        return scaled_sound

    return f


def read_audio(audio_path, target_fs=None, dtype='float64'):
    audio, fs = sf.read(audio_path, dtype=dtype)

    # if this is not a mono sounds file
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs

    return audio, fs


def save_object_as_pkl(full_path_name, obj):
    f = open(full_path_name, 'wb')
    pickle.dump(obj, f)
    f.close()


def restore_pkl_object(full_path_name):
    f = open(full_path_name, 'r')
    obj = pickle.load(f)
    f.close()

    return obj


def prepare_data_lists(data_roots, data_ground_truth):
    data_lists = []
    for data_root in data_roots:
        dict = {}
        dict['train'] = glob(os.path.join(data_root, 'train') + '/*')
        dict['val'] = glob(os.path.join(data_root, 'val') + '/*')
        dict['train'].sort(key=get_img_name)
        dict['val'].sort(key=get_img_name)
        data_lists.append(dict)

    dict = {}
    dict['train'] = glob(os.path.join(data_ground_truth, 'train') + '/*')
    dict['val'] = glob(os.path.join(data_ground_truth, 'val') + '/*')
    dict['train'].sort(key=get_img_name)
    dict['val'].sort(key=get_img_name)

    return data_lists, dict


def convert_list_to_str(arr):
    str1 = " ".join(str(x) for x in arr)
    return str1


def split_dataset_to_train_val(dataset, train_dataset_percent):

    shuffle(dataset)
    border = int(len(dataset)*train_dataset_percent)
    train_data = dataset[:border]
    val_data = dataset[border:]

    # train_data = dataset[:256]
    # val_data = dataset[256:288]

    return train_data, val_data
