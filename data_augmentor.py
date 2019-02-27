from utils import read_audio, save_object_as_pkl
import glob
import os
import librosa
import numpy as np


# For BC learning
def a_weight(fs, n_fft, min_db=-80.0):
    freq = np.linspace(0, fs // 2, n_fft // 2 + 1)
    freq_sq = np.power(freq, 2)
    freq_sq[0] = 1.0
    weight = 2.0 + 20.0 * (2 * np.log10(12194) + 2 * np.log10(freq_sq)
                           - np.log10(freq_sq + 12194 ** 2)
                           - np.log10(freq_sq + 20.6 ** 2)
                           - 0.5 * np.log10(freq_sq + 107.7 ** 2)
                           - 0.5 * np.log10(freq_sq + 737.9 ** 2))
    weight = np.maximum(weight, min_db)

    return weight


def compute_gain(sound, fs, min_db=-80.0, mode='A_weighting'):
    if fs == 16000:
        n_fft = 2048
    elif fs == 44100:
        n_fft = 4096
    else:
        raise Exception('Invalid fs {}'.format(fs))
    stride = n_fft // 2

    gain = []
    for i in xrange(0, len(sound) - n_fft + 1, stride):
        if mode == 'RMSE':
            g = np.mean(sound[i: i + n_fft] ** 2)
        elif mode == 'A_weighting':
            spec = np.fft.rfft(np.hanning(n_fft + 1)[:-1] * sound[i: i + n_fft])

            power_spec = np.abs(spec) ** 2
            a_weighted_spec = power_spec * np.power(10, a_weight(fs, n_fft) / 10)
            g = np.sum(a_weighted_spec)
        else:
            raise Exception('Invalid mode {}'.format(mode))
        gain.append(g)

    gain = np.array(gain)
    gain = np.maximum(gain, np.power(10, min_db / 10))
    gain_db = 10 * np.log10(gain)

    return gain_db


def mix(sound1, sound2, r, fs):
    gain1 = np.max(compute_gain(sound1, fs))  # Decibel
    gain2 = np.max(compute_gain(sound2, fs))
    t = 1.0 / (1 + np.power(10, (gain1 - gain2) / 20.) * (1 - r) / r)
    sound = ((sound1 * t + sound2 * (1 - t)) / np.sqrt(t ** 2 + (1 - t) ** 2))

    return sound


def zero_pad(audio, start_pos, new_audio_lenght, fs=44100):
    assert len(audio) <= new_audio_lenght*fs
    new_audio = np.zeros(int(new_audio_lenght*fs)+1)
    new_audio[start_pos:start_pos+len(audio)] = audio[:]
    return new_audio


def mix_audios(audios_with_info, mixed_audio_lenght=10, fs=44100):
#    assert np.max(audios_with_info[:]['end_pos']) <= mixed_audio_lenght
    mixed_audio = zero_pad(audios_with_info[0]['audio'], audios_with_info[0]['start_pos'], mixed_audio_lenght)
    for i in range(len(audios_with_info)):
        if i == 0:
            continue
        audio = zero_pad(audios_with_info[i]['audio'], audios_with_info[i]['start_pos'], mixed_audio_lenght)
        mixed_audio = mix(mixed_audio, audio, audios_with_info[i]['r'], fs)
    return mixed_audio


def random_crop(audio, duration=10, fs=44100):
    start_pos = int(np.floor(np.random.rand() * (len(audio) - duration*fs)))
#     print start_pos/fs
    cropped_audio = audio[start_pos:start_pos + duration * fs]
    return cropped_audio, start_pos


def random_segment_in_range(A, B, c):
    start_pos = A + int(np.floor(np.random.rand() * (B-A - c)))
    end_pos = start_pos + c
    return start_pos, end_pos


def mix_audio_combination(sounds_list_1, sounds_list_2, backgrounds_list, r1=0.5, r2=0.5, r3=0.8, duration=10, fs=44100):
    background_index = np.random.choice(len(backgrounds_list))
    background, _ = read_audio(backgrounds_list[background_index], target_fs=fs)
    background, _ = random_crop(background, duration, fs)

    sound1_index = np.random.choice(len(sounds_list_1))
    sound1, _ = read_audio(sounds_list_1[sound1_index], target_fs=fs)
    del sounds_list_1[sound1_index]

    sound1_start, sound1_end = random_segment_in_range(0, len(background), len(sound1))

    sound2_start = None
    sound2_end = None
    if sounds_list_2 is not None:
        sound2_index = np.random.choice(len(sounds_list_2))
        sound2, _ = read_audio(sounds_list_2[sound2_index], target_fs=fs)
        segments_where_sound2_could_be = []
        if sound1_start > len(sound2):
            segments_where_sound2_could_be.append((0, sound1_start - 1))
        if len(background) - sound1_end - 1 > len(sound2):
            segments_where_sound2_could_be.append((sound1_end + 1, len(background) - 1))
        # print[sound1_start, sound1_start + len(sound1)], segments_where_sound2_could_be, len(sound2)
        if len(segments_where_sound2_could_be) > 0:
            sound2_segment_index = np.random.choice(len(segments_where_sound2_could_be))
            segment_start, segment_end = segments_where_sound2_could_be[sound2_segment_index]
            sound2_start, sound2_end = random_segment_in_range(segment_start, segment_end, len(sound2))

    audios_with_info = []
    audios_with_info.append({'audio': sound1, 'start_pos': sound1_start, 'r': r1})
    if sound2_start is not None:
        audios_with_info.append({'audio': sound2, 'start_pos': sound2_start, 'r': r2})
        del sounds_list_2[sound2_index]
        sound2_end = sound2_start + len(sound2)

    audios_with_info.append({'audio': background, 'start_pos': 0, 'r': r3})

    return mix_audios(audios_with_info, mixed_audio_lenght=duration, fs=fs), sound1_start, sound1_end, sound2_start, sound2_end


def data_mixer(coughs_sounds_path, negative_sounds_path, background_noises_path, audio_duration, audio_fs, save_path):

    cough_sounds = glob.glob(os.path.join(coughs_sounds_path, '*'))
    negative_sounds = glob.glob(os.path.join(negative_sounds_path, '*'))
    background_noises = glob.glob(os.path.join(background_noises_path, '*'))

    mixed_audios = []
    # print cough_sounds, negative_sounds, background_noises

    while len(cough_sounds) > 0 or len(negative_sounds) > 0:
        print len(cough_sounds), len(negative_sounds)

        combination_varient = np.random.choice(6)

        if combination_varient == 0 and len(cough_sounds) > 1:
            mixed_audios.append(
                (mix_audio_combination(cough_sounds, cough_sounds, background_noises, audio_duration, audio_fs), True, True))
        elif combination_varient == 1 and len(cough_sounds) > 0 and len(negative_sounds) > 0:
            mixed_audios.append(
                (mix_audio_combination(cough_sounds, negative_sounds, background_noises, audio_duration, audio_fs), True, False))
        elif combination_varient == 2 and len(negative_sounds) > 0 and len(cough_sounds) > 0:
            mixed_audios.append(
                (mix_audio_combination(negative_sounds, cough_sounds, background_noises, audio_duration, audio_fs), False, True))
        elif combination_varient == 3 and len(negative_sounds) > 1:
            mixed_audios.append(
                (mix_audio_combination(negative_sounds, negative_sounds, background_noises, audio_duration, audio_fs), False, False))
        elif combination_varient == 4 and len(cough_sounds) > 0:
            mixed_audios.append(
                (mix_audio_combination(cough_sounds, None, background_noises, audio_duration, audio_fs), True, False))
        elif combination_varient == 5 and len(negative_sounds) > 0:
            mixed_audios.append(
                (mix_audio_combination(negative_sounds, None, background_noises, audio_duration, audio_fs), False, False))

    labels = []
    for i, mixed_audio_with_info in enumerate(mixed_audios):
        mixed_audio_with_info, is_sound1_cough, is_sound2_cough = mixed_audio_with_info
        audio, sound1_start, sound1_end, sound2_start, sound2_end = mixed_audio_with_info

        audio_name = 'mixed_audio_' + str(i) + '_'
        cough_sound_places = []
        if is_sound1_cough and sound1_start is not None:
            cough1_start = (sound1_start / float(audio_fs))
            cough1_end = (sound1_end / float(audio_fs))
            audio_name = audio_name + ('%.2f' % cough1_start) + '_' + ('%.2f' % cough1_end) + '_'
            cough_sound_places.append((cough1_start, cough1_end))
        if is_sound2_cough and sound2_start is not None:
            cough2_start = (sound2_start / float(audio_fs))
            cough2_end = (sound2_end / float(audio_fs))
            audio_name = audio_name + ('%.2f' % cough2_start) + '_' + ('%.2f' % cough2_end)
            cough_sound_places.append((cough2_start, cough2_end))
        audio_name += '.wav'

        full_path_name = os.path.join(save_path, audio_name)
        librosa.output.write_wav(full_path_name, audio, sr=audio_fs, norm=False)

        labels.append({'path': full_path_name, 'cough_sound_places': cough_sound_places})

    save_object_as_pkl(os.path.join(save_path, 'labels.pckl'), labels)


audio_duration = 10
audio_fs = 44100

coughs_sounds_path = '/home/sam/Desktop/first_model_with_data/data_source/cough_data'
negative_sounds_path = '/home/sam/Desktop/first_model_with_data/data_source/not_coughs_temp'
background_noises_path = '/home/sam/Desktop/first_model_with_data/data_source/background_noises'
save_path = '/home/sam/Desktop/first_model_with_data/data_source/mixed_data_audios'
# data_mixer(coughs_sounds_path=coughs_sounds_path, negative_sounds_path=negative_sounds_path,
#            background_noises_path=background_noises_path, audio_duration=audio_duration, audio_fs=audio_fs,
#            save_path=save_path)
