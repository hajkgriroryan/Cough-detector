from argparse import ArgumentParser
import tensorflow as tf
import time
import numpy as np
import os
import librosa
import sounddevice as sd
from utils import read_audio
from prepare_dataset import from_audio_to_spectogram
from data_augmentor import zero_pad
from data_augmentor import random_crop, mix_audios

parser = ArgumentParser()


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise parser.ArgumentTypeError('Boolean value expected.')


parser.add_argument('--model_path', type=str, help='pretrained model path', required=True)
# parser.add_argument('--audio_path', type=str, help='audio path', required=True)
# parser.add_argument('--ground_truth', type=str, help='ground truth of audio, if it is')
# parser.add_argument('--output_dir', type=str, help='test output direction', required=True)
parser.add_argument('--use_gpu', type=str2bool, default=False)

os.chdir(os.path.dirname(os.path.realpath(__file__)))


def load_graph(graph_path, graph_name="cough_detector"):
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(graph_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        output_nodes = tf.import_graph_def(graph_def, name=graph_name, return_elements=["input:0", "output:0"])

    return output_nodes


def get_predicted_cough_places(prediction, threshold, max_gap_lenght_s, min_cough_lenght_s,  audio_lenght_s=10):
    time_scale = audio_lenght_s / float(len(prediction))
    cough_places = []
    max_gap_lenght = int(max_gap_lenght_s / time_scale)
    min_cough_lenght = int(min_cough_lenght_s / time_scale)
    cough_start = None
    cough_end = None
    gap_count = 0
    for i, p in enumerate(prediction):
        if p > threshold:
            if cough_start is None:
                cough_start = i

            cough_end = i

        elif cough_start is not None:
            gap_count += 1
            if gap_count > max_gap_lenght:
                if cough_end - cough_start + 1 > min_cough_lenght:
                    cough_places.append((cough_start * time_scale, cough_end * time_scale))
                    cough_start = None
                    cough_end = None
                gap_count = 0

    if cough_start is not None and cough_end - cough_start + 1 > min_cough_lenght:
        cough_places.append((cough_start * time_scale, cough_end * time_scale))

    return cough_places


def do_audio_noramlization(audio, fs):
    norm_audio, fs_norm = read_audio('/home/hayk/PycharmProjects/data_source/not_coughs_temp/1-18655-A-31.wav')

    audios_with_info = [{'audio': audio, 'start_pos':0}, {'audio': norm_audio,
                                                          'start_pos': len(audio), 'r': 0.5}]

    mixed_audio = mix_audios(audios_with_info, int((len(audio) + len(norm_audio)) / float(fs)) + 2, fs)
    print int((len(audio) + len(norm_audio)) / float(fs)) + 2

    return mixed_audio[:len(audio)]


def main():
    options = parser.parse_args()
    if options.use_gpu:
        device = '/gpu:0'
    else:
        device = '/cpu:0'

    grop = tf.GraphOptions(optimizer_options=tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L1))
    config = tf.ConfigProto(device_count={"CPU": 1}, inter_op_parallelism_threads=1,
                            intra_op_parallelism_threads=1, use_per_session_threads=True, graph_options=grop)

    _, model_name = os.path.split(options.model_path)

    # if not os.path.exists(options.output_dir):
    #     os.makedirs(options.output_dir)

    with tf.Session(config=config) as sess:
        with tf.device(device):
            run_arg = load_graph(options.model_path)

            fs = 44100
            while(True):
                print 'press to start recording'
                raw_input()
                print 'recording is started'
                recording = sd.rec(int(10 * fs), samplerate=fs, channels=2, dtype='float64', blocking=True)

                print 'recording is finished'
                librosa.output.write_wav('./temp.wav', recording, sr=fs, norm=False)

                audio, fs = read_audio('./temp.wav', fs)
                # audio = do_audio_noramlization(audio, fs)
                # librosa.output.write_wav('./temp_normed.wav', audio, sr=fs, norm=False)
                # audio, fs = read_audio('./temp_normed.wav', fs)

                if len(audio) < 441000:
                    audio = zero_pad(audio, 0, 10, fs)
                elif len(audio) > 441000:
                    print 'audio lenght is bigger 10s, please try smaller audio'
                    exit()

                spectogram = from_audio_to_spectogram(audio, fs, 0.97, 0.064, 0.032, 4096, 64)
                spectogram = spectogram[np.newaxis, ..., np.newaxis]

                start = time.time()

                network_output = sess.run(run_arg[1], {run_arg[0]: spectogram})
                print network_output

                cough_places = get_predicted_cough_places(prediction=network_output[0, 0, :, 0], threshold=0.7, max_gap_lenght_s=0.2,
                                                          min_cough_lenght_s=0.1)

                print cough_places

                print(time.time() - start)


if __name__ == '__main__':
    main()
