import numpy
from utils import read_audio, restore_pkl_object, save_object_as_pkl


def from_audio_to_spectogram(signal, sample_rate, pre_emphasis, frame_size, frame_stride, NFFT, nfilt):
    emphasized_signal = numpy.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(numpy.ceil(
        float(numpy.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

    pad_signal_length = num_frames * frame_step + frame_length
    z = numpy.zeros((pad_signal_length - signal_length))
    # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal
    pad_signal = numpy.append(emphasized_signal, z)

    indices = numpy.tile(numpy.arange(0, frame_length), (num_frames, 1)) + numpy.tile(
        numpy.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(numpy.int32, copy=False)]

    frames *= numpy.hamming(frame_length)

    mag_frames = numpy.absolute(numpy.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum

    low_freq_mel = 0
    high_freq_mel = (2595 * numpy.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
    mel_points = numpy.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10 ** (mel_points / 2595) - 1))  # Convert Mel to Hz
    bin = numpy.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = numpy.zeros((nfilt, int(numpy.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])  # left
        f_m = int(bin[m])  # center
        f_m_plus = int(bin[m + 1])  # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = numpy.dot(pow_frames, fbank.T)
    filter_banks = numpy.where(filter_banks == 0, numpy.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * numpy.log10(filter_banks)  # dB

    return filter_banks.T


def prepare_spectograms(labels_pkl_path, fs=44100):
    labels = restore_pkl_object(labels_pkl_path)

    maxes = []
    minis = []
    for i, audio_info in enumerate(labels):
        audio, fs = read_audio(audio_path=audio_info['path'], target_fs=fs)
        spectogram = from_audio_to_spectogram(audio, fs, 0.97, 0.064, 0.032, 4096, 64)

        audio_info['spectogram'] = spectogram

        print str(len(labels)) + '/' + str(i+1), numpy.max(spectogram), numpy.min(spectogram)
        maxes.append(numpy.max(spectogram))
        minis.append(numpy.min(spectogram))

    save_object_as_pkl(labels_pkl_path, labels)
    # print labels
    print numpy.max(maxes), numpy.min(minis)


# prepare_spectograms('/home/sam/Desktop/first_model_with_data/labels.pckl')

