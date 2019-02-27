from torch.utils import data
import numpy as np
from utils import read_audio, convert_list_to_str


class SpectogramLoader(data.Dataset):
    def __init__(self, dataset, spectogram_lenght_in_seconds=10.):

        self.dataset = dataset
        self.dataset_size = len(dataset)
        self.spectogram_lenght_in_seconds = spectogram_lenght_in_seconds

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        # index = 0  # temp
        spectogram = self.dataset[index]['spectogram']

        ground_truth = np.zeros((1, spectogram.shape[1]))
        for range in self.dataset[index]['cough_sound_places']:
            start_range = int(np.round(range[0] / self.spectogram_lenght_in_seconds * spectogram.shape[1]))
            end_range = int(np.round(range[1] / self.spectogram_lenght_in_seconds * spectogram.shape[1]))
            ground_truth[0, start_range:end_range+1] = 1.0

        spectogram = spectogram[..., np.newaxis]
        ground_truth = ground_truth[..., np.newaxis]

        cough_places_str = convert_list_to_str(self.dataset[index]['cough_sound_places'])

        audio, fs = read_audio(self.dataset[index]['path'])

        return spectogram, ground_truth, audio, fs, cough_places_str
