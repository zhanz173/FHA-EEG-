import torch
import numpy as np
import os
import h5py
import mne
import pandas as pd
from torch.utils.data import Dataset
from functools import partial
from scipy.signal import resample

FHA_EEG_channels_ORDER =['C3','C4','Cz','F3','F4','F7','F8','Fz','Fp1','Fp2','Fpz','O1','O2','P3','P4','Pz','T3','T4','T5','T6']

class FHA_EEG_Loader(Dataset):
    def __init__(self, data_csv_path, dataset_folder, config):
        self.dataset_folder = dataset_folder
        self._set_config(config)
        self._load_csv(data_csv_path)
        self._sort_data() # sort the data by ChunksName for faster sequential loading
        self.start_timestamps = 0

    def __len__(self):
        return len(self.data_csv)

    def __getitem__(self, idx):
        file_path, ScanID = self._get_file_path(idx)
        data = self._load_EEG(file_path, ScanID)
        data = self._crop_data(data)
        data = self._resample_data(data)

        labels = self._extract_labels(idx)
        return self._pack_data(data, labels)
    
    #convert the data to dictionary
    def _pack_data(self, data, labels):
        data_dict = {'EEG_Raw': data}
        for i, key in enumerate(self.label_keys):
            data_dict[key] = labels[i]
        return data_dict
    
    def _extract_labels(self, idx):
        return self.data_csv.iloc[idx][self.label_keys].values

    def _get_file_path(self, idx):
        ScanID = self.data_csv.iloc[idx]['ScanID']
        chunk_name = self.data_csv.iloc[idx]['chunk_index']
        return os.path.join(self.dataset_folder, chunk_name), ScanID
    
    def _crop_data(self, data):
        # (channels, timepoints)
        # crop the data to the window size
        if data.shape[-1] < self.window_size:
            return data
        
        if self.access_pattern == 'random':
            start = np.random.randint(0, len(data) - self.window_size)
            data = data[..., start:start+self.window_size]
        elif self.access_pattern == 'sequential':
            data = data[..., self.start_timestamps:self.start_timestamps+self.window_size]

        return data
    
    #for sequential access pattern, after each epoch, increment the start location by window size
    def _on_epoch_end(self):
        self.total_iterations += 1
        if self.access_pattern == 'sequential':
            self.start_timestamps += self.window_size

    #for sequential access pattern, if the start location + window size is greater than the data length, reset the start location to 0
    def _on_epoch_start(self):
        if self.access_pattern == 'sequential':
            if self.start_timestamps + self.window_size >= self.max_length:
                self.start_timestamps = 0

    def _load_EEG(self, file_path, scan_id):
        with h5py.File(file_path, 'r') as f:
            data = f[scan_id][:]
        return data
    
    def _load_csv(self, data_csv_path):
        self.data_csv = pd.read_csv(data_csv_path)
    
    def _sort_data(self):
        self.data_csv = self.data_csv.sort_values(by='chunk_index')
        # select row with chunk_index not na
        self.data_csv = self.data_csv[self.data_csv['chunk_index'].notna()]
        return self.data_csv
    
    def _resample_data(self, data):
        if self.sampling_rate != 256:
            max_samples = min(data.shape[-1], self.window_size)
            data = resample(data, int(max_samples * self.sampling_rate / 256), axis=-1)
        return data.astype(np.float32)
    
    def _set_config(self, config):
        self.config = config
        self.channels = int(config['channels'])
        self.window_size = int(config['window_size'])
        self.sampling_rate = config['sampling_rate']
        self.access_pattern = config['access_pattern']
        self.label_keys = config['label']
        self.max_length = config['max_length']

        assert self.channels < 21 and self.channels > 0, "channels must be between 1 and 20"
        assert self.sampling_rate <= 256, "sampling rate must be less than 256 Hz"



if __name__ == '__main__':
    from functools import partial

    EEG_FS = 256
    TARGET_FS = 200
    EEG_LEN = 360
    workspace = r'H:\EEG\FHA\Resting\001_a01_01'
    annotations_file = [r"H:\EEG\FHA\annotations\processed.csv"]
    dataset_root = r''
    extension = r'_raw.fif'
    Hosptial = 'Burnaby'

    config={'channels': 20, 'sampling_rate': 200, 'window_size': 1e9, 'access_pattern': 'sequential','label': ['ScanID'], 'max_length': 360*256}
    EEG_DATA = FHA_EEG_Loader(r"H:\EEG\Annotations\FHA_annotation_full.csv", r"H:\EEG\FHA\Resting\chunks", config)