import torch
import numpy as np
import os
import h5py
import mne
from torch.utils import data
from torch.utils.data import Dataset
from functools import partial

FHA_EEG_channels_ORDER =['C3','C4','Cz','F3','F4','F7','F8','Fz','Fp1','Fp2','Fpz','O1','O2','P3','P4','Pz','T3','T4','T5','T6']

class FHA_EEG_Loader(Dataset):
    def __init__(self, config):
        self.dataset_root = config['dataset_root']
        self.tmax = config['tmax']
        self.sampling_rate = config['sampling_rate']
        
        self.file_list = self._load_files(self.dataset_root)
    def __len__(self):
       return len(self.file_list)

    def __getitem__(self, idx):
        return self._pack_data(idx)
    
    def _pack_data(self, idx):
        # read EEG file with mne
        file_path = self.file_list[idx]
        EEG_Id = os.path.basename(file_path).split('_')[0]  # Assuming filename format is "ScanID_raw.fif"
        try:
            raw = mne.io.read_raw_fif(file_path, preload=False, verbose=False)
            raw.pick(FHA_EEG_channels_ORDER)
            raw.resample(self.sampling_rate, npad="auto")
            data = raw.get_data(tmax=self.tmax) # n_channels, n_times
            data = data.astype(np.float32)
        except Exception as e:
            print(f"Error loading EEG file {file_path}: {e}")
            data = None

        data_dict = {'EEG_Raw': data, 'ScanID': EEG_Id}
        return data_dict

    def _load_files(self, root_dir):
        file_list = []
        for dirpath, dirnames, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.endswith('.fif'):
                    file_list.append(os.path.join(dirpath, filename))
        return file_list


if __name__ == '__main__':
    from functools import partial

    EEG_FS = 256
    TARGET_FS = 200
    EEG_LEN = 360
    EEG_FOLDER = r"H:\EEG\FHA\hyperventilation"

    config={'sampling_rate': 200, 'tmax': 360, 'dataset_root': EEG_FOLDER}
    test_loader = FHA_EEG_Loader(config)
    print(f"Total EEG files found: {len(test_loader)}")
    for idx in range(10):
        data_dict = test_loader._pack_data(idx)
        print(f"EEG data shape for file {idx}: {data_dict['EEG_Raw'].shape}")
        print(f"EEG ID for file {idx}: {data_dict['ScanID']}")

