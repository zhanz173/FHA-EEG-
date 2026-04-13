import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import mne
import os

"""
Handle EEG in fif format
Expected dataset structure:
dataset_folder/
--Hospital1/
----ScanID_1_raw.fif
----ScanID_2_raw.fif
--Hospital2/
----ScanID_3_raw.fif
"""

class FHA_EEG_Dataset(Dataset):
    EEG_CHANNELS_ORDER = ['C3','C4','Cz','F3','F4','F7','F8','Fz','Fp1','Fp2','Fpz','O1','O2','P3','P4','Pz','T3','T4','T5','T6']

    def __init__(self, data_csv_path, dataset_folder, resample_rate=200):
        self.dataset_folder = dataset_folder
        self.resample_rate = resample_rate
        self._load_csv(data_csv_path)

    def _load_csv(self, data_csv_path):
        self.data_csv = pd.read_csv(data_csv_path)

    def __len__(self):
        return len(self.data_csv)
    
    def _load_EEG(self, file_path, ScanID):
        # Implement this method to load the EEG data from the given file path and ScanID
        EEG_path = os.path.join(self.dataset_folder, file_path, f"{ScanID}_raw.fif")
       
        try:
            raw = mne.io.read_raw_fif(EEG_path, preload=False, verbose=False)
            raw.pick(FHA_EEG_Dataset.EEG_CHANNELS_ORDER)
            raw.resample(self.resample_rate, npad="auto")
            data = raw.get_data() # n_channels, n_times
            data = data.astype(np.float32)
        except Exception as e:
            print(f"Error loading EEG file {EEG_path}: {e}")
            data = None

        return data

    def __getitem__(self, idx):
        # Implement this method to retrieve a specific item from the dataset
        hospital = self.data_csv.iloc[idx]['Hospital']
        ScanID = self.data_csv.iloc[idx]['ScanID']

        EEG_raw_data = self._load_EEG(hospital, ScanID)

        data = {
            'EEG': EEG_raw_data,  # n_channels, n_times
            'ScanID': ScanID,
        }
        return data

if __name__ == "__main__":
    data_csv_path = "H:\\EEG\\Annotations\\EEG_hyperventilation.csv"
    dataset_folder = "H:\\EEG\\FHA\\hyperventilation"
    dataset = FHA_EEG_Dataset(data_csv_path, dataset_folder)

    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"EEG data shape: {sample['EEG'].shape}")
    print(f"ScanID: {sample['ScanID']}")