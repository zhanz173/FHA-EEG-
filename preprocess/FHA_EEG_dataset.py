import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import mne
import os
import re
import sys
from pathlib import Path

from typing import Dict, List, Optional, Tuple

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
def extract_site_and_sha256(input_root: Path, fif_path: Path) -> Tuple[str, str]:
    rel = fif_path.relative_to(input_root)
    if len(rel.parts) < 2:
        raise ValueError(f"Expected site/SHA256_raw.fif structure, got {fif_path}")
    site = rel.parts[0]
    sha256_id = re.sub(r"_raw\.fif$", "", fif_path.name)
    return site, sha256_id

def find_fif_files(input_root: Path) -> List[Path]:
    return sorted([p for p in input_root.rglob("*_raw.fif") if p.is_file()])

class FHA_EEG_Loader(Dataset):
    EEG_CHANNELS_ORDER = ['C3','C4','Cz','F3','F4','F7','F8','Fz','Fp1','Fp2','Fpz','O1','O2','P3','P4','Pz','T3','T4','T5','T6']

    def __init__(self, dataset_folder, resample_rate=200):
        self.resample_rate = resample_rate
        self.dataset_folder = dataset_folder
        self.data_files = find_fif_files(Path(dataset_folder))

    def __len__(self):
        return len(self.data_files)

    def _load_EEG(self, EEG_path: Path) -> Optional[np.ndarray]:
        # Implement this method to load the EEG data from the given file path and ScanID       
        try:
            raw = mne.io.read_raw_fif(EEG_path, preload=False, verbose=False)
            raw.pick(FHA_EEG_Loader.EEG_CHANNELS_ORDER)
            raw.resample(self.resample_rate, npad="auto")
            data = raw.get_data() # n_channels, n_times
            data = data.astype(np.float32)
        except Exception as e:
            print(f"Error loading EEG file {EEG_path}: {e}")
            data = None

        return data

    def __getitem__(self, idx):
        # Implement this method to retrieve a specific item from the dataset
        fif_path = self.data_files[idx]
        _, sha256_id = extract_site_and_sha256(Path(self.dataset_folder), fif_path)
        ScanID = sha256_id

        EEG_raw_data = self._load_EEG(fif_path)

        data = {
            'EEG_Raw': EEG_raw_data,  # n_channels, n_times
            'ScanID': ScanID,
        }
        return data

if __name__ == "__main__":
    dataset_folder = "H:\\EEG\\FHA\\hyperventilation"
    dataset = FHA_EEG_Loader(dataset_folder, resample_rate=200)

    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"EEG data shape: {sample['EEG_Raw'].shape}")
    print(f"ScanID: {sample['ScanID']}")