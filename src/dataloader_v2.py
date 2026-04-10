import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import h5py
import os

## WIP: Will replace EEGDatasetWithLabel in the future after refactoring
## The EEG fetures are stored in pyarrow sharded format, with an index parquet file and multiple h5 files containing the actual features. This dataset class is designed
class EEGFeatureDataset(Dataset):
    """
    One item = one EEG, containing all of its segments.

    Returns:
        {
            "labram": Tensor [N_seg, ...]   (optional)
            "welch": Tensor [N_seg, ...]    (optional)
            "eeg_id": str
            "segment_idx": Tensor [N_seg]
            "start_sec": Tensor [N_seg]
            "end_sec": Tensor [N_seg]
        }
    """
    def __init__(
        self,
        root_dir: str,
        EEG_metadata_path: str,
        index_file: str = "index.parquet",
        use_labram: bool = True,
        use_welch: bool = True,
        use_ordinal_label: bool = False,
    ):
        self.root_dir = root_dir
        self.index_path = os.path.join(root_dir, index_file)
        self.df = pd.read_parquet(self.index_path).reset_index(drop=True)

        self.use_labram = use_labram
        self.use_welch = use_welch
        self.use_ordinal_label = use_ordinal_label

        self._h5_cache = {}

        # group rows by eeg_id
        self.eeg_ids = []
        self.groups = []

        for eeg_id, g in self.df.groupby("eeg_id", sort=False):
            g = g.sort_values("segment_idx").reset_index(drop=True)
            self.eeg_ids.append(str(eeg_id))
            self.groups.append(g)

        self.metadata = self._read_metadata(EEG_metadata_path)
        self._filter_eeg_without_labels(self.metadata)

    def __len__(self):
        return len(self.groups)

    def _get_h5(self, shard_name: str):
        if shard_name not in self._h5_cache:
            shard_path = os.path.join(self.root_dir, shard_name)
            self._h5_cache[shard_name] = h5py.File(shard_path, "r")
        return self._h5_cache[shard_name]

    def __getitem__(self, idx: int):
        g = self.groups[idx]
        eeg_id = self.eeg_ids[idx]
        labels = self._read_label(eeg_id)

        out = {
            "eeg_id": eeg_id,
            "labels": torch.tensor(labels, dtype=torch.long),
            "segment_idx": torch.tensor(g["segment_idx"].to_numpy(), dtype=torch.long),
            "start_sec": torch.tensor(g["start_sec"].to_numpy(), dtype=torch.float32),
            "end_sec": torch.tensor(g["end_sec"].to_numpy(), dtype=torch.float32),
        }

        if self.use_labram:
            labram_list = []
            for _, row in g.iterrows():
                h5f = self._get_h5(row["shard"])
                labram_list.append(h5f["LaBraM"][int(row["row"])])
            out["labram"] = torch.tensor(np.stack(labram_list), dtype=torch.float32)

        if self.use_welch:
            welch_list = []
            for _, row in g.iterrows():
                h5f = self._get_h5(row["shard"])
                welch_list.append(h5f["Welch"][int(row["row"])].reshape(-1, 50)) # reshape to [n_channels, n_freq_bins]
            out["welch"] = torch.tensor(np.stack(welch_list), dtype=torch.float32)

        return out
    
    def _read_label(self, eeg_id):
        ## get [Focal Epi, Gen Epi, Focal Non-epi, Gen Non-epi, Abnormality] label vector
        row = self.metadata[self.metadata["ScanID"] == eeg_id]
        if len(row) == 0:
            raise KeyError(f"EEG ID {eeg_id} not found in metadata")
        ordinal_label = row.iloc[0][["Focal Epi", "Gen Epi", "Focal Non-epi", "Gen Non-epi", "Abnormality"]].values.astype('int32')
        if self.use_ordinal_label:
            return ordinal_label
        else:
            return (ordinal_label[:5] > 2).astype('int32') # convert to multi-label binary
        
    def _read_metadata(self, metadata_path):
        try:
            metadata = pd.read_csv(metadata_path)
        except Exception as e:
            print(f"Error reading metadata from {metadata_path}: {e}")
            raise
        return metadata
    
    def _filter_eeg_without_labels(self, metadata):
        valid_eeg_ids = set(metadata["ScanID"].unique())
        filtered_groups = []
        filtered_eeg_ids = []
        for eeg_id, g in zip(self.eeg_ids, self.groups):
            if eeg_id in valid_eeg_ids:
                filtered_groups.append(g)
                filtered_eeg_ids.append(eeg_id)
        self.groups = filtered_groups
        self.eeg_ids = filtered_eeg_ids
        print(f"Filtered EEGs without labels. Remaining EEGs: {len(self.eeg_ids)}")

    def close(self):
        for f in self._h5_cache.values():
            try:
                f.close()
            except Exception:
                pass
        self._h5_cache = {}

    def __del__(self):
        self.close()

## special collate_fn to handle variable number of segments across EEGs, will pad with zeros to the max number of segments in the batch, and create a mask to indicate valid segments
def eeg_collate_fn_variable_length(batch):
    """
    Pads variable number of segments across EEGs.

    Returns:
        {
            "x": Tuple of features for compatibility with existing model code, e.g. (labram, welch) where each is [B, N_max, ...]
            "y": Tensor [B, num_labels]
            "mask": [B, N_max]   True for valid segments
            "eeg_id": list[str]
            "lengths": LongTensor [B]
        }
    """
    B = len(batch)
    lengths = torch.tensor([item["segment_idx"].shape[0] for item in batch], dtype=torch.long)
    labels = torch.stack([item["labels"] for item in batch], dim=0) # [B, num_labels]
    N_max = int(lengths.max())

    out = {
        "x": (),
        "y": labels,
        "mask": torch.arange(N_max).unsqueeze(0) < lengths.unsqueeze(1),
        "eeg_id": [item["eeg_id"] for item in batch],
        "lengths": lengths,
    }

    if "labram" in batch[0]:
        feat_shape = batch[0]["labram"].shape[1:]
        labram_pad = torch.zeros((B, N_max, *feat_shape), dtype=batch[0]["labram"].dtype)
        for i, item in enumerate(batch):
            n = item["labram"].shape[0]
            labram_pad[i, :n] = item["labram"]
        out["x"] += (labram_pad,)

    if "welch" in batch[0]:
        feat_shape = batch[0]["welch"].shape[1:]
        welch_pad = torch.zeros((B, N_max, *feat_shape), dtype=batch[0]["welch"].dtype)
        for i, item in enumerate(batch):
            n = item["welch"].shape[0]
            welch_pad[i, :n] = item["welch"]
        out["x"] += (welch_pad,)

    return out

def test_EEGFeatureDataset():
    dataset = EEGFeatureDataset(root_dir=r"H:\EEG_features\EEG_features_labram_welch_HV")
    item = dataset[0]
    print(item.keys())
    print(item["x"][0].shape, item["x"][1].shape)


def test_dataset():
    dataset = EEGFeatureDataset(root_dir=r"H:\EEG_features\EEG_features_labram_welch_HV", EEG_metadata_path=r"E:\project\FHA-EEG-\data\EEG_HV_metadata.csv")
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=eeg_collate_fn_variable_length)
    for batch in dataloader:
        print(batch.keys())
        print(batch["x"][0].shape, batch["x"][1].shape, batch["mask"].shape)
        print(batch["y"].shape)
        print(f"EEG lengths: {batch['lengths']}")
        break

if __name__ == "__main__":
    test_dataset()
