import os
import h5py
import numpy as np
import pandas as pd
from pyparsing import Dict
from torch.utils.data import Dataset
from typing import Optional, Tuple, Dict

# max (Focal Epi,Gen Epi, Focal Non-epi, Gen Non-epi) <= Abnormality
# the maximum of the four confidence levels should not exceed the Abnormality confidence level
def check_confidence_levels(row):
    max_confidence = max(row['Focal Epi'], row['Gen Epi'], row['Focal Non-epi'], row['Gen Non-epi'])
    return max_confidence <= row['Abnormality']

class EEGDatasetWithLabel(Dataset):
    ## HDF5 dataset with LaBraM and Welch features
    ## This dataset is specifically designed for our current EEG feature storage format
    ## Expects a csv file with EEG metadata with matching eeg_id column
    ## Expects a directory with: pyarrow index + h5 shards
    ## Index columns: eeg_id, shard, row
    ## Expects each shard to have datasets: LaBraM, Welch, ids
    ## Return (LaBraM, Welch, label) tuples
    def __init__(self, root="h5_eeg_feats", metadata = None,ids=None, return_ids=False, return_neurologist_ids=False, return_ordinal=False):
        self.root = root
        df = pd.read_parquet(os.path.join(root, "index.parquet"))
        if ids is not None:
            df = df[df["eeg_id"].isin(set(ids))].reset_index(drop=True)

        self.index = df # index dataframe of EEG features
        self.return_ids = return_ids
        self.return_neurologist_ids = return_neurologist_ids
        self._files = {}  # cache: shard -> h5py.File
        self._hasLabels = False
        self._ordinal = return_ordinal
        self._read_metadata(metadata) # read metadata if provided

    def _read_metadata(self, metadata_path, drop_dupliate_EEG=True):
        ## matching metadata to index
        if metadata_path is None:
             ## unsupervised mode, return none labels
             ## rename eeg_id to ScanID for consistency
            self.index = self.index.rename(columns={"eeg_id": "ScanID"})
            return
        
        # check if the path is absolute or relative
        if os.path.isabs(metadata_path):
            abs_path = metadata_path
        else:
            abs_path = os.path.join(self.root, metadata_path)
            
        print(f"Loading metadata from {abs_path}")
        n_samples_pre = len(self.index)
        meta = pd.read_csv(abs_path)
        meta = meta.dropna(subset=['Hashed_ReportURN']) # No diagnosis available
        meta = meta.query('HourDiffURN <= 24') # keep only EEGs within 24 hours of admission

        # cross-filter, keep rows in index with matching EEG_ID in metadata
        merged = pd.merge(meta, self.index,  left_on='ScanID', right_on='eeg_id', how='inner')

        # Drop rows with missing labels to prevent NaN losses during training/validation
        label_cols = ["Focal Epi", "Gen Epi", "Focal Non-epi", "Gen Non-epi", "Abnormality"]
        if not set(label_cols).issubset(set(merged.columns)):
            missing = list(set(label_cols) - set(merged.columns))
            raise KeyError(f"Missing expected label columns in metadata: {missing}")
        merged = merged.dropna(subset=label_cols)

        # print how many samples are dropped after
        n_samples_post = len(merged)
        print(f"Dropped {n_samples_pre - n_samples_post} samples after preprocessing.")

        # Coerce to numeric and clip to [0,1] just in case metadata contains stray values
        for c in label_cols:
            merged[c] = pd.to_numeric(merged[c], errors='coerce')
        merged = merged.dropna(subset=label_cols)  # drop rows that became NaN after coercion

        # Age may be NaN; drop or fill if needed (not used in loss)
        if 'Age_YEARS' in merged.columns:
            merged['Age_YEARS'] = pd.to_numeric(merged['Age_YEARS'], errors='coerce')

        # ensure confidence levels are consistent
        confidence_check = merged.apply(check_confidence_levels, axis=1)
        n_inconsistent = (~confidence_check).sum()
        if n_inconsistent > 0:
            print(f"Dropping {n_inconsistent} samples with inconsistent confidence levels.")
            merged = merged[confidence_check]

        # check if label mask columns exist, if not create them with all True
        quality_cols = ["Focal Epi_clean", "Gen Epi_clean", "Focal Non-epi_clean", "Gen Non-epi_clean", "Abnormality_clean"]
        for col in quality_cols:
            if col not in merged.columns:
                merged[col] = True  # assume all labels are clean if no column provided

        # check if neurologist column exist, if not create with all "unknown"
        if "Physician" not in merged.columns:
            merged["Physician"] = "unknown"

        self.index = merged.reset_index(drop=True)
        self._hasLabels = True

    def __len__(self):
        return len(self.index)

    def _open(self, shard):
        f = self._files.get(shard)
        if f is None:
            f = h5py.File(os.path.join(self.root, shard), "r")
            self._files[shard] = f
        return f
    
    def _read_label(self, eeg_id):
        ## get [Focal Epi, Gen Epi, Focal Non-epi, Gen Non-epi, Abnormality, Age_YEARS] label vector
        assert self._hasLabels, "Metadata not loaded"
        row = self.index[self.index["ScanID"] == eeg_id]
        if len(row) == 0:
            raise KeyError(f"EEG ID {eeg_id} not found in metadata")
        return row.iloc[0][["Focal Epi", "Gen Epi", "Focal Non-epi", "Gen Non-epi", "Abnormality", "Age_YEARS"]].values.astype('float32')

    def _read_features(self,row):
        f = self._open(row.shard)
        r = int(row.row)
        x_lbr   = f["LaBraM"][r]  # type: ignore (T1, F1)
        x_welch = f["Welch"][r]    # type: ignore (T2, F2*20 channels)
        x_welch = x_welch.reshape(x_welch.shape[0], 20, -1) # type: ignore reshape to (T2, 20, F2)
        return x_lbr, x_welch * 1e3 # scale Welch features
    
    # weight for each sample based on label confidence
    # for now we just use simple rules: Confident no {1} -> 1, low confidence no {2} -> 0.75, low confident yes {3} -> 0.75, confident yes {4} -> 1
    # TODO: implement curiculum learning with mentorNet
    def _get_sample_weight(self, idx):
        row = self.index.iloc[idx]
        if "Focal Epi score" not in row.index:
            # if no score columns, return Confident no {1} -> 1, low confidence no {2} -> 0.75, low confident yes {3} -> 0.75, confident yes {4} -> 1
            weight = np.ones(5, dtype='float32')
            label_cols = ["Focal Epi", "Gen Epi", "Focal Non-epi", "Gen Non-epi", "Abnormality"]
            for i, col in enumerate(label_cols):
                label = row[col]
                if label == 2 or label == 3:
                    weight[i] = 0.5
            return weight
        weight = row[["Focal Epi score", "Gen Epi score", "Focal Non-epi score", "Gen Non-epi score", "Abnormality score"]].values.astype('float32')
        return weight
    
    # label csv files should contain a boolean value indicating label quality
    # Focal Epi_clean	Gen Epi_clean	Focal Non-epi_clean	Gen Non-epi_clean	Abnormality_clean : Optional[True/False | None]
    def _get_label_quality(self, idx):
        row = self.index.iloc[idx]
        if not self._hasLabels:
            return None
        quality_cols = ["Focal Epi_clean", "Gen Epi_clean", "Focal Non-epi_clean", "Gen Non-epi_clean", "Abnormality_clean"]
        label_is_clean = row[quality_cols].values.astype('bool')
        return label_is_clean
    
    def _build_neurologist_table(self) -> Dict[str, int]:
        """
        Build a mapping from neurologist IDs to integer indices.
        Neurologist with less than 200 labels -> 0, others assigned unique indices starting from 1.
        0 index is reserved for "unknown" neurologist.
        """
        neurologist_ids_value_counts = self.get_distinct_neurologist_value_counts()
        selected_neurologist_ids = neurologist_ids_value_counts[neurologist_ids_value_counts >= 200].index.tolist()
        neurologist_id_to_index = {nid: idx + 1 for idx, nid in enumerate(selected_neurologist_ids)} # str:int mapping
        return neurologist_id_to_index
    
    def _index_neurologist_id(self, neurologist_id: str) -> int:
        """
        Convert a neurologist ID to its corresponding index.
        Neurologist with less than 200 labels -> 0 (unknown).
        """
        if not hasattr(self, '_neurologist_id_to_index'):
            self._neurologist_id_to_index = self._build_neurologist_table()
        return self._neurologist_id_to_index.get(neurologist_id, 0)
    
    def __getitem__(self, i):
        data_sample = dict()
        row = self.index.iloc[i]
        x_lbr, x_welch = self._read_features(row)
        data_sample['x'] = (x_lbr, x_welch)
        if self._hasLabels:
            ordinal_label = row[["Focal Epi", "Gen Epi", "Focal Non-epi", "Gen Non-epi", "Abnormality"]].values.astype('float32')
            sample_weight = self._get_sample_weight(i)
            label_mask = self._get_label_quality(i)

            data_sample["sample_weight"] = sample_weight
            data_sample["labels_mask"] = label_mask
            data_sample["age"] = float(row["Age_YEARS"]) * 0.01 # 

            if self.return_ids:
                id = row["Hashed_ReportURN"]
                data_sample["id"] = id
            if self.return_neurologist_ids:
                neurologist_id = row["Physician"] # skip assert
                neurologist_index = self._index_neurologist_id(neurologist_id)
                data_sample["neurologist_id"] = neurologist_index # int index
            if self._ordinal:
                data_sample["y"] = ordinal_label - 1 # convert to 0-indexed ordinal labels
            else:
                data_sample["y"] = (ordinal_label > 2).astype('float32') # binary labels

        return data_sample

    def get_by_id(self, eeg_id):
        row = self.index[self.index["ScanID"] == eeg_id]
        if len(row) == 0:
            raise KeyError(f"EEG ID {eeg_id} not found in index")
        if len(row) > 1:
            raise KeyError(f"EEG ID {eeg_id} found multiple times in index")
        row = row.iloc[0]
        x_lbr, x_welch = self._read_features(row)
        return x_lbr, x_welch
    
    def get_all_labels(self):
        if not self._hasLabels:
            raise ValueError("Metadata not loaded, no labels available")
        labels = self.index[["Focal Epi", "Gen Epi", "Focal Non-epi", "Gen Non-epi", "Abnormality"]].values.astype('float32')
        return labels
    
    def get_distinct_neurologist_value_counts(self) -> pd.Series:
        if "Physician" not in self.index.columns:
            raise ValueError("Metadata does not contain Physician column")
        neurologist_ids_value_counts = self.index["Physician"].value_counts()
        return neurologist_ids_value_counts
    
    def get_valide_neurologist_counts(self) -> int:
        neurologist_ids_value_counts = self.get_distinct_neurologist_value_counts()
        selected_neurologist_ids = neurologist_ids_value_counts[neurologist_ids_value_counts >= 500].index.tolist()
        return len(selected_neurologist_ids) + 1 # +1 for average neurologist
    
    def get_positive_weights(self):
        ## compute positive weights for each class for balancing
        if not self._hasLabels:
            raise ValueError("Metadata not loaded, no labels available")
        labels = self.get_all_labels() > 2  # (N, K)
        mask = self.index[["Focal Epi_clean", "Gen Epi_clean", "Focal Non-epi_clean", "Gen Non-epi_clean", "Abnormality_clean"]].values.astype('bool')
        n_samples_clean = mask.sum(axis=0)  # (K,)
        n_samples_positive = (labels & mask).sum(axis=0)  # (K,)
        # torch style pos weight: n_negative/n_positive = (N - P) / P
        pos_weights = (n_samples_clean - n_samples_positive).astype('float32') / n_samples_positive.astype('float32')
        return pos_weights.astype('float32')
    

def unit_test(dataset):
    print(f"Dataset length: {len(dataset)}")
    for i in range(2):
        sample = dataset[i]
        x_lbr, x_welch = sample['x']
        label = sample['y']
        age = sample['age']
        print(f"Sample {i}: LaBraM shape: {x_lbr.shape}, Welch shape: {x_welch.shape}, Label: {label}, Age: {age}, Weight: {sample['sample_weight']}, ID: {sample['id']}, Labels Mask : {sample['labels_mask']}")
        # print datatype
        print(f"Data types - LaBraM: {x_lbr.dtype}, Welch: {x_welch.dtype}, Label: {label.dtype}, Age: {type(age)}, Weight: {type(sample['sample_weight'])}, ID: {type(sample['id'])}, Labels Mask: {type(sample['labels_mask'])}")
    #postive weights
    pos_weights = dataset.get_positive_weights()
    print(f"Positive weights for each class: {pos_weights}")

def test_compatibility_with_torchloader(dataset):
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=1)
    for batch in dataloader:
        x_lbr_batch, x_welch_batch = batch['x']
        y_batch = batch['y']
        print(f"Batch LaBraM shape: {x_lbr_batch.shape}, Welch shape: {x_welch_batch.shape}, Label shape: {y_batch.shape}")
        print(f"Mask shape: {batch['labels_mask'].shape}, Mask: {batch['labels_mask']}, Weights: {batch['sample_weight']}")
        print(f"Neurologist IDs: {batch['neurologist_id']}")
        print(f"Neurologist table size: {dataset.get_valide_neurologist_counts()}")
        print("-----------------")
        break


if __name__ == "__main__":
    dataset = EEGDatasetWithLabel(root=r"H:\\EEG_features\\EEG_features_labram_welch", metadata=r"H:\Thesis_Project\Essembles\checkpoints\train.csv", return_ids=True, return_ordinal=True, return_neurologist_ids=True)
    test_compatibility_with_torchloader(dataset)
    
    unit_test(dataset)
    try:
        dataset.get_by_id("cbcf2ad8-e1df-4275-857c-758705fa1d4c")
    except KeyError as e:
        print(e)
    ## expected output:
    ## Sample 0: LaBraM shape: (89, 200), Welch shape: (89, 20, 25), Label: [1. 1. 1. 1. 2.]
