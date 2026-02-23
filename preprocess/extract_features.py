from einops import rearrange
import torch 
import torch.nn as nn
import scipy.signal as signal
import numpy as np
import os
import h5py

import pyarrow as pa
import pyarrow.parquet as pq

from LaBraM.modeling_finetune import labram_base_patch200_200


standard_1020 = [
    'FP1', 'FPZ', 'FP2', 
    'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFZ', 'AF2', 'AF4', 'AF6', 'AF8', 'AF10', \
    'F9', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'F10', \
    'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', \
    'T9', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'T10', \
    'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', \
    'P9', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'P10', \
    'PO9', 'PO7', 'PO5', 'PO3', 'PO1', 'POZ', 'PO2', 'PO4', 'PO6', 'PO8', 'PO10', \
    'O1', 'OZ', 'O2', 'O9', 'CB1', 'CB2', \
    'IZ', 'O10', 'T3', 'T5', 'T4', 'T6', 'M1', 'M2', 'A1', 'A2', \
    'CFC1', 'CFC2', 'CFC3', 'CFC4', 'CFC5', 'CFC6', 'CFC7', 'CFC8', \
    'CCP1', 'CCP2', 'CCP3', 'CCP4', 'CCP5', 'CCP6', 'CCP7', 'CCP8', \
    'T1', 'T2', 'FTT9h', 'TTP7h', 'TPP9h', 'FTT10h', 'TPP8h', 'TPP10h', \
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F8", "F8-T8", "T8-P8", "P8-O2", "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2"
]


def get_input_chans(ch_names):
    input_chans = [0] # for cls token
    for ch_name in ch_names:
        input_chans.append(standard_1020.index(ch_name) + 1)
    return input_chans

def loadPretrainedLabram(number_of_classes = 1, **kwargs):
    from collections import OrderedDict
    import LaBraM.utils as utils
    from timm.models import create_model

    model = create_model(
        model_name = 'labram_base_patch200_200',
        init_values=0.1,
        num_classes = number_of_classes,
        **kwargs
    )
    checkpoint_model = None
    checkpoint = torch.load(r"preprocess\LaBraM\checkpoints\labram-base.pth", map_location='cpu', weights_only=False)

    for model_key in ['model','module']:
        if model_key in checkpoint:
            checkpoint_model = checkpoint[model_key]
            print("Load state_dict by model_key = %s" % model_key)
            break
    if checkpoint_model is None:
        checkpoint_model = checkpoint
    if (checkpoint_model is not None):
        all_keys = list(checkpoint_model.keys())
        new_dict = OrderedDict()
        for key in all_keys:
            if key.startswith('student.'):
                new_dict[key[8:]] = checkpoint_model[key]
            else:
                pass
        checkpoint_model = new_dict

    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    all_keys = list(checkpoint_model.keys())
    for key in all_keys:
        if "relative_position_index" in key:
            checkpoint_model.pop(key)
    utils.load_state_dict(model, checkpoint_model, prefix="")
    return model

class model_wrapper():
    def __init__(self, ch_names=[], n_times=4, patch_size=200,  n_classes=1, **kwargs):
        self.model = loadPretrainedLabram(n_classes, **kwargs)
        self.n_times = n_times
        self.patch_size = patch_size

        self.channel2index([channel.upper() for channel in ch_names])
    
    def channel2index(self, ch_names:list[str]):
        self.valid_channels, self.channel_idx_mask = self.filter_channels(ch_names)
        self.n_chans = len(self.valid_channels)
        self.ch_idx  = get_input_chans(self.valid_channels)
  
    def filter_channels(self, ch_names:list[str]):
        ''' remove channels that are not in the standard 10-20 system,
          and return the indices of the valid channels in the input list, as well as the names of the valid channels
        '''
        channel_idx = []
        valid_channels = []
        for i, ch in enumerate(ch_names):
            if ch in standard_1020:
                channel_idx.append(i)
                valid_channels.append(ch)
            else:
                print(f"Channel {ch} is not in the list of valid channels")
        return valid_channels, channel_idx

    def forward_features(self, x):
        self.model.eval()
        with torch.no_grad():
            x = self.segment(x, batch_size=x.shape[0]).contiguous()
            features = self.model.forward_features(x, input_chans=self.ch_idx)
        return features
    
    def segment(self, x, batch_size):
        x = x[:, self.channel_idx_mask,:] # select valid channels, shape (B, C_valid, T)
        x = x.view(
            batch_size,
            self.n_chans,
            self.n_times // self.patch_size,
            self.patch_size,
        )
        return x

    def to(self, device):
        self.model.to(device)

    def eval(self):
        self.model.eval()
    
def to_chunk(duration, data, overlap=0):
    if overlap == 0:
        ## simple rearrange without copy 20 * NT -> 20 * (N T)
        return rearrange(data[..., :data.shape[-1]//duration *duration], 'C (N T) -> N C T', T=duration)
    chunk = []
    for i in range(0, data.shape[-1] - duration + 1, duration - overlap):
        chunk.append(data[...,i:i+duration].copy())
    return np.stack(chunk, axis=0)

def run_PSD(data, fs):
    f, Pxx = signal.welch(data, fs=fs, nperseg=200, noverlap=100, detrend='constant', axis=-1)
    return f[...,:50], Pxx[...,:50] # batch, channels, freq (0-50Hz) at 1Hz interval

def run_labram(data,model,device='cuda'):
    model.to(device)
    model.eval()
    data = torch.tensor(data, dtype=torch.float32).to(device)
    with torch.no_grad():
        features = model.forward_features(data)
    return features.detach().cpu().numpy()

def run(data, model, duration, overlap, TARGET_FS=200 ):
    data = to_chunk(duration*TARGET_FS, data, overlap*TARGET_FS) # chunk the data into segments of length duration with overlap, shape (N, C, T), N is used as batch dimension for model inference
    f, PSD = run_PSD(data, TARGET_FS)
    LabraM_features = run_labram(data, model)
    N, D = LabraM_features.shape
    return LabraM_features, PSD.reshape(N, -1)

class Feature_generator:
    def __init__(self, dataset, model, duration=8, overlap=4, batch_size=32):
        self.dataset = dataset
        self.model = model
        self.duration = duration
        self.overlap = overlap
        self.batch_size = batch_size
    
    def __iter__(self):
        # return a generator that yields (lbr, welch, ids)
        # lbr: (B, T1, F1), welch: (B, T2, F2), ids: list of length B
        batch_lbr = []
        batch_welch = []
        batch_ids = []
        for i in range(len(self.dataset)):
            data = self.dataset[i]['EEG_Raw'] * 1e5
            eeg_id = self.dataset[i]['ScanID']
            features, psd = run(data, self.model, self.duration, self.overlap)
            batch_lbr.append(features.astype(np.float32))
            batch_welch.append(psd.astype(np.float32))
            batch_ids.append(eeg_id)
            if len(batch_ids) == self.batch_size:
                yield (np.stack(batch_lbr), np.stack(batch_welch), batch_ids)
                batch_lbr = []
                batch_welch = []
                batch_ids = []

        if len(batch_ids) > 0:
            yield (np.stack(batch_lbr), np.stack(batch_welch), batch_ids)
    def __len__(self):
        return len(self.dataset)

def build_h5_shards_from_batches(
    batch_iter,                   # yields (lbr, welch, ids)
    out_dir="h5_shards",
    rows_per_shard=20000,         # pick a multiple of your batch size
    compression="lzf",            # "lzf" (fast) or "gzip"
    compression_opts=None,        # e.g., 4 if you use gzip
    id_dtype=None,                 # leave None to use UTF-8 strings
    debug=False
):
    """
    batch_iter must yield tuples: (LaBraM, Welch, EEG_ID)
      - LaBraM: np.array or torch.Tensor, shape [B, T1, F1]
      - Welch : np.array or torch.Tensor, shape [B, T2, F2]
      - EEG_ID: list/array of length B (str-like)
    Creates sharded HDF5 files with datasets:
      - "LaBraM": (rows_per_shard, T1, F1) float32
      - "Welch" : (rows_per_shard, T2, F2) float32
      - "ids"   : (rows_per_shard,) string
    And an index parquet: columns ["eeg_id", "shard", "row"]
    """

    os.makedirs(out_dir, exist_ok=True)
    index_rows = []

    shard_id = -1
    f = d_lbr = d_welch = d_ids = None
    write_pos = 0
    first = True
    T1 = F1 = T2 = F2 = None
    current_shard_path = None

    def _np(a):
        # accepts torch or numpy; casts to np.float32 for feature arrays
        if hasattr(a, "detach"):   # torch tensor
            a = a.detach().cpu().numpy()
        return a

    def new_shard(T1_, F1_, T2_, F2_, verbose=False):
        if verbose:
            print(f"Creating new shard {shard_id+1} at {current_shard_path}, with shapes LaBraM: ({rows_per_shard}, {T1_}, {F1_}), Welch: ({rows_per_shard}, {T2_}, {F2_})")
        nonlocal shard_id, f, d_lbr, d_welch, d_ids, write_pos
        if f is not None:
            f.flush()
            f.close()
        shard_id += 1
        path = os.path.join(out_dir, f"shard_{shard_id:05d}.h5")
        f = h5py.File(path, "w")

        str_dt = id_dtype or h5py.string_dtype(encoding="utf-8")

        # Chunking: rows × time × feat — row-major chunks for fast row reads
        # Adjust T chunk if very long sequences to keep chunk sizes reasonable.
        t1_chunk = min(T1_, 256)
        t2_chunk = min(T2_, 256)

        kwargs = dict(compression=compression)
        if compression == "gzip" and compression_opts is not None:
            kwargs["compression_opts"] = compression_opts

        d_lbr = f.create_dataset(
            "LaBraM",
            shape=(rows_per_shard, T1_, F1_),
            dtype="float32",
            chunks=(min(64, rows_per_shard), t1_chunk, F1_),
            **kwargs
        )
        d_welch = f.create_dataset(
            "Welch",
            shape=(rows_per_shard, T2_, F2_),
            dtype="float32",
            chunks=(min(64, rows_per_shard), t2_chunk, F2_),
            **kwargs
        )
        d_ids = f.create_dataset(
            "ids",
            shape=(rows_per_shard,),
            dtype=str_dt,
            **kwargs
        )

        # Write shapes as file attrs for reference
        f.attrs["LaBraM_shape"] = (rows_per_shard, T1_, F1_)
        f.attrs["Welch_shape"]  = (rows_per_shard, T2_, F2_)

        write_pos = 0
        return path

    for lbr, welch, ids in batch_iter:
        # Convert to numpy
        lbr  = _np(lbr)     # (B, T1, F1)
        welch = _np(welch)  # (B, T2, F2)
        ids  = np.asarray(list(map(str, ids)))  # ensure array[str]

        if first:
            assert lbr.ndim == 3 and welch.ndim == 3, f"Expected 3D features; got {lbr.shape}, {welch.shape}"
            T1, F1 = lbr.shape[1], lbr.shape[2]
            T2, F2 = welch.shape[1], welch.shape[2]
            current_shard_path = new_shard(T1, F1, T2, F2)
            first = False

        B = lbr.shape[0]
        assert welch.shape[0] == B and ids.shape[0] == B, "Batch sizes must match for LaBraM/Welch/ids"

        start = 0
        while start < B:
            space = rows_per_shard - write_pos
            if space == 0:
                current_shard_path = new_shard(T1, F1, T2, F2)
                space = rows_per_shard
            take = min(space, B - start)

            sl = slice(write_pos, write_pos + take)
            bl = slice(start, start + take)

            d_lbr[sl, :, :] = lbr[bl].astype(np.float32, copy=False)
            d_welch[sl, :, :] = welch[bl].astype(np.float32, copy=False)
            d_ids[sl] = ids[bl]

            # index rows
            for i in range(take):
                index_rows.append({
                    "eeg_id": ids[start + i],
                    "shard": os.path.basename(current_shard_path),
                    "row": write_pos + i
                })

            write_pos += take
            start += take

        if debug:
            print(f"Processed batch of size {B}; wrote up to row {write_pos} in shard {current_shard_path}")
            if shard_id >= 1:
                break

    if f is not None:
        f.flush()
        f.close()

    if index_rows:
        pq.write_table(pa.Table.from_pylist(index_rows), os.path.join(out_dir, "index.parquet"))


if __name__ == "__main__":
    from EEG_loader import FHA_EEG_channels_ORDER, FHA_EEG_Loader
    import torch
    import argparse
    B = 32 # batch size,estimate 7MB per batch, you don't need to change this
    config={'channels': 20, 'sampling_rate': 200, 'window_size': 1e9, 'access_pattern': 'sequential','label': ['ScanID'], 'max_length': 360*256} # you don't need to change this config, just make sure the paths below are correct
    
    ## check if GPU is available for feature extraction, if not use CPU (will be slower)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device} for feature extraction")

    ## default parameters for feature extraction, you can adjust duration and overlap to get different segment lengths and overlapping segments
    patch_size = 200 # 1 second at 200Hz, is fixed during model pretraining
    overlap = 0 # in seconds, adjust if you want overlapping segments (e.g. 2s segments with 1s overlap -> duration=2, overlap=1)
    segment_duration = 4 # in seconds, adjust if you want longer segments (e.g. 4s segments with 0s overlap -> duration=4, overlap=0)
    EEG_feature_save_dir = r"H:\EEG_features\EEG_features_labram_welch_4s" # output directory for h5 shards and index parquet
    EEG_Data_dir = r"H:\EEG\FHA\Resting\chunks" # directory containing EEG data files
    EEG_Meta_dir = r"H:\EEG\Annotations\FHA_annotation_full.csv" # path to CSV file containing metadata and labels

    ## Command line arguments can be used to override the above parameters
    parser = argparse.ArgumentParser(description="Extract features from EEG data and save as HDF5 shards")
    parser.add_argument("--data_dir", type=str, default=EEG_Data_dir, help="Directory containing EEG data files")
    parser.add_argument("--meta_csv", type=str, default=EEG_Meta_dir, help="Path to CSV file containing metadata and labels")
    parser.add_argument("--output_dir", type=str, default=EEG_feature_save_dir, help="Output directory for HDF5 shards and index parquet")
    parser.add_argument("--duration", type=int, default=segment_duration, help="Segment duration in seconds")
    parser.add_argument("--overlap", type=int, default=overlap, help="Segment overlap in seconds")
    args = parser.parse_args()

    EEG_Data_dir = args.data_dir
    EEG_Meta_dir = args.meta_csv
    EEG_feature_save_dir = args.output_dir
    segment_duration = args.duration
    overlap = args.overlap

    ## Load dataset and initialize model
    EEG_DATA = FHA_EEG_Loader(EEG_Meta_dir, EEG_Data_dir, config)
    print(f"Dataset loaded with {len(EEG_DATA)} samples. Example keys: {list(EEG_DATA[0].keys())}")

    model = model_wrapper(ch_names=FHA_EEG_channels_ORDER, n_times=segment_duration*200, patch_size=patch_size,  n_classes=1)
    gen = Feature_generator(EEG_DATA, model, duration=segment_duration, overlap=overlap, batch_size=B)

    ## run feature extraction and save to HDF5 shards with index parquet
    build_h5_shards_from_batches(
        gen,
        EEG_feature_save_dir,
        rows_per_shard=B * 100, # 100*B samples per shard, 700MB per shard
        compression="lzf",
        debug=False)