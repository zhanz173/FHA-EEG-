from __future__ import annotations

import argparse
import os
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.signal as signal
import torch
from einops import rearrange


standard_1020 = [
    'FP1', 'FPZ', 'FP2',
    'AF9', 'AF7', 'AF5', 'AF3', 'AF1', 'AFZ', 'AF2', 'AF4', 'AF6', 'AF8', 'AF10',
    'F9', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'F10',
    'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10',
    'T9', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'T10',
    'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10',
    'P9', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'P10',
    'PO9', 'PO7', 'PO5', 'PO3', 'PO1', 'POZ', 'PO2', 'PO4', 'PO6', 'PO8', 'PO10',
    'O1', 'OZ', 'O2', 'O9', 'CB1', 'CB2',
    'IZ', 'O10', 'T3', 'T5', 'T4', 'T6', 'M1', 'M2', 'A1', 'A2',
    'CFC1', 'CFC2', 'CFC3', 'CFC4', 'CFC5', 'CFC6', 'CFC7', 'CFC8',
    'CCP1', 'CCP2', 'CCP3', 'CCP4', 'CCP5', 'CCP6', 'CCP7', 'CCP8',
    'T1', 'T2', 'FTT9h', 'TTP7h', 'TPP9h', 'FTT10h', 'TPP8h', 'TPP10h',
    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2'
]


@dataclass
class SegmentMeta:
    eeg_id: str
    segment_idx: int
    n_segments: int
    start_sample: int
    end_sample: int
    start_sec: float
    end_sec: float
    original_num_samples: int


@dataclass(frozen=True)
class SegmentConfig:
    duration_sec: int
    overlap_sec: int = 0
    target_fs: int = 200
    pad_last: bool = False

    @property
    def duration_samples(self) -> int:
        return self.duration_sec * self.target_fs

    @property
    def overlap_samples(self) -> int:
        return self.overlap_sec * self.target_fs

    def validate(self) -> None:
        validate_overlap(self.duration_samples, self.overlap_samples)


@dataclass(frozen=True)
class PreprocessingConfig:
    normalize_mode: str = 'zscore_per_channel'
    scale_factor: float = 1e5


@dataclass(frozen=True)
class ExtractionContext:
    channel_names: Sequence[str]
    segment_config: SegmentConfig


def get_input_chans(ch_names: Sequence[str]) -> List[int]:
    input_chans = [0]
    for ch_name in ch_names:
        input_chans.append(standard_1020.index(ch_name) + 1)
    return input_chans


def load_pretrained_labram(
    number_of_classes: int = 1,
    checkpoint_path: str = r"LaBraM\checkpoints\labram-base.pth",
    **kwargs,
):
    import LaBraM.utils as utils
    from timm.models import create_model
    from LaBraM.modeling_finetune import labram_base_patch200_200

    model = create_model(
        model_name='labram_base_patch200_200',
        init_values=0.1,
        num_classes=number_of_classes, # we only need the feature extractor part, so num_classes can be 1
        **kwargs,
    )

    checkpoint_model = None
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    for model_key in ['model', 'module']:
        if model_key in checkpoint:
            checkpoint_model = checkpoint[model_key]
            print(f'Load state_dict by model_key = {model_key}')
            break
    if checkpoint_model is None:
        checkpoint_model = checkpoint

    new_dict = OrderedDict()
    for key in checkpoint_model.keys():
        if key.startswith('student.'):
            new_dict[key[8:]] = checkpoint_model[key]
    checkpoint_model = new_dict

    state_dict = model.state_dict()
    for key in ['head.weight', 'head.bias']:
        if key in checkpoint_model and checkpoint_model[key].shape != state_dict[key].shape:
            print(f'Removing key {key} from pretrained checkpoint')
            del checkpoint_model[key]

    for key in list(checkpoint_model.keys()):
        if 'relative_position_index' in key:
            checkpoint_model.pop(key)

    utils.load_state_dict(model, checkpoint_model, prefix='')
    return model


class FeatureExtractor(ABC):
    name: str

    @abstractmethod
    def extract(self, chunks: np.ndarray, context: ExtractionContext) -> np.ndarray:
        raise NotImplementedError


class LabramFeatureExtractor(FeatureExtractor):
    name = 'LaBraM'

    def __init__(
        self,
        ch_names: Sequence[str],
        n_times: int,
        patch_size: int = 200,
        n_classes: int = 1,
        checkpoint_path: str = r"preprocess\LaBraM\checkpoints\labram-base.pth",
        device: Optional[str] = None,
        **kwargs,
    ):
        self.model = load_pretrained_labram(n_classes, checkpoint_path=checkpoint_path, **kwargs)
        self.n_times = n_times
        self.patch_size = patch_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        if self.n_times % self.patch_size != 0:
            raise ValueError(f'n_times={self.n_times} must be divisible by patch_size={self.patch_size}')

        self._set_channel_mapping([channel.upper() for channel in ch_names])
        self.to(self.device)
        self.eval()

    def _set_channel_mapping(self, ch_names: Sequence[str]) -> None:
        valid_channels, channel_idx_mask = self.filter_channels(ch_names)
        if not valid_channels:
            raise ValueError('No valid EEG channels matched standard_1020.')
        self.valid_channels = valid_channels
        self.channel_idx_mask = channel_idx_mask
        self.n_chans = len(valid_channels)
        self.ch_idx = get_input_chans(valid_channels)

    @staticmethod
    def filter_channels(ch_names: Sequence[str]) -> Tuple[List[str], List[int]]:
        channel_idx = []
        valid_channels = []
        for i, ch in enumerate(ch_names):
            if ch in standard_1020:
                channel_idx.append(i)
                valid_channels.append(ch)
            else:
                print(f'Channel {ch} is not in the list of valid channels and will be dropped.')
        return valid_channels, channel_idx

    def _segment_for_model(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, self.channel_idx_mask, :]
        if x.shape[-1] != self.n_times:
            raise ValueError(f'Expected segment length {self.n_times}, got {x.shape[-1]}')
        return x.reshape(
            x.shape[0],
            self.n_chans,
            self.n_times // self.patch_size,
            self.patch_size,
        )

    def extract(self, chunks: np.ndarray, context: ExtractionContext) -> np.ndarray:
        del context
        self.model.eval()
        x = torch.as_tensor(chunks, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            x = self._segment_for_model(x).contiguous()
            features = self.model.forward_features(x, input_chans=self.ch_idx)
        return features.detach().cpu().numpy().astype(np.float32)

    def to(self, device: str) -> None:
        self.device = device
        self.model.to(device)

    def eval(self) -> None:
        self.model.eval()


class WelchFeatureExtractor(FeatureExtractor):
    name = 'Welch'

    def __init__(
        self,
        nperseg: int = 200,
        noverlap: int = 100,
        max_freq: float = 50.0,
        detrend: str = 'constant',
        flatten: bool = True,
    ):
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.max_freq = max_freq 
        self.detrend = detrend
        self.flatten = flatten

    def extract(self, chunks: np.ndarray, context: ExtractionContext) -> np.ndarray:
        freqs, pxx = signal.welch(
            chunks,
            fs=context.segment_config.target_fs,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            detrend=self.detrend,
            axis=-1,
        )
        keep = freqs < self.max_freq
        features = pxx[..., keep].astype(np.float32)
        if self.flatten:
            features = features.reshape(features.shape[0], -1)
        return features


def build_labram_extractor(
    *,
    channel_names: Sequence[str],
    segment_config: SegmentConfig,
    checkpoint_path: str,
    device: str,
    patch_size: int,
    **_: Any,
) -> FeatureExtractor:
    return LabramFeatureExtractor(
        ch_names=channel_names,
        n_times=segment_config.duration_samples,
        patch_size=patch_size,
        n_classes=1,
        checkpoint_path=checkpoint_path,
        device=device,
    )


def build_welch_extractor(
    *,
    flatten_welch: bool,
    **_: Any,
) -> FeatureExtractor:
    return WelchFeatureExtractor(flatten=flatten_welch)


FEATURE_EXTRACTOR_REGISTRY = {
    'labram': build_labram_extractor,
    'welch': build_welch_extractor,
}


def validate_overlap(duration_samples: int, overlap_samples: int) -> None:
    if overlap_samples < 0:
        raise ValueError('overlap must be non-negative')
    if overlap_samples >= duration_samples:
        raise ValueError('overlap must be smaller than duration')


def to_chunk(
    duration_samples: int,
    data: np.ndarray,
    overlap_samples: int = 0,
    pad_last: bool = False,
) -> np.ndarray:
    validate_overlap(duration_samples, overlap_samples)

    n_channels, n_total = data.shape
    if n_total < duration_samples:
        if not pad_last:
            return np.zeros((0, n_channels, duration_samples), dtype=data.dtype)
        out = np.zeros((1, n_channels, duration_samples), dtype=data.dtype)
        out[0, :, :n_total] = data
        return out

    if overlap_samples == 0 and not pad_last:
        usable = (n_total // duration_samples) * duration_samples
        if usable == 0:
            return np.zeros((0, n_channels, duration_samples), dtype=data.dtype)
        return rearrange(data[:, :usable], 'C (N T) -> N C T', T=duration_samples)

    step = duration_samples - overlap_samples
    chunks = []
    for start in range(0, n_total - duration_samples + 1, step):
        chunks.append(data[:, start:start + duration_samples].copy())

    if pad_last:
        last_start = 0 if len(chunks) == 0 else len(chunks) * step
        if last_start < n_total:
            tail = np.zeros((n_channels, duration_samples), dtype=data.dtype)
            remain = data[:, last_start:]
            tail[:, :remain.shape[-1]] = remain
            chunks.append(tail)

    if len(chunks) == 0:
        return np.zeros((0, n_channels, duration_samples), dtype=data.dtype)
    return np.stack(chunks, axis=0)


def compute_segment_times(
    n_total_samples: int,
    n_segments: int,
    segment_config: SegmentConfig,
) -> List[SegmentMeta]:
    step = segment_config.duration_samples - segment_config.overlap_samples
    meta = []
    for seg_idx in range(n_segments):
        start_sample = seg_idx * step
        end_sample = min(start_sample + segment_config.duration_samples, n_total_samples)
        meta.append(
            SegmentMeta(
                eeg_id='',
                segment_idx=seg_idx,
                n_segments=n_segments,
                start_sample=start_sample,
                end_sample=end_sample,
                start_sec=start_sample / segment_config.target_fs,
                end_sec=end_sample / segment_config.target_fs,
                original_num_samples=n_total_samples,
            )
        )
    return meta


def normalize_eeg(data: np.ndarray, mode: str = 'zscore_per_channel') -> np.ndarray:
    if mode == 'none':
        return data
    if mode == 'zscore_per_channel':
        mean = data.mean(axis=-1, keepdims=True)
        std = data.std(axis=-1, keepdims=True)
        std = np.where(std < 1e-6, 1.0, std)
        return (data - mean) / std
    raise ValueError(f'Unknown normalization mode: {mode}')


class FeatureExtractionPipeline:
    def __init__(
        self,
        segment_config: SegmentConfig,
        preprocessing_config: PreprocessingConfig,
        extractors: Sequence[FeatureExtractor],
        channel_names: Sequence[str],
    ):
        if not extractors:
            raise ValueError('At least one feature extractor must be configured.')
        self.segment_config = segment_config
        self.preprocessing_config = preprocessing_config
        self.extractors = list(extractors)
        self.context = ExtractionContext(channel_names=channel_names, segment_config=segment_config)
        self.segment_config.validate()
        self._validate_extractor_names()

    @property
    def feature_names(self) -> List[str]:
        return [extractor.name for extractor in self.extractors]

    def _validate_extractor_names(self) -> None:
        names = self.feature_names
        if len(names) != len(set(names)):
            raise ValueError(f'Feature extractor names must be unique. Got: {names}')

    def prepare_chunks(self, data: np.ndarray) -> Tuple[np.ndarray, List[SegmentMeta]]:
        chunks = to_chunk(
            duration_samples=self.segment_config.duration_samples,
            data=data,
            overlap_samples=self.segment_config.overlap_samples,
            pad_last=self.segment_config.pad_last,
        )
        if chunks.shape[0] == 0:
            return chunks, []

        chunks = normalize_eeg(chunks, mode=self.preprocessing_config.normalize_mode).astype(np.float32)
        chunk_meta = compute_segment_times(
            n_total_samples=data.shape[-1],
            n_segments=chunks.shape[0],
            segment_config=self.segment_config,
        )
        return chunks, chunk_meta

    def extract(self, data: np.ndarray) -> Tuple[Dict[str, np.ndarray], List[SegmentMeta]]:
        chunks, chunk_meta = self.prepare_chunks(data)
        if chunks.shape[0] == 0:
            return {name: np.zeros((0, 0), dtype=np.float32) for name in self.feature_names}, []

        features = {}
        for extractor in self.extractors:
            features[extractor.name] = extractor.extract(chunks, self.context)
        return features, chunk_meta


class SegmentFeatureGenerator:
    """Yield buffered feature rows where each row is one fixed-length segment from any EEG."""

    def __init__(
        self,
        dataset,
        pipeline: FeatureExtractionPipeline,
        write_buffer_size: int = 256,
        skip_short_eeg: bool = False,
    ):
        self.dataset = dataset
        self.pipeline = pipeline
        self.write_buffer_size = write_buffer_size
        self.skip_short_eeg = skip_short_eeg

    def __iter__(self) -> Iterator[Tuple[Dict[str, np.ndarray], List[Dict[str, Any]]]]:
        batch_features: Dict[str, List[np.ndarray]] = {name: [] for name in self.pipeline.feature_names}
        batch_meta: List[Dict[str, Any]] = []
        min_samples = self.pipeline.segment_config.duration_samples

        for i in range(len(self.dataset)):
            item = self.dataset[i]
            data = np.asarray(item['EEG_Raw'], dtype=np.float32) * self.pipeline.preprocessing_config.scale_factor
            eeg_id = str(item['ScanID'])

            if self.skip_short_eeg and data.shape[-1] < min_samples:
                continue

            feature_map, meta_list = self.pipeline.extract(data)
            if not meta_list:
                continue

            n_segments = len(meta_list)
            for name, values in feature_map.items():
                if values.shape[0] != n_segments:
                    raise ValueError(f'Extractor {name} returned {values.shape[0]} rows for {n_segments} segments.')

            for seg_idx in range(n_segments):
                meta = meta_list[seg_idx]
                meta.eeg_id = eeg_id
                batch_meta.append(asdict(meta))
                for name in self.pipeline.feature_names:
                    batch_features[name].append(feature_map[name][seg_idx])

                if len(batch_meta) == self.write_buffer_size:
                    yield self._flush(batch_features, batch_meta)
                    batch_features = {name: [] for name in self.pipeline.feature_names}
                    batch_meta = []

        if batch_meta:
            yield self._flush(batch_features, batch_meta)

    def _flush(
        self,
        batch_features: Dict[str, List[np.ndarray]],
        batch_meta: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, np.ndarray], List[Dict[str, Any]]]:
        stacked_features = {
            name: np.stack(feature_rows).astype(np.float32)
            for name, feature_rows in batch_features.items()
        }
        return stacked_features, list(batch_meta)

    def __len__(self) -> int:
        return len(self.dataset)


def build_h5_shards_from_segment_batches(
    batch_iter: Iterable[Tuple[Dict[str, np.ndarray], List[Dict[str, Any]]]],
    out_dir: str = 'h5_shards',
    rows_per_shard: int = 20000,
    compression: str = 'lzf',
    compression_opts: Optional[int] = None,
    debug: bool = False,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    index_rows: List[Dict[str, Any]] = []

    shard_id = -1
    f = None
    datasets: Dict[str, Any] = {}
    write_pos = 0
    feature_shapes: Optional[Dict[str, Tuple[int, ...]]] = None
    current_shard_path: Optional[str] = None

    if debug:
        print(
            f'Starting to build HDF5 shards in {out_dir} with rows_per_shard={rows_per_shard}, '
            f'compression={compression}, compression_opts={compression_opts}'
        )

    def new_shard(shapes: Dict[str, Tuple[int, ...]]) -> str:
        nonlocal shard_id, f, datasets, write_pos, current_shard_path
        if f is not None:
            f.flush()
            f.close()

        shard_id += 1
        current_shard_path = os.path.join(out_dir, f'shard_{shard_id:05d}.h5')
        f = h5py.File(current_shard_path, 'w')
        datasets = {}

        kwargs = dict(compression=compression)
        if compression == 'gzip' and compression_opts is not None:
            kwargs['compression_opts'] = compression_opts

        for feature_name, shape in shapes.items():
            chunk_shape = (min(256, rows_per_shard), *shape)
            datasets[feature_name] = f.create_dataset(
                feature_name,
                shape=(rows_per_shard, *shape),
                dtype='float32',
                chunks=chunk_shape,
                **kwargs,
            )
            f.attrs[f'{feature_name}_shape_per_row'] = shape

        f.attrs['rows_per_shard'] = rows_per_shard
        f.attrs['feature_names'] = np.asarray(list(shapes.keys()), dtype='S')
        write_pos = 0
        return current_shard_path

    for feature_map, meta in batch_iter:
        if not feature_map:
            continue

        feature_map = {name: np.asarray(values, dtype=np.float32) for name, values in feature_map.items()}
        row_count = next(iter(feature_map.values())).shape[0]

        for name, values in feature_map.items():
            if values.shape[0] != row_count:
                raise ValueError(f'Batch row count mismatch for extractor {name}.')
        if row_count != len(meta):
            raise ValueError('Batch sizes do not match between feature arrays and metadata.')

        if feature_shapes is None:
            feature_shapes = {name: tuple(values.shape[1:]) for name, values in feature_map.items()}
            new_shard(feature_shapes)

        start = 0
        while start < row_count:
            space = rows_per_shard - write_pos
            if space <= 0:
                new_shard(feature_shapes)
                space = rows_per_shard

            take = min(space, row_count - start)
            shard_slice = slice(write_pos, write_pos + take)
            batch_slice = slice(start, start + take)

            for name, values in feature_map.items():
                datasets[name][shard_slice] = values[batch_slice]

            for i in range(take):
                row_meta = dict(meta[start + i])
                row_meta['shard'] = os.path.basename(current_shard_path)
                row_meta['row'] = int(write_pos + i)
                index_rows.append(row_meta)

            write_pos += take
            start += take
        if debug:
            # break after first shard for quick debugging
            if shard_id > 0:
                print(f'Created shard {current_shard_path} with {write_pos} rows.')
                break


    if f is not None:
        f.attrs['last_write_pos'] = write_pos
        f.flush()
        f.close()

    if index_rows:
        pq.write_table(pa.Table.from_pylist(index_rows), os.path.join(out_dir, 'index.parquet'))
    else:
        print('No rows written. Check segment duration, overlap, and input data lengths.')


def build_feature_extractors(
    extractor_names: Sequence[str],
    channel_names: Sequence[str],
    segment_config: SegmentConfig,
    checkpoint_path: str,
    device: str,
    patch_size: int,
    flatten_welch: bool,
) -> List[FeatureExtractor]:
    extractors: List[FeatureExtractor] = []
    for extractor_name in extractor_names:
        normalized_name = extractor_name.lower()
        extractor_builder = FEATURE_EXTRACTOR_REGISTRY.get(normalized_name)
        if extractor_builder is None:
            raise ValueError(
                f'Unknown extractor "{extractor_name}". Supported extractors: {sorted(FEATURE_EXTRACTOR_REGISTRY)}.'
            )
        extractors.append(
            extractor_builder(
                channel_names=channel_names,
                segment_config=segment_config,
                checkpoint_path=checkpoint_path,
                device=device,
                patch_size=patch_size,
                flatten_welch=flatten_welch,
            )
        )
    return extractors


if __name__ == '__main__':
    from EEG_loader import FHA_EEG_channels_ORDER, FHA_EEG_Loader

    default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {default_device} for feature extraction')

    patch_size = 200
    LABRAM_PRETRAINED_FS = 200
    overlap = 0
    segment_duration = 4
    eeg_feature_save_dir = r"EEG_features\EEG_features_labram_welch_HV'
    eeg_data_dir = r'EEG\FHA\hyperventilation'
    checkpoint_path = r'LaBraM\checkpoints\labram-base.pth'

    parser = argparse.ArgumentParser(description='Extract segment-level EEG features and save as HDF5 shards')
    parser.add_argument('--data_dir', type=str, default=eeg_data_dir)
    parser.add_argument('--output_dir', type=str, default=eeg_feature_save_dir)
    parser.add_argument('--duration', type=int, default=segment_duration, help='Segment duration in seconds')
    parser.add_argument('--overlap', type=int, default=overlap, help='Segment overlap in seconds')
    parser.add_argument( '--write_buffer_size', type=int, default=256, help='Number of segment rows buffered before writing to disk')
    parser.add_argument('--rows_per_shard', type=int, default=256 * 100)
    parser.add_argument('--device', type=str, default=default_device)
    parser.add_argument('--pad_last', action='store_true', help='Pad the last short segment instead of dropping it')
    parser.add_argument('--scale_factor', type=float, default=1e5, help='Multiply raw EEG by this factor before extraction')
    parser.add_argument('--normalize_mode', type=str, default='zscore_per_channel', choices=['none', 'zscore_per_channel'])
    parser.add_argument('--checkpoint_path', type=str, default=checkpoint_path)
    parser.add_argument('--compression', type=str, default='lzf', choices=['lzf', 'gzip'])
    parser.add_argument('--compression_opts', type=int, default=4, help='Compression level for gzip (1-9)')
    parser.add_argument('--extractors', nargs='+', default=['labram', 'welch'], choices=['labram', 'welch'])
    parser.add_argument('--skip_short_eeg', action='store_true', help='Skip EEGs shorter than one segment')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    dataloader_config = {
        'dataset_root': args.data_dir,
        'tmax': 360,
        'sampling_rate': LABRAM_PRETRAINED_FS, # fixed for LaBraM, will be resampled in loader if needed
    }
    dataset = FHA_EEG_Loader(dataloader_config)
    print(f'Dataset loaded with {len(dataset)} samples. Example keys: {list(dataset[0].keys())}')

    segment_config = SegmentConfig(
        duration_sec=args.duration,
        overlap_sec=args.overlap,
        target_fs=LABRAM_PRETRAINED_FS, 
        pad_last=args.pad_last,
    )
    preprocessing_config = PreprocessingConfig(
        normalize_mode=args.normalize_mode,
        scale_factor=args.scale_factor,
    )
    extractors = build_feature_extractors(
        extractor_names=args.extractors,
        channel_names=FHA_EEG_channels_ORDER,
        segment_config=segment_config,
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        patch_size=patch_size,
        flatten_welch=True,
    )
    pipeline = FeatureExtractionPipeline(
        segment_config=segment_config,
        preprocessing_config=preprocessing_config,
        extractors=extractors,
        channel_names=FHA_EEG_channels_ORDER,
    )

    gen = SegmentFeatureGenerator(
        dataset=dataset,
        pipeline=pipeline,
        write_buffer_size=args.write_buffer_size,
        skip_short_eeg=args.skip_short_eeg,
    )

    if args.debug:
        print('Debug mode enabled: only processing one batch of segments.')
        
    build_h5_shards_from_segment_batches(
        gen,
        out_dir=args.output_dir,
        rows_per_shard=args.rows_per_shard,
        compression=args.compression,
        compression_opts=args.compression_opts,
        debug=args.debug,
    )
