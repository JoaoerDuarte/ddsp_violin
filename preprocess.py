import os
import warnings
import pathlib
import numpy as np
import yaml
import librosa as li
import torch
from tqdm import tqdm
from os import makedirs, path
from effortless_config import Config
from ddsp_torch.core import extract_loudness, extract_pitch

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['METAL_DEVICE_WRITABLE_SHARED'] = '0'
warnings.filterwarnings('ignore', message='.*divide by zero.*')


def get_files(data_location, extension, **kwargs):
    print(f"\nSearching for files in: {data_location}")
    print(f"File extension: {extension}")
    files = sorted(list(pathlib.Path(data_location).rglob(f"*.{extension}")))
    print("\nFiles found in order of processing:")
    print("==================================")
    for i, file_path in enumerate(files, 1):
        print(f"{i}. {file_path}")
    print(f"\nTotal files found: {len(files)}")
    print("==================================\n")
    return files


def _nan_inf_safe(x, fill=0.0):
    if x is None:
        return None
    return np.nan_to_num(np.asarray(x), nan=fill, posinf=fill, neginf=fill)


def _align_to_length(x, required):
    x = np.asarray(x).reshape(-1)
    cur = x.shape[0]
    if cur == required:
        return x.astype(np.float32, copy=False)
    if cur > required:
        excess = cur - required
        if excess == 1:
            x = x[:-1]
        else:
            a = excess // 2
            b = excess - a
            x = x[a:cur - b]
    else:
        if cur == 0:
            x = np.zeros(required, dtype=np.float32)
        else:
            x = np.pad(x, (0, required - cur), mode='edge')
    return x.astype(np.float32, copy=False)


def preprocess_audio_file(file_path, sampling_rate, block_size, signal_length,
                          oneshot, crepe_model_path="", **kwargs):
    audio, _ = li.load(file_path, sr=sampling_rate)

    pad = (signal_length - len(audio) % signal_length) % signal_length
    if pad:
        audio = np.pad(audio, (0, pad))

    if oneshot:
        audio = audio[..., :signal_length]

    pitch = extract_pitch(audio, sampling_rate, block_size, crepe_model_path)
    loudness = extract_loudness(audio, sampling_rate, block_size)

    pitch = _nan_inf_safe(pitch, 0.0)
    loudness = np.nan_to_num(np.asarray(loudness), nan=-120.0, posinf=-120.0, neginf=-120.0)

    n_segments = audio.shape[0] // signal_length
    frames_per_segment = signal_length // block_size
    required_total = n_segments * frames_per_segment

    pitch = _align_to_length(pitch, required_total)
    loudness = _align_to_length(loudness, required_total)

    audio_segments = audio.reshape(n_segments, signal_length).astype(np.float32, copy=False)
    pitch_segments = pitch.reshape(n_segments, frames_per_segment)
    loudness_segments = loudness.reshape(n_segments, frames_per_segment)

    return audio_segments, pitch_segments, loudness_segments


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        super().__init__()
        self.signals = np.load(path.join(data_dir, "signals.npy"))
        self.pitches = np.load(path.join(data_dir, "pitches.npy"))
        self.loudness = np.load(path.join(data_dir, "loudness.npy"))

    def __len__(self):
        return self.signals.shape[0]

    def __getitem__(self, idx):
        signal = torch.from_numpy(self.signals[idx])
        pitch = torch.from_numpy(self.pitches[idx])
        loud = torch.from_numpy(self.loudness[idx])
        return signal, pitch, loud


def load_config(config_arg):
    if config_arg is None:
        config_path = "config.yaml"
    else:
        config_path = f"configs/{config_arg}.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def process_all_files(files, preprocess_config):
    progress_bar = tqdm(files)
    all_signals, all_pitches, all_loudness = [], [], []
    total_segments = 0

    for file_path in progress_bar:
        progress_bar.set_description(str(file_path))
        try:
            signals, pitches, loudness = preprocess_audio_file(file_path, **preprocess_config)
        except Exception as e:
            print(f"\nError processing {file_path}: {e}")
            continue
        all_signals.append(signals)
        all_pitches.append(pitches)
        all_loudness.append(loudness)
        total_segments += len(signals)

    return all_signals, all_pitches, all_loudness, total_segments


def save_preprocessed_data(signals_list, pitches_list, loudness_list, output_dir):
    signals = np.concatenate(signals_list, 0).astype(np.float32)
    pitches = np.concatenate(pitches_list, 0).astype(np.float32)
    loudness = np.concatenate(loudness_list, 0).astype(np.float32)

    makedirs(output_dir, exist_ok=True)
    np.save(path.join(output_dir, "signals.npy"), signals)
    np.save(path.join(output_dir, "pitches.npy"), pitches)
    np.save(path.join(output_dir, "loudness.npy"), loudness)

    print(f"\nSaved {len(signals)} segments to {output_dir}")
    print(f"Shapes — pitch {pitches.shape}, loudness {loudness.shape}, signals {signals.shape}")

    return len(signals)


def main():
    print("\n=== Starting Preprocessing ===")

    class args(Config):
        CONFIG = None

    args.parse_args()
    config = load_config(args.CONFIG)

    files = get_files(**config["data"])

    signals_list, pitches_list, loudness_list, total_segments = process_all_files(
        files, config["preprocess"]
    )

    final_segment_count = save_preprocessed_data(
        signals_list, pitches_list, loudness_list, config["preprocess"]["out_dir"]
    )

    print(f"\nPreprocessing Complete. {final_segment_count} segments saved to {config['preprocess']['out_dir']}")


if __name__ == "__main__":
    main()
