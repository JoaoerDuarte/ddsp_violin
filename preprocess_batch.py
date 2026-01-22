import os
import glob
import numpy as np
import librosa as li
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in log10")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in log10")

# Ensure this import works by running the script from the repo root
from ddsp_torch.core import extract_loudness, extract_pitch


def get_audio_files(directory):
    exts = ['*.wav', '*.mp3', '*.flac', '*.ogg']
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(directory, '**', ext), recursive=True))
    return sorted(files)


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


def preprocess_audio_file(file_path, sampling_rate=16000, block_size=64,
                          signal_length=64000, crepe_model_path=""):
    try:
        audio, _ = li.load(file_path, sr=sampling_rate)

        pad = (signal_length - len(audio) % signal_length) % signal_length
        if pad:
            audio = np.pad(audio, (0, pad))

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

    except Exception as e:
        print(f"\nError processing {file_path}: {e}")
        return None, None, None


def preprocess_folder(folder_path, output_dir, sampling_rate=16000,
                      block_size=64, signal_length=64000, crepe_model_path="",
                      print_shapes=True):
    files = get_audio_files(folder_path)
    if not files:
        print(f"WARNING: No audio files found in {folder_path}")
        return False

    print(f"Found {len(files)} audio files")

    sigs, pchs, lous = [], [], []

    for file_path in tqdm(files, desc="Processing files"):
        s, p, l = preprocess_audio_file(
            file_path, sampling_rate, block_size, signal_length, crepe_model_path
        )
        if s is not None:
            sigs.append(s)
            pchs.append(p)
            lous.append(l)

    if not sigs:
        print(f"ERROR: No valid audio processed in {folder_path}")
        return False

    signals = np.concatenate(sigs, 0).astype(np.float32)
    pitches = np.concatenate(pchs, 0).astype(np.float32)
    loudness = np.concatenate(lous, 0).astype(np.float32)

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "signals.npy"), signals)
    np.save(os.path.join(output_dir, "pitches.npy"), pitches)
    np.save(os.path.join(output_dir, "loudness.npy"), loudness)

    if print_shapes:
        print(f"Saved {len(signals)} segments to {output_dir}")
        print(f"Shapes — pitch {pitches.shape}, loudness {loudness.shape}, signals {signals.shape}")
    return True


def merge_string_datasets(base_output_dir, strings=['A', 'D', 'E', 'G']):
    print("\nMerging per-string datasets into full dataset...")

    sigs, pchs, lous = [], [], []

    for s in strings:
        d = os.path.join(base_output_dir, s)
        if not os.path.exists(d):
            print(f"WARNING: String folder {d} not found, skipping")
            continue

        sp = os.path.join(d, "signals.npy")
        pp = os.path.join(d, "pitches.npy")
        lp = os.path.join(d, "loudness.npy")
        if not all(os.path.exists(p) for p in [sp, pp, lp]):
            print(f"WARNING: Missing .npy files in {d}, skipping")
            continue

        sigs.append(np.load(sp))
        pchs.append(np.load(pp))
        lous.append(np.load(lp))

    if not sigs:
        print("ERROR: No valid string data found for merging")
        return False

    ms = np.concatenate(sigs, 0).astype(np.float32)
    mp = np.concatenate(pchs, 0).astype(np.float32)
    ml = np.concatenate(lous, 0).astype(np.float32)

    full = os.path.join(base_output_dir, "full")
    os.makedirs(full, exist_ok=True)
    np.save(os.path.join(full, "signals.npy"), ms)
    np.save(os.path.join(full, "pitches.npy"), mp)
    np.save(os.path.join(full, "loudness.npy"), ml)

    print(f"Merged dataset saved to {full}")
    print(f"Shapes — pitch {mp.shape}, loudness {ml.shape}, signals {ms.shape}")
    return True


def preprocess_physical_model_dataset(root_variation_path, preprocessed_root, strings=['A', 'D', 'E', 'G'], **preprocess_params):
    print(f"\n{'='*60}")
    print(f"Preprocessing: {root_variation_path}")
    print(f"Target root: {preprocessed_root}")
    print(f"{'='*60}")

    if not os.path.exists(root_variation_path):
        print(f"ERROR: Dataset path does not exist: {root_variation_path}")
        return False

    # Mirror dataset structure under preprocessed/
    rel = os.path.relpath(root_variation_path, start="dataset")
    out_root = os.path.join(preprocessed_root, rel)

    for s in strings:
        s_path = os.path.join(root_variation_path, s)
        if not os.path.exists(s_path):
            print(f"\nWARNING: String folder {s} not found in {root_variation_path}, skipping")
            continue
        print(f"\n--- Processing String {s} ---")
        s_out = os.path.join(out_root, s)
        preprocess_folder(s_path, s_out, print_shapes=False, **preprocess_params)

    merge_string_datasets(out_root, strings)
    return True


def preprocess_recordings_dataset(recordings_path, preprocessed_root, **preprocess_params):
    print(f"\n{'='*60}")
    print(f"Preprocessing: {recordings_path}")
    print(f"Target root: {preprocessed_root}")
    print(f"{'='*60}")

    if not os.path.exists(recordings_path):
        print(f"ERROR: Dataset path does not exist: {recordings_path}")
        return False

    # Mirror dataset structure under preprocessed/ (no "full" for recordings)
    rel = os.path.relpath(recordings_path, start="dataset")
    out_dir = os.path.join(preprocessed_root, rel)

    preprocess_folder(recordings_path, out_dir, print_shapes=True, **preprocess_params)
    return True


def main():
    PREPROCESS_PARAMS = {
        'sampling_rate': 16000,
        'block_size': 64,
        'signal_length': 64000,
        'crepe_model_path': ""
    }

    PREPROCESSED_ROOT = "preprocessed"

    print("="*60)
    print("DDSP Dataset Preprocessing (Dynamic Discovery)")
    print("="*60)

    # 1) DYNAMIC DISCOVERY: RECORDINGS
    # Scans dataset/recordings/*
    rec_root = os.path.join("dataset", "recordings")
    if os.path.exists(rec_root):
        print("\n\n### SCANNING RECORDINGS ###\n")
        for rec_name in sorted(os.listdir(rec_root)):
            rec_path = os.path.join(rec_root, rec_name)
            if not os.path.isdir(rec_path): continue
            
            # Check if done
            rel = os.path.relpath(rec_path, start="dataset")
            out_check = os.path.join(PREPROCESSED_ROOT, rel, "signals.npy")
            
            if os.path.exists(out_check):
                print(f"Skipping {rec_name} (already exists)")
                continue

            preprocess_recordings_dataset(rec_path, PREPROCESSED_ROOT, **PREPROCESS_PARAMS)

    # 2) DYNAMIC DISCOVERY: PHYSICAL MODEL VARIATIONS
    # Scans dataset/physical_model/{version}/{variation}
    pm_root = os.path.join("dataset", "physical_model")
    if os.path.exists(pm_root):
        print("\n\n### SCANNING PHYSICAL MODELS ###\n")
        # Iterate versions (e.g. 'new', 'old')
        for version in sorted(os.listdir(pm_root)):
            ver_path = os.path.join(pm_root, version)
            if not os.path.isdir(ver_path): continue

            # Iterate variations (e.g. 'convolved_Bernardel...', 'raw')
            for variation in sorted(os.listdir(ver_path)):
                var_path = os.path.join(ver_path, variation)
                if not os.path.isdir(var_path): continue

                # Check if done (For PMs, we look for the 'full' merged dataset)
                rel = os.path.relpath(var_path, start="dataset")
                out_check = os.path.join(PREPROCESSED_ROOT, rel, "full", "signals.npy")

                if os.path.exists(out_check):
                    print(f"Skipping {version}/{variation} (already exists)")
                    continue

                preprocess_physical_model_dataset(var_path, PREPROCESSED_ROOT, strings=['A', 'D', 'E', 'G'], **PREPROCESS_PARAMS)

    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
