# DDSP Violin - Codebase Guide

## Overview

This is a PyTorch implementation of DDSP (Differentiable Digital Signal Processing) specialized for violin synthesis. It supports two synthesis modes:
- **Standard DDSP**: Learned harmonic distributions
- **Violin/Helmholtz Mode**: Physics-guided synthesis using bow position, notch depth, brightness, and inharmonicity

The model learns to synthesize audio from pitch (F0) and loudness conditioning signals.

---

## Project Structure

```
ddsp_violin/
├── ddsp_torch/           # Core library (IMPORTANT - main code lives here)
├── configs/              # 220 YAML experiment configs (can ignore details)
├── runs/                 # Training outputs (logs, checkpoints, audio)
├── preprocessed/         # Preprocessed .npy datasets
├── model_parameters/     # Saved model weights (.npz files)
├── normalized_irs/       # Normalized impulse responses
├── plots/                # Analysis plots
├── plots_ir/             # Impulse response visualizations
│
├── train.py              # IMPORTANT: Main training entry point
├── preprocess.py         # IMPORTANT: Data preprocessing script
├── config.yaml           # Template/default configuration
├── combine.sh            # Utility script (minor)
│
├── *.ipynb               # Jupyter notebooks for analysis (can ignore)
└── repo.txt              # Repository metadata (ignore)
```

---

## Important Files (ddsp_torch/)

### Core Model Files
| File | Lines | Purpose | Priority |
|------|-------|---------|----------|
| `model.py` | 259 | **Main DDSP model class** - orchestrates encoder, decoder, synthesizers, filters | HIGH |
| `decoder.py` | 88 | Generates synthesis parameters from conditioning inputs | HIGH |
| `encoder.py` | 97 | Optional - extracts latent Z from audio via MFCCs + GRU | MEDIUM |
| `synth.py` | 78 | Harmonic and noise synthesis functions | HIGH |
| `model_utils.py` | 299 | Physics functions (bow notch, brightness) + activation helpers | HIGH for violin mode |

### DSP & Filtering
| File | Lines | Purpose | Priority |
|------|-------|---------|----------|
| `core.py` | 343 | Core DSP: pitch/loudness extraction, STFT, scaling, upsampling | HIGH |
| `filters.py` | 347 | ConvolutionalFilter, ARFilter, ARMAFilter implementations | MEDIUM |
| `dsp.py` | 144 | FFT convolution, impulse response conversion | MEDIUM |
| `filter_utils.py` | 43 | Reflection-to-AR coefficient conversion (Levinson-Durbin) | LOW |

### Training Infrastructure
| File | Lines | Purpose | Priority |
|------|-------|---------|----------|
| `train_util.py` | 634 | **Training utilities** - config loading, dataset, logging, optimizer | HIGH |
| `losses.py` | 174 | Loss functions: MultiScaleSTFTLoss, SourceVarianceLoss, ViolinPhysicsLoss | HIGH |
| `nn.py` | 48 | Basic building blocks: MLP, GRU, Normalize | LOW |

### Can Usually Ignore
- `__init__.py` - Empty package marker
- `.ipynb_checkpoints/` - Jupyter autosaves

---

## Entry Points

### Training
```bash
python train.py                     # Uses config.yaml
python train.py --config NAME       # Uses configs/NAME.yaml
```

### Preprocessing
```bash
python preprocess.py                # Uses config.yaml
python preprocess.py --config NAME  # Uses configs/NAME.yaml
```

---

## Data Flow Pipeline

```
1. PREPROCESSING (preprocess.py)
   Raw audio files → CREPE pitch extraction → A-weighted loudness → .npy files
   Output: signals.npy, pitches.npy, loudness.npy

2. TRAINING (train.py)
   Load .npy → DataLoader → Model forward → Loss → Backward → Optimizer step

3. MODEL FORWARD PASS (model.py)
   Inputs: pitch [B, T, 1], loudness [B, T, 1], audio [B, L] (optional)

   [Pitch, Loudness] → Scale → Decoder → Synthesis params
                                              ↓
   [Audio] → Encoder → Z (latent) ──────────→ ↑
                                              ↓
   Synthesis params → Harmonic Synth ──┐
                   → Noise Synth ──────┼→ Resonance Filter → Room Filter → Output
```

---

## Two Synthesis Modes

### Standard DDSP Mode
- Decoder outputs: `harmonic_distribution` (n_harmonic values)
- Directly learns amplitude for each harmonic
- Uses `SourceVarianceLoss` for regularization

### Violin (Helmholtz) Mode
- Decoder outputs: `β` (bow position), `γ` (notch depth), `α` (brightness), `residuals`
- Physics-based spectrum computation:
  - Baseline: 1/n amplitude rolloff
  - Bow notch: sin²(πnβ) creates notches where n×β is integer
  - Brightness tilt: n^(-α) spectral slope
  - Residuals: per-harmonic corrections
- Uses `ViolinPhysicsLoss` for smoothness regularization

---

## Key Classes

### DDSP (model.py:20)
Main model class. Key methods:
- `forward(pitch, loudness, audio=None)` - Main inference
- `_generate_violin_harmonic()` - Physics-guided synthesis
- `_generate_standard_harmonic()` - Learned harmonic distribution

### Decoder (decoder.py:11)
Conditioning → synthesis parameters.
Architecture: MLP stacks → GRU → Skip connection → MLP → Linear output

### Encoder (encoder.py:7)
Audio → latent Z.
Architecture: MFCC → Normalize → GRU → Linear → Interpolate to target length

### Filters (filters.py)
- `ConvolutionalFilter`: Learned FIR via FFT convolution
- `ARFilter`: IIR filter with reflection coefficients (guaranteed stable)
- `ARMAFilter`: Combined AR poles + MA zeros

---

## Configuration Structure

Configs are YAML files with these sections:

```yaml
train:           # Training hyperparameters
  name: ...      # Run name (creates runs/NAME/ directory)
  steps: ...     # Total training steps
  batch_size: ...

preprocess:      # Audio preprocessing settings
  sampling_rate: 16000
  block_size: 64      # Hop size for feature extraction
  signal_length: 64000  # Samples per segment (4 seconds at 16kHz)

data:            # Dataset location
  data_location: dataset/...
  extension: wav

model:           # Model architecture
  encoder: ...   # Optional encoder config
  decoder: ...   # Decoder config
  helmholtz: ... # Violin mode parameters
  use_resonance: true/false
  use_room: true/false

loss:            # Loss function config
  mssl: ...      # Multi-scale STFT loss scales
  violin_physics: ...  # VPL weights
```

---

## Config Naming Convention

Config files follow the pattern:
```
{VIOLIN}_{MIC}_{NORM}_{PITCH}__{FILTER}__{MODE}_{LOSS}.yaml
```

Examples:
- `Bernardel_ear_norm_A__vio.yaml` - Violin mode, no extra filter, note A
- `Bernardel_ear_norm_A__arma64x64__vio_vpl_w10.yaml` - ARMA filter, VPL weight=10

---

## Run Directory Structure

Each training run creates:
```
runs/RUN_NAME/
├── config.yaml          # Copy of config used
├── final_state.pth      # Model weights
├── loss_log.txt         # CSV of training metrics
├── events.out.tfevents* # TensorBoard logs
├── final_eval.wav       # Original/synthesized comparison audio
└── step_N_eval.wav      # Checkpoints at specified steps
```

---

## Code Patterns

### Tensor Shapes
- Audio: `[batch, samples]` or `[batch, samples, 1]`
- Frame-rate features: `[batch, n_frames, features]`
- Pitch/loudness: `[batch, n_frames, 1]`
- Harmonic amplitudes: `[batch, n_frames, n_harmonics]`

### Physics Parameters (Violin Mode)
- `β` (bow position): Range [β_min, β_max], typically [0.05, 0.5]
- `γ` (notch depth): Range [0, 1], controls how deep spectral notches are
- `α` (brightness): Range [α_min, α_max], controls spectral tilt
- `B` (inharmonicity): Range [0, B_max], string stiffness coefficient

### Activation Functions
- Raw decoder outputs → sigmoid/tanh → scaled to physical ranges
- `exp_sigmoid()` for amplitudes (always positive, smooth)

### Upsampling
Frame-rate params (e.g., 1000 frames) → sample-rate (64000 samples):
- `upsample(signal, factor, method='window')` - Hanning windowed overlap-add
- `upsample(signal, factor, method='linear')` - Simple interpolation

---

## Dependencies

Key libraries:
- PyTorch + torchaudio
- librosa (audio processing)
- crepe (pitch extraction)
- soundfile (audio I/O)
- tensorboard (logging)
- effortless_config (CLI arg parsing)
- numpy, tqdm, yaml

---

## Common Tasks

### Train a new model
```bash
cd ddsp_violin
python train.py --config YOUR_CONFIG_NAME
```

### Check training progress
- TensorBoard: `tensorboard --logdir=runs/`
- Loss log: `runs/RUN_NAME/loss_log.txt`
- Audio samples: `runs/RUN_NAME/*_eval.wav`

### Load a trained model
```python
from ddsp_torch.model import DDSP
import torch
import yaml

with open('runs/RUN_NAME/config.yaml') as f:
    config = yaml.safe_load(f)

model = DDSP(**config['model'])
model.load_state_dict(torch.load('runs/RUN_NAME/final_state.pth'))
model.eval()

# Inference
output = model(pitch_tensor, loudness_tensor)
audio = output['signal']
```

---

## Files to Ignore

- `*.ipynb` notebooks - Analysis/visualization, not core functionality
- `.ipynb_checkpoints/` - Jupyter autosaves
- `__pycache__/` - Python bytecode
- `repo.txt` - Metadata file
- `combine.sh` - Minor utility script
- Individual config files in `configs/` - Only look at structure, not details

---

## Project Guidelines

### Code Style
- Minimize changes to existing code
- Prefer small, focused modifications over refactors
- Always preserve existing comments and docstrings

### When Making Changes
- Explain what you're changing BEFORE doing it
- If a change touches more than 2 files, ask for confirmation first
- Never modify the data loading pipeline without explicit permission
