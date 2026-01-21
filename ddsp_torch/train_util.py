import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
import yaml
import numpy as np
import math
import subprocess
import sys
import traceback
from os import path, makedirs
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import _LRScheduler

from preprocess import Dataset
from ddsp_torch.model import DDSP
from ddsp_torch.losses import MultiScaleSTFTLoss, SourceVarianceLoss, ViolinPhysicsLoss


class TFExponentialDecay(_LRScheduler):
    """Learning rate scheduler mimicking TensorFlow's ExponentialDecay."""
    
    def __init__(self, optimizer: torch.optim.Optimizer, initial_lr: float, decay_steps: int, 
                 decay_rate: float, staircase: bool = False):
        self.initial_lr = initial_lr
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.staircase = staircase
        super().__init__(optimizer)

    def get_lr(self) -> list[float]:
        step = self._step_count

        if self.staircase:
            exponent = step // self.decay_steps
        else:
            exponent = step / self.decay_steps

        decay_factor = self.decay_rate ** exponent
        current_lr = self.initial_lr * decay_factor

        return [current_lr for _ in self.base_lrs]


def get_tensorflow_scheduler(optimizer: torch.optim.Optimizer, initial_lr: float = 0.001, 
                           decay_steps: int = 10000, decay_rate: float = 0.98, 
                           staircase: bool = False) -> TFExponentialDecay:
    return TFExponentialDecay(
        optimizer=optimizer,
        initial_lr=initial_lr,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=staircase
    )


def load_configuration():
    """Parse arguments and load configuration file."""
    from effortless_config import Config
    
    class args(Config):
        CONFIG = None

    args.parse_args()

    if args.CONFIG is None:
        config_path = "config.yaml"
        print(f"Using default configuration: {config_path}")
    else:
        config_path = f"configs/{args.CONFIG}.yaml"
        print(f"Using specified configuration: {config_path}")

    if not path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)

    try:
        with open(config_path, "r") as config_file:
            config = yaml.safe_load(config_file)
        print("Configuration loaded successfully.")
        return config, args.CONFIG
    except Exception as e:
        print(f"Error loading config file {config_path}: {e}")
        sys.exit(1)


def extract_training_parameters(config):
    """Extract and validate training parameters from config."""
    try:
        train_config = config["train"]
        return {
            'steps': int(train_config.get("steps", 30000)),
            'batch_size': int(train_config.get("batch_size", 4)),
            'loss_window': int(train_config.get("loss_window", 10)),
            'save_audio_steps': train_config.get("save_audio_steps", []),
            'save_audio_num_comparisons': int(train_config.get("save_audio_num_comparisons", 1)),
            'grad_clip_norm': float(train_config.get("gradient_clip_norm", 3.0)),
            'initial_lr': float(train_config.get("learning_rate", 0.001)),
            'decay_steps': int(train_config.get("lr_decay_steps", 10000)),
            'decay_rate': float(train_config.get("lr_decay_rate", 0.98)),
            'staircase': bool(train_config.get("lr_staircase", False)),
            'svl_activation_threshold': float(train_config.get("svl_activation_threshold", 10.0)),
            'num_workers': int(train_config.get("dataloader_num_workers", 4)),
            'pin_memory': bool(train_config.get("dataloader_pin_memory", True))
        }
    except KeyError as e:
        print(f"Error: Missing required key in 'train' config section: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: Invalid value type in 'train' config section: {e}")
        sys.exit(1)


def setup_run_directory(config):
    """Create run directory and save config."""
    train_config = config["train"]
    run_dir = path.join(train_config.get("root", "runs"), train_config["name"])
    makedirs(run_dir, exist_ok=True)
    print(f"Run directory: {run_dir}")
    
    try:
        with open(path.join(run_dir, "config.yaml"), "w") as f:
            yaml.safe_dump(config, f)
    except Exception as e:
        print(f"Warning: Could not save config to run directory: {e}")
    
    return run_dir


def initialize_model(config, device):
    """Initialize DDSP model and move to device."""
    try:
        model = DDSP(**config.get("model", {})).to(device)
        return model
    except KeyError as e:
        print(f"Error: Missing required key in 'model' config section: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error initializing DDSP model:")
        traceback.print_exc()
        sys.exit(1)


def setup_loss_functions(config, device):
    """Initialize loss functions based on configuration."""
    loss_config = config.get("loss", {})
    model_config = config.get("model", {})
    helmholtz_config = model_config.get("helmholtz", {})
    violin_loss_config = loss_config.get("violin_physics", {})
    
    use_violin_synthesis = bool(helmholtz_config.get("use_helmholtz_synthesis", False))
    use_vpl_config_flag = bool(loss_config.get("use_helmholtz_deviation_loss", True))
    
    try:
        mssl = MultiScaleSTFTLoss(
            scales=loss_config["mssl"]["scales"],
            overlap=loss_config["mssl"]["overlap"]
        ).to(device)

        svl = None
        use_svl_config_flag = bool(loss_config.get("use_source_variance", False))
        svl_active = use_svl_config_flag and not use_violin_synthesis

        if svl_active:
            svl_params = loss_config.get("source_variance", {})
            svl = SourceVarianceLoss(**svl_params).to(device)
            print("\nSource Variance Loss: Enabled (Standard Mode)")
        elif use_svl_config_flag and use_violin_synthesis:
            print("\nSource Variance Loss: Configured but INACTIVE (Violin Mode)")
        else:
            print("\nSource Variance Loss: Disabled")

        vpl = None
        vpl_weight = float(violin_loss_config.get("weight", 0.0))
        vpl_active = use_violin_synthesis and use_vpl_config_flag and vpl_weight > 0

        if vpl_active:
            vpl = ViolinPhysicsLoss(
                weight=vpl_weight,
                weight_β=float(violin_loss_config.get("smoothness_β", 0.01)),
                weight_α=float(violin_loss_config.get("smoothness_α", 0.01)),
                weight_γ=float(violin_loss_config.get("smoothness_γ", 0.01)),
                weight_B=float(violin_loss_config.get("smoothness_B", 0.01)),
                weight_residuals=float(violin_loss_config.get("residual_magnitude", 0.01)),
                residual_loss_type=str(violin_loss_config.get("residual_loss_type", "l1")),
                use_activation_filter=bool(violin_loss_config.get("use_activation_filter", True)),
                loudness_threshold=float(violin_loss_config.get("loudness_threshold", 0.2)),
                pitch_threshold=float(violin_loss_config.get("pitch_threshold", 20.0))
            ).to(device)
            print(f"Violin Physics Loss: Enabled (Weight: {vpl_weight})")
        elif use_violin_synthesis and (not use_vpl_config_flag or vpl_weight <= 0):
            status_reasons = []
            if not use_vpl_config_flag:
                status_reasons.append("Config Toggle is False")
            if vpl_weight <= 0:
                status_reasons.append("Weight is <= 0")
            print(f"Violin Physics Loss: Configured but INACTIVE ({', '.join(status_reasons)})")
        else:
            print("Violin Physics Loss: Disabled")

        return mssl, svl, vpl, svl_active, vpl_active

    except KeyError as e:
        print(f"Error: Missing required key in loss config: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error setting up loss functions: {e}")
        sys.exit(1)


def run_preprocessing_if_needed(config, config_name):
    """Run preprocessing subprocess if needed."""
    try:
        preprocess_dir, preprocess_exists = setup_data_pipeline(config)
    except Exception as e:
        print(f"Error setting up data pipeline: {e}")
        sys.exit(1)

    if not preprocess_exists:
        print("\nRunning subprocess for preprocessing...")
        cmd = ["python", "preprocess.py"]
        if config_name:
            cmd.extend(["--config", config_name])
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("Subprocess finished successfully.")
        except FileNotFoundError:
            print("Error: 'python' command not found. Ensure Python is in your PATH.")
            sys.exit(1)
        except subprocess.CalledProcessError as e:
            print(f"--- Preprocessing Subprocess Failed ---")
            print(f"Command: {' '.join(e.cmd)}")
            print(f"Return Code: {e.returncode}")
            print(f"Output:\n{e.stdout}")
            print(f"Error:\n{e.stderr}")
            sys.exit(1)

    return preprocess_dir


def create_dataset_and_dataloader(preprocess_dir, train_params):
    """Create dataset and dataloader."""
    try:
        dataset = Dataset(preprocess_dir)
        if len(dataset) == 0:
            print(f"Error: Preprocessed dataset in {preprocess_dir} is empty!")
            sys.exit(1)

        dataloader = torch.utils.data.DataLoader(
            dataset, 
            train_params['batch_size'], 
            shuffle=True, 
            drop_last=True,
            num_workers=train_params['num_workers'],
            pin_memory=train_params['pin_memory'] if torch.cuda.is_available() else False,
            persistent_workers=True if train_params['num_workers'] > 0 else False
        )
        
        print(f"\nDataset loaded from {preprocess_dir} ({len(dataset)} samples).")
        return dataset, dataloader

    except FileNotFoundError:
        print(f"Error: Could not load preprocessed files from {preprocess_dir}")
        sys.exit(1)
    except Exception as e:
        print(f"Error creating Dataset or DataLoader: {e}")
        sys.exit(1)


def setup_logging(run_dir, svl_active, vpl_active):
    """Initialize logging systems."""
    writer = None
    loss_log_path = None
    
    try:
        writer = SummaryWriter(run_dir, flush_secs=20)
        loss_log_path = path.join(run_dir, "loss_log.txt")
        
        loss_header_parts = ["step", "total_loss", "spectral_loss", "forward_ms", "backward_ms", "total_ms"]
        if svl_active:
            loss_header_parts.append("variance_loss")
        if vpl_active:
            loss_header_parts.append("violin_physics_loss")
        
        loss_header = ",".join(loss_header_parts) + "\n"
        with open(loss_log_path, "w") as f:
            f.write(loss_header)
            
        print(f"Logging initialized (TensorBoard: {run_dir}, File: {loss_log_path})")
        return writer, loss_log_path
    
    except Exception as e:
        print(f"Warning: Failed to initialize logging: {e}")
        return None, None


def setup_optimizer_and_scheduler(model, train_params):
    """Setup optimizer and learning rate scheduler."""
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    if not trainable_params:
        print("Warning: No trainable parameters found in model!")
    
    optimizer = torch.optim.Adam(trainable_params, lr=train_params['initial_lr'])
    
    scheduler = get_tensorflow_scheduler(
        optimizer=optimizer,
        initial_lr=train_params['initial_lr'],
        decay_steps=train_params['decay_steps'],
        decay_rate=train_params['decay_rate'],
        staircase=train_params['staircase']
    )
    
    return optimizer, scheduler, trainable_params


def calculate_loss(mssl, svl, vpl, model_outputs, target_signal, pitch, loudness,
                  svl_active, vpl_active, svl_threshold, recent_mssl_avg):
    """Calculate total loss from all components."""
    predicted_signal = model_outputs['signal'].squeeze(-1)
    target_for_loss = target_signal.squeeze(-1) if target_signal.dim() > 2 else target_signal
    
    spectral_loss = mssl(predicted_signal, target_for_loss)
    total_loss = spectral_loss
    
    variance_loss_value = None
    violin_physics_loss_value = None
    svl_applied = False
    vpl_applied = False
    
    if svl_active and model_outputs.get('harmonic_amplitudes') is not None and recent_mssl_avg < svl_threshold:
        variance_loss_component = svl(model_outputs['harmonic_amplitudes'], pitch, loudness)
        variance_loss_value = variance_loss_component.item()
        total_loss = total_loss + variance_loss_component
        svl_applied = True
    
    if vpl_active and model_outputs.get('violin_diagnostics') is not None:
        violin_physics_component = vpl(
            model_outputs['violin_diagnostics'],
            f0_hz=pitch,
            loudness=loudness
        )
        violin_physics_loss_value = violin_physics_component.item()
        total_loss = total_loss + violin_physics_component
        vpl_applied = True
    
    return total_loss, spectral_loss, variance_loss_value, violin_physics_loss_value, svl_applied, vpl_applied


def log_training_metrics(writer, loss_log_path, global_step, total_loss_value, spectral_loss_value,
                        forward_time, backward_time, total_time, variance_loss_value, 
                        violin_physics_loss_value, svl_active, vpl_active):
    """Log metrics to TensorBoard and CSV file."""
    try:
        if writer:
            log_training_step(
                writer, global_step, total_loss_value, spectral_loss_value,
                forward_time, backward_time, total_time,
                variance_loss_value, violin_physics_loss_value
            )

        if loss_log_path:
            log_items = [
                f"{global_step}", f"{total_loss_value:.6f}", f"{spectral_loss_value:.6f}",
                f"{forward_time*1000:.2f}", f"{backward_time*1000:.2f}", f"{total_time*1000:.2f}"
            ]
            if svl_active:
                log_items.append(f"{variance_loss_value if variance_loss_value is not None else 0.0:.6f}")
            if vpl_active:
                log_items.append(f"{violin_physics_loss_value if violin_physics_loss_value is not None else 0.0:.6f}")
            
            log_line = ",".join(log_items) + "\n"
            with open(loss_log_path, "a") as log_f:
                log_f.write(log_line)
                
    except Exception as e:
        print(f"\nError during logging at step {global_step}: {e}")


def save_final_results(model, run_dir, config, dataloader, device, train_params, global_step):
    """Save final model state and evaluation audio."""
    try:
        torch.save(model.state_dict(), path.join(run_dir, "final_state.pth"))
        print("Saving final evaluation audio...")
        save_evaluation_audio(
            model, model.state_dict(), run_dir, config,
            dataloader, device, "final",
            num_batches=train_params['save_audio_num_comparisons']
        )
    except Exception as e:
        print(f"Error saving final state/audio: {e}")


def print_training_info(config: dict, device: torch.device, steps: int, batch_size: int, 
                       epochs: int, dataset_size: int, dataloader_size: int):
    """Prints a summary of the key training configuration settings."""
    train_config = config.get("train", {})
    model_config = config.get("model", {})
    encoder_config = model_config.get("encoder", {})
    loss_config = config.get("loss", {})
    violin_model_config = model_config.get("helmholtz", {})
    violin_loss_config = loss_config.get("violin_physics", {})

    print("\n--- Training Configuration ---")
    print(f"- Run Name: {train_config.get('name', 'N/A')}")
    print(f"- Device: {device}")
    print(f"- Total Steps: {steps:,}")
    print(f"- Batch Size: {batch_size}")
    print(f"- Epochs (approx): {epochs}")
    print(f"- Dataset Size: {dataset_size:,} segments")
    print(f"- Steps/Epoch: {dataloader_size:,}")

    print("\n--- Audio Settings ---")
    print(f"- Sampling Rate: {model_config.get('sampling_rate', train_config.get('sampling_rate', 'N/A'))} Hz")
    print(f"- Signal Length: {model_config.get('signal_length', train_config.get('signal_length', 'N/A'))} samples")
    print(f"- Block Size: {model_config.get('block_size', train_config.get('block_size', 'N/A'))} samples")

    print("\n--- Learning Rate Schedule ---")
    print(f"- Scheduler: TensorFlow Exponential Decay")
    print(f"  - Initial LR: {train_config.get('learning_rate', 0.001):.1e}")
    print(f"  - Decay Steps: {int(train_config.get('lr_decay_steps', 10000)):,}")
    print(f"  - Decay Rate: {train_config.get('lr_decay_rate', 0.98)}")
    print(f"  - Staircase: {train_config.get('lr_staircase', False)}")

    print("\n--- Model Configuration Summary ---")
    use_violin = bool(violin_model_config.get('use_helmholtz_synthesis', False))
    print(f"- Synthesis Mode: {'Violin' if use_violin else 'Standard'}")
    
    if use_violin:
        print(f"  - β range: [{violin_model_config.get('β_min', 0.05)}, {violin_model_config.get('β_max', 0.50)}]")
        print(f"  - α range: [{violin_model_config.get('α_min', -1.0)}, {violin_model_config.get('α_max', 1.0)}]")
        print(f"  - Notch width: {violin_model_config.get('notch_width', 0.05)}")
        print(f"  - Residuals: {violin_model_config.get('n_residuals', 10)} (scale: {violin_model_config.get('residual_scale', 0.1)})")

    print(f"- Max Inharmonicity (B_max): {violin_model_config.get('inharmonicity_b_max', 'N/A')}")
    print(f"- Encoder Used: {bool(encoder_config.get('use_encoder', False))}")
    print(f"- Resonance Used: {bool(model_config.get('use_resonance', False))} (Type: {model_config.get('resonance_type', 'N/A')})")
    print(f"- Room Used: {bool(model_config.get('use_room', False))} (Type: {model_config.get('room_type', 'N/A')})")

    print("\n--- Loss Configuration ---")
    use_svl_config = bool(loss_config.get('use_source_variance', False))
    svl_active = use_svl_config and not use_violin
    vpl_weight = float(violin_loss_config.get("weight", 0.0))
    use_vpl_config_toggle = bool(loss_config.get("use_helmholtz_deviation_loss", True))
    vpl_active = use_violin and use_vpl_config_toggle and vpl_weight > 0

    print(f"- SVL Active for this Run: {svl_active}")
    if use_svl_config:
        print(f"  - SVL Configured Weight: {loss_config.get('source_variance', {}).get('weight', 'N/A')}")
        print(f"  - SVL Activation Threshold (MSSL): {train_config.get('svl_activation_threshold', 'N/A')}")

    print(f"- Violin Physics Loss Active for this Run: {vpl_active}")
    if use_violin:
        print(f"  - VPL Config Toggle: {use_vpl_config_toggle}")
        print(f"  - VPL Configured Weight: {vpl_weight}")
        print(f"  - VPL Loss Type: {violin_loss_config.get('residual_loss_type', 'N/A')}")
    print("-----------------------------")


def save_evaluation_audio(model: nn.Module, model_state: dict, run_dir: str, config: dict,
                          dataloader: torch.utils.data.DataLoader, device: torch.device,
                          prefix: str, num_batches: int = 1):
    """Generates and saves evaluation audio with beep separators between original/synthesized pairs."""
    original_training_state = model.training
    
    if model_state:
        model.load_state_dict(model_state)
    model.eval()

    model_config = config.get("model", {})
    encoder_config = model_config.get("encoder", {})
    train_config = config.get("train", {})

    sampling_rate = int(model_config.get('sampling_rate', train_config.get('sampling_rate', 16000)))
    beep_duration = 0.1
    beep_frequency = 1000
    beep_amplitude = 0.5
    
    time_beep = torch.linspace(0, beep_duration, int(beep_duration * float(sampling_rate)), dtype=torch.float32, device='cpu')
    beep_signal = beep_amplitude * torch.sin(2 * math.pi * beep_frequency * time_beep)
    beep_signal_np = beep_signal.numpy()

    audio_parts = []

    try:
        with torch.no_grad():
            data_iterator = iter(dataloader)

            for i in range(num_batches):
                try:
                    signals, pitches, loudness = next(data_iterator)
                except StopIteration:
                    if i == 0:
                        print("Warning: Ran out of evaluation data before processing any segment.")
                        return
                    print(f"Warning: Ran out of evaluation data after processing {i} pairs.")
                    break

                signals_device = signals.to(device)
                pitches_device = pitches.unsqueeze(-1).to(device)
                loudness_device = loudness.unsqueeze(-1).to(device)

                audio_input = signals_device if bool(encoder_config.get("use_encoder", False)) else None
                model_outputs = model(pitches_device, loudness_device, audio=audio_input)
                synthesized_signal = model_outputs['signal']

                if synthesized_signal is None:
                    print(f"Warning: Model output signal is None for pair {i+1}, skipping.")
                    continue

                original_segment = signals[0].cpu().numpy()
                synthesized_segment = synthesized_signal[0].squeeze(-1).cpu().numpy()

                min_length = min(len(original_segment), len(synthesized_segment))
                original_segment = original_segment[:min_length]
                synthesized_segment = synthesized_segment[:min_length]
                
                if min_length == 0:
                    print(f"Warning: Empty segment for pair {i+1}, skipping.")
                    continue

                audio_parts.append(original_segment)
                audio_parts.append(beep_signal_np)
                audio_parts.append(synthesized_segment)

                if i < num_batches - 1:
                    audio_parts.append(beep_signal_np)

            if not audio_parts:
                print("Warning: No audio segments were collected for saving.")
                return

            final_audio = np.concatenate(audio_parts)
            save_path = path.join(run_dir, f"{prefix}_eval.wav")
            sf.write(save_path, final_audio, sampling_rate)
            print(f"Evaluation audio saved to: {save_path}")

    finally:
        if original_training_state:
            model.train()
        else:
            model.eval()


def setup_data_pipeline(config: dict):
    """Determine dataset paths and check if preprocessed data exists."""
    if 'data' not in config:
        config['data'] = {}
    if 'preprocess' not in config:
        config['preprocess'] = {}
    if 'train' not in config:
        raise ValueError("Missing 'train' section in config.")

    train_cfg = config['train']
    data_cfg = config['data']
    prep_cfg  = config['preprocess']

    # Respect the “root + dataset tag” convention.
    root = str(train_cfg.get("preprocessed_data_dir", "preprocessed"))
    dataset_tag = data_cfg.get("dataset_name")

    # If no explicit tag, try to infer from data_location relative to "dataset/"
    data_location = data_cfg.get("data_location")
    if dataset_tag is None and data_location:
        try:
            # Makes "dataset/recordings/berlin" -> "recordings/berlin"
            after_dataset = data_location.split("dataset/")[1]
            dataset_tag = after_dataset.strip("/\\")
        except Exception:
            pass

    if dataset_tag is None:
        dataset_tag = "default_dataset"

    preprocess_dir = path.join(root, dataset_tag)

    # Keep these in config for downstream code
    prep_cfg["out_dir"] = preprocess_dir
    data_cfg["data_location"] = data_cfg.get("data_location", f"dataset/{dataset_tag}")
    data_cfg["extension"] = data_cfg.get("extension", "wav")

    required_files = ["signals.npy", "pitches.npy", "loudness.npy"]
    has_all = all(path.exists(path.join(preprocess_dir, f)) for f in required_files)

    if not has_all:
        print(f"\nPreprocessed files not found in {preprocess_dir}")
        makedirs(preprocess_dir, exist_ok=True)
        return preprocess_dir, False
    else:
        print(f"\nUsing existing preprocessed files from {preprocess_dir}")
        return preprocess_dir, True



def log_training_step(writer: SummaryWriter, step: int, total_loss: float, spectral_loss: float,
                     forward_time: float, backward_time: float, total_time: float,
                     variance_loss_value: float | None = None,
                     violin_physics_loss_value: float | None = None):
    """Logs training metrics for a single step to TensorBoard."""
    writer.add_scalar("Loss/Total", total_loss, step)
    writer.add_scalar("Loss/Spectral", spectral_loss, step)

    if variance_loss_value is not None:
        writer.add_scalar("Loss/Variance", variance_loss_value, step)

    if violin_physics_loss_value is not None:
        writer.add_scalar("Loss/ViolinPhysics", violin_physics_loss_value, step)

    writer.add_scalar("Time/Forward_ms", forward_time * 1000, step)
    writer.add_scalar("Time/Backward_ms", backward_time * 1000, step)
    writer.add_scalar("Time/Total_ms", total_time * 1000, step)


def format_progress_dict(epoch: int, spectral_loss_val: float, total_loss_val: float,
                         avg_mssl_loss: float, learning_rate: float,
                         variance_loss_value: float | None = None,
                         violin_physics_loss_value: float | None = None) -> dict:
    """Creates a dictionary for displaying progress in the tqdm progress bar."""
    progress_dict = {
        'Epoch': epoch + 1,
        'MSSL': f'{spectral_loss_val:.4f}',
        'Total': f'{total_loss_val:.4f}',
        'AvgMSSL': f'{avg_mssl_loss:.4f}',
        'LR': f'{learning_rate:.2e}',
    }
    
    if variance_loss_value is not None:
        progress_dict['SVL'] = f'{variance_loss_value:.4f}'

    if violin_physics_loss_value is not None:
        progress_dict['VPL'] = f'{violin_physics_loss_value:.4f}'

    return progress_dict