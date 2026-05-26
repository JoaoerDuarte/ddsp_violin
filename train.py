import torch
import time
import traceback
from collections import deque
from tqdm import tqdm
from torch.nn.utils.clip_grad import clip_grad_norm_

from ddsp_torch.train_util import (
    load_configuration, extract_training_parameters, setup_run_directory,
    initialize_model, setup_loss_functions, run_preprocessing_if_needed,
    create_dataset_and_dataloader, setup_logging, setup_optimizer_and_scheduler,
    calculate_loss, log_training_metrics, save_final_results,
    print_training_info, save_evaluation_audio, format_progress_dict,
)


def run_training_loop(model, dataloader, optimizer, scheduler, trainable_params,
                      mssl, hrl, train_params, config, run_dir, device,
                      hrl_active, writer, loss_log_path):
    """Main training loop: forward, loss, backward, log, save."""
    recent_mssl_losses = deque(maxlen=train_params['loss_window'])
    step_times = deque(maxlen=train_params['steps'])
    global_step = 0

    epochs = int(torch.ceil(torch.tensor(train_params['steps'] / len(dataloader), dtype=torch.float32))) if len(dataloader) > 0 else 1

    print_training_info(config, device, train_params['steps'], train_params['batch_size'],
                        epochs, len(dataloader.dataset), len(dataloader))
    print(f"Gradient Clipping Norm: {train_params['grad_clip_norm']}")
    if train_params['save_audio_steps']:
        print(f"\nWill save audio at steps: {train_params['save_audio_steps']}")

    print("\n--- Starting Training ---")
    pbar = tqdm(total=train_params['steps'], desc="Training", unit="step")
    training_failed = False

    try:
        for epoch in range(epochs):
            if global_step >= train_params['steps']:
                break

            for batch_idx, batch_data in enumerate(dataloader):
                if global_step >= train_params['steps']:
                    break

                try:
                    signals, pitches, loudness = batch_data
                except ValueError as e:
                    print(f"\nError unpacking batch {batch_idx} in epoch {epoch}: {e}")
                    continue

                model.train()
                non_blocking = train_params['pin_memory'] if device.type == 'cuda' else False
                signals_device = signals.to(device, non_blocking=non_blocking)
                pitches_device = pitches.unsqueeze(-1).to(device, non_blocking=non_blocking)
                loudness_device = loudness.unsqueeze(-1).to(device, non_blocking=non_blocking)

                step_start_time = time.perf_counter()

                try:
                    encoder_config = config.get("model", {}).get("encoder", {})
                    audio_input = signals_device if encoder_config.get("use_encoder", False) else None
                    model_outputs = model(pitches_device, loudness_device, audio=audio_input)
                    forward_time = time.perf_counter() - step_start_time
                except Exception:
                    print(f"\nError during forward pass at step {global_step}:")
                    traceback.print_exc()
                    training_failed = True
                    break

                try:
                    current_mssl_avg = sum(recent_mssl_losses) / len(recent_mssl_losses) if recent_mssl_losses else float('inf')

                    total_loss, spectral_loss, hrl_loss_value, hrl_applied = calculate_loss(
                        mssl, hrl, model_outputs, signals_device, pitches_device, loudness_device,
                        hrl_active,
                    )

                    recent_mssl_losses.append(spectral_loss.item())
                    total_loss_value = total_loss.item()
                    spectral_loss_value = spectral_loss.item()

                except Exception:
                    print(f"\nError during loss calculation at step {global_step}:")
                    traceback.print_exc()
                    training_failed = True
                    break

                backward_start_time = time.perf_counter()
                try:
                    optimizer.zero_grad()
                    total_loss.backward()
                    if trainable_params:
                        clip_grad_norm_(trainable_params, train_params['grad_clip_norm'])
                    optimizer.step()
                    scheduler.step()
                    backward_time = time.perf_counter() - backward_start_time
                except Exception:
                    print(f"\nError during backward pass at step {global_step}:")
                    traceback.print_exc()
                    training_failed = True
                    break

                total_step_time = time.perf_counter() - step_start_time
                step_times.append(total_step_time)

                log_training_metrics(
                    writer, loss_log_path, global_step, total_loss_value, spectral_loss_value,
                    forward_time, backward_time, total_step_time,
                    hrl_loss_value, hrl_active,
                )

                pbar.set_postfix(format_progress_dict(
                    epoch, spectral_loss_value, total_loss_value, current_mssl_avg,
                    optimizer.param_groups[0]["lr"],
                    hrl_loss_value if hrl_applied else None,
                ))
                pbar.update(1)

                if train_params['save_audio_steps'] and global_step in train_params['save_audio_steps']:
                    print(f"\nSaving evaluation audio at step {global_step}...")
                    save_evaluation_audio(
                        model, model.state_dict(), run_dir, config,
                        dataloader, device, f"step_{global_step}",
                        num_batches=train_params['save_audio_num_comparisons'],
                    )

                global_step += 1

            if training_failed:
                break

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        training_failed = True
    except Exception:
        print("\nUnexpected error during training:")
        traceback.print_exc()
        training_failed = True
    finally:
        if pbar:
            pbar.close()

    return global_step, training_failed, step_times


if __name__ == '__main__':
    config, config_name = load_configuration()
    train_params = extract_training_parameters(config)
    run_dir = setup_run_directory(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n--- Device Information ---")
    print(f"- Using {'GPU: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("------------------------")

    model = initialize_model(config, device)
    mssl, hrl, hrl_active = setup_loss_functions(config, device)

    preprocess_dir = run_preprocessing_if_needed(config, config_name)
    dataset, dataloader = create_dataset_and_dataloader(preprocess_dir, train_params)

    writer, loss_log_path = setup_logging(run_dir, hrl_active)
    optimizer, scheduler, trainable_params = setup_optimizer_and_scheduler(model, train_params)

    global_step, training_failed, step_times = run_training_loop(
        model, dataloader, optimizer, scheduler, trainable_params,
        mssl, hrl, train_params, config, run_dir, device,
        hrl_active, writer, loss_log_path,
    )

    if writer:
        writer.close()

    print("\n--- Training Loop Finished ---")

    if global_step > 0 and not training_failed:
        print("Saving final model state...")
        save_final_results(model, run_dir, config, dataloader, device, train_params, global_step)
    elif training_failed:
        print("Training failed or interrupted, final model not saved.")
    else:
        print("No training steps completed, final model not saved.")

    if step_times:
        avg_step_time = sum(step_times) / len(step_times)
        print(f"\nAverage step time: {avg_step_time*1000:.2f} ms")
        print(f"Total steps completed: {global_step}")
    print("-----------------------------")
