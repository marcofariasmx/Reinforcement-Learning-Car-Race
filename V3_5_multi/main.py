#!/usr/bin/env python3
import argparse
import threading
import pygame
import sys
import os
import platform
import time
import torch

from config import (MAX_SPEED, running, save_queue,
                    USE_MIXED_PRECISION, USE_GPU_FOR_INFERENCE,
                    hw_settings, update_settings_from_args,
                    print_current_settings, NUM_CARS)
from training import training_thread
from visualization import combined_rendering_thread, rendering_thread, dashboard_thread


def print_system_info():
    """Print detailed system information for debugging"""
    print("\n=== System Information ===")
    print(f"Platform: {platform.platform()}")
    print(f"Python version: {sys.version}")

    try:
        import pygame
        print(f"Pygame version: {pygame.version.ver}")
    except ImportError:
        print("Pygame not found")

    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Device count: {torch.cuda.device_count()}")

            # Print CUDA memory info
            device = torch.cuda.current_device()
            total_memory = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)
            print(f"Total GPU memory: {total_memory:.2f} GB")

            # Check if we can use mixed precision
            if hasattr(torch.cuda, 'amp') and USE_MIXED_PRECISION:
                print("Mixed precision training is available and enabled")
            else:
                print("Mixed precision training is not available or disabled")
    except ImportError:
        print("PyTorch not found")

    try:
        import numpy as np
        print(f"NumPy version: {np.__version__}")
    except ImportError:
        print("NumPy not found")

    print("===========================\n")


def main():
    global running, MAX_SPEED

    # Print system information
    print_system_info()

    # Create argument parser with better defaults (None means "use auto-detected value")
    parser = argparse.ArgumentParser(
        description='Car Reinforcement Learning with PPO',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Basic settings
    parser.add_argument('--max_speed', type=float, default=None,
                        help=f'Maximum car speed (auto-detected: {MAX_SPEED})')
    parser.add_argument('--batch_size', type=int, default=None,
                        help=f'Batch size for training (auto-detected: {hw_settings["batch_size"]})')
    parser.add_argument('--save_interval', type=int, default=None,
                        help=f'Save model every N episodes (auto-detected: {hw_settings["save_interval"]})')
    parser.add_argument('--ppo_epochs', type=int, default=None,
                        help=f'Number of PPO epochs per update (auto-detected: {hw_settings["ppo_epochs"]})')
    parser.add_argument('--num_cars', type=int, default=None,
                        help=f'Number of car instances to run in parallel (default: {NUM_CARS})')

    # Visualization options
    parser.add_argument('--no_gui', action='store_true',
                        help='Run without GUI for faster training')
    parser.add_argument('--headless', action='store_true',
                        help='Run in headless mode (no GUI, training only)')
    parser.add_argument('--separate_windows', action='store_true',
                        help='Use separate windows for simulation and dashboard')

    # Hardware utilization options
    parser.add_argument('--gpu_inference', action='store_true',
                        help='Force using GPU for inference')
    parser.add_argument('--cpu_inference', action='store_true',
                        help='Force using CPU for inference')
    parser.add_argument('--mixed_precision', action='store_true',
                        help='Enable mixed precision training if supported')
    parser.add_argument('--no_mixed_precision', action='store_true',
                        help='Disable mixed precision training')
    parser.add_argument('--async_save', action='store_true',
                        help='Enable asynchronous model saving')
    parser.add_argument('--sync_save', action='store_true',
                        help='Disable asynchronous model saving')
    parser.add_argument('--cpu', action='store_true',
                        help='Force using CPU even if GPU is available')

    # Performance monitoring options
    parser.add_argument('--profile', action='store_true',
                        help='Enable performance profiling')

    args = parser.parse_args()

    # Update settings based on command-line arguments (only if explicitly provided)
    update_settings_from_args(args)

    # Print the final settings after applying command-line overrides
    print_current_settings()

    # Headless mode overrides other options
    if args.headless:
        args.no_gui = True

    # Start training thread first
    print("Starting training thread...")
    train_thread = threading.Thread(target=training_thread, name="TrainingThread")
    train_thread.start()
    print("Training thread started")

    # If no GUI elements needed, just wait for training
    if args.no_gui:
        print("Running in headless mode (training only). Press Ctrl+C to stop.")
        try:
            # Just wait for training
            while running and train_thread.is_alive():
                time.sleep(1)

                # Periodically check if training is still running
                if not train_thread.is_alive():
                    print("Training thread has stopped")
                    break
        except KeyboardInterrupt:
            print("Training interrupted by user")
        finally:
            running = False
            # Signal the save thread to terminate
            try:
                save_queue.put(None)
            except:
                pass

            print("Waiting for training thread to finish...")
            try:
                train_thread.join(timeout=5.0)
                if train_thread.is_alive():
                    print("Training thread is still running after timeout, exiting anyway")
            except:
                print("Error while waiting for training thread")
        return

    # Initialize pygame here where we know we need it
    try:
        print("Initializing pygame...")
        # Initialize with better error handling
        if not pygame.get_init():
            pygame.init()

        if not pygame.display.get_init():
            pygame.display.init()

        print("Pygame initialized with driver:", pygame.display.get_driver())

        # Print available pygame drivers
        if hasattr(pygame.display, 'get_driver_list'):
            print("Available pygame drivers:", pygame.display.get_driver_list())
    except Exception as e:
        print(f"Failed to initialize pygame: {e}")
        print("Continuing in headless mode (training only)")
        # Continue with training if GUI fails
        try:
            while running and train_thread.is_alive():
                time.sleep(1)
        except KeyboardInterrupt:
            print("Training interrupted by user")
        finally:
            running = False
            train_thread.join(timeout=5.0)
        return

    try:
        # Use the combined window approach by default (unless separate windows requested)
        if args.separate_windows:
            print("Using separate windows for simulation and dashboard...")
            # This is the old approach - may not work well on all systems
            render_thread = threading.Thread(target=rendering_thread, name="RenderThread")
            dash_thread = threading.Thread(target=dashboard_thread, name="DashboardThread")
            render_thread.start()
            dash_thread.start()

            # Wait for rendering threads
            try:
                while running and (render_thread.is_alive() or dash_thread.is_alive()):
                    time.sleep(0.5)
            except KeyboardInterrupt:
                print("Interrupted by user")
                running = False
            finally:
                # Wait for threads to finish
                if render_thread.is_alive():
                    render_thread.join(timeout=2.0)
                if dash_thread.is_alive():
                    dash_thread.join(timeout=2.0)
        else:
            # New approach - single window with both components
            print("Starting combined simulation+dashboard window...")
            combined_rendering_thread()
    except Exception as e:
        print(f"Error in GUI: {e}")
        import traceback
        traceback.print_exc()

    # We only get here after windows are closed
    running = False

    # Signal the save thread to terminate
    try:
        save_queue.put(None)
    except:
        pass

    print("Windows closed, waiting for training thread to finish...")
    train_thread.join(timeout=5.0)

    if train_thread.is_alive():
        print("Training thread is still running after timeout. Exiting anyway.")
    else:
        print("Training thread has finished.")

    print("Exiting")


if __name__ == "__main__":
    main()