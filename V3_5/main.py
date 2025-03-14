#!/usr/bin/env python3
import argparse
import threading
import pygame
import sys
import os
import platform

from config import MAX_SPEED, running
from training import training_thread
from visualization import combined_rendering_thread, rendering_thread, dashboard_thread


def main():
    global running, MAX_SPEED

    # Print versions for debugging
    print(f"Python version: {sys.version}")
    print(f"Pygame version: {pygame.version.ver}")
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
    except ImportError:
        print("PyTorch not found")
    try:
        import numpy as np
        print(f"NumPy version: {np.__version__}")
    except ImportError:
        print("NumPy not found")

    # Allow command line arguments to adjust max speed
    parser = argparse.ArgumentParser(description='Car Reinforcement Learning with PPO')
    parser.add_argument('--max_speed', type=float, default=MAX_SPEED,
                        help=f'Maximum car speed (default: {MAX_SPEED})')
    parser.add_argument('--no_gui', action='store_true',
                        help='Run without GUI for faster training')
    parser.add_argument('--headless', action='store_true',
                        help='Run in headless mode (no GUI, training only)')
    parser.add_argument('--separate_windows', action='store_true',
                        help='Use separate windows for simulation and dashboard (not recommended)')
    args = parser.parse_args()

    # Update global max speed
    MAX_SPEED = args.max_speed
    print(f"Running with max speed: {MAX_SPEED}")

    # Headless mode overrides other options
    if args.headless:
        args.no_gui = True

    # Start training thread first
    print("Starting training thread...")
    train_thread = threading.Thread(target=training_thread)
    train_thread.start()
    print("Training thread started")

    # If no GUI elements needed, just wait for training
    if args.no_gui:
        print("Running in headless mode (training only). Press Ctrl+C to stop.")
        try:
            # Just wait for training
            train_thread.join()
        except KeyboardInterrupt:
            print("Training interrupted by user")
        finally:
            running = False
            try:
                train_thread.join(timeout=1.0)
            except:
                pass
        return

    # Initialize pygame here where we know we need it
    try:
        print("Initializing pygame...")
        pygame.init()
        pygame.display.init()
        print("Pygame initialized with driver:", pygame.display.get_driver())
    except Exception as e:
        print(f"Failed to initialize pygame: {e}")
        print("Continuing in headless mode (training only)")
        # Continue with training if GUI fails
        try:
            train_thread.join()
        except KeyboardInterrupt:
            print("Training interrupted by user")
        finally:
            running = False
            train_thread.join(timeout=1.0)
        return

    try:
        # Use the combined window approach by default (unless separate windows requested)
        if args.separate_windows:
            print("Using separate windows for simulation and dashboard (experimental)...")
            # This is the old approach - may not work well on all systems
            render_thread = threading.Thread(target=rendering_thread)
            dash_thread = threading.Thread(target=dashboard_thread)
            render_thread.start()
            dash_thread.start()
            render_thread.join()
            dash_thread.join()
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
    print("Windows closed, waiting for training thread to finish...")
    train_thread.join(timeout=2.0)
    print("Exiting")


if __name__ == "__main__":
    main()
