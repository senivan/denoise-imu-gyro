import os
import sys
from datetime import datetime
import subprocess

def get_latest_log(log_dir):
    """
    Get the latest log directory based on date and time in the folder names.
    """
    try:
        logs = [d for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))]
        if not logs:
            print(f"No logs found in {log_dir}.")
            return None
        logs.sort(key=lambda x: datetime.strptime(x, "%Y_%m_%d_%H_%M_%S"), reverse=True)
        return logs[0]
    except Exception as e:
        print(f"Error while finding the latest log: {e}")
        return None

def run_tensorboard(log_path):
    """
    Run TensorBoard with the specified log path.
    """
    try:
        print(f"Starting TensorBoard with log directory: {log_path}")
        subprocess.run(["tensorboard", "--logdir", log_path], check=True)
    except FileNotFoundError:
        print("TensorBoard is not installed or not found in PATH.")
    except subprocess.CalledProcessError as e:
        print(f"Error while running TensorBoard: {e}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python tensor_runner.py <folder_name> [log_name]")
        print("Example: python tensor_runner.py EUROC")
        print("Example: python tensor_runner.py EUROC 2025_03_14_21_42_37")
        sys.exit(1)

    folder_name = sys.argv[1]
    log_name = sys.argv[2] if len(sys.argv) > 2 else None

    base_dir = "./results/runs"
    log_dir = os.path.join(base_dir, folder_name)

    print(f"Log directory: {log_dir}")

    if not os.path.exists(log_dir):
        print(f"Folder {folder_name} does not exist in directory.")
        sys.exit(1)

    if log_name:
        log_path = os.path.join(log_dir, log_name)
        if not os.path.exists(log_path):
            print(f"Log {log_name} does not exist in directory.")
            sys.exit(1)
    else:
        log_name = get_latest_log(log_dir)
        if not log_name:
            sys.exit(1)
        log_path = os.path.join(log_dir, log_name)

    run_tensorboard(log_path)

if __name__ == "__main__":
    main()

