import os
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

def convert_euroc_to_tumvi(euroc_path, output_path):
    # Create output directories if they don't exist

    # Load Euroc IMU and ground truth data
    imu_file = os.path.join(euroc_path, 'mav0', 'imu0', 'data.csv')
    gt_file = os.path.join(euroc_path, 'mav0', 'state_groundtruth_estimate0', 'data.csv')

    if not os.path.isfile(imu_file) or not os.path.isfile(gt_file):
        print(f"Skipping {euroc_path}: Missing required files.")
        return

    imu_data = pd.read_csv(imu_file, comment='#', header=None)
    gt_data = pd.read_csv(gt_file, comment='#', header=None)

    # Set appropriate column names for readability
    imu_data.columns = ['timestamp', 'ang_vel_x', 'ang_vel_y', 'ang_vel_z', 'acc_x', 'acc_y', 'acc_z']
    gt_data.columns = [
        'timestamp', 'pos_x', 'pos_y', 'pos_z',
        'quat_w', 'quat_x', 'quat_y', 'quat_z',
        'vel_x', 'vel_y', 'vel_z',
        'bias_gyr_x', 'bias_gyr_y', 'bias_gyr_z',
        'bias_acc_x', 'bias_acc_y', 'bias_acc_z'
    ]


    # Convert timestamps to seconds
    imu_data['timestamp'] = imu_data['timestamp'] * 1e-9
    gt_data['timestamp'] = gt_data['timestamp'] * 1e-9

    # Synchronize timestamps using intersection
    common_timestamps = np.intersect1d(imu_data['timestamp'], gt_data['timestamp'])
    imu_data = imu_data[imu_data['timestamp'].isin(common_timestamps)].reset_index(drop=True)
    gt_data = gt_data[gt_data['timestamp'].isin(common_timestamps)].reset_index(drop=True)

    # Validate synchronization
    if len(imu_data) != len(gt_data):
        min_len = min(len(imu_data), len(gt_data))
        print(f"Warning: Length mismatch. Trimming to {min_len}.")
        imu_data = imu_data.iloc[:min_len]
        gt_data = gt_data.iloc[:min_len]

    print(f"IMU Data Shape: {imu_data.shape}")
    print(f"Ground Truth Shape: {gt_data.shape}")

    # Normalize quaternions
    quat_columns = ['quat_w', 'quat_x', 'quat_y', 'quat_z']
    quats = gt_data[quat_columns].to_numpy()
    quats /= np.linalg.norm(quats, axis=1).reshape(-1, 1)
    gt_data[quat_columns] = quats


    # Save IMU data in TUMVI format
    imu_output_file = os.path.join(output_path, 'imu.csv')
    print(imu_output_file)
    imu_data[['timestamp', 'ang_vel_x', 'ang_vel_y', 'ang_vel_z', 'acc_x', 'acc_y', 'acc_z']].to_csv(
        imu_output_file, index=False, header=[
            'Time', 'angular_velocity.x', 'angular_velocity.y', 'angular_velocity.z',
            'linear_acceleration.x', 'linear_acceleration.y', 'linear_acceleration.z'
        ]
    )

    # Save ground truth in TUMVI format
    gt_output_file = os.path.join(output_path, 'gt.csv')
    print(gt_output_file)
    gt_data[['timestamp', 'pos_x', 'pos_y', 'pos_z', 'quat_x', 'quat_y', 'quat_z', 'quat_w']].to_csv(
        gt_output_file, index=False, header=[
            'Time', 'transform.translation.x', 'transform.translation.y', 'transform.translation.z',
            'transform.rotation.x', 'transform.rotation.y', 'transform.rotation.z', 'transform.rotation.w'
        ]
    )

    print(f"Conversion complete for {euroc_path}! Data saved to {output_path}")

if __name__ == "__main__":
    dataset_paths = [
        "/datasets1/EUROC_dataset/V1_01_easy",
        "/datasets1/EUROC_dataset/V1_02_medium",
        "/datasets1/EUROC_dataset/V1_03_difficult",
        "/datasets1/EUROC_dataset/V2_01_easy",
        "/datasets1/EUROC_dataset/V2_02_medium",
        "/datasets1/EUROC_dataset/V2_03_difficult",
        "/datasets1/EUROC_dataset/MH_01_easy",
        "/datasets1/EUROC_dataset/MH_02_easy",
        "/datasets1/EUROC_dataset/MH_03_medium",
        "/datasets1/EUROC_dataset/MH_04_difficult",
        "/datasets1/EUROC_dataset/MH_05_difficult"
    ]

    output_paths = [
        "/datasets1/EUROC_dataset/V1_01_easy",
        "/datasets1/EUROC_dataset/V1_02_medium",
        "/datasets1/EUROC_dataset/V1_03_difficult",
        "/datasets1/EUROC_dataset/V2_01_easy",
        "/datasets1/EUROC_dataset/V2_02_medium",
        "/datasets1/EUROC_dataset/V2_03_difficult",
        "/datasets1/EUROC_dataset/MH_01_easy",
        "/datasets1/EUROC_dataset/MH_02_easy",
        "/datasets1/EUROC_dataset/MH_03_medium",
        "/datasets1/EUROC_dataset/MH_04_difficult",
        "/datasets1/EUROC_dataset/MH_05_difficult"
    ]

    for i in range(len(dataset_paths)):
        dataset_path = dataset_paths[i]
        output_root = output_paths[i]
        dataset_name = os.path.basename(dataset_path)
        convert_euroc_to_tumvi(dataset_path, output_root)
