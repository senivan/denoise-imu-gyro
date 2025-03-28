import torch
import numpy as np
from src.lie_algebra import SO3


def compute_alignment(R_gt, R_est):
    if isinstance(R_est, list):
        R_est = torch.stack(R_est, dim=0)
    if isinstance(R_gt, list):
        R_gt = torch.stack(R_gt, dim=0)
    
    # If the tensors are in quaternion format (M, 4), convert them
    if R_gt.dim() == 2 and R_gt.shape[1] == 4:
        R_gt = SO3.from_quaternion(R_gt)
    if R_est.dim() == 2 and R_est.shape[1] == 4:
        R_est = SO3.from_quaternion(R_est)
    
    # If the tensors are still 2D (but with 3 channels), convert from Lie algebra to rotation matrices
    if R_gt.dim() == 2 and R_gt.shape[1] == 3:
        R_gt = SO3.exp(R_gt)
    if R_est.dim() == 2 and R_est.shape[1] == 3:
        R_est = SO3.exp(R_est)
    
    # At this point, we expect R_gt and R_est to be (M, 3, 3)
    if R_est.dim() != 3 or R_est.shape[1:] != (3, 3):
        raise ValueError("R_est must be of shape (M,3,3), got {}".format(R_est.shape))
    if R_gt.dim() != 3 or R_gt.shape[1:] != (3, 3):
        raise ValueError("R_gt must be of shape (M,3,3), got {}".format(R_gt.shape))
    
    R_est0_inv = torch.inverse(R_est[0])
    alignment = torch.matmul(R_gt[0], R_est0_inv)
    R_est_aligned = torch.matmul(alignment, R_est)
    return R_est_aligned

def compute_aoe(R_gt, R_est):
    # Convert R_est to rotation matrices if in lie algebra (axis-angle: shape (M, 3)) or quaternion (shape (M, 4))
    if R_est.dim() == 2 and R_est.shape[1] == 3:
        R_est = SO3.exp(R_est)
    elif R_est.dim() == 2 and R_est.shape[1] == 4:
        R_est = SO3.from_quaternion(R_est)
    
    # Convert R_gt similarly
    if R_gt.dim() == 2 and R_gt.shape[1] == 3:
        R_gt = SO3.exp(R_gt)
    elif R_gt.dim() == 2 and R_gt.shape[1] == 4:
        R_gt = SO3.from_quaternion(R_gt)
    
    # Ensure shapes are (M, 3, 3)
    if R_gt.dim() != 3 or R_gt.shape[1:] != (3, 3):
        raise ValueError("R_gt must be of shape (M,3,3), got {}".format(R_gt.shape))
    if R_est.dim() != 3 or R_est.shape[1:] != (3, 3):
        raise ValueError("R_est must be of shape (M,3,3), got {}".format(R_est.shape))
    
    M = R_gt.shape[0]
    R_est_aligned = compute_alignment(R_gt, R_est)
    error_sum = 0.0
    for n in range(M):
        R_diff = torch.matmul(R_gt[n].transpose(0, 1), R_est_aligned[n])
        log_R = SO3.log(R_diff.unsqueeze(0)).squeeze(0)
        error_sum += torch.norm(log_R) ** 2
    return torch.sqrt(error_sum / M)

def cumulative_distance(positions):
    if isinstance(positions, torch.Tensor):
        positions = positions.cpu().numpy()
    dists = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    return np.concatenate(([0], np.cumsum(dists)))

def find_disp(positions, bar):
    cumulative = cumulative_distance(positions)
    indices = []
    M = positions.shape[0]
    for i in range(M):
        displace = cumulative - cumulative[i]
        j_cands = np.where(displace >= bar)[0]
        if len(j_cands) > 0:
            j = j_cands[0]
            indices.append((i, j))
    return indices

def compute_roe(R_gt, R_est):
    if R_est.dim() == 2 and R_est.shape[1] == 3:
        R_est = SO3.exp(R_est)
    elif R_est.dim() == 2 and R_est.shape[1] == 4:
        R_est = SO3.from_quaternion(R_est)
    
    if R_gt.dim() == 2 and R_gt.shape[1] == 3:
        R_gt = SO3.exp(R_gt)
    elif R_gt.dim() == 2 and R_gt.shape[1] == 4:
        R_gt = SO3.from_quaternion(R_gt)
    
    if R_gt.dim() != 3 or R_gt.shape[1:] != (3, 3):
        raise ValueError("R_gt must be of shape (M, 3, 3), got {}".format(R_gt.shape))
    if R_est.dim() != 3 or R_est.shape[1:] != (3, 3):
        raise ValueError("R_est must be of shape (M, 3, 3), got {}".format(R_est.shape))
    
    M = R_gt.shape[0]
    disp_pairs = find_disp(R_gt, 7)
    errors = []
    for (n, g) in disp_pairs:
        if n >= M or g >= M:
            continue
        delta_r_gt  = torch.matmul(R_gt[n].transpose(0, 1), R_gt[g])
        delta_r_est = torch.matmul(R_est[n].transpose(0, 1), R_est[g])
        R_diff = torch.matmul(delta_r_gt, torch.inverse(delta_r_est))
        log_R = SO3.log(R_diff.unsqueeze(0)).squeeze(0)
        errors.append(torch.norm(log_R))
    if len(errors) == 0:
        return torch.tensor(0.0)
    errors_tensor = torch.stack(errors)
    median = torch.median(errors_tensor)
    return median