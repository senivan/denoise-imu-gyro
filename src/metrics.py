import torch
import numpy as np
from src.lie_algebra import SO3


def compute_alignment(R_gt, R_est):
    if isinstance(R_est, list):
        R_est = torch.stack(R_est, dim=0)
    if isinstance(R_gt, list):
        R_gt = torch.stack(R_gt, dim=0)
    
    # Debug: ensure dimensions are correct
    if R_est.dim() != 3 or R_est.shape[1:] != (3, 3):
        raise ValueError("R_est must be of shape (M,3,3), got {}".format(R_est.shape))
    
    R_est0_inv = torch.inverse(R_est[0])
    alignment = torch.matmul(R_gt[0], R_est0_inv)
    R_est_aligned = torch.matmul(alignment, R_est)
    return R_est_aligned

def compute_aoe(R_gt, R_est):
    M = R_gt.shape[0]
    R_est_aligned = compute_alignment(R_gt, R_est)
    error_sum = 0.0
    for n in range(M):
        R_diff = torch.matmul(R_gt[n].transpose(0, 1), R_est_aligned[n])
        log_R = SO3.log(R_diff)
        error_sim += torch.norm(log_R) ** 2
    return torch.sqrt(error_sum / M)

def cumulative_distance(positions):
    dists = np.linalg.norm(np.diff(positions, axis=0), axis = 1)
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
    disp_pairs = find_disp(R_gt, 7)
    errors = []
    for (n, g) in disp_pairs:
        delta_r_gt  = torch.matmul(R_gt[n].transpose(0, 1), R_gt[g])
        delta_r_est = delta_R_est = torch.matmul(R_est[n].transpose(0, 1), R_est[g])
        R_diff = torch.matmul(delta_R_gt, torch.inverse(delta_R_est))
        log_R = SO3.log(R_diff)
        errors.append(torch.norm(log_R))
    errors_tensor = torch.stack(errors)
    median = torch.median(errors_tensor)
    return median
