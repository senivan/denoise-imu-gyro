import numpy as np
import torch
from src.utils import bmmt, bmv, bmtv, bbmv, bmtm
from src.lie_algebra import SO3
import matplotlib.pyplot as plt


class BaseLoss(torch.nn.Module):

    def __init__(self, min_N, max_N, dt):
        super().__init__()
        # windows sizes
        self.min_N = min_N
        self.max_N = max_N
        self.min_train_freq = 2 ** self.min_N
        self.max_train_freq = 2 ** self.max_N
        # sampling time
        self.dt = dt # (s)


class GyroLoss(BaseLoss):
    """Loss for low-frequency orientation increment"""

    def __init__(self, w, min_N, max_N, dt, target, huber):
        super().__init__(min_N, max_N, dt)
        # weights on loss
        self.w = w
        self.sl = torch.nn.SmoothL1Loss()
        if target == 'rotation matrix':
            self.forward = self.forward_with_rotation_matrices
        elif target == 'quaternion':
            self.forward = self.forward_with_quaternions
        elif target == 'rotation matrix mask':
            self.forward = self.forward_with_rotation_matrices_mask
        elif target == 'quaternion mask':
            self.forward = self.forward_with_quaternion_mask
        self.huber = huber
        self.weight = torch.ones(1, 1,
            self.min_train_freq).cuda()/self.min_train_freq
        self.N0 = 5 # remove first N0 increment in loss due not account padding

    def f_huber(self, rs):
        """Huber loss function"""
        loss = self.w*self.sl(rs/self.huber,
            torch.zeros_like(rs))*(self.huber**2)
        return loss

    def forward_with_rotation_matrices(self, xs, hat_xs):
        """Forward errors with rotation matrices"""
        N = xs.shape[0]
        # Xs = SO3.exp(xs[:, ::self.min_train_freq].reshape(-1, 3).double())
        Xs = SO3.exp(xs[:, ::self.min_train_freq, :3].reshape(-1, 3).double())
        hat_xs = self.dt*hat_xs.reshape(-1, 3).double()
        Omegas = SO3.exp(hat_xs[:, :3])
        # compute increment at min_train_freq by decimation
        for k in range(self.min_N):
            Omegas = Omegas[::2].bmm(Omegas[1::2])
        rs = SO3.log(bmtm(Omegas, Xs)).reshape(N, -1, 3)[:, self.N0:]
        loss = self.f_huber(rs)
        # compute increment from min_train_freq to max_train_freq
        for k in range(self.min_N, self.max_N):
            Omegas = Omegas[::2].bmm(Omegas[1::2])
            Xs = Xs[::2].bmm(Xs[1::2])
            rs = SO3.log(bmtm(Omegas, Xs)).reshape(N, -1, 3)[:, self.N0:]
            loss = loss + self.f_huber(rs)/(2**(k - self.min_N + 1))
        return loss

    def forward_with_quaternions(self, xs, hat_xs):
        """Forward errors with quaternion"""
        N = xs.shape[0]
        Xs = SO3.qexp(xs[:, ::self.min_train_freq].reshape(-1, 3).double())
        hat_xs = self.dt*hat_xs.reshape(-1, 3).double()
        Omegas = SO3.qexp(hat_xs[:, :3])
        # compute increment at min_train_freq by decimation
        for k in range(self.min_N):
            Omegas = SO3.qmul(Omegas[::2], Omegas[1::2])
        rs = SO3.qlog(SO3.qmul(SO3.qinv(Omegas), Xs)).reshape(N,
                -1, 3)[:, self.N0:]
        loss = self.f_huber(rs)
        # compute increment from min_train_freq to max_train_freq
        for k in range(self.min_N, self.max_N):
            Omegas = SO3.qmul(Omegas[::2], Omegas[1::2])
            Xs = SO3.qmul(Xs[::2], Xs[1::2])
            rs = SO3.qlog(SO3.qmul(SO3.qinv(Omegas), Xs))
            rs = rs.view(N, -1, 3)[:, self.N0:]
            loss = loss + self.f_huber(rs)/(2**(k - self.min_N + 1))
        return loss

    def forward_with_rotation_matrices_mask(self, xs, hat_xs):
        """Forward errors with rotation matrices"""
        N = xs.shape[0]
        masks = xs[:, :, 2].unsqueeze(1)
        # print("Masks: ", masks)
        masks = torch.nn.functional.conv1d(masks, self.weight, bias=None,
            stride=self.min_train_freq).double().transpose(1, 2)
        masks[masks < 1] = 0
        Xs = SO3.exp(xs[:, ::self.min_train_freq, :3].reshape(-1, 3).double())
        hat_xs = self.dt*hat_xs.reshape(-1, 3).double()
        Omegas = SO3.exp(hat_xs[:, :3])
        # compute increment at min_train_freq by decimation
        for k in range(self.min_N):
            Omegas = Omegas[::2].bmm(Omegas[1::2])
        rs = SO3.log(bmtm(Omegas, Xs)).reshape(N, -1, 3)[:, self.N0:]
        loss = self.f_huber(rs)
        # compute increment from min_train_freq to max_train_freq
        for k in range(self.min_N, self.max_N):
            Omegas = Omegas[::2].bmm(Omegas[1::2])
            Xs = Xs[::2].bmm(Xs[1::2])
            masks = masks[:, ::2] * masks[:, 1::2]
            rs = SO3.log(bmtm(Omegas, Xs)).reshape(N, -1, 3)[:, self.N0:]
            rs = rs[masks[:, self.N0:].squeeze(2) == 1]
            loss = loss + self.f_huber(rs[:,2])/(2**(k - self.min_N + 1))
        return loss

    def forward_with_quaternion_mask(self, xs, hat_xs):
        """Forward errors with quaternion"""
        N = xs.shape[0]
        masks = xs[:, :, 3].unsqueeze(1)
        masks = torch.nn.functional.conv1d(masks, self.weight, bias=None,
            stride=self.min_train_freq).double().transpose(1, 2)
        masks[masks < 1] = 0
        Xs = SO3.qexp(xs[:, ::self.min_train_freq, :3].reshape(-1, 3).double())
        hat_xs = self.dt*hat_xs.reshape(-1, 3).double()
        Omegas = SO3.qexp(hat_xs[:, :3])
        # compute increment at min_train_freq by decimation
        for k in range(self.min_N):
            Omegas = SO3.qmul(Omegas[::2], Omegas[1::2])
        rs = SO3.qlog(SO3.qmul(SO3.qinv(Omegas), Xs)).reshape(N,
                -1, 3)[:, self.N0:]
        rs = rs[masks[:, self.N0:].squeeze(2) == 1]
        loss = self.f_huber(rs)
        # compute increment from min_train_freq to max_train_freq
        for k in range(self.min_N, self.max_N):
            Omegas = SO3.qmul(Omegas[::2], Omegas[1::2])
            Xs = SO3.qmul(Xs[::2], Xs[1::2])
            masks = masks[:, ::2] * masks[:, 1::2]
            rs = SO3.qlog(SO3.qmul(SO3.qinv(Omegas), Xs)).reshape(N,
                -1, 3)[:, self.N0:]
            rs = rs[masks[:, self.N0:].squeeze(2) == 1]
            loss = loss + self.f_huber(rs)/(2**(k - self.min_N + 1))
        return loss


def quat_to_rot(q):
    """
    Converts a batch of quaternions in [x, y, z, w] format (shape: [B, 4])
    to rotation matrices of shape [B, 3, 3].
    """
    # Normalize the quaternion to ensure unit norm
    q = q / q.norm(dim=-1, keepdim=True)
    x, y, z, w = q.unbind(-1)
    # Compute each element of the rotation matrix
    m00 = 1 - 2*(y**2 + z**2)
    m01 = 2*(x*y - z*w)
    m02 = 2*(x*z + y*w)
    m10 = 2*(x*y + z*w)
    m11 = 1 - 2*(x**2 + z**2)
    m12 = 2*(y*z - x*w)
    m20 = 2*(x*z - y*w)
    m21 = 2*(y*z + x*w)
    m22 = 1 - 2*(x**2 + y**2)
    rot = torch.stack([
        torch.stack([m00, m01, m02], dim=-1),
        torch.stack([m10, m11, m12], dim=-1),
        torch.stack([m20, m21, m22], dim=-1)
    ], dim=-2)
    return rot

class LossImprovedAOE_ROE(BaseLoss):
    """Loss calculation with integrated AOE/ROE terms.
       This loss converts ground truth rotations from quaternion form to rotation matrices if needed.
    """
    def __init__(self, w, min_N, max_N, dt, target, huber, lambda_aoe=0.1, lambda_huber=0.1):
        super().__init__(min_N, max_N, dt)
        self.w = w
        self.sl = torch.nn.SmoothL1Loss()
        self.lambda_aoe = lambda_aoe    # Weight for the direct orientation error (AOE/ROE)
        self.lambda_huber = lambda_huber  # Weight for the decimation (SmoothL1) terms
        if target == 'rotation matrix':
            self.forward = self.forward_with_rotation_matrices
        elif target == 'quaternion':
            self.forward = self.forward_with_quaternions
        elif target == 'rotation matrix mask':
            self.forward = self.forward_with_rotation_matrices_mask
        elif target == 'quaternion mask':
            self.forward = self.forward_with_quaternion_mask
        self.huber = huber
        self.weight = torch.ones(1, 1, self.min_train_freq).cuda() / self.min_train_freq
        self.N0 = 5  # Remove first N0 increments due to padding

    def f_huber(self, rs):
        loss = self.w * self.sl(rs / self.huber, torch.zeros_like(rs)) * (self.huber ** 2)
        return loss

    def compute_aoe_roe(self, predicted, true):
        """
        Compute the mean squared angular error between predicted and true rotations.
        If either tensor has a last dimension of 4, it is assumed to be in quaternion form
        and is converted to a rotation matrix using SO3.from_quaternion with ordering 'wxyz'.
        After conversion, both tensors should have shape (M, 3, 3).
        """
        # If predicted is in quaternion form (shape [M,4]), convert it.
        if predicted.dim() == 2 and predicted.shape[1] == 4:
            predicted = SO3.from_quaternion(predicted, ordering='wxyz')
        # If true is in quaternion form (shape [M,4]), convert it.
        if true.dim() == 2 and true.shape[1] == 4:
            true = SO3.from_quaternion(true, ordering='wxyz')
        # If the inputs already have a batch dimension but with quaternions (e.g., [B, T, 4]),
        # you might need to reshape them. Here we assume inputs to this function are 2D tensors.
        # Crop both tensors to the minimum length in case they differ.
        min_len = min(predicted.shape[0], true.shape[0])
        predicted = predicted[:min_len]
        true = true[:min_len]
        # Now both tensors should have shape (M, 3, 3)
        relative_rotation = torch.bmm(predicted.transpose(1, 2), true)
        trace = torch.diagonal(relative_rotation, dim1=-2, dim2=-1).sum(-1)
        cos_angle = (trace - 1) / 2
        angle = torch.acos(torch.clamp(cos_angle, -1.0, 1.0))
        return torch.mean(angle ** 2)

    def forward_with_rotation_matrices(self, xs, hat_xs):
        """
        Forward method for rotation matrices and integrated AOE/ROE.
        xs: ground truth increments (tensor of shape (N, T, 4)); only the first 3 channels are used.
        hat_xs: network output (tensor of shape (N, T, 3))
        """
        N = xs.shape[0]
        # Use only the first 3 channels from xs
        Xs = SO3.exp(xs[:, ::self.min_train_freq, :3].reshape(-1, 3).double())
        hat_xs = self.dt * hat_xs.reshape(-1, 3).double()
        Omegas = SO3.exp(hat_xs[:, :3])
        
        # Decimation loop for min_N steps
        for k in range(self.min_N):
            if Omegas.size(0) % 2 == 1:
                Omegas = Omegas[:-1]
            Omegas = Omegas[::2].bmm(Omegas[1::2])
        
        predicted_rotations = Omegas
        true_rotations = Xs
        aoe_error = self.compute_aoe_roe(predicted_rotations, true_rotations)
        total_loss = self.lambda_aoe * aoe_error
        
        # Further decimation from min_N to max_N
        for k in range(self.min_N, self.max_N):
            if Omegas.size(0) % 2 == 1:
                Omegas = Omegas[:-1]
            if Xs.size(0) % 2 == 1:
                Xs = Xs[:-1]
            Omegas = Omegas[::2].bmm(Omegas[1::2])
            Xs = Xs[::2].bmm(Xs[1::2])
            # Crop to minimum length to ensure matching dimensions
            min_length = min(Omegas.size(0), Xs.size(0))
            Omegas = Omegas[:min_length]
            Xs = Xs[:min_length]
            rs = SO3.log(bmtm(Omegas, Xs)).reshape(N, -1, 3)[:, self.N0:]
            huber_term = self.f_huber(rs) / (2 ** (k - self.min_N + 1))
            total_loss += self.lambda_huber * huber_term
        return total_loss

    def forward_with_quaternions(self, xs, hat_xs):
        """
        Forward method for quaternions and integrated AOE/ROE.
        xs: ground truth increments (tensor of shape (N, T, 4)); use only the first 3 channels.
        hat_xs: network output (tensor of shape (N, T, 3))
        """
        N = xs.shape[0]
        Xs = SO3.qexp(xs[:, ::self.min_train_freq, :3].reshape(-1, 3).double())
        hat_xs = self.dt * hat_xs.reshape(-1, 3).double()
        Omegas = SO3.qexp(hat_xs[:, :3])
        for k in range(self.min_N):
            if Omegas.size(0) % 2 == 1:
                Omegas = Omegas[:-1]
            Omegas = SO3.qmul(Omegas[::2], Omegas[1::2])
        predicted_rotations = Omegas
        true_rotations = Xs
        aoe_error = self.compute_aoe_roe(predicted_rotations, true_rotations)
        total_loss = self.lambda_aoe * aoe_error
        for k in range(self.min_N, self.max_N):
            if Omegas.size(0) % 2 == 1:
                Omegas = Omegas[:-1]
            if Xs.size(0) % 2 == 1:
                Xs = Xs[:-1]
            Omegas = SO3.qmul(Omegas[::2], Omegas[1::2])
            Xs = SO3.qmul(Xs[::2], Xs[1::2])
            min_length = min(Omegas.size(0), Xs.size(0))
            Omegas = Omegas[:min_length]
            Xs = Xs[:min_length]
            rs = SO3.qlog(SO3.qmul(SO3.qinv(Omegas), Xs))
            rs = rs.view(N, -1, 3)[:, self.N0:]
            huber_term = self.f_huber(rs) / (2 ** (k - self.min_N + 1))
            total_loss += self.lambda_huber * huber_term
        return total_loss

    def forward_with_rotation_matrices_mask(self, xs, hat_xs):
        """
        Forward method for rotation matrices with mask and integrated AOE/ROE.
        xs: ground truth increments with mask (tensor of shape (N, T, 4)); if no mask is present, a dummy mask is used.
        hat_xs: network output (tensor of shape (N, T, 3))
        """
        N = xs.shape[0]
        if xs.shape[2] < 4:
            masks = torch.ones(xs.shape[0], xs.shape[1], 1, device=xs.device)
        else:
            masks = xs[:, :, 3].unsqueeze(1)
        masks = torch.nn.functional.conv1d(masks, self.weight, bias=None,
                                            stride=self.min_train_freq).double().transpose(1, 2)
        masks[masks < 1] = 0
        Xs = SO3.exp(xs[:, ::self.min_train_freq, :3].reshape(-1, 3).double())
        hat_xs = self.dt * hat_xs.reshape(-1, 3).double()
        Omegas = SO3.exp(hat_xs[:, :3])
        for k in range(self.min_N):
            if Omegas.size(0) % 2 == 1:
                Omegas = Omegas[:-1]
            Omegas = Omegas[::2].bmm(Omegas[1::2])
        predicted_rotations = Omegas
        true_rotations = Xs
        aoe_error = self.compute_aoe_roe(predicted_rotations, true_rotations)
        total_loss = self.lambda_aoe * aoe_error
        for k in range(self.min_N, self.max_N):
            if Omegas.size(0) % 2 == 1:
                Omegas = Omegas[:-1]
            if Xs.size(0) % 2 == 1:
                Xs = Xs[:-1]
            Omegas = Omegas[::2].bmm(Omegas[1::2])
            Xs = Xs[::2].bmm(Xs[1::2])
            if masks.size(1) % 2 == 1:
                masks = masks[:, :-1]
            masks = masks[:, ::2] * masks[:, 1::2]
            min_length = min(Omegas.size(0), Xs.size(0))
            Omegas = Omegas[:min_length]
            Xs = Xs[:min_length]
            rs = SO3.log(bmtm(Omegas, Xs)).reshape(N, -1, 3)[:, self.N0:]
            rs = rs[masks[:, self.N0:].squeeze(2) == 1]
            total_loss += self.lambda_huber * (self.f_huber(rs[:, 2]) / (2 ** (k - self.min_N + 1)))
        return total_loss

    def forward_with_quaternion_mask(self, xs, hat_xs):
        """
        Forward method for quaternions with mask and integrated AOE/ROE.
        xs: ground truth increments with mask (tensor of shape (N, T, 4)); if no mask is present, a dummy mask is used.
        hat_xs: network output (tensor of shape (N, T, 3))
        """
        N = xs.shape[0]
        if xs.shape[2] < 4:
            masks = torch.ones(xs.shape[0], xs.shape[1], 1, device=xs.device)
        else:
            masks = xs[:, :, 3].unsqueeze(1)
        masks = torch.nn.functional.conv1d(masks, self.weight, bias=None,
                                            stride=self.min_train_freq).double().transpose(1, 2)
        masks[masks < 1] = 0
        Xs = SO3.qexp(xs[:, ::self.min_train_freq, :3].reshape(-1, 3).double())
        hat_xs = self.dt * hat_xs.reshape(-1, 3).double()
        Omegas = SO3.qexp(hat_xs[:, :3])
        for k in range(self.min_N):
            if Omegas.size(0) % 2 == 1:
                Omegas = Omegas[:-1]
            Omegas = SO3.qmul(Omegas[::2], Omegas[1::2])
        predicted_rotations = Omegas
        true_rotations = Xs
        aoe_error = self.compute_aoe_roe(predicted_rotations, true_rotations)
        total_loss = self.lambda_aoe * aoe_error
        for k in range(self.min_N, self.max_N):
            if Omegas.size(0) % 2 == 1:
                Omegas = Omegas[:-1]
            if Xs.size(0) % 2 == 1:
                Xs = Xs[:-1]
            Omegas = SO3.qmul(Omegas[::2], Omegas[1::2])
            Xs = SO3.qmul(Xs[::2], Xs[1::2])
            min_length = min(Omegas.size(0), Xs.size(0))
            Omegas = Omegas[:min_length]
            Xs = Xs[:min_length]
            masks = masks[:, ::2] * masks[:, 1::2]
            rs = SO3.qlog(SO3.qmul(SO3.qinv(Omegas), Xs))
            rs = rs.view(N, -1, 3)[:, self.N0:]
            huber_term = self.f_huber(rs) / (2 ** (k - self.min_N + 1))
            total_loss += self.lambda_huber * huber_term
        return total_loss