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
        Xs = SO3.exp(xs[:, ::self.min_train_freq].reshape(-1, 3).double())
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
        masks = xs[:, :, 3].unsqueeze(1)
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


class LossImprovedAOE_ROE(BaseLoss):
    """Loss calculation with integrated AOE/ROE terms."""
    def __init__(self, w, min_N, max_N, dt, target, huber, lambda_aoe=0.2, lambda_huber=0.1):
        super().__init__(min_N, max_N, dt)
        self.w = w
        self.sl = torch.nn.SmoothL1Loss()
        self.lambda_aoe = lambda_aoe    # Weight for the direct orientation error (AOE/ROE)
        self.lambda_huber = lambda_huber  # Weight for the decimation/huber terms
        if target == 'rotation matrix':
            self.forward = self.forward_with_rotation_matrices
        elif target == 'quaternion':
            self.forward = self.forward_with_quaternions
        elif target == 'rotation matrix mask':
            self.forward = self.forward_with_rotation_matrices_mask
        elif target == 'quaternion mask':
            self.forward = self.forward_with_quaternion_mask
        self.huber = huber
        # Weight for computing convolutional masks
        self.weight = torch.ones(1, 1, self.min_train_freq).cuda() / self.min_train_freq
        self.N0 = 5  # Remove first N0 increments due to padding

    def f_huber(self, rs):
        """Compute the Huber (SmoothL1) loss term."""
        loss = self.w * self.sl(rs / self.huber, torch.zeros_like(rs)) * (self.huber ** 2)
        return loss

    def compute_aoe_roe(self, predicted, true):
        """
        Compute the orientation error between predicted and true rotations.
        Both predicted and true are assumed to be tensors of shape (M, 3, 3).
        Returns the mean squared angular error.
        """
        # Compute relative rotation: R_rel = predicted^T * true
        relative_rotation = torch.bmm(predicted.transpose(1, 2), true)
        # Calculate the trace of each relative rotation matrix
        trace = torch.diagonal(relative_rotation, dim1=-2, dim2=-1).sum(-1)
        # Compute cosine of the rotation angle
        cos_angle = (trace - 1) / 2
        # Clamp for numerical stability and compute angle in radians
        angle = torch.acos(torch.clamp(cos_angle, -1.0, 1.0))
        # Return mean squared error (squared angle)
        return torch.mean(angle ** 2)

    def forward_with_rotation_matrices(self, xs, hat_xs):
        """
        Forward method when using rotation matrices.
        xs: ground truth increments (tensor of shape (N_samples, 3) or convertible to rotation matrices)
        hat_xs: network output (tensor of shape (N_samples, 3))
        """
        N = xs.shape[0]
        # Convert the ground truth increments to rotation matrices.
        Xs = SO3.exp(xs[:, ::self.min_train_freq].reshape(-1, 3).double())
        # Scale network output by dt and reshape.
        hat_xs = self.dt * hat_xs.reshape(-1, 3).double()
        # Convert network output to rotation matrices.
        Omegas = SO3.exp(hat_xs[:, :3])
        # Decimation: compute increment at min_train_freq.
        for k in range(self.min_N):
            Omegas = Omegas[::2].bmm(Omegas[1::2])
        predicted_rotations = Omegas
        true_rotations = Xs
        # Compute the direct orientation error (AOE/ROE) term.
        aoe_error = self.compute_aoe_roe(predicted_rotations, true_rotations)
        total_loss = self.lambda_aoe * aoe_error
        # Compute additional loss terms by further decimating over scales.
        for k in range(self.min_N, self.max_N):
            Omegas = Omegas[::2].bmm(Omegas[1::2])
            Xs = Xs[::2].bmm(Xs[1::2])
            rs = SO3.log(bmtm(Omegas, Xs)).reshape(N, -1, 3)[:, self.N0:]
            huber_term = self.f_huber(rs) / (2 ** (k - self.min_N + 1))
            total_loss += self.lambda_huber * huber_term
        return total_loss

    def forward_with_quaternions(self, xs, hat_xs):
        """
        Forward method when using quaternions.
        xs: ground truth increments (tensor of shape (N_samples, 3) in rotation vector form)
        hat_xs: network output (tensor of shape (N_samples, 3))
        """
        N = xs.shape[0]
        Xs = SO3.qexp(xs[:, ::self.min_train_freq].reshape(-1, 3).double())
        hat_xs = self.dt * hat_xs.reshape(-1, 3).double()
        Omegas = SO3.qexp(hat_xs[:, :3])
        for k in range(self.min_N):
            Omegas = SO3.qmul(Omegas[::2], Omegas[1::2])
        predicted_rotations = Omegas
        true_rotations = Xs
        aoe_error = self.compute_aoe_roe(predicted_rotations, true_rotations)
        total_loss = self.lambda_aoe * aoe_error
        for k in range(self.min_N, self.max_N):
            Omegas = SO3.qmul(Omegas[::2], Omegas[1::2])
            Xs = SO3.qmul(Xs[::2], Xs[1::2])
            rs = SO3.qlog(SO3.qmul(SO3.qinv(Omegas), Xs))
            rs = rs.view(N, -1, 3)[:, self.N0:]
            huber_term = self.f_huber(rs) / (2 ** (k - self.min_N + 1))
            total_loss += self.lambda_huber * huber_term
        return total_loss

    def forward_with_rotation_matrices_mask(self, xs, hat_xs):
        """
        Forward method for rotation matrices with an additional mask.
        xs: ground truth increments with mask information (last channel holds mask)
        hat_xs: network output
        """
        N = xs.shape[0]
        masks = xs[:, :, 3].unsqueeze(1)
        masks = torch.nn.functional.conv1d(masks, self.weight, bias=None,
                                            stride=self.min_train_freq).double().transpose(1, 2)
        masks[masks < 1] = 0
        Xs = SO3.exp(xs[:, ::self.min_train_freq, :3].reshape(-1, 3).double())
        hat_xs = self.dt * hat_xs.reshape(-1, 3).double()
        Omegas = SO3.exp(hat_xs[:, :3])
        for k in range(self.min_N):
            Omegas = Omegas[::2].bmm(Omegas[1::2])
        predicted_rotations = Omegas
        true_rotations = Xs
        aoe_error = self.compute_aoe_roe(predicted_rotations, true_rotations)
        total_loss = self.lambda_aoe * aoe_error
        for k in range(self.min_N, self.max_N):
            Omegas = Omegas[::2].bmm(Omegas[1::2])
            Xs = Xs[::2].bmm(Xs[1::2])
            masks = masks[:, ::2] * masks[:, 1::2]
            rs = SO3.log(bmtm(Omegas, Xs)).reshape(N, -1, 3)[:, self.N0:]
            rs = rs[masks[:, self.N0:].squeeze(2) == 1]
            total_loss += self.lambda_huber * (self.f_huber(rs[:, 2]) / (2 ** (k - self.min_N + 1)))
        return total_loss

    def forward_with_quaternion_mask(self, xs, hat_xs):
        """
        Forward method for quaternions with an additional mask.
        xs: ground truth increments with mask (last channel holds mask information)
        hat_xs: network output
        """
        N = xs.shape[0]
        masks = xs[:, :, 3].unsqueeze(1)
        masks = torch.nn.functional.conv1d(masks, self.weight, bias=None,
                                            stride=self.min_train_freq).double().transpose(1, 2)
        masks[masks < 1] = 0
        Xs = SO3.qexp(xs[:, ::self.min_train_freq, :3].reshape(-1, 3).double())
        hat_xs = self.dt * hat_xs.reshape(-1, 3).double()
        Omegas = SO3.qexp(hat_xs[:, :3])
        for k in range(self.min_N):
            Omegas = SO3.qmul(Omegas[::2], Omegas[1::2])
        predicted_rotations = Omegas
        true_rotations = Xs
        aoe_error = self.compute_aoe_roe(predicted_rotations, true_rotations)
        total_loss = self.lambda_aoe * aoe_error
        for k in range(self.min_N, self.max_N):
            Omegas = SO3.qmul(Omegas[::2], Omegas[1::2])
            Xs = SO3.qmul(Xs[::2], Xs[1::2])
            masks = masks[:, ::2] * masks[:, 1::2]
            rs = SO3.qlog(SO3.qmul(SO3.qinv(Omegas), Xs))
            rs = rs.view(N, -1, 3)[:, self.N0:]
            rs = rs[masks[:, self.N0:].squeeze(2) == 1]
            total_loss += self.lambda_huber * (self.f_huber(rs) / (2 ** (k - self.min_N + 1)))
        return total_loss

