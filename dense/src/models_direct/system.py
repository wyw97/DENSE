# All rights reserved
# By Yiwen Wang, May 2024.
import torch
from asteroid.engine import System

class SystemInformed(System):
    def common_step(self, batch, batch_nb, train=True):
        inputs, targets, enrolls = batch
        
        est_targets = self(inputs, enrolls)
        # if not self.training:
        #     print("input shape: ", inputs.shape, est_targets.shape, est_targets[0, 0, :100])
        loss = self.loss_func(est_targets, targets)
        # loss = self.loss_func(est_targets.squeeze(1), targets.squeeze(1))
        return loss


class SystemPredicted(System):
    def common_step(self, batch, batch_nb, train=True):
        inputs, targets, enrolls, predicted = batch
        # print("shape: ", inputs.shape, targets.shape, enrolls.shape, predicted.shape)
        # shape: torch.Size([6, 24000]) torch.Size([6, 1, 24000]) torch.Size([6, 24000]) torch.Size([6, 24000])
        est_targets = self(inputs, enrolls, predicted)
        loss = self.loss_func(est_targets, targets)
        return loss
    

class SystemPredictedTeacherForcing(System):
    def common_step(self, batch, batch_nb, train=True):
        inputs, targets, enrolls, predicted = batch
        # print("shape: ", inputs.shape, targets.shape, enrolls.shape, predicted.shape)
        # shape: torch.Size([6, 24000]) torch.Size([6, 1, 24000]) torch.Size([6, 24000]) torch.Size([6, 24000])
        self.eval()
        with torch.no_grad():
            est_targets = self(inputs, enrolls, predicted)
        est_targets = est_targets.squeeze(1)
        shift_size = 128
        shift_est_targets = torch.roll(est_targets, shifts=shift_size, dims=-1)
        shift_est_targets[:, :shift_size] = 0
        mask_ratio = 1 - int(self.current_epoch / 15) * 0.1
        if mask_ratio < 0:
            mask_ratio = 0
        mask_gene = torch.rand_like(shift_est_targets) < mask_ratio
        mask_predict_input = torch.where(mask_gene, predicted, shift_est_targets)
        if self.training:
            self.train()
        est_targets = self(inputs, enrolls, mask_predict_input)
        loss = self.loss_func(est_targets, targets)
        return loss

class SystemPredictedTeacherForcingAutoRegression(System):
    def common_step(self, batch, batch_nb, train=True):
        inputs, targets, enrolls, predicted = batch
        iterative_time = self.current_epoch // 50
        shift_est_targets = predicted.clone()
        for _ in range(iterative_time):
            with torch.no_grad():
                est_targets = self(inputs, enrolls, shift_est_targets)
            est_targets = est_targets.squeeze(1)
            shift_size = 128
            shift_est_targets = torch.roll(est_targets, shifts=shift_size, dims=-1)
            shift_est_targets[:, :shift_size] = 0.0*torch.rand_like(shift_est_targets[:, :shift_size])
        est_targets = self(inputs, enrolls, shift_est_targets)
        loss = self.loss_func(est_targets, targets)
        return loss


class SystemPredictedPairs(System):
    def common_step(self, batch, batch_nb, train=True):
        inputs, targets, enrolls, predicted = batch
        shift_size = 16
        est_targets_first = self(inputs, enrolls, torch.zeros_like(predicted))
        est_targets_squeeze = est_targets_first.squeeze(1)
        shift_est_targets = torch.roll(est_targets_squeeze, shifts=shift_size, dims=-1)
        shift_est_targets[:, :shift_size] = 0.0*torch.rand_like(shift_est_targets[:, :shift_size])
        est_targets = self(inputs, enrolls, shift_est_targets)
        alpha_loss = 0.25 
        loss = alpha_loss*self.loss_func(est_targets_first, targets) + (1-alpha_loss)*self.loss_func(est_targets, targets)
        return loss
