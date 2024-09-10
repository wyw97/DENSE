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
    

class SystemPredictedTeacherForcingTripleTime(System):
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
        shift_est_targets[:, :shift_size] = (1e-6)*torch.rand_like(shift_est_targets[:, :shift_size])
        with torch.no_grad():
            est_targets_2 = self(inputs, enrolls, shift_est_targets)
        est_targets_2 = est_targets_2.squeeze(1)
        shift_est_targets_2 = torch.roll(est_targets_2, shifts=shift_size, dims=-1)
        shift_est_targets_2[:, :shift_size] = (1e-6)*torch.rand_like(shift_est_targets_2[:, :shift_size])
        # random number from 0 to 1
        # random_zero_one = torch.rand(1)
        # if random_zero_one < 0.5:
        #     # set zero for shift_est_targets_2
        #     shift_est_targets_2 = torch.zeros_like(shift_est_targets_2)
        if self.training:
            self.train()
        est_targets = self(inputs, enrolls, shift_est_targets_2)
        loss = self.loss_func(est_targets, targets)
        return loss


class SystemPredictedTeacherForcingAutoRegression(System):
    def common_step(self, batch, batch_nb, train=True):
        inputs, targets, enrolls, predicted = batch
        # print("shape: ", inputs.shape, targets.shape, enrolls.shape, predicted.shape)
        # shape: torch.Size([6, 24000]) torch.Size([6, 1, 24000]) torch.Size([6, 24000]) torch.Size([6, 24000])
        # self.eval()
        iterative_time = self.current_epoch // 50
        shift_est_targets = predicted.clone()
        for _ in range(iterative_time):
            with torch.no_grad():
                est_targets = self(inputs, enrolls, shift_est_targets)
            est_targets = est_targets.squeeze(1)
            shift_size = 128
            shift_est_targets = torch.roll(est_targets, shifts=shift_size, dims=-1)
            shift_est_targets[:, :shift_size] = 0.0*torch.rand_like(shift_est_targets[:, :shift_size])
        # get random number to set shift_est_targets as zero
        # rand_num = torch.rand(1)
        # if rand_num < 0.7:
        #     shift_est_targets = torch.zeros_like(shift_est_targets)
        # if self.training:
        #     self.train()
        est_targets = self(inputs, enrolls, shift_est_targets)
        loss = self.loss_func(est_targets, targets)
        return loss


class SystemPredictedTeacherForcingAutoRegression2chn(System):
    def common_step(self, batch, batch_nb, train=True):
        inputs, targets, enrolls, predicted = batch
        # print("shape: ", inputs.shape, targets.shape, enrolls.shape, predicted.shape)
        # shape: torch.Size([6, 24000]) torch.Size([6, 1, 24000]) torch.Size([6, 24000]) torch.Size([6, 24000])
        # self.eval()
        iterative_time = self.current_epoch // 50
        shift_est_targets = predicted.clone()
        # concatenate two channels
        # In the provided code, `shift_size` is a parameter used to determine the amount of shifting
        # applied to a tensor along a specific dimension. It is used in the context of rolling the
        # tensor along a particular axis.
        shift_size = 32
        shift_inputs = torch.roll(inputs.unsqueeze(1), shifts=shift_size, dims=-1)
        mix_inputs_predicts = torch.cat((shift_inputs, predicted.unsqueeze(1)), dim=1) # shape: torch.Size([6, 2, 24000])
        for _ in range(iterative_time):
            with torch.no_grad():
                est_targets = self(inputs, enrolls, mix_inputs_predicts)
            est_targets = est_targets.squeeze(1)
            
            shift_est_targets = torch.roll(est_targets, shifts=shift_size, dims=-1)
            shift_est_targets[:, :shift_size] = 0.0*torch.rand_like(shift_est_targets[:, :shift_size])
            mix_inputs_predicts = torch.cat((shift_inputs, shift_est_targets.unsqueeze(1)), dim=1)
        # get random number to set shift_est_targets as zero
        # rand_num = torch.rand(1)
        # if rand_num < 0.7:
            # mix_inputs_predicts = torch.zeros_like(mix_inputs_predicts)
            # mix_inputs_predicts[:, 1, :] = 0.0*torch.rand_like(mix_inputs_predicts[:, 1, :])
        # if self.training:
        #     self.train()
        est_targets = self(inputs, enrolls, mix_inputs_predicts)
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