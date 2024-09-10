# Copyright (c) 2021 Brno University of Technology
# Copyright (c) 2021 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved
# By Katerina Zmolikova, August 2021.

import torch
from asteroid.utils.torch_utils import pad_x_to_y, jitable_shape
from asteroid.models.base_models import _shape_reconstructed, _unsqueeze_to_3d
from asteroid.models.base_models import BaseEncoderMaskerDecoder
import time


class BaseEncoderMaskerDecoderInformed(BaseEncoderMaskerDecoder):
    """Base class for informed encoder-masker-decoder extraction models.
    Adapted from Asteroid calss BaseEncoderMaskerDecoder
    https://github.com/asteroid-team/asteroid/blob/master/asteroid/models/base_models.py

    Args:
        encoder (Encoder): Encoder instance.
        masker (nn.Module): masked network.
        decoder (Decoder): Decoder instance.
        auxiliary (nn.Module): auxiliary network processing enrollment.
        encoder_activation (optional[str], optional): activation to apply after encoder.
            see ``asteroid.masknn.activations`` for valid values.
    """
    def __init__(self, encoder, masker, decoder, auxiliary,
                 encoder_activation=None):
        super().__init__(encoder, masker, decoder, encoder_activation)
        self.auxiliary = auxiliary

    def forward(self, wav, enrollment):
        """Enc/Mask/Dec model forward

        Args:
            wav (torch.Tensor): waveform tensor. 1D, 2D or 3D tensor, time last.
            enrollment (torch.Tensor): enrollment information tensor. 1D, 2D or 3D tensor.

        Returns:
            torch.Tensor, of shape (batch, 1, time) or (1, time)
        """
        # Remember shape to shape reconstruction, cast to Tensor for torchscript
        shape = jitable_shape(wav)
        # Reshape to (batch, n_mix, time)
        wav = _unsqueeze_to_3d(wav)

        # Real forward
        tf_rep = self.forward_encoder(wav)
        enroll_emb = self.auxiliary(enrollment)
        est_masks = self.forward_masker(tf_rep, enroll_emb)
        masked_tf_rep = self.apply_masks(tf_rep, est_masks)
        decoded = self.forward_decoder(masked_tf_rep)

        reconstructed = pad_x_to_y(decoded, wav)
        return _shape_reconstructed(reconstructed, shape)

    def forward_masker(self, tf_rep: torch.Tensor, enroll: torch.Tensor) -> torch.Tensor:
        """Estimates masks from time-frequency representation.

        Args:
            tf_rep (torch.Tensor): Time-frequency representation in (batch,
                feat, seq).
            enroll (torch.Tensor): Time-frequency representation in (batch,
                feat, seq).

        Returns:
            torch.Tensor: Estimated masks
        """
        return self.masker(tf_rep, enroll)


    def forward_streaming(self, wav, enrollment, kernel_size=16, step_size=8, buffer_reception=1531, buffer_dim=512):
        """Enc/Mask/Dec model forward

        Args:
            wav (torch.Tensor): waveform tensor. 1D, 2D or 3D tensor, time last.
            enrollment (torch.Tensor): enrollment information tensor. 1D, 2D or 3D tensor.

        Returns:
            torch.Tensor, of shape (batch, 1, time) or (1, time)
        """
        # Remember shape to shape reconstruction, cast to Tensor for torchscript
        
        shape = jitable_shape(wav)
        # Reshape to (batch, n_mix, time)
        wav = _unsqueeze_to_3d(wav)
        # print("wav shape: ", wav.shape, wav.device)
        reconstructed = torch.zeros((wav.shape[0], wav.shape[-1]), device=wav.device)
        mixture_emb_buffer = torch.zeros((wav.shape[0], buffer_reception, buffer_dim), device=wav.device)
        # Real forward
        tf_rep = self.forward_encoder(wav)
        enroll_emb = self.auxiliary(enrollment)
        curr_idx = 0
        while curr_idx < tf_rep.shape[-1]:
            # print(curr_idx, wav.device, end=';')
            curr_tf_rep = tf_rep[:, :, curr_idx]
            td_begin = curr_idx * step_size
            td_end = td_begin + kernel_size
            mixture_emb_buffer = torch.roll(mixture_emb_buffer, shifts=-1, dims=1)
            # print("curr tf rep shape: ", curr_tf_rep.shape) # [batch, 512]
            mixture_emb_buffer[:, -1, :] = curr_tf_rep # .squeeze(-1)
            curr_est_mask = self.forward_masker(mixture_emb_buffer.permute(0, 2, 1), enroll_emb)
            # print("curr est mask shape: ", curr_est_mask.shape) # [batch, 1, 512, 1531]
            curr_masked_tf_rep = self.apply_masks(curr_tf_rep, curr_est_mask[..., -1])
            # print("tf rep shape: ", curr_masked_tf_rep.shape) # [6, 1, 512]
            curr_decoded = self.forward_decoder(curr_masked_tf_rep.unsqueeze(-1))
            # print("curr decoded shape: ", curr_decoded.shape)
            reconstructed[:, td_begin:td_end] += curr_decoded[:, 0, :]
            curr_idx += 1
        return _shape_reconstructed(reconstructed, shape)



class BaseEncoderMaskerDecoderInformedOutputHelpPredict(BaseEncoderMaskerDecoder):
    """Base class for informed encoder-masker-decoder extraction models.
    Adapted from Asteroid calss BaseEncoderMaskerDecoder
    https://github.com/asteroid-team/asteroid/blob/master/asteroid/models/base_models.py

    Args:
        encoder (Encoder): Encoder instance.
        masker (nn.Module): masked network.
        decoder (Decoder): Decoder instance.
        auxiliary (nn.Module): auxiliary network processing enrollment.
        encoder_activation (optional[str], optional): activation to apply after encoder.
            see ``asteroid.masknn.activations`` for valid values.
    """
    def __init__(self, encoder, masker, decoder, auxiliary, predict2vec, merge_embed,
                 encoder_activation=None, use_average_emb=False):
        super().__init__(encoder, masker, decoder, encoder_activation)
        self.auxiliary = auxiliary
        self.predict2vec = predict2vec
        self.merge_embed = merge_embed # use for merging two embedding layers
        self.use_average_emb = use_average_emb

    def forward(self, wav, enrollment, predicted_wav):
        """Enc/Mask/Dec model forward

        Args:
            wav (torch.Tensor): waveform tensor. 1D, 2D or 3D tensor, time last.
            enrollment (torch.Tensor): enrollment information tensor. 1D, 2D or 3D tensor.

        Returns:
            torch.Tensor, of shape (batch, 1, time) or (1, time)
        """
        # Remember shape to shape reconstruction, cast to Tensor for torchscript
        shape = jitable_shape(wav)
        # Reshape to (batch, n_mix, time)
        wav = _unsqueeze_to_3d(wav)

        # Real forward
        tf_rep = self.forward_encoder(wav)
        enroll_emb = self.auxiliary(enrollment)
        # print("shape : predict wav: ", predicted_wav.shape)
        predict_emb = self.predict2vec(predicted_wav)

        # add on 04/25/2024 
        # embedding should be cummutive
        if self.use_average_emb:
            cumulative_sum = predict_emb.cumsum(dim=1) 
            T = predict_emb.shape[1]
            timestep = torch.arange(1, T+1, device=predict_emb.device).float()
            predict_emb = cumulative_sum / timestep
        
        combine_emb = self.merge_embed(enroll_emb, predict_emb)
        # print("enroll_emb shape: ", enroll_emb.shape, predict_emb.shape, combine_emb.shape)
        # torch.Size([6, 256]) torch.Size([6, 1, 256, 2999]) torch.Size([6, 2999, 256])
        est_masks = self.forward_masker(tf_rep, combine_emb)
        masked_tf_rep = self.apply_masks(tf_rep, est_masks)
        # print("mask shape: ", tf_rep.shape, combine_emb.shape, masked_tf_rep.shape)
        # mask shape:  torch.Size([1, 512, 3769]) torch.Size([1, 3769, 256]) torch.Size([1, 1, 512, 3769])
        decoded = self.forward_decoder(masked_tf_rep)

        reconstructed = pad_x_to_y(decoded, wav)
        return _shape_reconstructed(reconstructed, shape)


    def forward_streaming(self, wav, enrollment, kernel_size=16, step_size=8, time_delay=16, buffer_length=1531, buffer_dim=512, enroll_dim=256):
        """Enc/Mask/Dec model forward

        Args:
            wav (torch.Tensor): waveform tensor. 1D, 2D or 3D tensor, time last.
            enrollment (torch.Tensor): enrollment information tensor. 1D, 2D or 3D tensor.

        Returns:
            torch.Tensor, of shape (batch, 1, time) or (1, time)
        """
        # Remember shape to shape reconstruction, cast to Tensor for torchscript
        self.eval()
        shape = jitable_shape(wav)
        # Reshape to (batch, n_mix, time)
        wav = _unsqueeze_to_3d(wav)
        # Real forward
        tf_rep = self.forward_encoder(wav) # tf_rep shape:  torch.Size([1, 512, 3769]) 30160/8 = 3770
        # set time-domain predict wav and delay predict wav
        total_predict_wav = torch.zeros((wav.shape[0], wav.shape[-1]), device=wav.device)
        total_predict_wav_delay = torch.zeros((wav.shape[0], wav.shape[-1]+time_delay), device=wav.device)

        # get enroll embedding
        enroll_emb = self.auxiliary(enrollment)
        predict_emb_buffer = torch.zeros((wav.shape[0], buffer_length, buffer_dim), device=wav.device)
        mixture_emb_buffer = torch.zeros((wav.shape[0], buffer_length, buffer_dim), device=wav.device)
        enroll_emb_buffer = torch.zeros((wav.shape[0], buffer_length, enroll_dim), device=wav.device)
        curr_idx = 0
        
        # initialize with enroll embedding
        enroll_tf_emb = self.forward_encoder(enrollment)
        # start_time = time.time()
        # total_frame_time = 0.
        while curr_idx < tf_rep.shape[-1]:
            # curr_idx: frame number
            curr_tf_rep = tf_rep[:, :, curr_idx]
            td_begin = curr_idx * step_size
            td_end = td_begin + kernel_size
            # tt = time.time()
            predict_wav_emb = self.predict2vec[:2](total_predict_wav_delay[:,  td_begin:td_end]) # [1, 512, 1]
            
            predict_emb_buffer = torch.roll(predict_emb_buffer, shifts=-1, dims=1)
            predict_emb_buffer[:, -1, :] = predict_wav_emb.squeeze(-1)
            mixture_emb_buffer = torch.roll(mixture_emb_buffer, shifts=-1, dims=1)
            mixture_emb_buffer[:, -1, :] = curr_tf_rep.squeeze(-1)
            predict_emb = self.predict2vec[2:](predict_emb_buffer.permute(0, 2, 1))
            combine_emb = self.merge_embed(enroll_emb, predict_emb[..., -1:])
            enroll_emb_buffer = torch.roll(enroll_emb_buffer, shifts=-1, dims=1)
            # print("combind emb shape: ", combine_emb.shape)
            enroll_emb_buffer[:, -1, :] = combine_emb.squeeze(1)
            curr_est_mask = self.forward_masker(mixture_emb_buffer.permute(0, 2, 1), enroll_emb_buffer)
            curr_masked_tf_rep = self.apply_masks(curr_tf_rep, curr_est_mask[..., -1])
            curr_decoded = self.forward_decoder(curr_masked_tf_rep.unsqueeze(-1))
            # total_frame_time += time.time() - tt
            total_predict_wav[:, td_begin:td_end] += curr_decoded[:, 0, :]
            total_predict_wav_delay[:, td_begin+time_delay:td_end+time_delay] += curr_decoded[:, 0, :]
            curr_idx += 1
        # end_time = time.time()
        # avg_frame_time = total_frame_time/tf_rep.shape[-1]

        # print(f"Totoal chunk processing time: {total_frame_time:.4f}s")
        # print(f"Averaged processing time per chunk (Processing latency) : {avg_frame_time:.4f}s")
        # print(f"Ideal latency: {kernel_size/8000:.4f}s")
        # print(f"Actual latency: {kernel_size/8000 + avg_frame_time:.4f}s")
        # print(f"Real-time factor (RTF): {avg_frame_time/(step_size/8000):.4f}")
        return _shape_reconstructed(total_predict_wav[:, :], shape)

    def forward_masker(self, tf_rep: torch.Tensor, enroll: torch.Tensor) -> torch.Tensor:
        """Estimates masks from time-frequency representation.

        Args:
            tf_rep (torch.Tensor): Time-frequency representation in (batch,
                feat, seq).
            enroll (torch.Tensor): Time-frequency representation in (batch,
                feat, seq).

        Returns:
            torch.Tensor: Estimated masks
        """
        return self.masker(tf_rep, enroll)
