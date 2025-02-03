# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Loss functions."""

import numpy as np
import torch
from .torch_utils.ops import conv2d_gradfix, conv1d_gradfix

#----------------------------------------------------------------------------

class Loss:
    def loss(self, real_img, real_logits, gen_img, gen_logits, gen_ws): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(self, device, r1_gamma=10, pl_weight=0, pl_decay=0.01, pl_no_weight_grad=False):
        super().__init__()
        self.device             = device
        self.r1_gamma           = r1_gamma
        self.pl_weight          = pl_weight
        self.pl_decay           = pl_decay
        self.pl_no_weight_grad  = pl_no_weight_grad
        self.pl_mean            = torch.zeros([], device=device)

    def loss_2D(self, real_img, real_logits, gen_img, gen_logits, gen_ws):
        # Gmain: Maximize logits for generated images.
        loss_Gmain = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))
        G_loss = loss_Gmain
        # training_stats.report('Loss/G/loss', loss_Gmain)
        # Gpl: Apply path length regularization.
        pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
        with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients(self.pl_no_weight_grad):
            pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
        pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
        pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
        self.pl_mean.copy_(pl_mean.detach())
        pl_penalty = (pl_lengths - pl_mean).square()
        # training_stats.report('Loss/pl_penalty', pl_penalty)
        loss_Gpl = pl_penalty * self.pl_weight
        # training_stats.report('Loss/G/reg', loss_Gpl)
        # Dmain: Minimize logits for generated images.
        loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))
        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
        D_loss = loss_Dgen + loss_Dreal
        # training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)
        with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
            r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img], create_graph=True, only_inputs=True)[0]
        r1_penalty = r1_grads.square().sum([1,2,3])
        loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
        # training_stats.report('Loss/r1_penalty', r1_penalty)
        # training_stats.report('Loss/D/reg', loss_Dr1)

        return G_loss, D_loss

    def loss_1D(self, real_img, real_logits, gen_img, gen_logits, gen_ws):
        # Gmain: Maximize logits for generated images.
        loss_Gmain = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))
        # training_stats.report('Loss/G/loss', loss_Gmain)
        # Gpl: Apply path length regularization.
        pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
        with torch.autograd.profiler.record_function('pl_grads'), conv1d_gradfix.no_weight_gradients(self.pl_no_weight_grad):
            pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
        pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
        pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
        self.pl_mean.copy_(pl_mean.detach())
        pl_penalty = (pl_lengths - pl_mean).square()
        # training_stats.report('Loss/pl_penalty', pl_penalty)
        loss_Gpl = pl_penalty * self.pl_weight
        # training_stats.report('Loss/G/reg', loss_Gpl)
        # Dmain: Minimize logits for generated images.
        loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))
        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
        # training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)
        with torch.autograd.profiler.record_function('r1_grads'), conv1d_gradfix.no_weight_gradients():
            r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img], create_graph=True, only_inputs=True)[0]
        r1_penalty = r1_grads.square().sum([1,2,3])
        loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
        # training_stats.report('Loss/r1_penalty', r1_penalty)
        # training_stats.report('Loss/D/reg', loss_Dr1)

        return loss_Gmain+loss_Gpl+loss_Dgen+loss_Dreal+loss_Dr1
#----------------------------------------------------------------------------
