# Copyright 2021 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import argparse
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from solo.methods.base import BaseMomentumMethod
from solo.utils.momentum import initialize_momentum_params
from solo.losses.dual_temperature_loss import dual_temperature_loss_func


class SimMoCo_DualTemperature(BaseMomentumMethod):
    queue: torch.Tensor

    def __init__(
        self,
        proj_output_dim: int,
        proj_hidden_dim: int,
        temperature: float,
        dt_m: float,
        plus_version: bool,
        **kwargs
    ):
        """Implements simmoco with dual temperature.

        Args:
            proj_output_dim (int): number of dimensions of projected features.
            proj_hidden_dim (int): number of neurons of the hidden layers of the projector.
            temperature (float): temperature for the softmax in the contrastive loss.
            queue_size (int): number of samples to keep in the queue.
        """

        super().__init__(**kwargs)

        self.temperature = temperature
        self.dt_m = dt_m
        self.plus_version = plus_version
        

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )

        # momentum projector
        self.momentum_projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )

        initialize_momentum_params(self.projector, self.momentum_projector)


    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parent_parser = super(SimMoCo_DualTemperature, SimMoCo_DualTemperature).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("simmoco_dual_temperature")

        # projector
        parser.add_argument("--proj_output_dim", type=int, default=128)
        parser.add_argument("--proj_hidden_dim", type=int, default=2048)

        # parameters
        parser.add_argument("--temperature", type=float, default=0.1)
        parser.add_argument("--dt_m", type=float, default=10)

        # train the plus version which uses symmetric loss
        parser.add_argument("--plus_version", action="store_true")
        
        return parent_parser

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector parameters together with parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [{"params": self.projector.parameters()}]
        return super().learnable_params + extra_learnable_params

    @property
    def momentum_pairs(self) -> List[Tuple[Any, Any]]:
        """Adds (projector, momentum_projector) to the parent's momentum pairs.

        Returns:
            List[Tuple[Any, Any]]: list of momentum pairs.
        """

        extra_momentum_pairs = [(self.projector, self.momentum_projector)]
        return super().momentum_pairs + extra_momentum_pairs


    def forward(self, X: torch.Tensor, *args, **kwargs) -> Dict[str, Any]:
        """Performs the forward pass of the online encoder and the online projection.

        Args:
            X (torch.Tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of the parent and the projected features.
        """

        out = super().forward(X, *args, **kwargs)
        q = F.normalize(self.projector(out["feats"]), dim=-1)
        return {**out, "q": q}

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """
        Training step for MoCo reusing BaseMomentumMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the
                format of [img_indexes, [X], Y], where [X] is a list of size self.num_crops
                containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of MOCO loss and classification loss.

        """

        if self.plus_version: 
            out = super().training_step(batch, batch_idx)
            class_loss = out["loss"]
            feats1, feats2 = out["feats"]
            momentum_feats1, momentum_feats2 = out["momentum_feats"]

            q1 = self.projector(feats1)
            q2 = self.projector(feats2)

            q1 = F.normalize(q1, dim=-1)
            q2 = F.normalize(q2, dim=-1)
            with torch.no_grad():
                k1 = self.momentum_projector(momentum_feats1)
                k2 = self.momentum_projector(momentum_feats2)
                k1 = F.normalize(k1, dim=-1).detach()
                k2 = F.normalize(k2, dim=-1).detach()


            nce_loss = (
                dual_temperature_loss_func(q1, k2,
                                temperature=self.temperature,
                                dt_m=self.dt_m)
                + dual_temperature_loss_func(q2, k1,
                                temperature=self.temperature,
                                dt_m=self.dt_m)
            ) / 2

            # calculate std of features
            z1_std = F.normalize(q1, dim=-1).std(dim=0).mean()
            z2_std = F.normalize(q2, dim=-1).std(dim=0).mean()
            z_std = (z1_std + z2_std) / 2

            metrics = {
                "train_nce_loss": nce_loss,
                "train_z_std": z_std,
            }
            self.log_dict(metrics, on_epoch=True, sync_dist=True)

            return nce_loss + class_loss

        else:
            out = super().training_step(batch, batch_idx)
            class_loss = out["loss"]
            feats1, _ = out["feats"]
            _, momentum_feats2 = out["momentum_feats"]

            q1 = self.projector(feats1)

            q1 = F.normalize(q1, dim=-1)

            with torch.no_grad():
                k2 = self.momentum_projector(momentum_feats2)
                k2 = F.normalize(k2, dim=-1).detach()

            nce_loss = dual_temperature_loss_func(q1, k2,
                                temperature=self.temperature,
                                dt_m=self.dt_m)

            # calculate std of features
            z1_std = F.normalize(q1, dim=-1).std(dim=0).mean()
            z_std = z1_std

            metrics = {
                "train_nce_loss": nce_loss,
                "train_z_std": z_std,
            }
            self.log_dict(metrics, on_epoch=True, sync_dist=True)

            return nce_loss + class_loss
