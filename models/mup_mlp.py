from typing import List

import numpy as np
import torch
from torch import nn
import mup  # Microsoft μP library
from mup import MuReadout, set_base_shapes, make_base_shapes

from .activation import switchable_activation


class MuMLP(nn.Module):
    """A μP-aware MLP.

    Flattens all dimensions except batch and uses GELU nonlinearities.
    """

    def __init__(
        self,
        n_features,
        type_token_list,
        hidden_channels,
        hidden_layers,
        out_shape=1,
        activation="gelu",
        transforms=None,
        fv_input=True,
        num_groups=1,
        dropout_prob=None,
        gain=1.0,
        loss='MSE',
        init_bias=True,
        batchnorm=False,
        BN_eps=1e-5, #default PyTorch value
        BN_momentum=0.1, #default PyTorch value
    ):
        super().__init__()

        if not hidden_layers > 0:
            raise NotImplementedError("Only supports > 0 hidden layers")

        self.in_shape = n_features
        self.out_shape = out_shape
        self.loss = loss

        layers: List[nn.Module] = []

        # Input layer
        layers.append(nn.Linear(np.prod(self.in_shape), hidden_channels))
        if dropout_prob is not None:
            layers.append(nn.Dropout(dropout_prob))
        if batchnorm: 
            layers.append(nn.BatchNorm1d(hidden_channels, eps=BN_eps, momentum=BN_momentum))

        # Hidden layers
        for _ in range(hidden_layers - 1):
            layers.append(switchable_activation(activation=activation, num_groups=1))
            layers.append(nn.Linear(hidden_channels, hidden_channels))
            if dropout_prob is not None:
                layers.append(nn.Dropout(dropout_prob))
            if batchnorm: 
                layers.append(nn.BatchNorm1d(hidden_channels, eps=BN_eps, momentum=BN_momentum))

        # Last activation
        layers.append(switchable_activation(activation=activation, num_groups=1))

        # Output layer
        if self.loss in ('HETEROSC', 'MAE_TO_HETEROSC'):
            layers.append(MuReadout(hidden_channels, np.prod(2 * self.out_shape)))
        else:
            layers.append(MuReadout(hidden_channels, np.prod(self.out_shape)))

        self.mlp = nn.Sequential(*layers)
        self.mlp.apply(lambda m: self.init_weights(m, activation=activation, init_bias=init_bias))

    @staticmethod
    def init_weights(m, activation='gelu', init_bias=True):
        """
        μP-compatible initialization for MLP layers.

        Args:
            m: layer (nn.Linear or MuReadout)
            activation: activation function name ('gelu', 'relu', etc.)
            init_bias: whether to initialize biases
        """
        if isinstance(m, MuReadout):
            # Readout layer: weights and biases initialized to zero
            if m.weight is not None:
                nn.init.zeros_(m.weight)
            if (m.bias is not None) and init_bias:
                nn.init.zeros_(m.bias)

        elif isinstance(m, nn.Linear):
            # Hidden layers: Kaiming normal with activation gain
            #gain = nn.init.calculate_gain(activation)
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            #m.weight.data.mul_(gain)

            # Bias initialization
            if (m.bias is not None): #and init_bias:
                fan_in = m.weight.size(1)
                bound = 1 / (fan_in ** 0.5)
                nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, inputs: torch.Tensor):
        """Forward pass of μP-aware MLP."""
        x = self.mlp(inputs)

        if self.loss in ('HETEROSC', 'MAE_TO_HETEROSC'):
            # Split last `out_shape` dimensions, apply softplus for positivity
            x_sigmas = torch.max(torch.nn.functional.softplus(x[:, -self.out_shape:]),
                                 torch.tensor(1e-15, device=x.device))
            x = torch.cat((x[:, :-self.out_shape], x_sigmas), dim=1)
            return x
        else:
            return x