# Copyright 2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""OmniSafe tools package."""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from typing import Any

import numpy as np
import torch
import torch.backends.cudnn
import yaml
from torch import nn


def get_flat_params_from(model: list[nn.Parameter]) -> torch.Tensor:
    """This function is used to get the flattened parameters from the model.

    .. note::
        Some algorithms need to get the flattened parameters from the model, such as the
        :class:`TRPO` and :class:`CPO` algorithm. In these algorithms, the parameters are flattened
        and then used to calculate the loss.

    Examples:
        >>> model = torch.nn.Linear(2, 2)
        >>> model.weight.data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> get_flat_params_from(model)
        tensor([1., 2., 3., 4.])

    Args:
        model (torch.nn.Module): model to be flattened.

    Returns:
        Flattened parameters.

    Raises:
        AssertionError: If no gradients were found in model parameters.
    """
    flat_params = []
    for param in model:
        if param.requires_grad:
            data = param.data
            data = data.view(-1)  # flatten tensor
            flat_params.append(data)
    assert flat_params, 'No gradients were found in model parameters.'
    return torch.cat(flat_params)


def get_flat_gradients_from(model: list[nn.Parameter]) -> torch.Tensor:
    """This function is used to get the flattened gradients from the model.

    .. note::
        Some algorithms need to get the flattened gradients from the model, such as the
        :class:`TRPO` and :class:`CPO` algorithm. In these algorithms, the gradients are flattened
        and then used to calculate the loss.

    Args:
        model (list[nn.Parameter]): The model to be flattened.

    Returns:
        Flattened gradients.

    Raises:
        AssertionError: If no gradients were found in model parameters.
    """
    grads = []
    for param in model:
        if param.requires_grad and param.grad is not None:
            grad = param.grad
            grads.append(grad.view(-1))  # flatten tensor and append
        else:
            # print(f"Gradient for parameter '{param.name}' is None")
            grads.append(torch.zeros_like(param).view(-1))
    assert grads, 'No gradients were found in model parameters.'
    return torch.cat(grads)


def set_param_values_to_model(model: list[nn.Parameter], vals: torch.Tensor) -> None:
    """This function is used to set the parameters to the model.

    .. note::
        Some algorithms (e.g. TRPO, CPO, etc.) need to set the parameters to the model, instead of
        using the ``optimizer.step()``.

    Examples:
        >>> model = torch.nn.Linear(2, 2)
        >>> model.weight.data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        >>> vals = torch.tensor([1.0, 2.0, 3.0, 4.0])
        >>> set_param_values_to_model(model, vals)
        >>> model.weight.data
        tensor([[1., 2.],
                [3., 4.]])

    Args:
        model (list[nn.Parameter]): The model to be set.
        vals (torch.Tensor): The parameters to be set.

    Raises:
        AssertionError: If the instance of the parameters is not ``torch.Tensor``, or the lengths of
            the parameters and the model parameters do not match.
    """
    assert isinstance(vals, torch.Tensor)
    i: int = 0
    for param in model:
        if param.requires_grad:  # param has grad and, hence, must be set
            orig_size = param.size()
            size = np.prod(list(param.size()))
            new_values = vals[i : int(i + size)]
            # set new param values
            new_values = new_values.view(orig_size)
            param.data = new_values
            i += int(size)  # increment array position
    assert i == len(vals), f'Lengths do not match: {i} vs. {len(vals)}'