import collections
import random
from typing import Any

import numpy as np
import torch
from acados_template.acados_ocp_iterate import (
    AcadosOcpFlattenedBatchIterate,
    AcadosOcpFlattenedIterate,
    AcadosOcpIterate,
)
from leap_c.mpc import MPCParameter
from torch.utils._pytree import tree_map_only
from torch.utils.data._utils.collate import collate, default_collate_fn_map


class ReplayBuffer:
    def __init__(
        self, buffer_limit: int, device: str, tensor_dtype: torch.dtype = torch.float32
    ):
        """
        Args:
            buffer_limit: The maximum number of transitions that can be stored in the buffer.
                If the buffer is full, the oldest transitions are discarded when putting in a new one.
            device: The device to which all sampled tensors will be cast.
            tensor_dtype: The data type to which the tensors in the observation will be cast.
        """
        self.buffer = collections.deque(maxlen=buffer_limit)
        self.device = device
        self.tensor_dtype = tensor_dtype

        self.custom_collate_fn_map = self.create_collate_fn_map()

    def put(self, data: Any):
        """Put the data into the replay buffer. If the buffer is full, the oldest data is discarded.

        Parameters:
            data: The data to put into the buffer.
                It should be collatable according to the custom_collate function.
        """
        self.buffer.append(data)

    def sample(self, n: int) -> Any:
        """
        Sample a mini-batch from the replay buffer,
        collate the mini-batch according to self.custom_collate_map
        and cast all tensors in the collated mini-batch (must be a pytree structure)
        to the device and dtype of the buffer.

        Parameters:
            n: The number of samples to draw.
        """
        mini_batch = random.sample(self.buffer, n)
        return self.pytree_tensor_to(
            collate(mini_batch, collate_fn_map=self.custom_collate_fn_map)
        )

    @staticmethod
    def _safe_collate_possible_nones(
        field_data: list[None] | list[np.ndarray],
    ) -> None | np.ndarray:
        """Checks whether the given list contains only Nones or only non-Nones.
        If it contains only Nones, it returns None, otherwise it uses np.stack on the given list.
        If a mixture of Nones and non-Nones is detected, a ValueError is raised."""
        any_none = False
        all_none = True
        for data in field_data:
            if data is None:
                any_none = True
            else:
                all_none = False
            if any_none and not all_none:
                raise ValueError("All or none of the data must be None.")
        if all_none:
            return None
        else:
            return np.stack(field_data, axis=0)  # type:ignore

    def create_collate_fn_map(self):
        """Create the collate function map for the collate function.
        By default, this is the default_collate_fn_map of pytorch, with an additional
        rule for MPCParameter, AcadosOcpFlattenedIterate and AcadosOcpIterate."""
        custom_collate_map = default_collate_fn_map.copy()

        # NOTE: If MPCParameter should also be tensorified, you can turn mpcparam_fn off
        # and use the following code for handling the Nones
        # def none_fn(batch, *, collate_fn_map=None):
        #     # Collate nones into one none but throws an error if batch contains something else than none.
        #     if any(x is not None for x in batch):
        #         raise ValueError("None collate function can only collate Nones.")
        #     return None
        # custom_collate_map[type_(None)]=none_fn

        # Keeps MPCParameter as np.ndarray
        def mpcparam_fn(batch, *, collate_fn_map=None):
            # Collate MPCParameters by stacking the p_global and p_stagewise parts, but do not convert them to tensors.

            glob_data = [x.p_global for x in batch]
            stag_data = [x.p_stagewise for x in batch]
            idx_data = [x.p_stagewise_sparse_idx for x in batch]

            return MPCParameter(
                p_global=ReplayBuffer._safe_collate_possible_nones(glob_data),
                p_stagewise=ReplayBuffer._safe_collate_possible_nones(stag_data),
                p_stagewise_sparse_idx=ReplayBuffer._safe_collate_possible_nones(
                    idx_data
                ),
            )

        def acados_flattened_iterate_fn(batch, *, collate_fn_map=None):
            return AcadosOcpFlattenedBatchIterate(
                x=np.stack([x.x for x in batch], axis=0),
                u=np.stack([x.u for x in batch], axis=0),
                z=np.stack([x.z for x in batch], axis=0),
                sl=np.stack([x.sl for x in batch], axis=0),
                su=np.stack([x.su for x in batch], axis=0),
                pi=np.stack([x.pi for x in batch], axis=0),
                lam=np.stack([x.lam for x in batch], axis=0),
                N_batch=len(batch),
            )

        def acados_iterate_fn(batch, *, collate_fn_map=None):
            # NOTE: Could also be a FlattenedBatchIterate (which has a parallelized set in the batch solver),
            # but this seems more intuitive. If the user wants to have a flattened batch iterate, he can
            # just put in AcadosOcpIterate.flatten into the buffer.
            return list(batch)

        custom_collate_map[MPCParameter] = mpcparam_fn
        custom_collate_map[AcadosOcpFlattenedIterate] = acados_flattened_iterate_fn
        custom_collate_map[AcadosOcpIterate] = acados_iterate_fn

        return custom_collate_map

    def pytree_tensor_to(self, pytree: Any) -> Any:
        """Convert all tensors in the pytree to self.tensor_dtype and
        move them to self.device."""
        return tree_map_only(
            torch.Tensor,
            lambda t: t.to(device=self.device, dtype=self.tensor_dtype),
            pytree,
        )

    def __len__(self):
        return len(self.buffer)
