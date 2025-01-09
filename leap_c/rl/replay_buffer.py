import collections
import random
from typing import Any

import numpy as np
import torch
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
            obs_dtype: The data type to which the tensors in the observation will be cast.
            NOTE: Only works if the observations contain np.ndarrays or torch.tensors as "leaf"-data.
            If the observations contain other types (e.g. floats (!), ints, bool), the collate function will use the default collate
            for these types, which creates a tensor without calling the collate for torch.Tensor.
        """
        self.buffer = collections.deque(maxlen=buffer_limit)
        self.device = device
        self.tensor_dtype = tensor_dtype

        self.custom_collate_map = self.create_collate_map()

    def put(self, data: Any):
        """Put the data into the replay buffer. If the buffer is full, the oldest data is discarded.

        Parameters:
            data: The data to put into the buffer.
                It should be collatable according to the collate function.
        """
        self.buffer.append(data)

    def sample(self, n: int) -> Any:
        """
        Sample a mini-batch from the replay buffer,
        collated according to the collate function of this class
        and cast to the device and dtype of the buffer according to the
        pytree_tensor_to function.

        Parameters:
            n: The number of samples to draw.
        """
        mini_batch = random.sample(self.buffer, n)
        return collate(mini_batch)

        return (
            self.collate_obs(obs_lst),
            torch.from_numpy(np.array(a_lst, dtype=np.float32)).to(device=self.device),
            torch.from_numpy(np.array(r_lst, dtype=np.float32))
            .unsqueeze(1)
            .to(device=self.device),
            self.collate_obs(obs_next_lst),
            torch.from_numpy(np.array(done_lst, dtype=np.float32))
            .unsqueeze(1)
            .to(device=self.device),
        )

    def create_collate_map(self):
        custom_collate_map = default_collate_fn_map.copy()

        # NOTE: If MPCParameter should also be tensorified, you can use this and turn mpcparam_fn off
        # def none_fn(batch, *, collate_fn_map=None):
        #     # Collate nones into one none but throws an error if batch contains something else than none.
        #     if any(x is not None for x in batch):
        #         raise ValueError("None collate function can only collate Nones.")
        #     return None
        # custom_collate_map[type_(None)]=none_fn

        # Keeps MPCParameter as np.ndarray
        def mpcparam_fn(batch, *, collate_fn_map=None):
            # Collate MPCParameters by stacking the p_global and p_stagewise parts, but do not convert them to tensors.
            # Only works if all elements for a field are np.arrays or nones.

            glob_data = [x.p_global for x in batch]
            stag_data = [x.p_stagewise for x in batch]
            idx_data = [x.p_stagewise_sparse_idx for x in batch]

            return MPCParameter(
                p_global=safe_collate_field(glob_data),
                p_stagewise=safe_collate_field(stag_data),
                p_stagewise_sparse_idx=safe_collate_field(idx_data),
            )

        custom_collate_map[MPCParameter] = mpcparam_fn

        return custom_collate_map

    def collate(self, data: Any) -> Any:
        """Collate the input and cast all final tensors to the device and dtype of the buffer."""
        return self.pytree_tensor_to(
            collate(data, collate_fn_map=self.custom_collate_map)
        )

    def pytree_tensor_to(self, pytree: Any) -> Any:
        """Convert all tensors of the pytree to self.tensor_dtype and
        move them to self.device."""
        return tree_map_only(
            torch.Tensor,
            lambda t: t.to(device=self.device, dtype=self.tensor_dtype),
            pytree,
        )

    def size(self):
        return len(self.buffer)
