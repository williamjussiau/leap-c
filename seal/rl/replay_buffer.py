import collections
import random

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, buffer_limit: int, device: str):
        self.buffer = collections.deque(maxlen=buffer_limit)
        self.device = device

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_lst.append([done])

        return (
            torch.from_numpy(np.array(s_lst, dtype=np.float32)).to(self.device),
            torch.from_numpy(np.array(a_lst, dtype=np.float32)).to(self.device),
            torch.from_numpy(np.array(r_lst, dtype=np.float32)).to(self.device),
            torch.from_numpy(np.array(s_prime_lst, dtype=np.float32)).to(self.device),
            torch.from_numpy(np.array(done_lst, dtype=np.float32)).to(self.device),
        )

    def size(self):
        return len(self.buffer)
