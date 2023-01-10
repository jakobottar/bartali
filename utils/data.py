"""
data utils
"""
import torch


class MultiplyBatchSampler(torch.utils.data.sampler.BatchSampler):
    multiplier = 2

    def __iter__(self):
        for batch in super().__iter__():
            yield batch * self.multiplier
