"""Shared, round-frozen QMC qLogNEI for allocation and final selection."""
import numpy as np
import torch
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler
from .gp import GPPerformanceSurrogate


class ReliableGPPerformanceSurrogate(GPPerformanceSurrogate):
    minimum_acquisition_samples = 1024

    def __init__(self,space):
        super().__init__(space)
        self.begin_round(0)

    def begin_round(self,seed):
        self.round_seed=int(seed);self._acquisition=None;self._sample_count=None

    def fit(self,configs,utilities):
        super().fit(configs,utilities)
        self.begin_round(0)

    def log_noisy_expected_improvement(self,configs,samples=128,seed=0):
        if not configs:return np.empty(0)
        count=max(self.minimum_acquisition_samples,int(samples))
        if self._acquisition is None or self._sample_count!=count:
            self._acquisition=qLogNoisyExpectedImprovement(self._require_model(),
                X_baseline=self.model.train_inputs[0],
                sampler=SobolQMCNormalSampler(torch.Size([count]),seed=self.round_seed),
                prune_baseline=True)
            self._sample_count=count
        out=[]
        with torch.no_grad():
            for start in range(0,len(configs),64):
                x=torch.as_tensor(self.space.encode_many(configs[start:start+64]),dtype=torch.double)
                out.extend(self._acquisition(x.unsqueeze(-2)).cpu().numpy().reshape(-1))
        return np.asarray(out)
