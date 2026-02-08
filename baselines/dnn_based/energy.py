from baselines.base import BaseBaseline
from baselines.registry import register_baseline

import torch
import numpy as np
from tqdm import tqdm


@register_baseline("energy")
class Energy(BaseBaseline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model.eval()
        self.T = 1.0

    @torch.no_grad()
    def eval(self, data_loader):
        result = []
        
        for (images, _) in tqdm(data_loader):
            images = images.to(self.device)

            logits = self.model.get_output(images)
            scores = self.T * torch.logsumexp(logits / self.T, dim=1)

            result.append(scores.cpu().numpy())

        return np.concatenate(result)

