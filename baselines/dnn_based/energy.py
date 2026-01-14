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
        self.T = 1

    @torch.no_grad()
    def eval(self, data_loader):
        result = []
        
        for (images, _) in tqdm(data_loader):
            images = images.to(self.device)
            output = self.model.get_output(images)

            energy = self.T * torch.logsumexp(output / self.T, dim=1).data.cpu().numpy()
            result.append(energy)

        return np.concatenate(result)

