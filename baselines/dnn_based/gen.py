from baselines.base import BaseBaseline
from baselines.registry import register_baseline

import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm


@register_baseline("gen")
class GEN(BaseBaseline):     
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model.eval()
        self.gamma = 0.1
        self.M = 100

    @torch.no_grad()
    def eval(self, data_loader):
        result = []
        
        for (images, _) in tqdm(data_loader):
            images = images.to(self.device)
            output = self.model.get_output(images)
            smax = (F.softmax(output, dim=1)).data.cpu().numpy()
            probs_sorted = np.sort(smax, axis=1)[:,self.M:]
            scores = np.sum(probs_sorted ** self.gamma * (1 - probs_sorted) ** self.gamma, axis=1)

            result.append(-scores)

        return np.concatenate(result)

