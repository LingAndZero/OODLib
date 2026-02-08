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

            logits = self.model.get_output(images)
            probs = F.softmax(logits, dim=1)
            probs_sorted, _ = torch.sort(probs, dim=1, descending=False)
            probs_sorted = probs_sorted[:, self.M:]
            scores = torch.sum((probs_sorted ** self.gamma) * ((1.0 - probs_sorted) ** self.gamma), dim=1)

            result.append(-scores.cpu().numpy())

        return np.concatenate(result)

