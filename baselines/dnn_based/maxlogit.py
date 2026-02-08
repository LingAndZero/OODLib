from baselines.base import BaseBaseline
from baselines.registry import register_baseline

import torch.nn.functional as F
import torch
import numpy as np
from tqdm import tqdm


@register_baseline("maxlogit")
class MaxLogit(BaseBaseline):
    
    @torch.no_grad()
    def eval(self, data_loader):
        self.model.eval()
        result = []
        
        for (images, _) in tqdm(data_loader):
            images = images.to(self.device)

            logits = self.model.get_output(images)
            scores, _ = torch.max(logits, dim=1)

            result.append(scores.cpu().numpy())

        return np.concatenate(result)
