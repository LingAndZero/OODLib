from baselines.base import BaseBaseline
from baselines.registry import register_baseline

import torch.nn.functional as F
import torch
import numpy as np
from tqdm import tqdm


@register_baseline("mcm")
class MCM(BaseBaseline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.t = 1


    @torch.no_grad()
    def eval(self, data_loader):

        T = 1
        
        self.model.eval()
        result = []
        
        for (images, _) in tqdm(data_loader):
            images = images.to(self.device)
            output = self.model.get_output(images)

            smax = (F.softmax(output / T, dim=1)).data.cpu().numpy()
            output = np.max(smax, axis=1)

            result.append(output)

        return np.concatenate(result)
