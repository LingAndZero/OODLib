from baselines.base import BaseBaseline
from baselines.registry import register_baseline

import torch
import numpy as np
from tqdm import tqdm


@register_baseline("react")
class ReAct(BaseBaseline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model.eval()
        self.T = 1
        self.p = 90
        
        self.threshold = self.get_threshold()
        
    def get_threshold(self):
        train_features = self.get_train_feature()

        vals = []
        for t in train_features:
            vals.append(t.float().cpu().numpy().reshape(-1))

        all_vals = np.concatenate(vals, axis=0)
        threshold = np.percentile(all_vals, self.p)
        return threshold
        
    @torch.no_grad()
    def eval(self, data_loader):
        result = []
        
        for (images, _) in tqdm(data_loader):
            images = images.to(self.device)

            feat = self.model.get_feature(images)
            # clip activations/features
            feat = torch.clamp(feat, max=self.threshold)
            feat = feat.view(feat.size(0), -1)

            logits = self.model.linear(feat)

            energy = self.T * torch.logsumexp(logits / self.T, dim=1)
            result.append(energy.detach().cpu().numpy())

        return np.concatenate(result)
