from baselines.base import BaseBaseline
from baselines.registry import register_baseline

import torch
import numpy as np
from tqdm import tqdm


@register_baseline("dice")
class DICE(BaseBaseline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model.eval()
        self.T = 1
        self.p = 70

        self.mask = self.get_mask()
        with torch.no_grad():
            self.model.linear.weight.mul_(self.mask)
    
    def get_mask(self):
        train_features = self.get_train_feature()

        feats = []
        for t in train_features:
            feats.append(t.cpu())
        feats = torch.cat(feats, dim=0)

        mean_feat = feats.mean(dim=0)
        weight = self.model.linear.weight.detach().cpu()
        contrib = mean_feat.unsqueeze(0) * weight
        threshold = np.percentile(contrib, self.p)
        
        mask = (contrib > threshold).float().to(self.device)
        return mask
    
    @torch.no_grad()
    def eval(self, data_loader):
        result = []
        
        for (images, _) in tqdm(data_loader):
            images = images.to(self.device)

            logits = self.model.get_output(images)
            scores = self.T * torch.logsumexp(logits / self.T, dim=1)
            
            result.append(scores.cpu().numpy())

        return np.concatenate(result)
