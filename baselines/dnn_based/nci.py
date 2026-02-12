from baselines.base import BaseBaseline
from baselines.registry import register_baseline

import torch
import numpy as np
from tqdm import tqdm


@register_baseline("nci")
class NCI(BaseBaseline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model.eval()
        self.alpha = 0.001
        self.w = self.model.linear.weight
        self.mean_feat = self.get_mean_feat().to(self.device)

    @torch.no_grad()
    def get_mean_feat(self):
        train_features = self.get_train_feature()

        feats = []
        for t in train_features:
            feats.append(t.cpu())
        feats = torch.cat(feats, dim=0)

        mean_feat = feats.mean(dim=0)
        return mean_feat
    
    @torch.no_grad()
    def eval(self, data_loader):
        result = []
        
        for (images, _) in tqdm(data_loader):
            images = images.to(self.device)

            logits = self.model.get_output(images)
            feats = self.model.get_feature(images)
            pred = logits.argmax(dim=1) 
            scores = torch.sum(self.w[pred] * (feats - self.mean_feat), axis=1) / torch.norm(feats - self.mean_feat, dim=1) + self.alpha * feats.norm(dim=1, p=1)
            
            result.append(scores.cpu().numpy())

        return np.concatenate(result)
