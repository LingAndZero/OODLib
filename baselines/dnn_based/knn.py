from baselines.base import BaseBaseline
from baselines.registry import register_baseline

import torch
import numpy as np
from tqdm import tqdm


@register_baseline("knn")
class KNN(BaseBaseline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model.eval()

        self.K = 1000
        self.alpha = 0.01

        train_feats = []
        for t in self.get_train_feature():
            train_feats.append(t)
        train_feats = torch.cat(train_feats, dim=0)
        train_feats = train_feats / train_feats.norm(p=2, dim=1, keepdim=True)
        
        N = train_feats.size(0)
        n_sub = int(N * self.alpha)
        rand_ind = torch.randperm(N)[:n_sub]
        self.bank = train_feats[rand_ind].to(self.device)
    
    @torch.no_grad()
    def eval(self, data_loader):
        result = []
        
        for (images, _) in tqdm(data_loader):
            images = images.to(self.device)

            feats = self.model.get_feature(images)
            feats = feats / feats.norm(p=2, dim=1, keepdim=True)
            dist = torch.cdist(feats, self.bank)
            scores = dist.topk(self.K, dim=1, largest=False).values[:, -1]
            
            result.append(-(scores ** 2).cpu().numpy())

        return np.concatenate(result)
