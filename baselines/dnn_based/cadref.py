from baselines.base import BaseBaseline
from baselines.registry import register_baseline

import torch
import numpy as np
from tqdm import tqdm


@register_baseline("cadref")
class CADRef(BaseBaseline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model.eval()
        
        # energy score
        self.T = 1.0
        w = self.model.linear.weight.detach().to(self.device)
        self.w_sign = w.sign()
        self.class_means = self.get_class_means().to(self.device)
        self.global_score = self.get_global_score().to(self.device)

    @torch.no_grad()
    def get_class_means(self):
        train_features = self.get_train_feature()

        class_means = []
        for feats_c in train_features:
            class_means.append(feats_c.mean(dim=0))
        
        class_means = torch.stack(class_means, dim=0)
        return class_means
    
    def get_global_score(self):
        train_logits = self.get_train_logit()

        scores = []
        for t in train_logits:
            scores.append(self.T * torch.logsumexp(t / self.T, dim=1))
        global_score = torch.cat(scores, dim=0).mean()

        return global_score
    
    @torch.no_grad()
    def eval(self, data_loader):
        result = []
        
        for (images, _) in tqdm(data_loader):
            images = images.to(self.device)

            logits, feats = self.model.get_output(images, return_feature=True)
            class_ids = logits.argmax(dim=1)
            class_proto = self.class_means.index_select(0, class_ids)
            
            # Feature Decoupling
            dist = class_proto - feats
            sg = self.w_sign.index_select(0, class_ids)
  
            ep_dist = torch.clamp(dist * sg, min=0)
            en_dist = torch.clamp(dist * (-sg), min=0)

            feat_norm = feats.norm(p=1, dim=1).clamp_min(1e-12)
            ep_error = ep_dist.norm(p=1, dim=1) / feat_norm
            en_error = en_dist.norm(p=1, dim=1) / feat_norm
            
            #  Error Scaling
            logit_score = self.T * torch.logsumexp(logits / self.T, dim=1)
            scores = ep_error / logit_score + en_error / self.global_score
            
            result.append(-scores.cpu().numpy())

        return np.concatenate(result)
