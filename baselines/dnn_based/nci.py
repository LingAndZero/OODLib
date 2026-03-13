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
        self.alpha = getattr(self.args, "nci_alpha", 0.001)
        self.w = self.model.linear.weight.detach().to(self.device)

        train_feats = [t for t in self.get_train_feature() if t.numel() > 0]
        train_feats = torch.cat(train_feats, dim=0).to(self.device)
        self.train_mean = train_feats.mean(dim=0)

    @torch.no_grad()
    def eval(self, data_loader):
        result = []

        for (images, _) in tqdm(data_loader):
            images = images.to(self.device)

            logits, feats = self.model.get_output(images, return_feature=True)
            pred = logits.argmax(dim=1)
            diff = feats - self.train_mean
            class_w = self.w.index_select(0, pred)
            proj_score = (class_w * diff).sum(dim=1) / diff.norm(p=2, dim=1)
            scores = proj_score + self.alpha * feats.norm(p=1, dim=1)

            result.append(scores.cpu().numpy())

        return np.concatenate(result)
