from baselines.base import BaseBaseline
from baselines.registry import register_baseline

import numpy as np
import torch
from tqdm import tqdm


def scale(x, percentile=65):
    x_input = x.clone()
    b, c, h, w = x.shape

    s1 = x.sum(dim=(1, 2, 3))
    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile / 100.0))
    t = x.view(b, c * h * w)
    v, i = torch.topk(t, k, dim=1)
    t.zero_().scatter_(dim=1, index=i, src=v)
    s2 = x.sum(dim=(1, 2, 3))
    scale_factor = s1 / s2

    return x_input * torch.exp(scale_factor[:, None, None, None])


@register_baseline("scale")
class Scale(BaseBaseline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model.eval()
        self.percentile = getattr(self.args, "scale_percentile", getattr(self.args, "percentile", 85))

    @torch.no_grad()
    def eval(self, data_loader):
        result = []

        for (images, _) in tqdm(data_loader):
            images = images.to(self.device)

            feats = self.model.get_feature(images)
            feats = scale(feats.view(feats.size(0), -1, 1, 1), self.percentile)
            feats = feats.view(feats.size(0), -1)
            logits = self.model.linear(feats)
            scores = torch.logsumexp(logits, dim=1)

            result.append(scores.cpu().numpy())

        return np.concatenate(result)
