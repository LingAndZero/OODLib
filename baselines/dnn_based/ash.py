from baselines.base import BaseBaseline
from baselines.registry import register_baseline

import torch
import numpy as np
from tqdm import tqdm


@register_baseline("ash")
class ASH(BaseBaseline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model.eval()
        self.T = 1
        self.p = 90

    @torch.no_grad()
    def eval(self, data_loader):
        result = []
        
        for (images, _) in tqdm(data_loader):
            images = images.to(self.device)
            feature = self.model.get_feature(images)

            feature = ash_s(feature, self.p)
            # output = ash_b(feature, self.p)
            # output = ash_p(feature, self.p)
            feature = feature.view(feature.size(0), -1)
            logits = self.model.linear(feature)

            energy = self.T * torch.logsumexp(logits / self.T, dim=1)
            result.append(energy.detach().cpu().numpy())

        return np.concatenate(result)


def ash_b(x, percentile):
    assert x.dim() == 4
    assert 0 <= percentile <= 100
    b, c, h, w = x.shape

    s1 = x.sum(dim=[1, 2, 3])

    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile / 100.0))
    t = x.view((b, c * h * w))
    v, i = torch.topk(t, k, dim=1)
    fill = s1 / k
    fill = fill.unsqueeze(dim=1).expand(v.shape)
    t.zero_().scatter_(dim=1, index=i, src=fill)
    return x


def ash_p(x, percentile):
    assert x.dim() == 4
    assert 0 <= percentile <= 100

    b, c, h, w = x.shape

    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile / 100.0))
    t = x.view((b, c * h * w))
    v, i = torch.topk(t, k, dim=1)
    t.zero_().scatter_(dim=1, index=i, src=v)

    return x


def ash_s(x, percentile):
    assert x.dim() == 4
    assert 0 <= percentile <= 100
    b, c, h, w = x.shape

    s1 = x.sum(dim=[1, 2, 3])
    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile / 100.0))
    t = x.view((b, c * h * w))
    v, i = torch.topk(t, k, dim=1)
    t.zero_().scatter_(dim=1, index=i, src=v)

    s2 = x.sum(dim=[1, 2, 3])

    scale = s1 / s2
    x = x * torch.exp(scale[:, None, None, None])

    return x