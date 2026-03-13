from baselines.base import BaseBaseline
from baselines.registry import register_baseline

import torch
import numpy as np
from numpy.linalg import pinv, norm
from sklearn.covariance import EmpiricalCovariance
from tqdm import tqdm


@register_baseline("vim")
class ViM(BaseBaseline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model.eval()

        self.T = 1.0
        NS, alpha, u = self.get_ns()
        self.NS = torch.from_numpy(NS).to(self.device)
        self.u  = torch.from_numpy(u).to(self.device)
        self.alpha = float(alpha)

    @torch.no_grad()
    def get_ns(self):
        w = self.model.linear.weight.detach().cpu().numpy()
        b = self.model.linear.bias.detach().cpu().numpy()
        u = -(pinv(w) @ b)

        train_feats = torch.cat([t for t in self.get_train_feature()], dim=0).detach().cpu().numpy()
        logit_id_train = train_feats @ w.T + b

        if train_feats.shape[-1] >= 2048:
            DIM = 1000
        elif train_feats.shape[-1] >= 768:
            DIM = 512
        else:
            DIM = train_feats.shape[-1] // 2

        print(f'{DIM=}')
        print('computing principal space...')
        ec = EmpiricalCovariance(assume_centered=True)
        ec.fit(train_feats - u)
        eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
        NS = np.ascontiguousarray(
            (eigen_vectors.T[np.argsort(eig_vals * -1)[DIM:]]).T)

        print('computing alpha...')
        vlogit_id_train = norm((train_feats - u) @ NS, axis=-1)
        alpha = logit_id_train.max(axis=-1).mean() / vlogit_id_train.mean()
        print(f'{alpha=:.4f}')

        return NS, alpha, u
 
    @torch.no_grad()
    def eval(self, data_loader):
        result = []
        
        for (images, _) in tqdm(data_loader):
            images = images.to(self.device)

            logits, feats = self.model.get_output(images, return_feature=True)

            proj = (feats - self.u) @ self.NS
            vlogit = proj.norm(p=2, dim=1) * self.alpha
            scores = -vlogit + self.T * torch.logsumexp(logits / self.T, dim=1)
            
            result.append(scores.cpu().numpy())

        return np.concatenate(result)
