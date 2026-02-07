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
        self.p = 90

        train_features = self.get_train_feature()

        train_all_features = []
        for i in range(len(train_features)):
            train_all_features.extend(train_features[i].cpu().numpy())
                    
        self.threshold = np.percentile(train_all_features, self.p)
   
    @torch.no_grad()
    def eval(self, data_loader):
        result = []
        
        for (images, _) in tqdm(data_loader):
            images = images.to(self.device)
            feature = self.model.get_feature(images)

            feature = feature.clip(max=self.threshold)
            feature = feature.view(feature.size(0), -1)
            output = self.model.linear(feature)

            energy = self.T * torch.logsumexp(output / self.T, dim=1).data.cpu().numpy()
            result.append(energy)

        return np.concatenate(result)
