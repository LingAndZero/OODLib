from baselines.base import BaseBaseline
from baselines.registry import register_baseline

import torch
import numpy as np
from tqdm import tqdm


@register_baseline("caref")
class CARef(BaseBaseline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model.eval()

        self.class_means = self.get_class_means().to(self.device)

    @torch.no_grad()
    def get_class_means(self):
        train_features = self.get_train_feature()

        class_means = []
        for feats_c in train_features:
            class_means.append(feats_c.mean(dim=0))
        
        class_means = torch.stack(class_means, dim=0)
        return class_means
    
    @torch.no_grad()
    def eval(self, data_loader):
        result = []
        
        for (images, _) in tqdm(data_loader):
            images = images.to(self.device)

            logits, feats = self.model.get_output(images, return_feature=True)
            class_ids = logits.argmax(dim=1)
            class_proto = self.class_means.index_select(0, class_ids)
            scores = (class_proto - feats).norm(dim=1, p=1) / feats.norm(dim=1, p=1)
            
            result.append(-scores.cpu().numpy())

        return np.concatenate(result)
