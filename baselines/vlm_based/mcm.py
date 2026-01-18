from baselines.base import BaseBaseline
from baselines.registry import register_baseline

import torch.nn.functional as F
import torch
import numpy as np
from tqdm import tqdm


@register_baseline("mcm")
class MCM(BaseBaseline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.model.eval()

        prompt = "a photo of a {}"
        texts = self.model.get_texts(prompt, self.ind_dataset.class_names).to(self.device)
        self.text_features = self.model.get_text_feature(texts)

        self.T = 1

    @torch.no_grad()
    def eval(self, data_loader):
        result = []
        
        for (images, _) in tqdm(data_loader):
            images = images.to(self.device)
            image_features = self.model.get_image_feature(images)

            output = image_features @ self.text_features.T

            smax = (F.softmax(output / self.T, dim=1)).data.cpu().numpy()
            output = np.max(smax, axis=1)

            result.append(output)

        return np.concatenate(result)
