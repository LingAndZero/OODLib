from matplotlib import image
from models.base_models import BaseModel
from models.registry import register_model

from clip import load, tokenize
import torch

@register_model("clip")
class CLIP(BaseModel):
    def __init__(self, num_classes, device):
        super().__init__(num_classes, device)
        self.model, _ = load("ViT-B/16", device)
        self.texts = None

    def eval(self):
        self.model.eval()

    def get_texts(self, prompt, labels):
        return tokenize([prompt.format(label) for label in labels])

    def get_output(self, image, text, return_feature=False):
        image_features = self.model.encode_image(image)
        text_features = self.model.encode_text(text)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        logits =  image_features @ text_features.t()
        if return_feature:
            features = {
                 "image": image_features,
                 "text": text_features,
             }
            return logits, features
        return logits

    def get_text_feature(self, x):
        with torch.no_grad():
            text_features = self.model.encode_text(x)
            text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features
    
    def get_image_feature(self, x):
        image_features = self.model.encode_image(x)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        
        return image_features
