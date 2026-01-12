from matplotlib import image
from models.base_models import BaseModel
from models.registry import register_model

from clip import load, tokenize


@register_model("clip")
class CLIP(BaseModel):
    def __init__(self, num_classes, device):
        super().__init__(num_classes, device)
        self.model = load("ViT-B/16", device)
        # texts = tokenize([f"a photo of a {label}" for label in ID_labels]).to(device)

    def get_output(self, image, text):
        image_features = self.model.encode_image(image)
        text_features = self.model.encode_text(text)

    # def get_text_feature(self, x):
    #     return self.model.encode_text(x)
    
    # def get_image_feature(self, x):
    #     return self.model.encode_image(x)