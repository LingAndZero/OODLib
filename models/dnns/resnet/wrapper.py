from models.base_models import BaseModel
from models.registry import register_model

from .architecture import resnet50


@register_model("resnet")
class ResNet(BaseModel):
    def __init__(self, num_classes, device):
        super().__init__(num_classes, device)
        self.model = resnet50(num_classes=num_classes, pretrained=True).to(device)
        self.linear = self.model.fc

    def eval(self):
        self.model.eval()

    def get_output(self, x, return_feature=False):
        feat = self.model.get_feature(x)
        logits = self.linear(feat)
        if return_feature:
            return logits, feat
        return logits
    
    def get_feature(self, x):
        return self.model.get_feature(x)
