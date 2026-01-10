from models.base_models import BaseModel
from models.registry import register_model

from .architecture import resnet50


@register_model("resnet")
class ResNet(BaseModel):
    def __init__(self, num_classes):
        super().__init__(num_classes)
        self.model = resnet50(num_classes=num_classes, pretrained=True).cuda()

    def get_output(self, x):
        return self.model(x)
    
    def get_feature(self, x):
        pass
