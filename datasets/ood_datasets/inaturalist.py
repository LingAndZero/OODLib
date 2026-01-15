import os

from datasets.base_dataset import BaseDataset
from datasets.registry import register_dataset

from torchvision import datasets
from torchvision import transforms


@register_dataset("iNaturalist")
class iNaturalist(BaseDataset):
    
    def __init__(self, root="./data"):
        super().__init__(root)

        # for CLIP
        normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                         std=(0.26862954, 0.26130258, 0.27577711))  
        test_largescale = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])

        # test_largescale = transforms.Compose([
        #     transforms.Resize(256),
        #     transforms.CenterCrop(224),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        # ])

        self.dataset = datasets.ImageFolder(
            root=os.path.join(self.root, "iNaturalist"),
            transform=test_largescale,
        )

        self.class_names = self.dataset.classes
