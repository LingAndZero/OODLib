import os

from datasets.base_dataset import BaseDataset
from datasets.registry import register_dataset

from torchvision import datasets
from torchvision import transforms


@register_dataset("Places")
class Places(BaseDataset):
    
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

        self.test_dataset = datasets.ImageFolder(
            root=os.path.join(self.root, "Places"),
            transform=test_largescale,
        )

        self.class_names = self.dataset.classes
