import os

from datasets.base_dataset import BaseDataset
from datasets.registry import register_dataset

from torchvision import datasets
from torchvision import transforms


@register_dataset("cifar10")
class CIFAR10(BaseDataset):
    
    def __init__(self, root, split):
        super().__init__(root)
        self.train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        ])
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        ])

        self.dataset = datasets.CIFAR10(
            root=os.path.join(self.root, "cifar10"),
            train=(split == "train"),
            transform=self.train_transform,
            download=True,
        )

        self.class_name = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']