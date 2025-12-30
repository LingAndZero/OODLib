import os

from datasets.base_dataset import BaseDataset
from datasets.registry import register_dataset

from torchvision import datasets
from torchvision import transforms


@register_dataset("cifar100")
class CIFAR100(BaseDataset):
    
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

        self.dataset = datasets.CIFAR100(
            root=os.path.join(self.root, "cifar100"),
            train=(split == "train"),
            transform=self.train_transform,
            download=True,
        )

        self.class_name = [
            'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee',
            'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus',
            'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
            'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab',
            'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish',
            'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
            'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man',
            'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom',
            'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
            'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
            'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
            'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
            'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper',
            'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
            'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale',
            'willow_tree', 'wolf', 'woman', 'worm'
        ]