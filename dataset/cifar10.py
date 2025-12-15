from dataset.base_dataset import BaseDataset
from dataset.registry import register_dataset
import torch


@register_dataset("cifar10")
class CIFAR10(BaseDataset):
    pass