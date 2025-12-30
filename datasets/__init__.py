from .registry import register_dataset, get_dataset, list_datasets

from .cifar10 import CIFAR10
from .cifar100 import CIFAR100

__all__ = [
    "cifar10",
    "cifar100"
]