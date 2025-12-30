from baselines import get_baseline
from datasets import get_dataset, list_datasets


if __name__ == "__main__":

    dataset = get_dataset("cifar100", root="./data", split="train")
    print(dataset.class_name)