from baselines import get_baseline
from datasets import get_dataset, list_datasets


if __name__ == "__main__":

    dataset = get_dataset(
        "cifar10",
        root="./data",
    )
    print(dataset[0])