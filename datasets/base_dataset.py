from torch.utils.data import Dataset


class BaseDataset(Dataset):

    def __init__(self, root):
        super().__init__()
        self.root = root
        self.test_dataset = None
        self.num_classes = None
        self.class_names = None

    def __len__(self):
        if self.test_dataset is None:
            raise RuntimeError("Dataset is not initialized.")
        return len(self.test_dataset)

    def __getitem__(self, idx):
        if self.test_dataset is None:
            raise RuntimeError("Dataset is not initialized.")
        return self.test_dataset[idx]