from torch.utils.data import Dataset


class BaseDataset(Dataset):

    def __init__(self, root):
        super().__init__()
        self.root = root
        self.dataset = None
        self.num_classes = None
        self.class_names = None

    def __len__(self):
        if self.dataset is None:
            raise RuntimeError("Dataset is not initialized.")
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.dataset is None:
            raise RuntimeError("Dataset is not initialized.")
        return self.dataset[idx]