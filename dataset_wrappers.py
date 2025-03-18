from torch.utils.data import Dataset
from torchvision import datasets, transforms


class TransformedDataset(Dataset):
    def __init__(self, base_dataset, transform=None):
        self.base_dataset = base_dataset
        self.transform = transform

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.base_dataset)

    @property
    def dataset(self):
        if hasattr(self.base_dataset, "dataset"):
            return self.base_dataset.dataset
        return self.base_dataset