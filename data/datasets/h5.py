import h5py
from torch.utils.data.dataset import Dataset


class H5Dataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

    def __len__(self):
        return 0

    def __getitem(self, index):
        return None