import h5py
from torch.utils.data.dataset import Dataset


class H5Dataset(Dataset):
    def __init__(self, path, split='train', transform=None):
        self.path = path
        self.split = split
        self.transform = transform

        self.__file = h5py.File(path, 'r')
        self.__dataset = self.__file[split]
        self.__len = self.__dataset['y'].shape[0]

    def __del__(self):
        self.__file.close()

    def __len__(self):
        return self.__len

    def __getitem__(self, index):
        sample = self.__dataset['x'][index]
        output = self.__dataset['y'][index]
        
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, output
