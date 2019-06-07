import torch
from PIL import Image
from torch.utils.data.dataset import Dataset


class PatchDataset(Dataset):
	def __init__(self, root, split='train', transform=None):
		self.root = root
		self.transform = transform
		self.patches = self.__load_patches()

	def __load_patches(self):


	def __len__(self):
		return len(self.patches)

	def __getitem__(self, x):
		path, (x, y, w, h), class_id = self.patches[x]
		mat = np.asarray(Image.open(path))
		
		x = Image.fromarray(mat[y:y + h, x:x + w, :])
		y = torch.tensor([class_id])

		if self.transform is not None:
			x = self.transform(x)

		return x, y