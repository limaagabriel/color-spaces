import h5py
from data.datasets.patch import PatchDataset


def save_patch_dataset_to_h5(root, path, transform=None, splits=['train', 'valid', 'test']):
	sets = map(lambda x: (x, PatchDataset(root, split=x, transform=transform)), splits)
	
	def get_data(dataset, index):
		data = []

		for patch in dataset.patches:
			item = patch[index]
			data.append(item)

		return np.array(data)

	with h5py.File(path, 'w') as f:
		for split, dataset in sets:
			group = f.create_group(split)
			group.create_dataset('x', data=get_data(dataset, 0))
			group.create_dataset('y', data=get_data(dataset, 1))
