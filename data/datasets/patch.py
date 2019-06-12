import os
import time
import torch
import numpy as np
from PIL import Image
from lxml import etree
from torch.utils.data.dataset import Dataset


class PatchDataset(Dataset):
	def __init__(self, root, split='train', transform=None):
		self.transform = transform
		self.root = os.path.join(root, split)
		self.class_mapper = self.__get_class_mapper(root)
		self.patches = self.__load_patches()

		print('[PatchDataset] {} dataset from {} loaded!'.format(split, root))

	@property
	def num_classes(self):
		return len(self.class_mapper)

	def __get_class_mapper(self, root):
		i = 0
		mapper = {}

		with open(os.path.join(root, 'classes.txt'), 'r+') as f:
			for line in f:
				if line not in mapper:
					mapper[line.strip()] = i
					i = i + 1

		return mapper

	def __get_samples(self):
		def filter_fn(x):
			path = os.path.join(self.root, x)
			return os.path.isfile(path)

		def make_samples_generator(files):
			other_files = list(filter(lambda x: not '.xml' in x, files))
			annotation_files = list(filter(lambda x: '.xml' in x, files))

			for annotation_file in annotation_files:
				annotation_name = os.path.splitext(annotation_file)[0]
				for f in other_files:
					f_name = os.path.splitext(f)[0]
					if f_name == annotation_name:
						a = os.path.join(self.root, f)
						b = os.path.join(self.root, annotation_file)
						yield a, b
			
		content = os.listdir(self.root)
		files = list(filter(filter_fn, content))
		return make_samples_generator(files)

	def __load_patches(self):
		data = []
		patches = []

		for image_path, annotation_path in self.__get_samples():
			tree = etree.parse(annotation_path)
			root = tree.getroot()

			for obj in root.findall('object'):
				box = obj.find('bndbox')
				x = int(box.find('xmin').text)
				y = int(box.find('ymin').text)
				w = int(box.find('xmax').text) - int(box.find('xmin').text)
				h = int(box.find('ymax').text) - int(box.find('ymin').text)
				class_id = self.class_mapper.get(obj.find('name').text.strip())

				data.append((image_path, (x, y, w, h), class_id))

		for path, (x, y, w, h), class_id in data:
			mat = np.asarray(Image.open(path))
			sample = mat[y:y + h, x:x + w, :]
			patches.append((sample, class_id))

		return patches

	def __len__(self):
		return len(self.patches)

	def __getitem__(self, index):
		sample, class_id = self.patches[index]

		if self.transform is not None:
			sample = self.transform(sample)

		return sample, class_id

if __name__ == '__main__':
	import h5py
	transform = None
	train_dataset = PatchDataset(os.environ.get('OBJECT_DETECTION_DATASET_PATH'),
								split='train', transform=transform)
	valid_dataset = PatchDataset(os.environ.get('OBJECT_DETECTION_DATASET_PATH'),
								split='valid', transform=transform)
	test_dataset = PatchDataset(os.environ.get('OBJECT_DETECTION_DATASET_PATH'),
								split='test', transform=transform)
	
	def get_data(dataset, index):
		data = []

		for patch in dataset.patches:
			item = patch[index]
			data.append(item)

		return np.array(data)

	with h5py.File('helmintos.h5', 'w') as f:
		splits = [
			('train', train_dataset),
			('valid', valid_dataset),
			('test', test_dataset)
		]

		for split, dataset in splits:
			group = f.create_group(split)
			group.create_dataset('x', data=get_data(dataset, 0))
			group.create_dataset('y', data=get_data(dataset, 1))

