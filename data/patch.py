import os
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

	@property
	def num_classes(self):
		return len(self.class_mapper)

	def __get_class_mapper(self, root):
		i = 0
		mapper = {}

		with open(os.path.join(root, 'classes.txt'), 'r+') as f:
			for line in f:
				if line not in mapper:
					i = i + 1
					mapper[line.strip()] = i

		return mapper

	def __get_samples(self):
		def filter_fn(x):
			path = os.path.join(self.root, x)
			return os.path.isfile(path)

		def make_samples_generator(files):
			samples = [] 
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

				patches.append((image_path, (x, y, w, h), class_id))
		return patches

	def __len__(self):
		return len(self.patches)

	def __getitem__(self, index):
		path, (x, y, w, h), class_id = self.patches[index]
		mat = np.asarray(Image.open(path))
		
		sample = Image.fromarray(mat[y:y + h, x:x + w, :])
		output = torch.tensor([class_id])

		if self.transform is not None:
			sample = self.transform(sample)

		return sample, output