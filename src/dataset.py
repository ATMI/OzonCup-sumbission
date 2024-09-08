import argparse
import dataclasses
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Callable, TypeAlias

import torch
from PIL import Image
from torch.utils import data
from torchvision.transforms import InterpolationMode, transforms
import torch.nn.functional as f

Constructor: TypeAlias = Callable[[int, InterpolationMode], "Dataset"]
BiConstructor: TypeAlias = Callable[[int, InterpolationMode], Tuple["Dataset", "Dataset"]]


class Dataset(data.Dataset):
	@dataclass
	class X:
		path: str
		image: torch.Tensor
		normalized: torch.Tensor

	@dataclass
	class Y:
		label: torch.Tensor
		encoding: torch.Tensor

	def __init__(
		self,

		path: Path,
		train: bool,
		items: List[str],

		image_size: int,
		interpolation: InterpolationMode = InterpolationMode.BILINEAR,
	):
		self.path = path.resolve()
		self.train = train
		self.items = items

		self.image_size = image_size
		self.interpolation = interpolation

		if self.train:
			self.transform = transforms.Compose([
				transforms.RandomHorizontalFlip(p=0.35),
				# transforms.RandomVerticalFlip(p=0.25),
				transforms.RandomAffine(degrees=(-10, 10), translate=(0.06, 0.06), scale=(1, 1.2), shear=(-10, 10)),
				transforms.Resize(size=(image_size, image_size), interpolation=interpolation),
				# transforms.RandomApply(transforms=[transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.5)], p=0.3),
				# transforms.RandomApply(transforms=[transforms.GaussianBlur(kernel_size=3, sigma=1.2)], p=0.2),
				transforms.RandomGrayscale(p=0.1),
				transforms.ToTensor(),
			])
		else:
			self.transform = transforms.Compose([
				transforms.Resize(size=(image_size, image_size), interpolation=interpolation),
				transforms.ToTensor(),
			])

		self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

	def __len__(self) -> int:
		return len(self.items)

	def __getitem__(self, index: int):
		path = self.items[index]

		image = Image.open(self.path / path)
		if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
			white_background = Image.new("RGB", image.size, (255, 255, 255))
			image = image.convert("RGBA")

			white_background.paste(image, mask=image.split()[3])
			image = white_background
		else:
			image = image.convert("RGB")

		image = self.transform(image)
		normalized = self.normalize(image)

		cls = Path(path).parts[0]
		cls = int(cls)

		label = torch.tensor([cls])
		# encoding = f.one_hot(label, 2).float()
		encoding = label.float()

		x = Dataset.X(path, image, normalized)
		y = Dataset.Y(label, encoding)

		x = dataclasses.asdict(x)
		y = dataclasses.asdict(y)

		return x, y

	@staticmethod
	def __save__(path: Path, train: bool, items: List[str]):
		filename = path / ("train" if train else "test")
		with open(filename, "w") as f:
			for item in items:
				f.write(item + "\n")

	@staticmethod
	def load_items(path: str | Path) -> List[str]:
		if isinstance(path, str):
			path = Path(path)

		with open(path, "r") as f:
			items = f.readlines()
		items = [item.rstrip("\n") for item in items]

		return items

	@staticmethod
	def load(path: str | Path, train: bool) -> Constructor:
		def constructor(image_size: int, interpolation: InterpolationMode):
			filename = path / ("train" if train else "test")
			items = Dataset.load_items(filename)
			dataset = Dataset(path, train, items, image_size, interpolation)
			return dataset

		return constructor

	@staticmethod
	def create(path: Path, ratio: float = 0.8) -> BiConstructor:
		train_items = []
		test_items = []

		for cls in [0, 1]:
			for dir_path, subdir_list, file_list in os.walk(path / str(cls)):
				dir_path = Path(dir_path).relative_to(path)
				items = [str(dir_path / file) for file in file_list]

				if len(items) > 0:
					random.shuffle(items)

					idx = int(len(items) * ratio)
					train_items += items[:idx]
					test_items += items[idx:]

		random.shuffle(train_items)
		random.shuffle(test_items)

		Dataset.__save__(path, True, train_items)
		Dataset.__save__(path, False, test_items)

		def constructor(image_size: int, interpolation: InterpolationMode):
			train_set = Dataset(path, True, train_items, image_size, interpolation)
			test_set = Dataset(path, False, test_items, image_size, interpolation)
			return train_set, test_set

		return constructor

	def save(self):
		Dataset.__save__(self.path, self.train, self.items)


def clean_dataset(src: Path, dst: Path, min_size: int):
	good_dir = dst / "good"
	bad_dir = dst / "bad"

	for dir_path, subdir_list, file_list in os.walk(src):
		for file_name in file_list:
			file_path = Path(dir_path) / file_name
			file_rel_path = file_path.relative_to(src)

			try:
				image = Image.open(file_path)
				image = image.convert("RGB")
				image.verify()

				if min(image.width, image.height) < min_size:
					raise ValueError(f"Image is too small: {file_path}")

				file_dst = good_dir / file_rel_path
			except Exception as e:
				file_dst = bad_dir / file_rel_path

			file_dst.parent.mkdir(parents=True, exist_ok=True)
			shutil.copyfile(file_path, file_dst)


def update_dataset(src: Path, dst: Path, ratio: float = 0.8):
	train_items = Dataset.load_items(src / "train")
	test_items = Dataset.load_items(src / "test")

	train_items = [item for item in train_items if (src / item).is_file()]
	test_items = [item for item in test_items if (src / item).is_file()]

	train_items = set(train_items)
	test_items = set(test_items)

	newbies = []
	for cls in [0, 1]:
		items = []
		for dir_path, subdir_list, file_list in os.walk(src / str(cls)):
			dir_path = Path(dir_path).relative_to(src)
			items += [str(dir_path / file) for file in file_list]
		items = set(items)
		newbies.append(items)

	newbies = [(cls - train_items).intersection(cls - test_items) for cls in newbies]

	train_items = list(train_items)
	test_items = list(test_items)

	for cls in newbies:
		cls = list(cls)
		random.shuffle(cls)

		idx = int(len(cls) * ratio)
		train_items += cls[:idx]
		test_items += cls[idx:]

	random.shuffle(train_items)
	random.shuffle(test_items)

	Dataset.__save__(dst, True, train_items)
	Dataset.__save__(dst, False, test_items)


def main():
	parser = argparse.ArgumentParser(description="Dataset utils")
	subparsers = parser.add_subparsers(dest="command")

	clean_parser = subparsers.add_parser("clean", help="Removes too small, broken image files")
	clean_parser.add_argument("src", type=Path, help="Path to the source folder")
	clean_parser.add_argument("dst", type=Path, help="Path to the destination folder")
	clean_parser.add_argument("min_size", type=int, help="Lower bound for smallest image dim")

	create_parser = subparsers.add_parser("create", help="Creates dataset")
	create_parser.add_argument("path", type=Path, help="Path to the source folder")
	create_parser.add_argument("ratio", type=float, help="Split ratio")

	update_parser = subparsers.add_parser("update", help="Updates dataset based on the files")
	update_parser.add_argument("src", type=Path, help="Path to the source dataset")
	update_parser.add_argument("dst", type=Path, help="Path to save the updated dataset")
	update_parser.add_argument("ratio", type=float, help="Split ratio")

	args = parser.parse_args()

	match args.command:
		case "clean":
			clean_dataset(args.src, args.dst, args.min_size)
		case "create":
			Dataset.create(args.path, args.ratio)
		case "update":
			update_dataset(args.src, args.dst, args.ratio)


if __name__ == "__main__":
	main()
