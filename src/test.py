import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as f
import torchvision.transforms.functional as tf
from torch import nn
from torch.utils import data
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from dataset import Dataset
from model import Model
from util.metrics import ConfMatrix, conf_matrix


def report_miss(
	report_dir: Path,
	batch: int,
	paths: List[str],
	images: torch.Tensor,
	miss: torch.Tensor
):
	if miss.sum().item() == 0:
		return

	matching = (report_dir / "matching.txt").open("a")

	miss = miss.cpu().squeeze()
	paths = [i for i, m in zip(paths, miss) if m]
	images = images[miss]

	for idx, (image, path) in enumerate(zip(images, paths)):
		image_name = f"{batch}_{idx}.jpg"
		image_path = report_dir / image_name
		matching.write(f"{image_name} -> {path}\n")

		image = tf.to_pil_image(image)
		image.save(image_path)


def test(
	model: nn.Module,
	device: torch.device,
	loader: data.DataLoader,
	test_dir: Optional[Path] = None,
) -> Tuple[float, ConfMatrix]:
	if test_dir is None:
		fp_dir = None
		fn_dir = None
	else:
		if test_dir.exists():
			print("Test directory exists, skipping test")
			exit(1)
		else:
			test_dir.mkdir(parents=True)

			fp_dir = test_dir / "fp"
			fn_dir = test_dir / "fn"

			fp_dir.mkdir()
			fn_dir.mkdir()

	with torch.no_grad():
		model.eval()
		running_loss = 0.0
		conf_mat = ConfMatrix()

		for batch, (X, Y) in enumerate(tqdm(loader, leave=False)):
			x = X["normalized"].to(device)
			y = Y["encoding"].to(device)

			y_prob = model(x)

			l = Y["label"].to(device)
			weight = torch.where(l == 0, 1, 3)

			loss = f.binary_cross_entropy(y_prob, y, weight)
			running_loss += loss.item()

			paths = X["path"]
			images = X["image"]
			y_pred = y_prob > 0.5

			tp, tn, fp, fn = conf_matrix(y_pred, l)
			if fp_dir is not None:
				report_miss(fp_dir, batch, paths, images, fp)
			if fn_dir is not None:
				report_miss(fn_dir, batch, paths, images, fn)

			conf_mat += ConfMatrix.from_mat((tp, tn, fp, fn))

	running_loss /= len(loader)
	return running_loss, conf_mat


def main():
	parser = argparse.ArgumentParser(description="Tests the model")
	parser.add_argument("checkpoint", type=Path, help="Model to test")
	parser.add_argument("dataset", type=Path, help="The dataset to use")
	parser.add_argument("batch_size", type=int, help="The batch size to use")
	parser.add_argument("--train_set", help="Test on train set", action="store_true")

	args = parser.parse_args()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = Model()
	model.to(device)

	checkpoint = torch.load(args.checkpoint, map_location=device)
	model.load_state_dict(checkpoint["model"])
	epoch = checkpoint["epoch"]

	now = datetime.now()
	now = now.strftime("%m-%d_%H-%M-%S")
	test_dir = Path("tests") / now

	train_set = bool(args.train_set)
	dataset = Dataset.load(args.dataset, train_set)
	dataset = dataset(384, InterpolationMode.BILINEAR)
	loader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

	loss, conf = test(model, device, loader, test_dir)

	log = test_dir / "log.txt"
	log = log.open("w")

	msg = f"[Epoch: {epoch}]"
	print(msg, file=log)
	print(msg)

	msg = f"[{now}] loss: {loss:.4f}, {conf}"
	print(msg, file=log)
	print(msg)

	log.close()


if __name__ == "__main__":
	main()
