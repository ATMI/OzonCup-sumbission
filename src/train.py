import argparse
from pathlib import Path
from typing import Tuple, Optional

import torch
import torch.nn.functional as f
from torch import nn, optim
from torch.utils import data
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

from dataset import Dataset
from model import Model
from util.yes_no_prompt import yes_no_prompt
from util.metrics import ConfMatrix
from test import test


def train_epoch(
	model: nn.Module,
	device: torch.device,
	loader: data.DataLoader,
	optimizer: optim.Optimizer
) -> Tuple[float, ConfMatrix]:
	model.train()
	running_loss = 0.0
	conf_mat = ConfMatrix()

	for X, Y in tqdm(loader, leave=False):
		x = X["normalized"].to(device)
		y = Y["encoding"].to(device)

		y_prob = model(x)

		l = Y["label"].to(device)
		weight = torch.where(l == 0, 1, 3)

		loss = f.binary_cross_entropy(y_prob, y, weight)
		running_loss += loss.item()

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		y_pred = y_prob > 0.5
		conf_mat += ConfMatrix.from_pred(y_pred, l)

	running_loss /= len(loader)
	return running_loss, conf_mat


def save_checkpoint(
	checkpoints_dir: Path,
	epoch: int,
	model: nn.Module,
	optimizer: optim.Optimizer
) -> None:
	obj = {
		"epoch": epoch,
		"model": model.state_dict(),
		"optimizer": optimizer.state_dict(),
	}
	torch.save(obj, str(checkpoints_dir / f"{epoch}.pth"))


def progress_message(train: bool, loss: float, conf: ConfMatrix):
	prefix = "Train" if train else "Valid"
	msg = f"[{prefix}] loss: {loss:.4f}, {conf}"
	return msg


def log_print(msg: str, file) -> None:
	print(msg)
	print(msg, file=file)
	file.flush()


def train(
	model: nn.Module,
	optimizer: optim.Optimizer,

	dataset: Path,
	epochs: int,
	batch_size: int,
	checkpoint: Optional[Path],
):
	log = open("log.txt", "w")
	checkpoints = Path("checkpoints")

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)

	if checkpoints.exists():
		if checkpoint is None:
			epoch = 0
		else:
			checkpoint = torch.load(checkpoint, map_location=device)

			epoch = checkpoint["epoch"]
			epochs += epoch

			model.load_state_dict(checkpoint["model"])
			optimizer.load_state_dict(checkpoint["optimizer"])
	else:
		checkpoints.mkdir()
		epoch = 0

	train_set = Dataset.load(dataset, True)
	test_set = Dataset.load(dataset, False)

	train_set = train_set(384, InterpolationMode.BILINEAR)
	test_set = test_set(384, InterpolationMode.BILINEAR)

	train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
	test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

	while epoch < epochs:
		try:
			epoch += 1
			msg = f"[Epoch: {epoch}]"
			log_print(msg, log)

			loss, conf = train_epoch(model, device, train_loader, optimizer)
			msg = progress_message(True, loss, conf)
			log_print(msg, log)

			save_checkpoint(checkpoints, epoch, model, optimizer)

			loss, conf = test(model, device, test_loader)
			msg = progress_message(False, loss, conf)
			log_print(msg, log)
		except KeyboardInterrupt:
			if yes_no_prompt("Stop training?", default="no"):
				break

			epoch -= 1
			epoch = max(epoch, 0)

	log.close()


def main():
	parser = argparse.ArgumentParser(description="Trains the model")
	parser.add_argument("dataset", type=Path, help="The dataset to use")
	parser.add_argument("epochs", type=int, help="The number of epochs to train")
	parser.add_argument("batch_size", type=int, help="The batch size to use")
	parser.add_argument("--checkpoint", type=Path, help="Checkpoint")
	parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")

	args = parser.parse_args()

	model = Model()
	optimizer = optim.SGD(model.parameters(), lr=args.lr)

	train(model, optimizer, args.dataset, args.epochs, args.batch_size, args.checkpoint)


if __name__ == "__main__":
	main()
