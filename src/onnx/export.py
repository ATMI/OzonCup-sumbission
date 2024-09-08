import argparse
from pathlib import Path

import torch
import torch.onnx

from src.model import Model

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("checkpoint", type=Path, help="Path to checkpoint")

	args = parser.parse_args()

	with torch.no_grad():
		device = torch.device("cpu")
		model = Model()
		model.eval()

		checkpoint = torch.load(args.checkpoint, map_location=device)
		model.load_state_dict(checkpoint["model"])

		x = torch.randn(1, 3, 384, 384)
		torch.onnx.export(model, x, "model.onnx", input_names=["x"], output_names=["y"])
