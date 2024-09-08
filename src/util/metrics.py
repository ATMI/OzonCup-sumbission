from typing import Tuple

import torch


def conf_matrix(y_pred, y):
	tp = (y_pred == 1) & (y == 1)
	tn = (y_pred == 0) & (y == 0)
	fp = (y_pred == 1) & (y == 0)
	fn = (y_pred == 0) & (y == 1)

	return tp, tn, fp, fn


def metrics(tp, tn, fp, fn):
	pred_p = tp + fp
	true_p = tp + fn

	total = tp + tn + fp + fn
	accuracy = (tp + tn) / total if total > 0 else 0
	precision = tp / pred_p if pred_p > 0 else 0
	recall = tp / true_p if true_p > 0 else 0

	pr = precision + recall
	f1 = 2 * (precision * recall) / pr if pr > 0 else 0

	return accuracy, precision, recall, f1


class ConfMatrix:
	def __init__(self, tp=0, tn=0, fp=0, fn=0):
		self.tp, self.tn, self.fp, self.fn = tp, tn, fp, fn

	@staticmethod
	def from_pred(y_pred, y):
		conf = conf_matrix(y_pred, y)
		conf = ConfMatrix.from_mat(conf)
		return conf

	@staticmethod
	def from_mat(mat: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
		mat = tuple(i.sum().item() for i in mat)
		conf = ConfMatrix(*mat)
		return conf

	def __add__(self, other):
		result = ConfMatrix(
			self.tp + other.tp,
			self.tn + other.tn,
			self.fp + other.fp,
			self.fn + other.fn,
		)
		return result

	def metrics(self):
		ms = metrics(self.tp, self.tn, self.fp, self.fn)
		ms = [m * 100.0 for m in ms]
		return ms

	def __str__(self):
		acc, pre, rec, f1 = self.metrics()
		s = f"acc: {acc:.2f}, pre: {pre:.2f}, rec: {rec:.2f}, f1: {f1:.2f}"
		return s
