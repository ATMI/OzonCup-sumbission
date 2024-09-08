from torch import nn
from torchvision import models


class Model(nn.Module):
	def __init__(self):
		super(Model, self).__init__()

		self.model = models.efficientnet_v2_s(
			weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
		)
		self.model.classifier = nn.Sequential(
			nn.Dropout(0.2),
			nn.Linear(1280, 1),
			nn.Sigmoid(),
		)

	def forward(self, x):
		x = self.model(x)
		return x
