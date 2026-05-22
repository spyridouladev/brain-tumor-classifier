import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class TumorClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        # Load pretrained ResNet
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # Replace final layer
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)