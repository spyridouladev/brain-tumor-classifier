import torch
import os
from libs.model import TumorClassifier
from libs.utils import get_base_path

def load_model(device):
    base_path = get_base_path()
    model_path = os.path.join(base_path,"models","tumor_classifier.pth")

    model = TumorClassifier(num_classes=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model