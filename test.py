import torch
import pandas as pd

from libs.utils import get_base_path
from libs.my_transforms import get_eval_transform
from libs.data_loader import get_test_loader
from libs.model_loader import load_model
from libs.evaluate import evaluate

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    base_path = get_base_path()
    eval_transform = get_eval_transform()

    test_loader, test_dataset = get_test_loader(base_path, eval_transform)

    model = load_model(device)

    acc, cm, _, _ = evaluate(model, test_loader, device)

    print("Testing samples:", len(test_dataset))
    print("Test accuracy:", acc)

    df_cm = pd.DataFrame(cm, index=test_dataset.classes, columns=test_dataset.classes)
    print(df_cm)


if __name__ == "__main__":
    test()