import torch
import torch.nn as nn
import torch.optim as optim

from libs.model import TumorClassifier
from libs.my_transforms import get_train_transform, get_eval_transform
from libs.utils import get_base_path
from libs.data_loader import get_dataloaders
from libs.evaluate import evaluate

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    base_path = get_base_path()

    train_transform = get_train_transform()
    eval_transform  = get_eval_transform()

    train_loader, val_loader = get_dataloaders(
        base_path, train_transform, eval_transform
    )

    model = TumorClassifier(num_classes=4).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.001
    )

    num_epochs = 20

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()            
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs} Loss: {running_loss / len(train_loader)}")

        val_acc, _, _, _ = evaluate(model, val_loader, device)
        print("Validation accuracy:", val_acc)

    torch.save(model.state_dict(), "models/tumor_classifier.pth")


if __name__ == "__main__":
    train()