from torchvision import datasets
import torch
import os

def get_dataloaders(base_path, train_transform, eval_transform, batch_size=32):
    train_path = os.path.join(base_path, "dataset_split", "train")
    val_path   = os.path.join(base_path, "dataset_split", "val")

    train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
    val_dataset   = datasets.ImageFolder(val_path, transform=eval_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader


def get_test_loader(base_path, eval_transform, batch_size=32):
    test_path = os.path.join(base_path, "dataset_split", "test")

    test_dataset = datasets.ImageFolder(test_path, transform=eval_transform)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return test_loader, test_dataset