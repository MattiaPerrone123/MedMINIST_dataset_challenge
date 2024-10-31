import os
import PIL
import torch
from torchvision import transforms

def prepare_output_dir(output_root, data_flag):
    """
    Prepares and returns a unique directory for storing output files
    """
    output_dir = os.path.join(output_root, data_flag)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def get_data_transform(resize):
    """
    Returns appropriate data transformations based on the resize flag
    """
    if resize:
        return transforms.Compose([
            transforms.Resize((224, 224), interpolation=PIL.Image.NEAREST),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

def load_datasets(DataClass, transform, download, as_rgb, size):
    """
    Loads and returns train, validation, and test datasets
    """
    return (
        DataClass(split='train', transform=transform, download=download, as_rgb=as_rgb, size=size),
        DataClass(split='val', transform=transform, download=download, as_rgb=as_rgb, size=size),
        DataClass(split='test', transform=transform, download=download, as_rgb=as_rgb, size=size)
    )

def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size):
    """
    Creates and returns data loaders for train, validation, and test sets
    """
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    train_loader_at_eval = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, train_loader_at_eval, val_loader, test_loader
