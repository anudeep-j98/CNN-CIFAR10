# Import necessary packages.
import torch
from torchvision import datasets, transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from torch.utils.data import DataLoader, Dataset

def get_data_loader(batch_size):

    def calculate_mean_std(dataset):
        data = np.concatenate([np.asarray(dataset[i][0]) for i in range(len(dataset))], axis=0)
        mean = data.mean(axis=(0, 1, 2)) / 255.0
        std = data.std(axis=(0, 1, 2)) / 255.0
        return mean, std

    class CIFAR10Albumentations(Dataset):
        def __init__(self, dataset, transform):
            self.dataset = dataset
            self.transform = transform

        def __len__(self):
            return len(self.dataset)

        @staticmethod
        def transform_to_float(tensor):
          # Ensure the tensor is float32
          return tensor.float()

        def __getitem__(self, idx):
            image, label = self.dataset[idx]
            image = np.array(image)  # Convert PIL image to numpy array
            augmented = self.transform(image=image)
            image = augmented['image']
            image_tensor = self.transform_to_float(image)
            return image_tensor, label

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True)
    mean, std = calculate_mean_std(train_dataset)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True)


    transform_train = A.Compose([
        A.Resize(64, 64),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
        A.CoarseDropout(
            max_holes=1, max_height=16, max_width=16,
            min_holes=1, min_height=16, min_width=16,
            fill_value=mean, mask_fill_value=None
        ),
        A.ColorJitter(brightness=0.10, contrast=0.1, saturation=0.10, hue=0.1),
        A.Normalize(mean=mean, std=std),  # Normalization
        A.pytorch.ToTensorV2(),  # Convert to PyTorch tensor
    ])

    transform_test = A.Compose([
        A.Resize(64, 64),
        A.Normalize(mean=mean, std=std),  # Normalization
        A.pytorch.ToTensorV2(),  # Convert to PyTorch tensor
    ])



    train_dataset_alb = CIFAR10Albumentations(train_dataset, transform_train)
    test_dataset_alb = CIFAR10Albumentations(test_dataset, transform_test)

    train_loader = DataLoader(train_dataset_alb, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset_alb, batch_size=batch_size, shuffle=True, num_workers=0)

    return train_loader, test_loader