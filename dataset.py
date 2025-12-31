import os

from PIL import Image
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class CustomDataset(datasets.ImageFolder):
    """带缓存的ImageFolder数据集，避免重复读取图片"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_data = []
        for img_path, label in tqdm(self.samples, desc="Caching data"):
            with open(img_path, "rb") as f:
                img = Image.open(f)
                img = img.convert("RGB")
                self.cache_data.append((img, label))

    def __getitem__(self, index: int) -> tuple[Image.Image, int]:
        img, label = self.cache_data[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    def __len__(self) -> int:
        return len(self.cache_data)


def get_data_loaders(
    data_dir: str,
    image_size: tuple[int, int] = (32, 32),
    batch_size: int = 128,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader]:
    transform_train = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAffine(
                degrees=0, translate=(0.1, 0.1), shear=10, scale=(0.8, 1.2)
            ),
            transforms.ToTensor(),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ]
    )
    train_dataset = CustomDataset(
        root=os.path.join(data_dir, "train"), transform=transform_train
    )
    test_dataset = CustomDataset(
        root=os.path.join(data_dir, "test"), transform=transform_test
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
    )

    return train_loader, test_loader
