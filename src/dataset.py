import os
import logging
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torch

# Enable loading of truncated images, common in real-world datasets
ImageFile.LOAD_TRUNCATED_IMAGES = True

COMMON_IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.tiff')
DEFAULT_NORM_MEAN = [0.485, 0.456, 0.406]
DEFAULT_NORM_STD = [0.229, 0.224, 0.225]

logger = logging.getLogger(__name__)

class CustomImageDataset(Dataset):
    """
    A custom dataset class for loading images from a directory structure
    where subdirectories represent classes.
    """
    def __init__(self, root_dir: str, transform=None):
        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f"Dataset root directory not found: {root_dir}")

        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        self.classes = []

        self._load_dataset()

    def _load_dataset(self):
        """Scans the root directory and populates image paths and labels."""
        logger.info(f"Scanning dataset in: {self.root_dir}")
        class_names = sorted([d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))])

        if not class_names:
            raise ValueError(f"No class subdirectories found in: {self.root_dir}")

        for i, class_name in enumerate(class_names):
            class_path = os.path.join(self.root_dir, class_name)
            self.class_to_idx[class_name] = i
            self.classes.append(class_name)
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                if os.path.isfile(img_path) and img_name.lower().endswith(COMMON_IMAGE_EXTENSIONS):
                    self.image_paths.append(img_path)
                    self.labels.append(i)

        if not self.image_paths:
            raise ValueError(f"No valid images found in the specified directory: {self.root_dir}. "
                             "Please check your dataset structure and image extensions.")

        logger.info(f"Detected {len(self.classes)} classes: {self.classes}")
        logger.info(f"Mapping: {self.class_to_idx}")
        logger.info(f"Total images found: {len(self.image_paths)}")


    def __len__(self) -> int:
        """Returns the total number of images in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        """
        Retrieves an image and its corresponding label at the given index.
        Applies transformations if provided. Handles potential image loading errors.
        """
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}. Returning a black image placeholder.")
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            if self.transform:
                image = self.transform(image)
            return image, self.labels[idx]

        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def prepare_dataloaders(config):
    """
    Prepares and returns training and validation DataLoaders, along with class names.
    Applies appropriate transformations for training and validation.
    """
    img_size = config.image_size

    train_transform = transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(DEFAULT_NORM_MEAN, DEFAULT_NORM_STD)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(DEFAULT_NORM_MEAN, DEFAULT_NORM_STD)
    ])

    full_dataset = CustomImageDataset(root_dir=config.data_dir)
    class_names = full_dataset.classes

    train_size = int(config.train_split_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size

    if train_size == 0 or val_size == 0:
        raise ValueError(f"Dataset split resulted in empty train or validation set. "
                         f"Total images: {len(full_dataset)}, Train ratio: {config.train_split_ratio}")

    generator = torch.Generator().manual_seed(config.seed)
    train_indices, val_indices = random_split(range(len(full_dataset)), [train_size, val_size], generator=generator)

    train_dataset_final = CustomImageDataset(root_dir=config.data_dir, transform=train_transform)
    train_dataset_final.image_paths = [full_dataset.image_paths[i] for i in train_indices.indices]
    train_dataset_final.labels = [full_dataset.labels[i] for i in train_indices.indices]

    val_dataset_final = CustomImageDataset(root_dir=config.data_dir, transform=val_transform)
    val_dataset_final.image_paths = [full_dataset.image_paths[i] for i in val_indices.indices]
    val_dataset_final.labels = [full_dataset.labels[i] for i in val_indices.indices]

    train_loader = DataLoader(
        train_dataset_final,
        batch_size=config.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset_final,
        batch_size=config.batch_size,
        shuffle=False,
    )

    logger.info(f"Training samples: {len(train_dataset_final)} (Batch size: {config.batch_size})")
    logger.info(f"Validation samples: {len(val_dataset_final)} (Batch size: {config.batch_size})")

    return train_loader, val_loader, class_names