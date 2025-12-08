import os
from PIL import Image
import torch
import torch.utils.data
import torchvision
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt


### GLOBAL VARIABLES ###
BATCH_SIZE = 10


class SegmentationDataset(torch.utils.data.Dataset):
    """Class for getting individual transformations and data
    Args:
        images_dir = path of input images
        labels_dir = path of labeled images
        transformI = Input Images transformation (default: None)
        transformM = Input Labels transformation (default: None)
    Output:
        img = Transformed images
        label = Transformed labels"""

    def __init__(self, images_dir, labels_dir, transformI=None, transformM=None):
        self.images = sorted(os.listdir(images_dir))
        self.labels = sorted(os.listdir(labels_dir))
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transformI = transformI
        self.transformM = transformM

        if self.transformI:
            self.tx = self.transformI
        else:
            self.tx = torchvision.transforms.Compose([
                torchvision.transforms.Resize((128, 128)),
                torchvision.transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        if self.transformM:
            self.lx = self.transformM
        else:
            self.lx = torchvision.transforms.Compose([
                torchvision.transforms.Resize((128, 128)),
                torchvision.transforms.Grayscale(),
                torchvision.transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        i1 = Image.open(self.images_dir + self.images[i])
        l1 = Image.open(self.labels_dir + self.labels[i])

        img = self.tx(i1)
        label = self.lx(l1)

        return img, label


if __name__ == '__main__':
    PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))
    print(PROJECT_ROOT)
    # Modify paths as needed
    train_path = os.path.join(PROJECT_ROOT, "../data_samples/images/")
    train_mask_path = os.path.join(PROJECT_ROOT, "../data_samples/masks/")
    
    Training_Data = SegmentationDataset(train_path, train_mask_path)

    num_train = len(Training_Data)
    indices = list(range(num_train))
    split = int(np.floor(0.1 * num_train))

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(Training_Data, batch_size=BATCH_SIZE, sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(Training_Data, batch_size=1, sampler=valid_sampler)
    
    # Display image and label.
    train_features, train_labels = next(iter(train_loader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    
    img = train_features[0].permute(1, 2, 0).numpy()
    label = train_labels[0].permute(1, 2, 0).numpy()
    
    plt.subplot(2, 1, 1)
    plt.imshow(img)
    plt.subplot(2, 1, 2)
    plt.imshow(label, cmap="gray")
    plt.show()

    print("finish loading")
