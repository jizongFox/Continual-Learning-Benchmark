import os

import torchvision
from torchvision import transforms

from .augment import RandomShift, DownSample
from .haxio_dataset import ImageFolder
from .wrapper import CacheClassLabel


def MNIST(dataroot, train_aug=False):
    # Add padding to make 32x32
    # normalize = transforms.Normalize(mean=(0.1307,), std=(0.3081,))  # for 28x28
    normalize = transforms.Normalize(mean=(0.1000,), std=(0.2752,))  # for 32x32

    val_transform = transforms.Compose([
        transforms.Pad(2, fill=0, padding_mode='constant'),
        transforms.ToTensor(),
        normalize,
    ])
    train_transform = val_transform
    if train_aug:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            normalize,
        ])

    train_dataset = torchvision.datasets.MNIST(
        root=dataroot,
        train=True,
        download=True,
        transform=train_transform
    )
    train_dataset = CacheClassLabel(train_dataset)

    val_dataset = torchvision.datasets.MNIST(
        dataroot,
        train=False,
        transform=val_transform
    )
    val_dataset = CacheClassLabel(val_dataset)

    return train_dataset, val_dataset


def CIFAR10(dataroot, train_aug=False):
    normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    train_transform = val_transform
    if train_aug:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=dataroot,
        train=True,
        download=True,
        transform=train_transform
    )
    train_dataset = CacheClassLabel(train_dataset)

    val_dataset = torchvision.datasets.CIFAR10(
        root=dataroot,
        train=False,
        download=True,
        transform=val_transform
    )
    val_dataset = CacheClassLabel(val_dataset)

    return train_dataset, val_dataset


def CIFAR100(dataroot, train_aug=False):
    normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    train_transform = val_transform
    if train_aug:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    train_dataset = torchvision.datasets.CIFAR100(
        root=dataroot,
        train=True,
        download=True,
        transform=train_transform
    )
    train_dataset = CacheClassLabel(train_dataset)

    val_dataset = torchvision.datasets.CIFAR100(
        root=dataroot,
        train=False,
        download=True,
        transform=val_transform
    )
    val_dataset = CacheClassLabel(val_dataset)

    return train_dataset, val_dataset


def HaxioDataset(dataroot, train_aug=False):
    # sorted_classes = ["Good", "Damage", "Cap", "Hair", "Black"]
    sorted_classes = ["Good", "Aesthetic", "Empty", "Hair", "NoCap", "Particle", "Unfilled"]

    val_transform = transforms.Compose([
        DownSample(5),
        transforms.ToTensor(),
    ])
    train_transform = val_transform
    if train_aug:
        train_transform = transforms.Compose([
            DownSample(5),
            RandomShift(0.1, 0.1),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

    train_dataset = ImageFolder(
        root=os.path.join(dataroot, "train"),
        transform=train_transform, customized_classes=sorted_classes
    )
    val_dataset = ImageFolder(
        root=os.path.join(dataroot, "val"), customized_classes=sorted_classes,
        transform=val_transform
    )
    assert train_dataset.classes == val_dataset.classes

    train_dataset = CacheClassLabel(train_dataset)
    val_dataset = CacheClassLabel(val_dataset)

    return train_dataset, val_dataset
