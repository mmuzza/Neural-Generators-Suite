import torch
from torchvision import datasets, transforms
import os

data_dir = "./data"
os.makedirs(data_dir, exist_ok=True)


transform = transforms.Compose([
    transforms.ToTensor(),
])


def check_dataset(dataset, name, train_len=60000, test_len=10000):
    train_size = len(dataset(root=data_dir, train=True, download=False, transform=transform))
    test_size = len(dataset(root=data_dir, train=False, download=False, transform=transform))

    print(f"{name} train_size: {train_size}")
    print(f"{name} test_size: {test_size}")
    if train_size != train_len:
        print('Unexpected train size. please try redownloading')
    else:
        print(f'train dataset downloaded successfully for {name}')
    if test_size != test_len:
        print('Unexpected test size. please try redownloading')
    else:
        print(f'test dataset downloaded successfully for {name}')


# MNIST
def download_mnist():
    print("Downloading MNIST dataset")
    datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    check_dataset(datasets.MNIST, 'MNIST')

# Fashion MNIST
def download_fashion_mnist():
    print("Downloading Fashion MNIST dataset")
    datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
    datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)
    check_dataset(datasets.FashionMNIST, 'Fashion MNIST')



download_mnist()
download_fashion_mnist()

