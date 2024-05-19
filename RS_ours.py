import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PPIDSG.models import Generator
import numpy as np
import argparse
import cv2
import os
import matplotlib.pyplot as plt
from PPIDSG.options import args_parser
# get dataset name from command line
args = args_parser()
datas = args.dataset
# hyperparameter setting
# datas = 'FMNIST' # mnist, fmnist, cifar10, svhn
dummys = args.setting  # dummy data: the random input, test:the image from test dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
to_pil_image = transforms.ToPILImage()

if datas == "FMNIST" or datas == "MNIST":
    target_model = Generator(1, 32, 1, 6).to(device)
else:
    target_model = Generator(3, 32, 3, 6).to(device)

# load the trained generater parameters from the client's sharing
if datas == "CIFAR10":
    modelPath = args.model_dir
    target_model.load_state_dict(torch.load(modelPath + "/generator_param.pkl"))
elif datas == "MNIST":
    modelPath = args.model_dir
    target_model.load_state_dict(torch.load(modelPath + "/generator_param.pkl"))
elif datas == "SVHN":
    modelPath = args.model_dir
    target_model.load_state_dict(torch.load(modelPath + "/generator_param.pkl"))
elif datas == "FMNIST":
    modelPath = args.model_dir
    target_model.load_state_dict(torch.load(modelPath + "/generator_param.pkl"))

apply_transform = transforms.Compose([transforms.ToTensor()])

apply_transform_cifar = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

apply_transform_svhn = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
)

apply_transform_mnist = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

if datas == "SVHN":
    train_dataset = datasets.SVHN(
        root="./data/svhn/",
        split="train",
        download=True,
        transform=apply_transform_svhn,
    )
    test_dataset = datasets.SVHN(
        root="./data/svhn/", split="test", download=True, transform=apply_transform_svhn
    )
elif datas == "CIFAR10":
    train_dataset = datasets.CIFAR10(
        root="./data/cifar/", train=True, download=True, transform=apply_transform_cifar
    )
    test_dataset = datasets.CIFAR10(
        root="./data/cifar/",
        train=False,
        download=True,
        transform=apply_transform_cifar,
    )
elif datas == "MNIST":
    train_dataset = datasets.MNIST(
        root="./data/", train=True, download=True, transform=apply_transform_mnist
    )
    test_dataset = datasets.MNIST(
        root="./data/", train=False, download=True, transform=apply_transform_mnist
    )
elif datas == "FMNIST":
    train_dataset = datasets.FashionMNIST(
        root="./data/fmnist/",
        train=True,
        download=True,
        transform=apply_transform_mnist,
    )
    test_dataset = datasets.FashionMNIST(
        root="./data/fmnist/",
        train=False,
        download=True,
        transform=apply_transform_mnist,
    )

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# image
test_loader_X = next(iter(test_loader))[0].clone().detach()
test_loader_X = test_loader_X.to(device)

# label
if datas == 'MNIST' or datas == 'FMNIST':
    test_loader_Y = test_loader_X[1].view(1, 1, 28, 28)
elif datas == 'CIFAR10':
    test_loader_Y = test_loader_X[1].view(1, 3, 32, 32)
elif datas == 'SVHN':
    test_loader_Y = test_loader_X[17].view(1, 3, 32, 32)

if dummys == 'dummy_data':
    orig = torch.randn(test_loader_Y.size()).to(device)
else:
    orig = test_loader_X

_, pred_output = target_model(orig)
img = pred_output[0]
image_np = pred_output[0].permute(1, 2, 0).cpu().detach().numpy()
plt.axis("off")
if image_np.shape[2]==1:
    plt.imshow(image_np,cmap='Greys')
    plt.savefig(f"dummy_{datas}_{dummys}.jpg",bbox_inches='tight',pad_inches=-0.1)
else:
    plt.imshow(image_np)
    plt.savefig(f"dummy_{datas}_{dummys}.jpg",bbox_inches='tight',pad_inches=-0.1)

orig = (orig+1)/2*255
img = (img+1)/2*255
mse = torch.mean((orig-img)**2)

max_pixel = 255
psnr = 20*torch.log(max_pixel/torch.sqrt(mse))/torch.log(torch.tensor(10))
print(psnr.item())
