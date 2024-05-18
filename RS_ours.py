import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PPIDSG.models import Generator
import numpy as np
import argparse

# get dataset name from command line
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    required=False,
    default="mnist",
    help="input dataset: mnist, cifar, svhn, fmnist",
)

parser.add_argument(
    "--model_dir",
    required=False,
    default="cifar_model",
)
args = parser.parse_args()
datas = args.dataset
# hyperparameter setting
# datas = 'FMNIST' # mnist, fmnist, cifar10, svhn
dummys = "test"  # dummy data: the random input, test:the image from test dataset
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

psnrl = []
for i, (images, labels) in enumerate(test_loader):
    # print(i)
    if datas == "MNIST" or datas == "FMNIST":
        test_loader_X = images.view(images.size(0), 1, 28, 28)
    elif datas == "CIFAR10" or datas == "SVHN":
        test_loader_X = images.view(images.size(0), 3, 32, 32)

    _, pred_output = target_model(test_loader_X.to(device))
    img = pred_output.to(device)
    orig = (test_loader_X.to(device) + 1) / 2 * 255
    img = (img + 1) / 2 * 255
    # show reconstructed image
    if i==0:
      simg = to_pil_image(pred_output[0].cpu())
      simg.show()
      simg.save(f'./pictures/dummy_{datas}.jpg')
    delta = orig - img
    delta = delta.reshape(delta.shape[0], -1)
    mse = torch.mean(delta**2, dim=1)
    max_pixel = 255
    psnr = 20 * torch.log(max_pixel / torch.sqrt(mse)) / (torch.log(torch.tensor(10)))
    psnr = torch.sum(psnr) / len(test_loader_X)
    psnrl.append(psnr.item())
    del pred_output, img, orig, psnr, delta

print(sum(psnrl) / len(psnrl))
