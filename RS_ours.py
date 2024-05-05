import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PPIDSG.models import Generator
import numpy as np
import argparse

#get dataset name from command line
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False, default='mnist', help='input dataset: mnist, cifar, svhn, fmnist')

datas = parser.parse_args().dataset
# hyperparameter setting
# datas = 'FMNIST' # mnist, fmnist, cifar10, svhn
dummys = 'test' # dummy data: the random input, test:the image from test dataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
to_pil_image = transforms.ToPILImage()

if datas == 'FMNIST' or datas == 'MNIST':
    target_model = Generator(1, 32, 1, 6).to(device)
else:
    target_model = Generator(3, 32, 3, 6).to(device)

# load the trained generater parameters from the client's sharing
if datas == 'CIFAR10':
    modelPath = './cifar_model/'
    target_model.load_state_dict(torch.load(modelPath + 'generator_param.pkl'))
elif datas == 'MNIST':
    modelPath = './mnist_model/'
    target_model.load_state_dict(torch.load(modelPath + 'generator_param.pkl'))
elif datas == 'SVHN':
    modelPath = './svhn_model/'
    target_model.load_state_dict(torch.load(modelPath + 'generator_param.pkl'))
elif datas == 'FMNIST':
    modelPath = './fmnist_model/'
    target_model.load_state_dict(torch.load(modelPath + 'generator_param.pkl'))

apply_transform = transforms.Compose([transforms.ToTensor()])

apply_transform_cifar = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

apply_transform_svhn = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ])

apply_transform_mnist = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
            ])

if datas == 'SVHN':
    train_dataset = datasets.SVHN(root='./data/svhn/', split='train', download=True, transform=apply_transform_svhn)
    test_dataset = datasets.SVHN(root='./data/svhn/', split='test', download=True, transform=apply_transform_svhn)
elif datas == 'CIFAR10':
    train_dataset = datasets.CIFAR10(root='./data/cifar/', train=True,download=True ,transform=apply_transform_cifar)
    test_dataset = datasets.CIFAR10(root='./data/cifar/', train=False,download=True ,transform=apply_transform_cifar)
elif datas == 'MNIST':
    train_dataset = datasets.MNIST(root='./data/', train=True,download=True ,transform=apply_transform_mnist)
    test_dataset = datasets.MNIST(root='./data/', train=False,download=True ,transform=apply_transform_mnist)
elif datas == 'FMNIST':
    train_dataset = datasets.FashionMNIST(root='./data/fmnist/', train=True,download=True ,transform=apply_transform_mnist)
    test_dataset = datasets.FashionMNIST(root='./data/fmnist/', train=False,download=True ,transform=apply_transform_mnist)

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

img = to_pil_image(test_loader_Y[0].cpu())
# img.show()

# get the input of generator
if dummys == 'dummy_data':
    dummy_data = torch.randn(test_loader_Y.size()).to(device)
    img = to_pil_image(dummy_data[0].cpu())
    img.save('./picture/fmnist.jpg')
else:
    dummy_data = test_loader_X
    orig = dummy_data[0].cpu()
    orig_pil = to_pil_image(orig)
    orig_pil.save('./picture/fmnist.jpg')

# RS
_, pred_output = target_model(dummy_data)
# print(_.shape,pred_output.shape)
# show reconstructed image
img = pred_output[0].cpu()
img_pil = to_pil_image(img)
# img.show()
img_pil.save('./picture/dummy_fmnist.jpg')

orig = (orig+1)/2*255
img = (img+1)/2*255
mse = torch.mean((orig-img)**2)

max_pixel = 255
print(max_pixel)
psnr = 20*torch.log(max_pixel/torch.sqrt(mse))/torch.log(torch.tensor(10))
print(psnr)
