import copy
import torch
import random
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset
from torch.autograd import Variable
import glob
import os


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode="train"):
        self.transform = transforms.Compose(transforms_)
        self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))
        if mode == "train":
            self.files.extend(sorted(glob.glob(os.path.join(root, "test") + "/*.*")))

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        w, h = img.size
        img_A = img.crop((0, 0, w / 2, h))
        img_B = img.crop((w / 2, 0, w, h))

        if np.random.random() < 0.5:
            img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
            img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return {"A": img_A, "B": img_B}

    def __len__(self):
        return len(self.files)


def dataset_non_iid_dirichlet(dataset, num_users, alpha=0.5):
    num_classes = len(set(dataset.targets))
    dict_users = {i: np.array([]) for i in range(num_users)}
    
    labels = np.array(dataset.targets)
    class_indices = [np.where(labels == i)[0] for i in range(num_classes)]

    for c in range(num_classes):
        np.random.seed(1234 + c)
        proportions = np.random.dirichlet(np.repeat(alpha, num_users))
        proportions = (np.cumsum(proportions) * len(class_indices[c])).astype(int)[:-1]
        
        class_data_split = np.split(class_indices[c], proportions)
        for user in range(num_users):
            dict_users[user] = np.concatenate((dict_users[user], class_data_split[user]), axis=0)

    for user in range(num_users):
        dict_users[user] = dict_users[user].astype(int)
    
    return dict_users


def split_dataset(dataset, user_groups):
    user_datasets = []
    for user_indices in user_groups.values():
        user_datasets.append(Subset(dataset, user_indices))
    return user_datasets


def get_dataset(args):
    if args.dataset == "cifar":
        data_dir = "./data/cifar/"
        apply_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = datasets.CIFAR100(data_dir, train=True, download=True, transform=apply_transform)
        test_dataset = datasets.CIFAR100(data_dir, train=False, download=True, transform=apply_transform)
        user_groups = dataset_non_iid_dirichlet(train_dataset, args.num_users, alpha=args.alpha)

    elif args.dataset == "mnist":
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
        ])
        data_dir = "./data/"
        train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=apply_transform)
        test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=apply_transform)
        user_groups = dataset_non_iid_dirichlet(train_dataset, args.num_users, alpha=args.alpha)

    elif args.dataset == "fmnist":
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
        ])
        data_dir = "./data/fmnist/"
        train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True, transform=apply_transform)
        test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True, transform=apply_transform)
        user_groups = dataset_non_iid_dirichlet(train_dataset, args.num_users, alpha=args.alpha)

    elif args.dataset == "svhn":
        data_dir = "./data/svhn/"
        apply_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])
        train_dataset = datasets.SVHN(data_dir, split="train", download=True, transform=apply_transform)
        test_dataset = datasets.SVHN(data_dir, split="test", download=True, transform=apply_transform)
        user_groups = dataset_non_iid_dirichlet(train_dataset, args.num_users, alpha=args.alpha)

    user_datasets = split_dataset(train_dataset, user_groups)

    return train_dataset, test_dataset, user_datasets


def average_weights(w):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def average_weights_new(w, p):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        w_avg[key] = torch.mul(w[1][key], (1 / p)) + torch.mul(w_avg[key], (1 - (1 / p)))
    return w_avg


# ImagePool class to load historical images
class ImagePool:
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images.data:
            image = torch.unsqueeze(image, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs += 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return Variable(torch.cat(return_images, 0))
