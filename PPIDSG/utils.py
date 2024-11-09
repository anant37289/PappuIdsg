import copy
import torch
import random
import glob
import os
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import Dataset

# Non-IID Dirichlet Distribution Function for Data Splitting
def dataset_non_iid_dirichlet(dataset, num_users, alpha=0.5):
    """
    Sample non-I.I.D. client data from the dataset using Dirichlet distribution.
    :param dataset: The dataset to split
    :param num_users: Number of users (clients)
    :param alpha: Dirichlet concentration parameter (smaller value means more heterogeneity)
    :return: dict of image index for each user
    """
    num_classes = len(set(dataset.targets))  # Dynamically handles the number of classes
    dict_users = {i: np.array([]) for i in range(num_users)}
    
    # Get indices for each class
    labels = np.array(dataset.targets)
    class_indices = [np.where(labels == i)[0] for i in range(num_classes)]

    for c in range(num_classes):
        # Use Dirichlet distribution to split class c data among clients
        np.random.seed(1234 + c)
        proportions = np.random.dirichlet(np.repeat(alpha, num_users))
        proportions = (np.cumsum(proportions) * len(class_indices[c])).astype(int)[:-1]
        
        # Split and assign to clients
        class_data_split = np.split(class_indices[c], proportions)
        for user in range(num_users):
            dict_users[user] = np.concatenate((dict_users[user], class_data_split[user]), axis=0)

    for user in range(num_users):
        dict_users[user] = dict_users[user].astype(int)
    
    return dict_users


def get_dataset(args):
    if args.dataset == "cifar":
        data_dir = "./data/cifar/"

        apply_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        train_dataset = datasets.CIFAR10(
            data_dir, train=True, download=True, transform=apply_transform
        )
        test_dataset = datasets.CIFAR10(
            data_dir, train=False, download=True, transform=apply_transform
        )

        # Sample training data amongst users using Dirichlet distribution for heterogeneity
        user_groups = dataset_non_iid_dirichlet(train_dataset, args.num_users, alpha=args.alpha)

    elif args.dataset == "svhn":
        data_dir = "./data/svhn/"

        apply_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

        train_dataset = datasets.SVHN(
            data_dir, split="train", download=True, transform=apply_transform
        )
        test_dataset = datasets.SVHN(
            data_dir, split="test", download=True, transform=apply_transform
        )

        # Sample training data amongst users using Dirichlet distribution for heterogeneity
        user_groups = dataset_non_iid_dirichlet(train_dataset, args.num_users, alpha=args.alpha)

    elif args.dataset == "mnist" or args.dataset == "fmnist":
        apply_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))]
        )

        if args.dataset == "mnist":
            data_dir = "./data/"
            train_dataset = datasets.MNIST(
                data_dir, train=True, download=True, transform=apply_transform
            )
            test_dataset = datasets.MNIST(
                data_dir, train=False, download=True, transform=apply_transform
            )
        else:
            data_dir = "./data/fmnist/"
            train_dataset = datasets.FashionMNIST(
                data_dir, train=True, download=True, transform=apply_transform
            )
            test_dataset = datasets.FashionMNIST(
                data_dir, train=False, download=True, transform=apply_transform
            )

        # Sample training data amongst users using Dirichlet distribution for heterogeneity
        user_groups = dataset_non_iid_dirichlet(train_dataset, args.num_users, alpha=args.alpha)

    return train_dataset, test_dataset, user_groups

# Function to Average the Weights (for Federated Averaging)
def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

# Function to load images into a pool (for GAN/Training use case)
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
                self.num_imgs = self.num_imgs + 1
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
        return_images = Variable(torch.cat(return_images, 0))
        return return_images
