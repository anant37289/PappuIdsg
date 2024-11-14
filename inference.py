import torch.nn as nn
import numpy as np
import torchvision
import torchvision.transforms as transforms
import copy
import torch
import time
import os
import random

from tqdm import tqdm
from PPIDSG.options import args_parser
import torch.nn.functional as F
from PPIDSG.update import LocalUpdate, test_inference
from PPIDSG.models import (
    Generator,
    Discriminator,
    AutoEncoder_VGG,
    VGG16_classifier,
    AutoEncoder_VGG_mnist,
    VGG16_classifier_mnist,
)
from PPIDSG.utils import get_dataset, average_weights
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import classification_report
from src.MIA.utils import train_attack_model
from src.models import AttackMLP

args = args_parser()

if __name__ == "__main__":
    start_time = time.time()
    args = args_parser()
    model_dir = f"./{args.model_dir}/"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # BUILD MODEL
    if args.dataset == "mnist":
        G = Generator(1, args.ngf, 1, args.num_resnet)
        D_B = Discriminator(1, args.ndf, 1)
        global_model = AutoEncoder_VGG_mnist().to(device)
        C = VGG16_classifier_mnist().to(device)
    elif args.dataset == "fmnist":
        G = Generator(1, args.ngf, 1, args.num_resnet)
        D_B = Discriminator(1, args.ndf, 1)
        global_model = AutoEncoder_VGG_mnist().to(device)
        C = VGG16_classifier_mnist().to(device)
    else:
        G = Generator(3, args.ngf, 3, args.num_resnet)
        D_B = Discriminator(3, args.ndf, 1)
        global_model = AutoEncoder_VGG().to(device)
        C = VGG16_classifier().to(device)

    G.normal_weight_init(mean=0.0, std=0.02)
    D_B.normal_weight_init(mean=0.0, std=0.02)
    G.to(device)
    D_B.to(device)
    
    #testing
    state_dict_G = torch.load(model_dir + "generator_param.pkl", weights_only=True)
state_dict_global = torch.load(model_dir + "_extractor_param.pkl", weights_only=True)
state_dict_C = torch.load(model_dir + "_classifier_param.pkl", weights_only=True)

# 2. Then load the state dict into the models
G.load_state_dict(state_dict_G)
global_model.load_state_dict(state_dict_global)
C.load_state_dict(state_dict_C)

# 3. Now you can move to device and use the models
G = G.to(device)
global_model = global_model.to(device)
C = C.to(device)

# 4. Call your test inference
test_acc = test_inference(G, global_model, C, test_dataset)
    