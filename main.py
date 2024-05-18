import copy
import torch
import time
import os
import random

from tqdm import tqdm

from PPIDSG.options import args_parser
from PPIDSG.update import LocalUpdate, test_inference
from PPIDSG.models import (
    Generator,
    Discriminator,
    AutoEncoder_VGG,
    VGG16_classifier,
    AutoEncoder_VGG_mnist,
    VGG16_classifier_mnist,
)
from PPIDSG.utils import get_dataset, average_weights, exp_details


if __name__ == "__main__":
    start_time = time.time()
    args = args_parser()
    exp_details(args)

    model_dir = args.model_dir

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # BUILD MODEL
    if args.dataset == "mnist" or args.dataset == "fmnist":
        # mnist, f-mnist:1; svhn, cifar10:3 (both input and output)
        G = Generator(1, args.ngf, 1, args.num_resnet)
        # mnist, f-mnist:1; svhn, cifar10:3 (only the input parameter)
        D_B = Discriminator(1, args.ndf, 1)
        global_model = AutoEncoder_VGG_mnist().to(device)
        C = VGG16_classifier_mnist().to(device)
    else:
        # mnist, f-mnist:1; svhn, cifar10:3 (both input and output)
        G = Generator(3, args.ngf, 3, args.num_resnet)
        # mnist, f-mnist:1; svhn, cifar10:3 (only the input parameter)
        D_B = Discriminator(3, args.ndf, 1)
        global_model = AutoEncoder_VGG().to(device)
        C = VGG16_classifier().to(device)

    G.normal_weight_init(mean=0.0, std=0.02)
    D_B.normal_weight_init(mean=0.0, std=0.02)
    G.to(device)
    D_B.to(device)

    # copy weights
    global_weights = global_model.state_dict()
    G_weights = G.state_dict()
    D_B_weights = D_B.state_dict()
    C_weights = C.state_dict()

    # Training
    global_model_local_weights = [global_weights for i in range(args.num_users)]
    D_B_local_weights = [D_B_weights for i in range(args.num_users)]
    C_local_weights = [C_weights for i in range(args.num_users)]
    g_glb = copy.deepcopy(G_weights)
    g_glb_prime = copy.deepcopy(G_weights)
    for key in g_glb.keys():
        g_glb[key] = 0
    for key in g_glb_prime.keys():
        g_glb_prime[key] = 0

    # Global Weight Constants
    w_glb_prime = copy.deepcopy(G_weights)
    w_glb_double_prime = copy.deepcopy(G_weights)
    test_acc_arr = []
    for epoch in tqdm(range(args.num_epochs)):
        G_local_weights = []
        print(f"\n | Global Training Round : {epoch + 1} |\n")

        for idx in range(args.num_users):
            global_model.load_state_dict(global_model_local_weights[idx])
            D_B.load_state_dict(D_B_local_weights[idx])
            C.load_state_dict(C_local_weights[idx])
            local_G = copy.deepcopy(G)
            local_Gwt = local_G.state_dict()
            for layer,key in enumerate(local_Gwt.keys()):
                mask1 = torch.ones(local_Gwt[key].shape[0]).to(device)
                mask05 = torch.ones(local_Gwt[key].shape[0]).to(device)
                mask_size = mask1.size()[0]

                perm = torch.randperm(mask_size)
                mask1[perm[0:mask_size//4]]=-1
                mask1[perm[mask_size//4:mask_size//2]]=1
                mask1[perm[mask_size//2:]]=0

                mask05[perm[0:mask_size//2]]=0
                mask05[perm[mask_size//2:3*mask_size//4]]=2
                mask05[perm[3*mask_size//4:]]=-2
                if "bias" in key:
                  local_Gwt[key] = local_Gwt[key] + args.alpha*mask1*g_glb[key]+args.alpha**2*mask05*g_glb_prime[key]
                if "weight" in key:
                  local_Gwt[key] = local_Gwt[key] + args.alpha*mask1[:,None,None,None]*g_glb[key]+args.alpha**2*mask05[:,None,None,None]*g_glb_prime[key]
            local_G.load_state_dict(local_Gwt)
            local_model = LocalUpdate(
                args=args,
                dataset=train_dataset,
                G=local_G,
                D_B=copy.deepcopy(D_B),
                idxs=user_groups[idx],
            )
            w, v, u, z = local_model.update_weights(
                D_A_model=copy.deepcopy(global_model),
                C=copy.deepcopy(C),
                global_round=epoch,
            )
            G_local_weights.append(copy.deepcopy(v))
            D_B_local_weights[idx] = copy.deepcopy(u)
            global_model_local_weights[idx] = copy.deepcopy(w)

        # update global weights and local weights
        w_glb_double_prime = copy.deepcopy(w_glb_prime)
        w_glb_prime = copy.deepcopy(G_weights)
        G_weights = average_weights(G_local_weights)  # federated train
        for key in G_weights.keys():
            g_glb[key] = G_weights[key] - w_glb_prime[key]
        for key in G_weights.keys():
            g_glb_prime[key] = G_weights[key] - w_glb_double_prime[key]
        G.load_state_dict(G_weights)  # each client generator

        test_acc = test_inference(G, global_model, C, test_dataset)  # test accuracy
        test_acc_arr.append("{:.2f}".format(100 * test_acc))
        print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))
        torch.save(G_weights, model_dir + "/generator_param.pkl")
    
    print(test_acc_arr)
    print("\n Total Run Time: {0:0.4f}".format(time.time() - start_time))
