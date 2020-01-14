import argparse
import os
import json
import shutil
import random
from itertools import islice

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import pdb

import numpy as np

from tqdm import tqdm

from datasets import get_CIFAR10, get_CIFAR100, get_SVHN
from model import Glow

from sklearn.metrics import roc_auc_score

def check_manual_seed(seed):
    seed = seed or random.randint(1, 10000)
    random.seed(seed)
    torch.manual_seed(seed)

    print("Using seed: {seed}".format(seed=seed))

def check_dataset(dataset, dataroot, augment, download):
    if dataset == "cifar10":
        cifar10 = get_CIFAR10(augment, dataroot, download)
        input_size, num_classes, train_dataset, test_dataset = cifar10
    if dataset == "cifar100":
        cifar100 = get_CIFAR100(augment, dataroot, download)
        input_size, num_classes, train_dataset, test_dataset = cifar100
    if dataset == "svhn":
        svhn = get_SVHN(augment, dataroot, download)
        input_size, num_classes, train_dataset, test_dataset = svhn

    return input_size, num_classes, train_dataset, test_dataset


def main(args):

    device = "cuda"

    check_manual_seed(args.seed)

    print("Getting datasets")
    image_shape, num_classes, _, test_dataset_cifar10 = \
        check_dataset("cifar10", "/data/sauravkadavath/cifar10-dataset/", args.augment, args.download)

    _, _, _, test_dataset_cifar100 = \
        check_dataset("cifar100", "/data/sauravkadavath/cifar10-dataset/", args.augment, args.download)

    test_loader_cifar10 = data.DataLoader(
        test_dataset_cifar10,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.n_workers,
        drop_last=False,
    )

    test_loader_cifar100 = data.DataLoader(
        test_dataset_cifar100,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.n_workers,
        drop_last=False,
    )

    models = []
    for model_path in args.saved_model:
        print("Initializing empty model")
        model = Glow(
            image_shape,
            args.hidden_channels,
            args.K,
            args.L,
            args.actnorm_scale,
            args.flow_permutation,
            args.flow_coupling,
            args.LU_decomposed,
            num_classes,
            args.learn_top,
            args.y_condition,
        ).cuda()
        
        print("Loading model from {0}".format(model_path))
        model.load_state_dict(torch.load(model_path))
        model.set_actnorm_init()
        models.append(model)

    cifar_10_losses_all = []
    for model in models:
        cifar10_losses = []
        for batch in tqdm(test_loader_cifar10):
            losses = eval_step(model, batch, args.y_condition)
            losses = list(losses)
            losses = [l.item() for l in losses]
            cifar10_losses.extend(losses)
        cifar_10_losses_all.append(cifar10_losses)

    cifar_100_losses_all = []
    for model in models:
        cifar100_losses = []
        for batch in tqdm(test_loader_cifar100):
            losses = eval_step(model, batch, args.y_condition)
            losses = list(losses)
            losses = [l.item() for l in losses]
            cifar100_losses.extend(losses)
        cifar_100_losses_all.append(cifar100_losses)

    cifar10_losses = np.array(cifar_10_losses_all)
    cifar100_losses = np.array(cifar_100_losses_all)

    cifar10_loss_mean = np.mean(cifar10_losses, axis=0)
    cifar100_loss_mean = np.mean(cifar100_losses, axis=0)

    cifar10_loss_variance = np.var(cifar10_losses, axis=0)
    cifar100_loss_variance = np.var(cifar100_losses, axis=0)

    cifar_10_scores = cifar10_loss_mean + cifar10_loss_variance
    cifar_100_scores = cifar100_loss_mean + cifar100_loss_variance

    pdb.set_trace()

    AUROC = roc_auc_score(
        [0 for _ in range(10000)] + [1 for _ in range(10000)],
        list(cifar_10_scores) + list(cifar_100_scores)
    )

    print(AUROC)

def eval_step(model, batch, y_condition):
    model.eval()

    x, y = batch
    x = x.cuda()

    with torch.no_grad():
        if y_condition:
            raise NotImplementedError()
        else:
            z, nll, y_logits = model(x, None)
            losses = nll

    return losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--download", action="store_true", help="downloads dataset")

    parser.add_argument(
        "--no_augment",
        action="store_false",
        dest="augment",
        help="Augment training data",
    )

    parser.add_argument(
        "--hidden_channels", type=int, default=512, help="Number of hidden channels"
    )

    parser.add_argument("--K", type=int, default=32, help="Number of layers per block")

    parser.add_argument("--L", type=int, default=3, help="Number of blocks")

    parser.add_argument(
        "--actnorm_scale", type=float, default=1.0, help="Act norm scale"
    )

    parser.add_argument(
        "--flow_permutation",
        type=str,
        default="invconv",
        choices=["invconv", "shuffle", "reverse"],
        help="Type of flow permutation",
    )

    parser.add_argument(
        "--flow_coupling",
        type=str,
        default="affine",
        choices=["additive", "affine"],
        help="Type of flow coupling",
    )

    parser.add_argument(
        "--no_LU_decomposed",
        action="store_false",
        dest="LU_decomposed",
        help="Train with LU decomposed 1x1 convs",
    )


    parser.add_argument(
        "--y_condition", action="store_true", help="Train using class condition"
    )

    parser.add_argument(
        "--y_weight", type=float, default=0.01, help="Weight for class condition loss"
    )

    parser.add_argument(
        "--n_workers", type=int, default=6, help="number of data loading workers"
    )

    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=512,
        help="batch size used during evaluation",
    )
    
    parser.add_argument(
        "--no_learn_top",
        action="store_false",
        help="Do not train top layer (prior)",
        dest="learn_top",
    )
    
    parser.add_argument(
        "--saved_model",
        action='append',
        help="Path to model to load for continuing training",
    )

    parser.add_argument(
        "--saved_optimizer",
        default="",
        help="Path to optimizer to load for continuing training",
    )

    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Use --distributed if you want your model to train on multiple GPUs"
    )

    parser.add_argument("--seed", type=int, default=0, help="manual seed")

    args = parser.parse_args()

    main(args)
