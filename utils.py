import torch
import os
import copy
import numpy as np
import pandas as pd
import shutil
from scipy import stats
import collections
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from collections import OrderedDict
import pynvml
import types

import model


def get_parameters(net, squeeze=True,):
    trainable = []
    trainable_name = [name for (name, param) in net.named_parameters()]
    state = net.state_dict()
    for i, name in enumerate(state.keys()):
        if name in trainable_name:
            trainable.append(state[name])

    if squeeze:
        trainable = torch.cat([i.reshape([-1]) for i in trainable])
    return trainable


def consistent_type(model,
                    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                    squeeze=True,):
    # this function takes in directory to where model is saved, model weights as a list of numpy array,
    # or a torch model and outputs model weights as a list of numpy array
    weights = get_parameters(model, squeeze=squeeze)
    weights = weights.to(device)

    return weights


def compute_distance(a, b, order, numpy=True, dim=None):
    if order == 'inf':
        order = np.inf
    if order == 'cos' or order == 'cosine':
        if dim is None:
            dim = 0
        metric = torch.nn.CosineSimilarity(dim=dim)
        dist = 1 - metric(a, b)
        if numpy:
            dist = dist.cpu().numpy()
        return dist
    else:
        if order != np.inf:
            try:
                order = int(order)
            except:
                raise TypeError("input metric for distance is not understandable")
        dist = torch.norm(a - b, p=order, dim=dim)
        if numpy:
            dist = dist.cpu().numpy()
        return dist


def parameter_distance(model1, model2, order=2, ):
    # compute the difference between 2 checkpoints
    weights1 = model1
    weights2 = model2
    if not isinstance(order, list):
        orders = [order]
    else:
        orders = order
    res_list = []
    for o in orders:
        res = compute_distance(weights1, weights2, o)
        if isinstance(res, np.ndarray):
            res = float(res)
        res_list.append(res)
    return res_list


def load_dataset(dataset, train, download=False, apply_transform=True):
    try:
        dataset_class = eval(f"torchvision.datasets.{dataset}")
    except:
        raise NotImplementedError(f"Dataset {dataset} is not implemented by pytorch.")

    if train and apply_transform:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    try:
        data = dataset_class(root='./data', train=train, download=download, transform=transform)
    except:
        if train:
            data = dataset_class(root='./data', split="train", download=download, transform=transform)
        else:
            data = dataset_class(root='./data', split="test", download=download, transform=transform)

    return data


def get_optimizer(dataset, net, lr, privacy_engine=None, optimizer="sgd"):
    if dataset == 'CIFAR10' and optimizer == "sgd":
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

    else:
        print("using adam")
        optimizer = optim.Adam(net.parameters(), lr=lr)
    if privacy_engine is not None:
        privacy_engine.attach(optimizer)
    return optimizer

