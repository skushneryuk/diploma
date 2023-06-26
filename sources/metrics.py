# Библиотеки для обучения
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def dice_metric(target, preds, threshold=None, reduce=False):
    assert target.shape == preds.shape
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()
    if torch.is_tensor(preds):
        preds = preds.detach().cpu().numpy()

    if threshold is not None:
        preds = (preds >= threshold).astype(int)

    numer = (preds * target).mean(axis=(-1, -2))
    denom = preds.mean(axis=(-1, -2)) + target.mean(axis=(-1, -2))
    dice = np.where(denom == 0, 0, 2 * numer / denom)

    if reduce:
        return dice.mean()
    return dice.squeeze()


def iou_metric(target, preds, threshold=None, reduce=False):
    assert target.shape == preds.shape
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()
    if torch.is_tensor(preds):
        preds = preds.detach().cpu().numpy()

    if threshold is not None:
        preds = (preds >= threshold).astype(int)

    numer = (preds * target).mean(axis=(-1, -2))
    denom = preds.mean(axis=(-1, -2)) + target.mean(axis=(-1, -2)) - numer
    iou = np.where(denom == 0, 1, numer / denom)

    if reduce:
        return iou.mean()
    return iou.squeeze()


def recall_metric(target, preds, threshold=None, reduce=False):
    assert target.shape == preds.shape
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()
    if torch.is_tensor(preds):
        preds = preds.detach().cpu().numpy()

    if threshold is not None:
        preds = (preds >= threshold).astype(int)

    numer = (preds * target).mean(axis=(-1, -2))
    denom = target.mean(axis=(-1, -2))
    recall = np.where(denom == 0, 1, numer / denom)

    if reduce:
        return recall.mean()
    return recall.squeeze()


def precision_metric(target, preds, threshold=None, reduce=False):
    assert target.shape == preds.shape
    if torch.is_tensor(target):
        target = target.detach().cpu().numpy()
    if torch.is_tensor(preds):
        preds = preds.detach().cpu().numpy()

    if threshold is not None:
        preds = (preds >= threshold).astype(int)

    numer = (preds * target).mean(axis=(-1, -2))
    denom = preds.mean(axis=(-1, -2))
    precision = np.where(denom == 0, 1, numer / denom)

    if reduce:
        return precision.mean()
    return precision.squeeze()
