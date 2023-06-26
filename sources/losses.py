# Библиотеки для обучения
import torch
import torch.nn.functional as F


def dice_loss(target, logits, smooth=1e-4):
    probs = None
    if len(logits.shape) == 4 and logits.shape[-3] != 1:
        probs = F.softmax(logits, dim=-3)
    else:
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probs = torch.cat([neg_prob, pos_prob], dim=-3)

    target_onehot = torch.eye(2).type(logits.type())[target.long().squeeze(1)].permute(0, 3, 1, 2)

    dims = (0,) + tuple(range(2, target.ndimension()))
    numer = (probs * target_onehot).sum(dim=dims)
    denom = (probs + target_onehot).sum(dim=dims)
    dice_score = (2. * numer / (denom + smooth)).mean()
    return (1 - dice_score)


def tversky_loss(target, logits, alpha=0.3, beta=0.7, smooth=1e-4, reduce=True):
    probs = None
    if len(logits.shape) == 4 and logits.shape[-3] != 1:
        probs = F.softmax(logits, dim=-3)
    else:
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probs = torch.cat([neg_prob, pos_prob], dim=-3)

    target_onehot = torch.eye(2).type(logits.type())[target.long().squeeze(1)].permute(0, 3, 1, 2)

    dims = (0,) + tuple(range(2, target.ndimension()))

    numer = (probs * target_onehot).sum(dim=dims)
    denom = (alpha * probs + beta * target_onehot).sum(dim=dims)
    if alpha + beta != 1.:
        denom += numer * (1 - alpha - beta)

    tversky_index = numer / (denom + smooth)
    if reduce:
        tversky_index = tversky_index.mean()

    return (1 - tversky_index)


def focal_tversky_loss(target, logits, alpha=0.3, beta=0.7, gamma=4/3, smooth=1e-4):
    nr_loss = tversky_loss(target, logits, alpha=alpha, beta=beta,
                           smooth=smooth, reduce=False)
    return torch.mean(torch.pow(nr_loss, 1 / gamma))


def bce_loss(target, logits, weight=torch.tensor([.5, .5]), smooth=1e-4):
    p_t = None
    if len(logits.shape) == 4 and logits.shape[-3] != 1:
        probs = F.softmax(logits, dim=-3)
        p_t = torch.where(target.to(bool), probs[:, 1], probs[:, 0])
    else:
        probs = F.sigmoid(logits)
        p_t = torch.where(target.to(bool), probs, 1 - probs)

    weight_t = weight.type(logits.type())[target.long()]
    return -(weight_t * torch.log(p_t + smooth)).mean()


def focal_loss(target, logits, gamma=2., weight=torch.tensor([.5, .5]), smooth=1e-4):
    p_t = None
    log_p_t = None
    if len(logits.shape) == 4 and logits.shape[-3] != 1:
        probs = F.softmax(logits, dim=-3)
        p_t = torch.where(target.to(bool), probs[:, 1], probs[:, 0])
        
        log_probs = F.log_softmax(logits + smooth, dim=-3)
        log_p_t = torch.where(target.to(bool), log_probs[:, 1], log_probs[:, 0])
    else:
        probs = F.sigmoid(logits)
        p_t = torch.where(target.to(bool), probs, 1 - probs)
        log_p_t = torch.log(p_t + smooth)
    weight_t = weight.type(logits.type())[target.long()]
    return -(weight_t * log_p_t * torch.pow(1 - p_t + smooth, gamma)).mean()


def combo_loss(target, logits, bce_weight=torch.tensor([.5, .5]), alpha=.5, smooth=1e-4):
    bce = bce_loss(target, logits, bce_weight, smooth=smooth)
    dice = dice_loss(target, logits, smooth=smooth)
    return alpha * bce + (1 - alpha) * dice


def unified_focal_loss(target, logits, delta=0.6, gamma=0.75, lmbda=0.5, smooth=1e-4):
    return lmbda * focal_loss(target, logits, weight=torch.tensor([1 - delta, delta]), gamma=1-gamma, smooth=smooth)\
         + (1 - lmbda) * focal_tversky_loss(target, logits, alpha=1 - delta, beta=delta, gamma=1/gamma, smooth=smooth)