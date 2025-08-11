from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class JointCrossEntropy(nn.Module):
    def __init__(self, weight=None) -> None:
        super(JointCrossEntropy, self).__init__()
        self.weight = weight

    def forward(self, prediction: Tensor, localization: Tensor,
                solubility: Tensor, solubility_known: bool, args) -> Tuple[Tensor, Tensor, Tensor]:
        """

            Args:
                prediction: output of the network with 12 logits where the last two are for the solubility
                localization: true label for localization
                solubility: true label for
                solubility_known: tensor on device whether or not the solubility is known such that the solubility loss is set
                to 0 the solubility is unknown.
                args: training arguments containing the weighting for the solubility loss

            Returns:
                loss: the overall loss
                loc_loss: loss of localization information
                sol_loss: loss of the solubility prediction

            """
        localization_loss = F.cross_entropy(prediction[..., :10], localization, weight=self.weight)
        solubility_loss = F.cross_entropy(prediction[..., -2:], solubility, reduction='none')
        solubility_loss = (solubility_loss * solubility_known).mean() * args.solubility_loss
        return localization_loss + solubility_loss, localization_loss, solubility_loss


class LocCrossEntropy(nn.Module):
    def __init__(self, weight=None) -> None:
        '''
            essentially the same as torch.nn.CrossEntropyLoss
        Args:
            weight: weights for the individual classes
        '''
        super(LocCrossEntropy, self).__init__()
        self.weight = weight

    def forward(self, prediction: Tensor, localization: Tensor,
                solubility: Tensor, solubility_known: bool, args) -> Tuple[Tensor, Tensor, Tensor]:
        """
            This is just a wrapper for the standard torch.nn.functional.cross_entropy.
            So essentially the same as torch.nn.CrossEntropyLoss
            Args:
                prediction: output of the network with 12 logits where the last two are for the solubility
                localization: true label for localization


            Returns:
                loss: the overall loss

            """
        localization_loss = F.cross_entropy(prediction, localization, weight=self.weight)
        return localization_loss, localization_loss, torch.tensor([0])


def compute_kl_loss(p, q, pad_mask=None):

    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.sum()
    q_loss = q_loss.sum()

    loss = (p_loss + q_loss) / 2
    return loss


class SolCrossEntropy(nn.Module):
    def __init__(self, weight=None) -> None:
        super(SolCrossEntropy, self).__init__()
        self.weight = weight

    def forward(self, prediction: Tensor, localization: Tensor,
                solubility: Tensor, solubility_known: bool, args) -> Tuple[Tensor, Tensor, Tensor]:
        """

            Args:
                prediction: output of the network with 12 logits where the last two are for the solubility
                localization: true label for localization


            Returns:
                loss: the overall loss

            """
        solubility_loss = F.cross_entropy(prediction[..., -2:], solubility)
        return solubility_loss, torch.tensor([0]), solubility_loss


class FocalLoss(nn.Module):

    def __init__(self, weight=None, reduction='mean', gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    def forward(self, prediction, target):
        logp = self.ce(prediction, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()