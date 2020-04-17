# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, weights=[1.0, 1.0, 1.0, 0.1, 0.03, 0.03], epsilon=0.1):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.weights = weights
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)


    def _forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        targets = targets.to(inputs.device)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        if not isinstance(inputs, tuple):
            inputs_tuple = (inputs,)
        else:
            inputs_tuple = inputs

        total_loss = self._forward(inputs_tuple[0], targets) * self.weights[0]
        # total_loss = self._forward(inputs_tuple[0], targets)

        for i in range(1, len(inputs_tuple)):
            total_loss = total_loss + self._forward(inputs_tuple[i], targets) * self.weights[i]
            # total_loss = total_loss + self._forward(inputs_tuple[i], targets)

        return total_loss
        # return total_loss / len(inputs_tuple)


