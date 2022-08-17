import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss
import numpy as np
import math

class ClfDistillLossFunction(nn.Module):
    """Torch classification debiasing loss function"""

    def forward(self, hidden, logits, bias, teach_probs, labels):
        """
        :param hidden: [batch, n_features] hidden features from the model
        :param logits: [batch, n_classes] logit score for each class
        :param bias: [batch, n_classes] log-probabilties from the bias for each class
        :param teach_probs: techer prediction probabilities # FIXME check values
        :param labels: [batch] integer class labels # FIXME nechapem co su labels
        :return: scalar loss
        """
        raise NotImplementedError()


class SmoothedDistillLoss(ClfDistillLossFunction):
    def forward(self, hidden, logits, bias_probs, teacher_probs, labels):
        softmaxf = torch.nn.Softmax(dim=1)
        probs = softmaxf(logits)

        one_hot_labels = torch.eye(logits.size(1)).cuda()[labels]
        weights = (1 - (one_hot_labels * torch.exp(bias_probs)).sum(1))
        weights = weights.unsqueeze(1).expand_as(teacher_probs)

        exp_teacher_probs = teacher_probs ** weights
        norm_teacher_probs = exp_teacher_probs / exp_teacher_probs.sum(1).unsqueeze(1).expand_as(teacher_probs)

        example_loss = -(norm_teacher_probs * probs.log()).sum(1)
        batch_loss = example_loss.mean()

        return batch_loss