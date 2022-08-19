import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss
import numpy as np
import math

class ClfDistillLossFunction(nn.Module):
    """Torch classification debiasing loss function"""

    def forward(self, logits, bias, teach_probs, labels):
        """
        :param logits: [batch, n_classes] logit score for each class
        :param bias: [batch, n_classes] log-probabilties from the bias for each class
        :param teach_probs: techer prediction probabilities # FIXME check values
        :param labels: [batch] integer class labels # FIXME nechapem co su labels
        :return: scalar loss
        """
        raise NotImplementedError()


class SmoothedDistillLoss(ClfDistillLossFunction):

    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, logits, bias_logits, teacher_logits, labels):
        softmaxf = torch.nn.Softmax(dim=1)
        probs = softmaxf(logits)
        teacher_probs = softmaxf(teacher_logits)
        bias_probs = softmaxf(bias_logits)

        one_hot_labels = torch.eye(logits.size(1)).to(self.device)[labels]
        weights = (1 - (one_hot_labels * torch.exp(bias_probs)).sum(1))
        weights = weights.unsqueeze(1).expand_as(teacher_probs)

        exp_teacher_probs = teacher_probs ** weights
        norm_teacher_probs = exp_teacher_probs / exp_teacher_probs.sum(1).unsqueeze(1).expand_as(teacher_probs)

        example_loss = -(norm_teacher_probs * probs.log()).sum(1)
        batch_loss = example_loss.mean()

        return batch_loss
