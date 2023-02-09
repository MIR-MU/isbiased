import torch
from torch import nn


class ClfDistillLossFunction(nn.Module):
    """Torch classification debiasing loss function"""

    def forward(self, logits, bias, teach_probs, labels):
        """
        :param logits: [batch, n_classes] logit score for each class
        :param bias: [batch, n_classes] log-probabilties from the bias for each class
        :param teach_probs: techer prediction probabilities
        :param labels: [batch] integer class labels
        :return: scalar loss
        """
        raise NotImplementedError()


class LearnedMixinHLoss(ClfDistillLossFunction):

    def __init__(self,
                 penalty: float,
                 model_hidden_size: int = 768,
                 scaling_portion: int = 10):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.learned_bias_scalar = torch.nn.Linear(in_features=model_hidden_size,
                                                   out_features=1,
                                                   device=self.device)
        self.scaling_portion = scaling_portion
        self.penalty = penalty

    def forward(self, logits, bias_logits, hidden_states, labels):

        bias_scale = self.learned_bias_scalar(hidden_states)
        bias_scale = torch.nn.functional.softplus(bias_scale)
        scaled_bias_logits = bias_logits * bias_scale.squeeze(-1)

        biased_start_logprobs = scaled_bias_logits.log_softmax(1)

        cross_entropy_loss = torch.nn.CrossEntropyLoss()
        lmix_loss = cross_entropy_loss(logits + scaled_bias_logits / self.scaling_portion, labels)

        entropy = -(torch.exp(biased_start_logprobs) * biased_start_logprobs).sum(1).mean(0)
        entropy_penalty = self.penalty * entropy

        return lmix_loss + entropy_penalty
