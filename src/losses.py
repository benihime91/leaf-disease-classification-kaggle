# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/02_losses.ipynb (unless otherwise specified).

__all__ = ['LabelSmoothingCrossEntropy', 'log_t', 'exp_t', 'compute_normalization_fixed_point', 'compute_normalization',
           'tempered_softmax', 'bi_tempered_logistic_loss', 'BiTemperedLogisticLoss']

# Cell
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss

# Cell
class LabelSmoothingCrossEntropy(_WeightedLoss):
    "label smoothing loss"
    def __init__(self, eps:float=0.1, reduction='mean', weight=None):
        super().__init__(weight=weight, reduction=reduction)
        self.eps,self.reduction = eps,reduction
        self.weight = weight

    def forward(self, output, target):
        c = output.size()[1]
        log_preds = F.log_softmax(output, dim=1)
        if self.reduction=='sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=1) #We divide by that size at the return line so sum and not mean
            if self.reduction=='mean':  loss = loss.mean()
        return loss*self.eps/c + (1-self.eps) * F.nll_loss(log_preds, target.long(), reduction=self.reduction)

# Cell
###########################################################################
#From : https://github.com/mlpanda/bi-tempered-loss-pytorch/blob/master/bi_tempered_loss.py
###########################################################################
def log_t(u, t):
    """Compute log_t for `u`."""

    if t == 1.0:
        return torch.log(u)
    else:
        return (u ** (1.0 - t) - 1.0) / (1.0 - t)


def exp_t(u, t):
    """Compute exp_t for `u`."""

    if t == 1.0:
        return torch.exp(u)
    else:
        return torch.relu(1.0 + (1.0 - t) * u) ** (1.0 / (1.0 - t))


def compute_normalization_fixed_point(activations, t, num_iters=5):
    """Returns the normalization value for each example (t > 1.0).
    Args:
    activations: A multi-dimensional tensor with last dimension `num_classes`.
    t: Temperature 2 (> 1.0 for tail heaviness).
    num_iters: Number of iterations to run the method.
    Return: A tensor of same rank as activation with the last dimension being 1.
    """

    mu = torch.max(activations, dim=-1).values.view(-1, 1)
    normalized_activations_step_0 = activations - mu

    normalized_activations = normalized_activations_step_0
    i = 0
    while i < num_iters:
        i += 1
        logt_partition = torch.sum(exp_t(normalized_activations, t), dim=-1).view(-1, 1)
        normalized_activations = normalized_activations_step_0 * (logt_partition ** (1.0 - t))

    logt_partition = torch.sum(exp_t(normalized_activations, t), dim=-1).view(-1, 1)

    return -log_t(1.0 / logt_partition, t) + mu


def compute_normalization(activations, t, num_iters=5):
    """Returns the normalization value for each example.
    Args:
    activations: A multi-dimensional tensor with last dimension `num_classes`.
    t: Temperature 2 (< 1.0 for finite support, > 1.0 for tail heaviness).
    num_iters: Number of iterations to run the method.
    Return: A tensor of same rank as activation with the last dimension being 1.
    """

    if t < 1.0:
        return None # not implemented as these values do not occur in the authors experiments...
    else:
        return compute_normalization_fixed_point(activations, t, num_iters)


def tempered_softmax(activations, t, num_iters=5):
    """Tempered softmax function.
    Args:
    activations: A multi-dimensional tensor with last dimension `num_classes`.
    t: Temperature tensor > 0.0.
    num_iters: Number of iterations to run the method.
    Returns:
    A probabilities tensor.
    """

    if t == 1.0:
        normalization_constants = torch.log(torch.sum(torch.exp(activations), dim=-1))
    else:
        normalization_constants = compute_normalization(activations, t, num_iters)

    return exp_t(activations - normalization_constants, t)

def bi_tempered_logistic_loss(activations, labels, t1, t2, label_smoothing=0.0, num_iters=5, reduction = 'mean'):
    """Bi-Tempered Logistic Loss."""

    if len(labels.shape)<len(activations.shape): #not one-hot
        labels_onehot = torch.zeros_like(activations)
        labels_onehot.scatter_(1, labels[..., None], 1)
    else:
        labels_onehot = labels

    if label_smoothing > 0:
        num_classes = labels_onehot.shape[-1]
        labels_onehot = ( 1 - label_smoothing * num_classes / (num_classes - 1) ) \
                * labels_onehot + \
                label_smoothing / (num_classes - 1)

    probabilities = tempered_softmax(activations, t2, num_iters)

    loss_values = labels_onehot * log_t(labels_onehot + 1e-10, t1) \
            - labels_onehot * log_t(probabilities, t1) \
            - labels_onehot.pow(2.0 - t1) / (2.0 - t1) \
            + probabilities.pow(2.0 - t1) / (2.0 - t1)
    loss_values = loss_values.sum(dim = -1) #sum over classes

    if reduction == 'none':
        return loss_values
    if reduction == 'sum':
        return loss_values.sum()
    if reduction == 'mean':
        return loss_values.mean()


class BiTemperedLogisticLoss(_WeightedLoss):
    "Bi-tempered-logistic-loss"
    def __init__(self, eps:float=0.1, t1=0.6, t2=1.4, num_iters=5, reduction='mean'):
        super().__init__(reduction=reduction)
        self.eps = eps
        self.t1, self.t2 = t1, t2
        self.num_iters = num_iters
        self.reduction = reduction

    def forward(self, output, target):
        loss=bi_tempered_logistic_loss(output, target, self.t1, self.t2,
                                      self.eps, self.num_iters, self.reduction)
        return loss