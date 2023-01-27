import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    """
    Softmax and sigmoid focal loss
    https://github.com/lonePatient/TorchBlocks/blob/master/torchblocks/losses/focal_loss.py
    """

    def __init__(self,
                 num_labels,
                 gamma=2.0,
                 alpha=0.25,
                 epsilon=1.e-9,
                 reduction='mean',
                 activation_type='softmax'
                 ):

        super(FocalLoss, self).__init__()
        self.num_labels = num_labels
        self.gamma = gamma  # gamma 为 0 时即退化为交叉熵
        self.alpha = alpha  
        self.epsilon = epsilon
        self.activation_type = activation_type
        self.reduction = reduction
    
    def sigmoid_focal_loss(
            self,
            inputs: torch.Tensor,
            targets: torch.Tensor
        ) -> torch.Tensor:
        """
        https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/focal_loss.py
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
            alpha: (optional) Weighting factor in range (0,1) to balance
                    positive vs negative examples. Default = -1 (no weighting).
            gamma: Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples.
            reduction: 'none' | 'mean' | 'sum'
                    'none': No reduction will be applied to the output.
                    'mean': The output will be averaged.
                    'sum': The output will be summed.
        Returns:
            Loss tensor with the reduction option applied.
        """
        inputs = inputs.float()
        targets = targets.float()
        p = torch.sigmoid(inputs)
        ce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss

    def forward(self, preds, target):
        return self.sigmoid_focal_loss(
            inputs=preds, targets=target)

    def forward_old(self, preds, target):
        """
        Args:
            logits: model's output, shape of [batch_size, num_cls]
            target: ground truth labels, shape of [batch_size]
        Returns:
            shape of [batch_size]
        """
        if self.activation_type == 'softmax':
            idx = target.view(-1, 1).long()
            one_hot_key = torch.zeros(idx.size(0), self.num_labels, dtype=torch.float32, device=idx.device)
            one_hot_key = one_hot_key.scatter_(1, idx, 1)
            logits = torch.softmax(preds, dim=-1)
            loss = -self.alpha * one_hot_key * torch.pow((1 - logits), self.gamma) * (logits + self.epsilon).log()
            loss = loss.sum(1)
        elif self.activation_type == 'sigmoid':
            multi_hot_key = target
            logits = torch.sigmoid(preds)  # logits -> prob (p_t)
            zero_hot_key = 1 - multi_hot_key
            loss = -self.alpha * multi_hot_key * torch.pow((1 - logits), self.gamma) * (logits + self.epsilon).log()
            loss += -(1 - self.alpha) * zero_hot_key * torch.pow(logits, self.gamma) * (1 - logits + self.epsilon).log()
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'none':
            pass
        return loss


class FocalCosineLoss(nn.Module):
    """Implementation Focal cosine loss.
    [Data-Efficient Deep Learning Method for Image Classification
    Using Data Augmentation, Focal Cosine Loss, and Ensemble](https://arxiv.org/abs/2007.07805).
    Source : <https://www.kaggle.com/c/cassava-leaf-disease-classification/discussion/203271>
    """

    def __init__(self, alpha=1, gamma=2, xent=0.1, reduction="mean"):
        """Constructor for FocalCosineLoss.
        """
        super(FocalCosineLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.xent = xent
        self.reduction = reduction

    def forward(self, logits, target):
        """Forward Method."""
        cosine_loss = nn.functional.cosine_embedding_loss(
            logits,
            torch.nn.functional.one_hot(target, num_classes=logits.size(-1)),
            torch.tensor([1], device=target.device),
            reduction=self.reduction,
        )

        cent_loss = nn.functional.cross_entropy(
            nn.functional.normalize(logits), target, reduction="none")
        pt = torch.exp(-cent_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * cent_loss

        if self.reduction == "mean":
            focal_loss = torch.mean(focal_loss)
        return cosine_loss + self.xent * focal_loss


_REDUCTION = 'mean'
LOSS_FN_LIB = {
    # -w_i(y_i \cdot \log x_i + (1-y_i) \ log (1-x_i))
    'bce': nn.BCELoss(  # (sigmoid(logits).squeeze(-1), target.double())
        reduction=_REDUCTION),
    'bcel': nn.BCEWithLogitsLoss(  # (logits.squeeze(-1), target.float())
        reduction=_REDUCTION),
    'ce': nn.CrossEntropyLoss(  # (logits, target.long())
        ignore_index=0,  # 0 for [PAD]
        reduction=_REDUCTION),  
    # - \alpha_t \cdot (1-p_t)^{\gamma} \log (p_t)
    'focal': FocalLoss(  # (logits.squeeze(-1), target)
        num_labels=None,  # alpha=-1, 
        activation_type='sigmoid',
        reduction=_REDUCTION),
    'mse': nn.MSELoss(  # (sigmoid(logits).squeeze(-1), target.float())
        reduction=_REDUCTION),
    'nll': nn.NLLLoss(  # (log(softmax(logits)), target)
        ignore_index=-1, 
        reduction=_REDUCTION),
}