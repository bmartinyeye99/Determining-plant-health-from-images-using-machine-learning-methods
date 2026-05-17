import torch
import torch.nn as nn
import torch.nn.functional as F


CLASS_INDEX = {
    "dead":     0,
    "severe":   1,
    "moderate": 2,
    "healthy":  3,
}


class HybridFocalDiceLoss(nn.Module):
    def __init__(
        self,
        weight=None,
        focal_weight=0.33,
        dice_weight=0.5,
        num_classes=4,
        focal_gamma=2.92,
        label_smoothing=0.05,
        betas=None,
    ):

        super().__init__()
        self.weight = weight
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.num_classes = num_classes
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing

        if betas is None:
            betas = [0.5] * num_classes
        self.register_buffer('beta_t', torch.tensor(betas, dtype=torch.float32))

    def forward(self, inputs, targets):
        device = inputs.device

        # --- Focal Loss ---
        ce_loss = F.cross_entropy(
            inputs, targets,
            weight=self.weight,
            reduction='none',
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.focal_gamma * ce_loss).mean()

        # --- Tversky Loss ---
        inputs_soft = F.softmax(inputs, dim=1)
        targets_one_hot = (
            F.one_hot(targets, num_classes=self.num_classes)
            .permute(0, 3, 1, 2)
            .float()
            .to(device)
        )

        dims = (0, 2, 3)
        intersection = torch.sum(inputs_soft * targets_one_hot, dims)
        fp = torch.sum(inputs_soft * (1.0 - targets_one_hot), dims)
        fn = torch.sum((1.0 - inputs_soft) * targets_one_hot, dims)

        beta_t = self.beta_t.to(device)
        alpha_t = 1.0 - beta_t

        tversky_index = (intersection + 1e-6) / (
            intersection + alpha_t * fp + beta_t * fn + 1e-6
        )
        dice_loss = (1.0 - tversky_index).mean()

        return self.focal_weight * focal_loss + self.dice_weight * dice_loss