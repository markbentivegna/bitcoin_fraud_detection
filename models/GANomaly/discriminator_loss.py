import torch
from torch import nn

class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_bce = nn.BCELoss()

    def forward(self, pred_real, pred_fake):
        error_discriminator_real = self.loss_bce(pred_real, torch.ones(size=pred_real.shape, dtype=torch.float32, device=pred_real.device))
        error_discriminator_fake = self.loss_bce(pred_fake, torch.zeros(size=pred_fake.shape, dtype=torch.float32, device=pred_fake.device))

        loss_discriminator = (error_discriminator_real + error_discriminator_fake) * 0.5
        return loss_discriminator