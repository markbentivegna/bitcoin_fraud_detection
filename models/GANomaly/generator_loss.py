from torch import nn

class GeneratorLoss(nn.Module):
    def __init__(self, w_adversarial=1, w_contextual=50, w_encoder=1):
        super().__init__()

        self.loss_adversarial = nn.MSELoss()
        self.loss_contextual = nn.L1Loss()
        self.loss_encoder = nn.SmoothL1Loss()

        self.w_adversarial = w_adversarial
        self.w_contextual = w_contextual
        self.w_encoder = w_encoder

    def forward(self, latent_input, latent_output, input_data, generated_data, pred_real, pred_fake):
        error_adversarial = self.loss_adversarial(pred_real, pred_fake)
        error_contextual = self.loss_contextual(input_data, generated_data)
        error_encoder = self.loss_encoder(latent_input, latent_output)
        loss = (error_adversarial * self.w_adversarial) + (error_contextual * self.w_contextual) + (error_encoder * self.w_encoder)
        return loss