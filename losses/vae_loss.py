import torch
import torch.nn as nn
import torch.nn.functional as F


class VAELoss(nn.Module):
    def __init__(self, beta=1, recon_loss='l2', reduction='mean', return_losses=False):
        super(VAELoss, self).__init__()
        self.beta = beta
        if recon_loss.lower() == 'l2':
            self.reconstruction_loss = nn.MSELoss(reduction=reduction)
        if recon_loss.lower() =='l1':
            self.reconstruction_loss = nn.L1Loss(reduction=reduction)
        if recon_loss.lower() == 'bce':
            self.reconstruction_loss = nn.BCEWithLogitsLoss(reduction=reduction)

        self.return_losses = return_losses


    def forward(self, reconstructed, original, mu, logvar):
        loss, loss_recon, loss_kl = None, None, None

        # 1
        loss_recon = self.reconstruction_loss(reconstructed, original)
        loss_recon = loss_recon * 784
        
        # 2
        loss_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # 3
        loss = loss_recon + self.beta * loss_kl


        
        if self.return_losses:
            return loss, loss_recon, loss_kl
        else:
            return loss