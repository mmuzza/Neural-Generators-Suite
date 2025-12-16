import torch
import torch.nn as nn
import torch.nn.functional as F
from .decoder import BasicDecoder

class BasicEncoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(BasicEncoder, self).__init__()

        """
            args:
                input_dim: dim of input image
                hidden_dim: dim of hidden layer
                latent_dim: dim of latent vectors mu and logvar
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

    
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        

    def forward(self, x):   
        """ forward pass for vae encoder.
            args: 
                x: [N, input_dim]

            outputs:
                mu: [N, latent_dim]
                logvar: [N, latent_dim]
        """
        
        mu, logvar = None, None

   
        # Hidden layer
        h = F.relu(self.fc1(x))

        # Computing mean and log variance
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar

    
        return mu, logvar


class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=256):

        """
            args:
                input_dim: dim of input image
                hidden_dim: dim of hidden layer
                latent_dim: dim of latents

            students implement Basic VAE using encoder and decoder. 
            1. instantiate encoder and pass in variables.
            2. instantiate decoder and pass in variables.
        """
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim



        self.encoder = BasicEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = BasicDecoder(latent_dim, hidden_dim, input_dim)
 

    def reparameterize(self, mu, logvar):
        """
            args:
                mu: [N,latent_dim]
                logvar: [N, latent_dim]
            outputs:
                z: reparameterized representation [N, latent_dim]

        """

        # z = None

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps

        return z
    
    @torch.jit.export
    def encode(self, x):
        """
            args:
                x: input [N, C, H, W]
            outputs:
                z: latent representation [N, latent_dim]
                mu: mean latent [N, latent_dim]
                logvar: log variance latent [N, latent_dim]
        """
        # z, mu, logvar = None, None, None


        x = x.view(x.size(0), -1)
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)

        return (z, mu, logvar)
    
    def forward(self, x):
        """
            args:
                x: input [N, C, H, W]
            outputs:
                z: latent representation
                mu: mean latent
                logvar: log variance latent
        """

        # out, mu, logvar = None, None, None


        x = x.view(x.size(0), -1)
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        out = self.decoder(z)
        return (out, mu, logvar)

  
        return (out, mu, logvar)
        
    @torch.jit.export
    @torch.no_grad()
    def generate(self, z):
        """
            args:
                x: input [N, latent_dim]
            outputs:
                out: (N, input_dim)
        """
        out = None
        out = self.decoder(z)
        return out
