from trainer_gan import GANTrainer
import unittest
import torch
from config import Config

class TestGAN(unittest.TestCase):
    
    def setUp(self):
        config_dict = {"data": {"dataset": "mnist"}, 
               "train": {"batch_size": 64, 
                         "lr": 0.0001, 
                         "n_epochs": 20,
                         "num_workers": 4},
                "network":{"model": "diffusion", 
                           "hidden_dim": 400,
                           "latent_dim": 128},
                "vae": {"vae_recon_loss": "l2", "beta": 2},
                "optimizer": {"type": "adamw", 
                              "weight_decay": 0.0},
                "gan": {"leaky": False},
                "diffusion": {"time_dim": 128, "timesteps": 500}}
        config = Config(config_dict)
        torch.manual_seed(42)
        self.trainer = GANTrainer(config, device='cpu') 

    def test_compute_loss_real(self):
        data = torch.ones(self.trainer.batch_size, 1, 28, 28)
        loss = self.trainer.compute_loss_real(data, self.trainer.batch_size)
        self.assertAlmostEqual(loss.item(), 0.5430, places=2)

    def test_compute_loss_fake(self):
        loss = self.trainer.compute_loss_fake(self.trainer.batch_size, self.trainer.config.network.latent_dim)
        self.assertAlmostEqual(loss.item(), 0.7849, places=2)

    def test_compute_loss_gen(self):
        loss = self.trainer.compute_loss_gen(self.trainer.batch_size, self.trainer.config.network.latent_dim)
        self.assertAlmostEqual(loss.item(), 0.6091, places=2)


if __name__ == '__main__':
    unittest.main()