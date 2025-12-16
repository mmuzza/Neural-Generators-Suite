import unittest, os
import numpy as np
import torch
from losses.vae_loss import VAELoss

class TestVAELoss(unittest.TestCase):

    def setUp(self):
        self.loss_fn = VAELoss(beta=1, recon_loss='l2')
        self.batch_size = 16
        self.input_dim = 784
        self.latent_dim = 20
        # TODO: test this dir on windows
        self.load_dir = 'tests/assets/'

    def test_shape(self):
        original = torch.randn(self.batch_size, self.input_dim)
        reconstructed = torch.randn(self.batch_size, self.input_dim)
        mu = torch.randn(self.batch_size, self.latent_dim)
        logvar = torch.randn(self.batch_size, self.latent_dim)

        loss = self.loss_fn(reconstructed, original, mu, logvar)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, (), "Loss should be a scalar")

    def test_loss_computation(self):
        
        original = torch.tensor([[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]])
        reconstructed = torch.tensor([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
        mu = torch.tensor([[0.0, 0.0], [1.0, -1.0]])
        logvar = torch.tensor([[0.0, 0.0], [0.0, 0.0]])  

        expected_loss = 5881.
        computed_loss = self.loss_fn(reconstructed, original, mu, logvar)

        self.assertAlmostEqual(computed_loss.item(), expected_loss, delta=1e-3)

    def test_beta_scaling(self):
        beta = 5
        loss_func = VAELoss(beta=beta, recon_loss='l2')

        original = torch.tensor([[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]])
        reconstructed = torch.tensor([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
        mu = torch.tensor([[0.0, 0.0], [1.0, -1.0]])
        logvar = torch.tensor([[0.0, 0.0], [0.0, 0.0]])

        expected_loss = 5885
        computed_loss = loss_func(reconstructed, original, mu, logvar)

        self.assertAlmostEqual(computed_loss.item(), expected_loss, delta=1e-5)

    def test_loss_with_zero_kl(self):
        original_path = os.path.join(self.load_dir, "original.npy")
        reconstructed_path = os.path.join(self.load_dir, "reconstructed.npy")
        
        if not os.path.exists(original_path) or not os.path.exists(reconstructed_path):
            raise FileNotFoundError(f"test files not found at {self.load_dir}. double check things.")
        
        original = torch.from_numpy(np.load(original_path))
        reconstructed = torch.from_numpy(np.load(reconstructed_path))
        mu = torch.zeros(self.batch_size, self.latent_dim) 
        logvar = torch.zeros(self.batch_size, self.latent_dim) 

        expected_loss = 1590.9205

        computed_loss = self.loss_fn(reconstructed, original, mu, logvar)

        self.assertAlmostEqual(computed_loss.item(), expected_loss, delta=1e-2)

if __name__ == '__main__':
    unittest.main()
