import unittest
import torch
from models.VAE import VAE 

class TestVAE(unittest.TestCase):
    
    def setUp(self):
        self.model = VAE(input_dim=784, hidden_dim=400, latent_dim=20)
        self.batch_size = 16
        
    
    def test_output_shapes_encoder(self):
        batch_size = 16
        input_tensor = torch.randn(batch_size, 784)
        z, mu, logvar = self.model.encode(input_tensor)

        self.assertEqual(z.shape, (batch_size, 20), f"Expected mu shape (16, 20), got {z.shape}")
        self.assertEqual(mu.shape, (batch_size, 20), f"Expected mu shape (16, 20), got {mu.shape}")
        self.assertEqual(logvar.shape, (batch_size, 20), f"Expected var shape (16, 20), got {logvar.shape}")
    
    def test_output_shapes_decoder(self):
        batch_size = 16
        z = torch.randn(batch_size, 20)
        out = self.model.generate(z)

        self.assertEqual(out.shape, (batch_size, 784), f"Expected out shape (16, 784), got {out.shape}")

    def test_forward_pass(self):
        input_tensor = torch.randn(32, 784)
        out, mu, logvar = self.model(input_tensor)
        self.assertIsInstance(out, torch.Tensor, "output should be a tensor")
        self.assertIsInstance(mu, torch.Tensor, "mu should be a tensor")
        self.assertIsInstance(logvar, torch.Tensor, "var should be a tensor")

    def test_reparam_shape(self):
        mu = torch.randn(self.batch_size, 20)
        logvar = torch.randn(self.batch_size, 20)

        z = self.model.reparameterize(mu, logvar)

        self.assertEqual(z.shape, mu.shape, "reparam of z should have the same shape as mu")

    def test_reparam_zero_logvar(self):
        mu = torch.randn(self.batch_size, 20)
        logvar = torch.zeros(self.batch_size, 20)

        z = self.model.reparameterize(mu, logvar)

        eps_hat = z - mu

        self.assertAlmostEqual(eps_hat.var().item(), 1.0, delta=0.2, msg='variance should be near 1 when logvar=0')

    def test_reparam_nonzero_logvar(self):
        mu = torch.randn(self.batch_size, 20)
        logvar = torch.ones(self.batch_size, 20)

        z = self.model.reparameterize(mu, logvar)

        self.assertFalse(torch.allclose(z, mu, atol=1e-5), "z != mu when logvar is nonzero")

    def test_reparam_stochasticity(self):
        mu = torch.randn(self.batch_size, 20)
        logvar = torch.randn(self.batch_size, 20)

        z1 = self.model.reparameterize(mu, logvar)
        z2 = self.model.reparameterize(mu, logvar)

        self.assertFalse(torch.allclose(z1, z2, atol=1e-5), "reparam should produce stochastic results with same mu & logvar")

if __name__ == '__main__':
    unittest.main()
