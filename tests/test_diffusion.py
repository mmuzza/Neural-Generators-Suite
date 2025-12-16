import unittest, os
import numpy as np
import torch, yaml
from train import Config
from trainer_diffusion import DiffusionTrainer



class TestDiffusion(unittest.TestCase):

    def setUp(self):
        
        self.load_dir = 'tests/assets/'
        config_file = './tests/assets/config_diffusion.yaml'
        with open(config_file, 'r') as file:
            config_dict = yaml.safe_load(file)
            self.config = Config(config_dict=config_dict)

        self.trainer = DiffusionTrainer(config=self.config, device='cpu')

    def test_noise_schedule(self):
        beta = self.trainer.beta
        expected_beta = torch.tensor([1.0000e-04, 2.3111e-03, 4.5222e-03, 6.7333e-03, 8.9444e-03, 1.1156e-02,
                            1.3367e-02, 1.5578e-02, 1.7789e-02, 2.0000e-02], device=self.trainer.device)
        
        self.assertTrue(torch.allclose(beta, expected_beta, atol=1e-3), "Beta schedule calculated incorrectly.")

        alpha = self.trainer.alpha
        expected_alpha = torch.tensor([0.9999, 0.9977, 0.9955, 0.9933, 0.9911, 0.9888, 0.9866, 0.9844, 0.9822,
                            0.9800], device=self.trainer.device)
        self.assertTrue(torch.allclose(alpha, expected_alpha, atol=1e-3), "Alpha calculated incorrectly")

        alphas_bar = self.trainer.alphas_bar
        expected_alphas_bar = torch.tensor([0.9999, 0.9976, 0.9931, 0.9864, 0.9776, 0.9667, 0.9537, 0.9389, 0.9222,
                                        0.9037], device=self.trainer.device)

        self.assertTrue(torch.allclose(alphas_bar, expected_alphas_bar, atol=1e-3), "Alphas bar calculated incorrectly.")
        
    def test_forward_diffusion_shape(self):
        x_0 = torch.randn((self.trainer.batch_size, 1, self.trainer.height, self.trainer.width), device=self.trainer.device)
        t = torch.randint(0, self.config.diffusion.timesteps, (self.trainer.batch_size,), device=self.trainer.device)

        x_t, noise = self.trainer.forward_diffusion(x_0, t)

        self.assertEqual(x_t.shape, x_0.shape, "x_t shape mismatch")
        self.assertEqual(noise.shape, x_0.shape, "Noise shape mismatch")

    def test_forward_diffusion_values(self):
        x_0 = torch.load(os.path.join(self.load_dir, 'fwd_diff_input.pth'), weights_only=True, map_location='cpu')
        x_0 = x_0.to(self.trainer.device)
        t = torch.tensor([3, 9, 0, 4], device=self.trainer.device)

        torch.manual_seed(42) 
        x_t, noise = self.trainer.forward_diffusion(x_0, t)

        expected_noise = torch.load(os.path.join(self.load_dir, 'fwd_diff_noise.pth'), weights_only=True, map_location='cpu')
        expected_x_t = torch.load(os.path.join(self.load_dir, 'fwd_diff_output.pth'), weights_only=True, map_location='cpu')

        self.assertTrue(torch.allclose(x_t, expected_x_t, atol=1e-1), "Forward diffusion expected output mismatch")
        self.assertTrue(torch.allclose(noise, expected_noise, atol=1e-1), "Forward diffusion expected noise mismatch")

    def test_sample_timestep_t0(self):
        x = torch.load(os.path.join(self.load_dir, 'fwd_diff_output.pth'), weights_only=True, map_location='cpu')
        x = x.to(self.trainer.device)
        t = torch.zeros((self.trainer.batch_size,), dtype=torch.long, device=self.trainer.device)

        torch.manual_seed(42) 
        x_prev = self.trainer.sample_timestep(x,t)   
        x_prev_expected = torch.load(os.path.join(self.load_dir, 'sample_timestep_t0.pth'), weights_only=True, map_location='cpu')

        self.assertTrue(torch.allclose(x_prev, x_prev_expected, atol=1e-1), "sample_timestep calculation does not match expected output when t=0.")


    def test_sample_timestep(self):
        x = torch.load(os.path.join(self.load_dir, 'fwd_diff_output.pth'), weights_only=True, map_location='cpu')
        x = x.to(self.trainer.device)
        t = torch.ones((self.trainer.batch_size,), dtype=torch.long, device=self.trainer.device)

        torch.manual_seed(42) 
        x_prev = self.trainer.sample_timestep(x,t)

        x_prev_expected = torch.load(os.path.join(self.load_dir, 'sample_timestep.pth'), weights_only=True, map_location='cpu')

        self.assertTrue(torch.allclose(x_prev, x_prev_expected, atol=1e-1), "sample_timestep calculation does not match expected output.")

 
    def test_noise_prediction(self):
        x_0 = torch.load(os.path.join(self.load_dir, 'fwd_diff_input.pth'), weights_only=True, map_location='cpu')
        x_0 = x_0.to(self.trainer.device)
        t = torch.tensor([3, 9, 0, 4], device=self.trainer.device)
        
        torch.manual_seed(42) 
        x_t, noise = self.trainer.forward_diffusion(x_0, t)
        out = self.trainer.net(x_t, t)

        expected_out = torch.load(os.path.join(self.load_dir, 'net_output.pth'), weights_only=True, map_location='cpu')

        expected_loss_oracle = torch.tensor(1.2927354574203491)
        loss_noise = self.trainer.criterion(out, noise)
        loss_out = self.trainer.criterion(out, expected_out)

        self.assertTrue(torch.allclose(out, expected_out, atol=1e-1), "predicted noise does not match expected output.")
        self.assertTrue(torch.allclose(expected_loss_oracle, loss_noise, atol=1e-1), "expected loss between model and noise does not match.")

   


if __name__ == '__main__':
    unittest.main()






