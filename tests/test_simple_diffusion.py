import unittest
import torch, os
from models import SimpleNoisePredictor
import torch.nn.functional as F
from simple_diffusion import SimpleDiffusionTrainer 


class TestSimpleDiffusionTrainer(unittest.TestCase):
    
    def setUp(self):
        self.device = 'cpu'
        self.load_dir = 'tests/assets/'
        self.image = torch.rand(1, 28, 28) 
        self.trainer = SimpleDiffusionTrainer(self.image, device=self.device)
    
    def test_basic_checks(self):
        initial_params = [p.clone().detach() for p in self.trainer.model.parameters()]
        
        loss, pred_noise = self.trainer.train_step()
        
        self.assertIsInstance(loss, float)
        self.assertGreater(loss, 0.0)
        
        self.assertEqual(pred_noise.shape, self.trainer.sample_noise().shape)
        
        for param_before, param_after in zip(initial_params, self.trainer.model.parameters()):
            self.assertFalse(torch.equal(param_before, param_after), msg="model params did not update. recheck implementation")
   
    def test_step_loss(self):
        torch.manual_seed(42)
        student_trainer = SimpleDiffusionTrainer(self.image, device=self.device)
        
        torch.manual_seed(42)
        student_loss, _ = student_trainer.train_step()
        
        ref_loss = 21.6167
        
        self.assertAlmostEqual(student_loss, ref_loss, places=1)
    
    def test_step_noise_prediction(self):
        torch.manual_seed(42)
        student_trainer = SimpleDiffusionTrainer(self.image, device=self.device)

        ref_pred_noise = torch.load(os.path.join(self.load_dir, 'ref_pred_noise.pt'), map_location='cpu')
        
        torch.manual_seed(42)
        _, student_pred_noise = student_trainer.train_step()
        
        self.assertTrue(torch.allclose(student_pred_noise, ref_pred_noise, atol=1e-2))



if __name__ == "__main__":
    unittest.main()