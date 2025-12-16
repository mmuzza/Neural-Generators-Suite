import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision.utils as vutils
import os
from utils.data_utils import load_pt_data, get_device
from models import SimpleNoisePredictor

os.makedirs('outputs', exist_ok=True)

class SimpleDiffusionTrainer:
    def __init__(self, image, device='cuda'):
        self.device = device
        self.model = SimpleNoisePredictor().to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        self.image = image.to(device)
        self.output_dir = f"./outputs/simple_diffusion"
        os.makedirs(self.output_dir, exist_ok=True,)
        
        # sample fixed noise
        torch.manual_seed(42)
        num_noises = 1
        noise_amplitude = 8
        self.fixed_noises = [torch.rand_like(self.image)*noise_amplitude for i in range(num_noises)]


    def sample_noise(self, idx=None, seed=42):
        torch.manual_seed(seed)
        if idx is None:
            idx = torch.randint(0,len(self.fixed_noises),(1,)).item()
        return self.fixed_noises[idx]
        
    def train_step(self):
        
        # DEBUG: Check first layer weights
        first_weight = list(self.model.parameters())[0]
        print(f"First weight sum: {first_weight.sum().item():.6f}")

        self.model.train()
        torch.manual_seed(42)
        image = self.image
        fixed_noise = self.sample_noise()
        loss = None
        pred_noise = None


        #############################################################################
        # TODO:                                                                     #
        #  
        #     1. create noisy image by taking the image defined above and adding the fixed noise to it.
        #     2. predict noise given noisy image using the model defined in the trainer class.
        #     3. compute loss between ground truth noise and predicted noise. and then apply backprop.
        #     4. use the variables for loss and pred_noise defined above as those are returned.                                  #
        #############################################################################

        noisy_image = image + fixed_noise

        pred_noise = self.model(noisy_image)
        pred_noise = pred_noise.view(fixed_noise.shape)

        loss = F.mse_loss(pred_noise, fixed_noise)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return loss.item(), pred_noise
    
    @torch.no_grad()
    def reverse_step(self, step):
        
        self.model.eval()
        # create noisy image.
        fixed_noise = self.sample_noise()
        noisy_image = self.image + fixed_noise
        denoised_image = None # use this variable to save the denoised image.
        pred_noise = None # use this variable to save the predicted noise.


        #############################################################################
        # # TODO:                                                                     #
        #     1. you are given a noisy image. predict the noise 
        #     2. use the predicted noise to denoise the image.   
        #      3. use the denoised_image and pred_noise variable to store the respective data #
        #############################################################################

        pred_noise_flat = self.model(noisy_image)
        pred_noise = pred_noise_flat.view(self.image.shape)
        denoised_image = noisy_image - pred_noise

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        
        # visualization
        grid = torch.cat([
            self.image,        # orig image
            noisy_image,       # noisy image
            fixed_noise,    # fixed noise (target)
            pred_noise,        # predicted noise
            denoised_image      # denoised image
        ], dim=0)
        
        vutils.save_image(grid.unsqueeze(1), os.path.join(self.output_dir,f"noise_test_{step:04d}.png"), nrow=5, normalize=True)
        return grid

def train():
    images, _ = map(list, zip(*load_pt_data('mnist_1_shots.pt')))
    image = images[0]
    
    # normalize to [-1,1]
    single_image = 2 * image - 1 

    device = get_device()
    trainer = SimpleDiffusionTrainer(single_image, device=device)
    
    trainer.reverse_step(step=0)
    # training loop
    for step in range(100):
        loss, pred_noise = trainer.train_step()
        
        if (step+1) % 25 == 0:
            print(f"Step {step}, Loss: {loss:.6f}")
            trainer.reverse_step(step)
    

if __name__ == "__main__":
    train()