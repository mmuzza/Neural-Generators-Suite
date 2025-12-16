from models.VAE import VAE
from losses import VAELoss
from utils.data_utils import set_seed, get_device, AverageMeter
from utils.trainer import Trainer
import os, torch, time, argparse
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import yaml


class VAETrainer(Trainer):
    def __init__(self, config, output_dir=None):
        super().__init__(config, output_dir=output_dir)

        self.net, self.criterion = self._init_vae()
        self.net.to(device=self.device)
        self.optimizer = self._init_optimizer(self.net)
        self.fixed_eval_latents = torch.randn((16, self.config.network.latent_dim), device=self.device)

 

    @staticmethod
    def _build_vae(input_dim, hidden_dim, latent_dim):
        """
        build basic vae given inputs
        """
        return VAE(input_dim=input_dim, 
                   hidden_dim=hidden_dim,
                   latent_dim=latent_dim)

    def _init_vae(self):
        """
        instantiate vae and criterion and return them.
        """
        vae = self._build_vae(input_dim=self.input_dim, 
                              hidden_dim=self.config.network.hidden_dim, 
                              latent_dim=self.config.network.latent_dim)
        criterion = VAELoss(beta = self.config.vae.beta,
                             recon_loss=self.config.vae.vae_recon_loss,
                             return_losses=True) 
        return vae, criterion
    

    def train(self):
        start = time.time()
        loss_meter = AverageMeter()
        iter_meter = AverageMeter()
        loss_recon_meter = AverageMeter()
        loss_kl_meter = AverageMeter()

        self.fixed_eval_batch.to(self.device)
        self.fixed_train_batch = None

        self.evaluate(epoch=-1)

        loss_func = self.criterion
        model = self.net
        
        # TODO: use the following variables to store the output of the network and the loss terms.
        loss, l2, l_kl = None, None, None 
        out = None

        for epoch in range(self.n_epochs):
           
            for i, (data, _ ) in enumerate(self.train_loader):
                start = time.time()

                data = data.to(self.device)

                self.optimizer.zero_grad()

                # 1
                out, mu, logvar = model(data.view(data.size(0), -1))

                # 2
                im_out = out.reshape(out.size(0), self.height, self.width)
                loss, l2, l_kl = loss_func(im_out, data.squeeze(), mu, logvar)

                # 3
                loss.backward()
                self.optimizer.step()
      
                loss_meter.update(loss.item(), out.size(0))
                iter_meter.update(time.time()-start)
                loss_kl_meter.update(l_kl.item(), out.size(0))
                loss_recon_meter.update(l2, out.size(0))

                if self.fixed_train_batch is None:
                    self.fixed_train_batch = data[:min(self.batch_size, 8)].clone()

                if i % 1000 == 0:
                    print(
                        f'Epoch: [{epoch}][{i}/{len(self.train_loader)}]\t'
                        f'Loss {loss_meter.val:.3f} ({loss_meter.avg:.3f})\t'
                        f'Loss_recon {loss_recon_meter.val:.3f} ({loss_recon_meter.avg:.3f})\t'
                        f'Loss_kl {loss_kl_meter.val:.3f} ({loss_kl_meter.avg:.3f})\t'
                        f'Time {iter_meter.val:.3f} ({iter_meter.avg:.3f})\t'
                        )

            if (epoch+1) % 5 == 0:  
                filename = f"{self.output_dir}/train_recon_epoch_{epoch+1}.png"  
                filename_samples = f"{self.output_dir}/samples_{epoch}.png"                
                self.sample_and_save(self.fixed_eval_latents, filename=filename_samples, n=self.fixed_eval_latents.size(0))
                self.evaluate(epoch=epoch)

        filename = f"{self.output_dir}/final_recon.png"  
        self.reconstruct_and_save(self.fixed_train_batch.view(self.fixed_train_batch.size(0), -1), filename, n=self.fixed_train_batch.size(0))
        self.sample_and_save(self.fixed_eval_latents, filename=filename_samples, n=self.fixed_eval_latents.size(0)) 
        print(f"Completed in {(time.time()-start):.3f}")
        self.save_model(self.net, f'{self.output_dir}/vae_{self.dataset.lower()}.pth')

    def evaluate(self, epoch):
        
        filename = f"{self.output_dir}/eval_epoch_{epoch+1}.png"               
        self.reconstruct_and_save(self.fixed_eval_batch.view(self.fixed_eval_batch.size(0), -1), filename, n=self.fixed_eval_batch.size(0))
 
        loss_meter = AverageMeter()
        iter_meter = AverageMeter()

        epoch_start = time.time()
        
        self.net.eval()
        with torch.no_grad():  
            for i, (data, _ ) in enumerate(self.test_loader):
                start = time.time()

                data = data.to(self.device)
                out, mu, logvar = self.net(data.view(data.size(0), -1))
                im_out = out.reshape(out.size(0), self.height, self.width)
                loss, l2, l_kl = self.criterion(im_out, data.squeeze(), mu, logvar)

                loss_meter.update(loss.item(), out.size(0))
                iter_meter.update(time.time()-start)

                if i % 500 == 0:
                    print(
                        f'Val Epoch: [{epoch}][{i}/{len(self.train_loader)}]\t'
                        f'Loss {loss_meter.val:.3f} ({loss_meter.avg:.3f})\t'
                        f'Time {iter_meter.val:.3f} ({iter_meter.avg:.3f})\t'
                        )
                    
        print(f"Val completed in {(time.time()-epoch_start)/60:.3f} min. Loss {loss_meter.val:.3f}")
        return loss_meter.val



    def reconstruct_and_save(self, data, filename, n):
        self.net.eval()  
        with torch.no_grad():
            res, _, _ = self.net(data.to(self.device))
        save_reconstruction(data, res, filename, n)
        self.net.train()

    def sample_and_save(self, data, filename, n):
        self.net.eval()  
        with torch.no_grad():
            res = self.net.generate(data.to(self.device))
        vutils.save_image(res.view(-1,1,28,28), filename, nrow=n)
        self.net.train()

def save_reconstruction(data, res, filename, nrow=8):
    data = data[:nrow].cpu().view(-1, 28, 28).numpy()  
    res = res[:nrow].cpu().view(-1, 28, 28).numpy()    

    fig, axes = plt.subplots(2, nrow, figsize=(nrow * 2, 4))
    
    for i, ax in enumerate(axes[0]):
        ax.imshow(data[i], cmap='gray')
        ax.axis('off')
    axes[0, 0].set_title('Original', fontsize=10, loc='left', pad=10)
    
    for i, ax in enumerate(axes[1]):
        ax.imshow(res[i], cmap='gray')
        ax.axis('off')
    axes[1, 0].set_title('Reconstructed', fontsize=10, loc='left', pad=10)
    
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()