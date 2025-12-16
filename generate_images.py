import os, torch, time, argparse
import yaml
from trainer_vae import VAETrainer
from trainer_gan import GANTrainer
from trainer_diffusion import DiffusionTrainer
from config import Config

import torchvision.utils as vutils

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="trainer")
    parser.add_argument('--config_file', type=str, default='configs/config.yaml', help="path to YAML config")
    parser.add_argument('--output_dir', type=str, default=None, help="path to output directory (optional); defaults to outputs/model_name")
    args = parser.parse_args()

    # Load YAML configuration
    with open(args.config_file, 'r') as file:
        config_dict = yaml.safe_load(file)
        config = Config(config_dict=config_dict)

    if config.network.model.lower() == 'vae':
        net_trainer = VAETrainer(config=config, output_dir=args.output_dir)
    elif config.network.model.lower() == 'gan':
        net_trainer = GANTrainer(config=config, output_dir=args.output_dir)
    elif config.network.model.lower() =='diffusion':
        net_trainer = DiffusionTrainer(config=config, output_dir=args.output_dir)
    else:
        raise ValueError(f"{config.network.model} not supported.") 
    
    args.output_dir = net_trainer.output_dir
    
    fixed_eval_latents = torch.randn((2000, config.network.latent_dim), device = net_trainer.device)

    if config.network.model.lower() == 'gan':
        model = net_trainer.load_model(f"{args.output_dir}/generator_{net_trainer.dataset.lower()}.pth", map_location=net_trainer.device)
        
        model.eval()  
        with torch.no_grad():
            res = model(fixed_eval_latents.to(net_trainer.device))
        all_images = res.view(-1, 1, 28, 28)

    if config.network.model.lower() == 'vae':
        model = net_trainer.load_model(f"{args.output_dir}/vae_{net_trainer.dataset.lower()}.pth", map_location=net_trainer.device)
        model.eval()  
        with torch.no_grad():
            res = model.decoder(fixed_eval_latents.to(net_trainer.device))
        all_images = res.view(-1, 1, 28, 28)

    if config.network.model.lower() == 'diffusion':
        model = net_trainer.load_model(f"{args.output_dir}/diffusion_net_{net_trainer.dataset.lower()}.pth", map_location=net_trainer.device)
        net_trainer.net = model
        model.eval()  
        with torch.no_grad():
            res = net_trainer.generate(2000)
        all_images = res.view(-1, 1, 28, 28)

    
    vutils.save_image(all_images[0:144,:,:,:], f"{args.output_dir}/grid.png", nrow=12)
    os.makedirs(f"{args.output_dir}/images/", exist_ok=True)
    for i in range(fixed_eval_latents.size(0)):
        vutils.save_image(all_images[i,:,:,:], f"{args.output_dir}/images/output{i}.png")
 
