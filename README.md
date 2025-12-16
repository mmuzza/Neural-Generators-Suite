# Generative Models: VAE, GAN, and Diffusion

A comprehensive PyTorch implementation of three fundamental generative models trained on MNIST and Fashion-MNIST datasets.

## 🎯 Overview

This project implements three core generative modeling approaches from scratch:

- **Variational Autoencoder (VAE)** - Learns a probabilistic latent representation with continuous generation capabilities
- **Generative Adversarial Network (GAN)** - Uses adversarial training between generator and discriminator networks
- **Diffusion Model** - Implements denoising diffusion with a UNet architecture for high-quality image generation

All models are trained and evaluated on MNIST and Fashion-MNIST datasets with configurable hyperparameters.

## 📁 Project Structure

```
.
├── models/
│   ├── vae.py              # VAE encoder, decoder, and full model
│   ├── gan.py              # GAN generator and discriminator
│   ├── decoder.py          # Shared decoder/generator architecture
│   ├── Unet.py             # UNet architecture for diffusion
│   └── noise_predictor.py  # Simple noise prediction network
├── losses/
│   └── loss.py             # VAE loss with KL divergence and reconstruction
├── configs/
│   ├── config_vae.yaml     # VAE training configuration
│   ├── config_gan.yaml     # GAN training configuration
│   └── config_diffusion.yaml  # Diffusion training configuration
└── README.md
```

## 🏗️ Model Architectures

### Variational Autoencoder (VAE)

The VAE consists of an encoder that maps inputs to a latent distribution and a decoder that reconstructs from samples:

- **Encoder**: `784 → 1024 → (μ, log σ²)` with 80-dimensional latent space
- **Decoder**: `80 → 1024 → 784` with sigmoid output
- **Loss**: Reconstruction (L2/L1) + β × KL divergence

The reparameterization trick (`z = μ + σ × ε`) enables backpropagation through stochastic sampling.

### Generative Adversarial Network (GAN)

The GAN uses adversarial training between two networks:

- **Generator**: `20 → 400 → 784` transforms random noise to images
- **Discriminator**: `784 → 400 → 1` classifies real vs. fake images
- Supports both ReLU and LeakyReLU activations

### Diffusion Model

Implements denoising diffusion with modern architectural choices:

- **UNet**: Residual blocks with skip connections, time embeddings, and group normalization
- **Features**: 
  - 3-level encoder-decoder with 64→128→256 channels
  - Time conditioning via MLP embeddings
  - Cosine noise schedule with 1000 timesteps
  - DDIM sampling with 50 steps for fast generation
  - Optional classifier-free guidance (scale 2.0)
  - EMA for stable training (decay 0.9999)

## ⚙️ Configuration

Each model uses YAML configuration files for hyperparameters. Key settings:

### VAE Configuration (Best Results)

```yaml
train:
  batch_size: 128
  lr: 0.0005
  n_epochs: 150

network:
  hidden_dim: 1024
  latent_dim: 80

vae:
  vae_recon_loss: l2
  beta: 0.00005  # Low beta for sharp reconstructions
```

### Diffusion Configuration

```yaml
train:
  batch_size: 128
  lr: 0.0002
  n_epochs: 10

diffusion:
  timesteps: 1000
  noise_schedule: cosine
  sample:
    method: ddim
    ddim_steps: 50
```

## 🚀 Training

Configure your model in the appropriate YAML file, then train:

```bash
python train.py --config configs/config_vae.yaml
python train.py --config configs/config_gan.yaml
python train.py --config configs/config_diffusion.yaml
```

## 📊 Results & Performance

### VAE Results (MNIST)

Achieved excellent reconstruction quality through extensive hyperparameter tuning:

- **Final Loss**: ~40-50 (L2 reconstruction)
- **Key Insight**: Lower β values (0.00005) produce sharper reconstructions while maintaining reasonable latent structure

### Training Notes

The config files include extensive hyperparameter exploration history (commented out), showing the optimization process:

- **β tuning**: Started at 1.0, progressively reduced to 0.00005
- **Latent dimensions**: Tested 8, 16, 20, 32, 64, 80 (settled on 80)
- **Loss functions**: Compared L1, L2, and BCE (L2 performed best)
- **Architecture**: Increased hidden_dim from 400 to 1024 for better capacity

## 💡 Key Implementation Details

### VAE Reparameterization

```python
std = torch.exp(0.5 * logvar)
eps = torch.randn_like(std)
z = mu + std * eps
```

### VAE Loss Components

```python
loss_recon = MSE(reconstructed, original) × 784
loss_kl = -0.5 × Σ(1 + log σ² - μ² - σ²)
loss = loss_recon + β × loss_kl
```

### Diffusion Time Embedding

The UNet uses learned time embeddings that are injected at multiple scales:

```python
t_emb = MLP(t) → [time_dim]
# Injected at different resolutions via addition
```

## 📦 Dependencies

```bash
torch
torchvision
pyyaml
numpy
```

Install via:

```bash
pip install torch torchvision pyyaml numpy
```

## 🔮 Future Improvements

- Add FID score evaluation
- Implement conditional generation
- Add latent space interpolation visualizations
- Support for custom datasets
- Attention mechanisms in UNet for diffusion
- Progressive training for diffusion models

## 📄 License

MIT

## 🙏 Acknowledgments

Implementations based on foundational papers:

- **VAE**: Kingma & Welling (2013) - "Auto-Encoding Variational Bayes"
- **GAN**: Goodfellow et al. (2014) - "Generative Adversarial Networks"
- **Diffusion**: Ho et al. (2020) - "Denoising Diffusion Probabilistic Models", Song et al. (2020) - "Denoising Diffusion Implicit Models"
