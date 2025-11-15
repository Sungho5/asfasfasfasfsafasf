# Dental X-ray Anomaly Detection via Flow+Diffusion Normalizer

Unsupervised anomaly detection for panoramic dental X-rays using VAE + Flow Matching + Diffusion Models.

## Overview

This pipeline learns a **normalizer** `G(x)` that maps any input X-ray to its "nearest normal version". Anomalies are detected by comparing the input with its normalized version:

```
Anomaly Map = |x - G(x)| + λ_grad |∇x - ∇G(x)| + λ_HF |Laplacian(x) - Laplacian(G(x))|
```

**Key Idea**:
- Normal X-rays: `G(x) ≈ x` → Small anomaly map
- Abnormal X-rays: `G(x)` removes lesions → Large anomaly map highlights pathology

---

## Pipeline Architecture

```
Phase 1: VAE Fine-tuning
  [Normal X-rays] → VAE (Decoder fine-tuned) → Latent 4×32×32

Phase 2: Latent Dataset Generation
  [Normal X-rays] → Encoder (frozen) → {z₁, z₂, ..., zₙ} (normal latent manifold)

Phase 3: Flow + Diffusion Training
  UNet learns:
    - Diffusion: noise prediction ε on latent space
    - Flow Matching: velocity prediction v for fast sampling
    - Identity Constraint: G(x_normal) ≈ x_normal

Phase 4: Inference
  Test X-ray → Encode → Add noise → Denoise → Decode → Normalized X-ray
  Anomaly Map = Multi-scale difference (L1 + Gradient + HF)
```

---

## File Structure

```
.
├── config.py                # All hyperparameters (Phase 1-4)
├── dataset.py               # DICOM X-ray dataset loader
├── model.py                 # VAE fine-tuning model + losses
├── train_vae.py             # Phase 1: VAE training script
├── extract_latents.py       # Phase 2: Extract latent representations
├── latent_dataset.py        # Latent dataset loader for Phase 3
├── flowdiff_model.py        # Flow+Diff UNet architecture
├── flowdiff_loss.py         # Diffusion + Flow + Identity losses
├── train_flowdiff.py        # Phase 3: Flow+Diff training (curriculum)
├── inference.py             # Phase 4: Anomaly detection
└── README.md                # This file
```

---

## Installation

```bash
# Create conda environment
conda create -n dental-anomaly python=3.10
conda activate dental-anomaly

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install diffusers transformers accelerate
pip install pydicom albumentations scikit-learn
pip install wandb matplotlib tqdm
```

---

## Usage

### Phase 1: VAE Fine-tuning

Fine-tune the decoder (and optionally encoder) of a pretrained Stable Diffusion VAE on **normal X-rays only**.

**Config**: Edit `config.py` → `VAEConfig`
- Loss weights: `λ_pix=1.0, λ_grad=0.5, λ_HF=0.3, λ_ssim=0.1, β_KL=1e-6`
- Encoder freeze: `freeze_encoder=True` (Decoder-only training)

**Run**:
```bash
python train_vae.py
```

**Output**:
- Checkpoint: `./outputs/phase1_vae/checkpoint_best.pt`
- Logs: Train/val losses, PSNR, SSIM, KL divergence

**What to Check**:
- ✅ Reconstruction quality: Sharp edges, preserved texture
- ✅ KL divergence: Should stay small (~0.01-0.05)
- ✅ PSNR > 30 dB, SSIM > 0.95

---

### Phase 2: Latent Dataset Generation

Extract latent representations `z = μ(x)` for all normal X-rays using the fine-tuned VAE.

**Config**: Edit `config.py` → `LatentConfig`
- VAE checkpoint: `vae_checkpoint='./outputs/phase1_vae/checkpoint_best.pt'`

**Run**:
```bash
python extract_latents.py
```

**Output**:
- `./outputs/phase2_latents/latents_train.npy` (N, 4, 32, 32)
- `./outputs/phase2_latents/latents_val.npy`
- Also saves `.pkl` with file paths for tracking

**What to Check**:
- ✅ Latent statistics: Mean ≈ 0, Std ≈ 1 (approximately)
- ✅ No NaNs or extreme outliers

---

### Phase 3: Flow + Diffusion Training

Train a UNet with dual heads (diffusion + flow) on the normal latent manifold.

**Curriculum Learning**:
1. **Phase 1 (0-100k steps)**: Pure generative
   - `L_gen = λ_diff·L_diff + λ_flow·L_flow`
   - Learns normal latent distribution

2. **Phase 2 (100k-300k steps)**: Add identity constraint
   - `L_total = L_gen + λ_id·L_id`
   - `L_id = ||x - G(x)||` for normal X-rays
   - Warm-up: Steps 90k-110k, linearly ramp `λ_id` from 0 to 0.1

**Config**: Edit `config.py` → `FlowDiffTrainingConfig`
- Phase 1 steps: `phase1_steps=100_000`
- Total steps: `total_steps=300_000`
- Identity warm-up: `identity_warmup_start=90_000`, `identity_warmup_end=110_000`

**Run**:
```bash
python train_flowdiff.py
```

**Output**:
- Checkpoint: `./outputs/phase3_flowdiff/checkpoint_best.pt`
- EMA weights included for inference

**What to Check**:
- ✅ Loss convergence: `L_diff` and `L_flow` stabilize
- ✅ Identity loss: `L_id` decreases during Phase 2
- ✅ Visual samples: Generated X-rays look realistic
- ✅ Identity test: `G(x_normal) ≈ x_normal` for validation samples

---

### Phase 4: Inference & Anomaly Detection

Run the normalizer on test X-rays and generate anomaly heatmaps.

**Config**: Edit `config.py` → `InferenceConfig`
- Noise level: `noise_level=0.4` (tune between 0.3-0.5)
- Anomaly map weights: `alpha_l1=1.0, beta_grad=0.5, gamma_hf=0.3`

**Run**:
```bash
python inference.py
```

**Output**:
- Visualizations: `./outputs/phase4_inference/result_XXX.png`
- Each figure shows:
  - Row 1: Original, Normalized, Pixel Residual
  - Row 2: L1 Component, Gradient Component, Final Anomaly Map

**What to Check**:
- ✅ Normal X-rays: Anomaly map should be mostly dark (low values)
- ✅ Abnormal X-rays: Lesions/cavities highlighted in anomaly map
- ✅ Normalized version: Lesions should be "filled in" or smoothed

---

## Hyperparameter Tuning Guide

### VAE (Phase 1)

| Parameter | Default | Tuning Notes |
|-----------|---------|--------------|
| `λ_pix` | 1.0 | Baseline, don't change |
| `λ_grad` | 0.5 | ↑ if edges too blurry |
| `λ_HF` | 0.3 | ↑ if texture lost |
| `β_KL` | 1e-6 | ↑ if latent diverges, ↓ if reconstruction poor |
| `lr` | 5e-5 | Lower if training unstable |

### Flow+Diff (Phase 3)

| Parameter | Default | Tuning Notes |
|-----------|---------|--------------|
| `λ_diff` | 1.0 | Main density learning |
| `λ_flow` | 0.5 | ↑ if flow convergence slow |
| `λ_id_global` | 0.1 | ↑ if normal X-rays change too much |
| `identity_noise_level` | 0.4 | Lower (0.3) for milder normalization |
| `phase1_steps` | 100k | Can reduce to 80k if converges fast |

### Inference (Phase 4)

| Parameter | Default | Tuning Notes |
|-----------|---------|--------------|
| `noise_level` | 0.4 | **Critical**: 0.3-0.5. Lower = less change, higher = more aggressive |
| `alpha_l1` | 1.0 | Baseline |
| `beta_grad` | 0.5 | ↑ if edge anomalies important |
| `gamma_hf` | 0.3 | ↑ if texture anomalies important |

---

## Monitoring & Debugging

### Phase 1 (VAE)

**Good signs**:
- `loss_pix` decreases steadily
- `loss_kl` stays small (<0.1)
- Reconstructions preserve bone texture and tooth roots

**Bad signs**:
- `loss_kl` explodes (>1.0) → Lower `β_KL`
- Blurry reconstructions → Increase `λ_grad` or `λ_HF`
- Mode collapse (all reconstructions similar) → Lower `β_KL`

### Phase 3 (Flow+Diff)

**Good signs**:
- `loss_diff` and `loss_flow` converge smoothly
- Generated samples look like X-rays (not noise)
- Identity loss decreases in Phase 2
- `G(x_normal) ≈ x_normal` on validation

**Bad signs**:
- Loss doesn't decrease → Check learning rate, batch size
- Generated samples are noise → Train longer in Phase 1
- Identity loss increases → Lower `λ_id_global`
- All outputs identical → Increase `identity_noise_level`

### Phase 4 (Inference)

**Good signs**:
- Normal X-rays: Low anomaly scores, dark heatmap
- Abnormal X-rays: High scores at lesion locations
- Normalized X-rays look realistic

**Bad signs**:
- All anomaly maps empty → Increase `noise_level`
- All anomaly maps saturated → Decrease `noise_level`
- Artifacts in normalized X-rays → Retrain VAE or Flow+Diff

---

## Citation & References

**Flow Matching**:
- Lipman et al., "Flow Matching for Generative Modeling" (2023)

**Diffusion Models**:
- Ho et al., "Denoising Diffusion Probabilistic Models" (2020)

**VAE for Latent Diffusion**:
- Rombach et al., "High-Resolution Image Synthesis with Latent Diffusion Models" (2022)

**Anomaly Detection**:
- Schlegl et al., "f-AnoGAN: Fast unsupervised anomaly detection" (2019)

---

## License

MIT License. See LICENSE file for details.

---

## Contact

For questions or issues, please open a GitHub issue or contact [your email/info].

---

## Acknowledgments

- Pretrained VAE from [Stability AI](https://huggingface.co/stabilityai/sd-vae-ft-mse)
- Inspired by recent advances in flow matching and diffusion-based anomaly detection
