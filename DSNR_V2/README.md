# DSRN: Dual-Stream Selective Reconstruction Network

**새로운 접근 방식으로 치과 파노라마 X-ray의 병변을 정상화하는 네트워크**

## 핵심 아이디어

기존 방법들의 문제점:
- **VAE/Flow**: 전체를 latent space로 압축 → 정상 부분도 blur 발생
- **Inpainting**: 명시적 mask 필요 → Unsupervised learning 불가
- **GAN**: 학습 불안정, 정상 부분도 변형

우리의 해결책: **Dual-Stream Architecture**
- **Stream 1 (Normal)**: 정상 부분은 그대로 통과 (Identity pathway)
- **Stream 2 (Abnormal)**: 비정상 부분만 완전 재구성 (Perfect reconstruction)
- **Soft Selection**: Anomaly scorer가 자동으로 두 stream을 pixel-wise로 혼합

## 아키텍처

```
Input X-ray
    │
    ├─────────────────┐
    │                 │
[Feature            │
Extractor]          │
    │                 │
[Anomaly            │
 Scorer] ────┐       │
    │        │       │
┌───┴────┐   │       │
│        │   │       │
[Normal  [Abnormal   │
Stream]  Stream]     │
│        │   │       │
└───┬────┘   │       │
    │        │       │
[Soft Select]◄┘      │
    │                │
[Fusion]◄────────────┘
    │
Output (Normalized)
```

## 주요 특징

### 1. Self-Supervised Learning
- **정상 데이터만 사용**: 실제 병변 데이터 불필요
- **Synthetic Lesion**: 다양한 형태의 가짜 병변 생성
  - Circular, Elliptical, Irregular, Multilocular, Root resorption 등
- **학습 목표**: 가짜 병변 제거 → 원본 정상 복원

### 2. Dual-Stream Design
- **Normal Stream**:
  - Alpha = 0.05의 매우 작은 residual connection
  - 정상 부분은 거의 그대로 통과

- **Abnormal Stream**:
  - Texture Synthesizer: Normal prototypes를 참고하여 정상 texture 생성
  - Perfect Reconstruction: 병변을 완전히 정상 조직으로 변환

### 3. Spatial Anomaly Scorer
- **Distance-based**: Normal prototypes까지의 거리 계산
- **Learned**: CNN 기반 anomaly scoring
- **Multi-scale**: 3가지 scale에서 aggregation
- **Output**: Pixel-wise anomaly map [0, 1]

### 4. Soft Fusion
- Anomaly map에 따라 두 stream을 부드럽게 혼합
- Score = 0 (정상) → 100% Normal stream
- Score = 1 (병변) → 100% Abnormal stream

## 디렉토리 구조

```
DSRN/
├── config.py                    # Configuration
├── models/
│   ├── __init__.py
│   ├── feature_extractor.py    # Multi-scale feature extraction
│   ├── anomaly_scorer.py       # Spatial anomaly scoring
│   ├── normal_stream.py        # Identity pathway
│   ├── abnormal_stream.py      # Perfect reconstruction
│   ├── fusion.py               # Soft fusion
│   └── dsrn.py                 # Main model
├── data/
│   ├── __init__.py
│   ├── dataset.py              # Dataset with synthetic lesions
│   └── lesion_synthesizer.py  # Diverse lesion synthesizer
├── train.py                     # Training script
├── test.py                      # Testing script
├── utils/
│   ├── __init__.py
│   └── visualization.py        # Visualization tools
└── README.md
```

## 설치

```bash
# Dependencies
pip install torch torchvision
pip install opencv-python scipy numpy
pip install matplotlib tensorboard
pip install tqdm
```

## 사용법

### 1. Configuration

`config.py`에서 설정 변경:

```python
@dataclass
class DSRNConfig:
    # Data
    data_root: str = "/dataset/panorama_total"  # 정상 X-ray 경로
    image_size: int = 256

    # Training
    batch_size: int = 8
    num_epochs: int = 100
    learning_rate: float = 1e-4

    # Loss weights
    lambda_recon: float = 1.0
    lambda_anomaly: float = 0.5
    lambda_identity: float = 0.3
    lambda_perceptual: float = 0.2
```

### 2. Training

```bash
# Train DSRN
python train.py

# Checkpoint saved to: ./checkpoints/
# Logs saved to: ./logs/
# Visualizations saved to: ./outputs/
```

Training은 두 단계로 진행:
- **Phase 1**: Normal Prototype Learning (정상 feature 저장)
- **Phase 2**: Self-Supervised Reconstruction (synthetic lesion 제거 학습)

### 3. Testing

#### 전체 데이터셋 테스트
```bash
python test.py --mode dataset --checkpoint ./checkpoints/best.pth --num_samples 10
```

#### 단일 이미지 테스트
```bash
python test.py --mode single --checkpoint ./checkpoints/best.pth --input /path/to/image.png
```

#### 디렉토리 전체 테스트
```bash
python test.py --mode directory --checkpoint ./checkpoints/best.pth --input /path/to/images/ --output_dir ./results/
```

### 4. Visualization

Training 중 자동으로 시각화:
- **Original**: 원본 정상 이미지
- **Input**: Synthetic lesion이 추가된 입력
- **GT Mask**: Ground truth lesion mask
- **Anomaly Map**: 예측된 anomaly map
- **Fusion Weights**: 두 stream의 혼합 비율
- **Reconstruction**: 최종 복원 결과
- **Residual**: |Input - Reconstruction|

## 학습 과정

### Phase 1: Normal Prototype Learning (자동)

```python
# 정상 이미지로 prototype 학습
for x_normal in normal_dataset:
    features = feature_extractor(x_normal)
    f4 = features['f4']  # [B, 512, 32, 32]

    # Add to memory bank
    prototypes.append(f4.mean(dim=[2,3]))  # [B, 512]

# Select top 1000 diverse prototypes
normal_prototypes = select_diverse(prototypes, k=1000)
```

### Phase 2: Self-Supervised Training (자동)

```python
for x_normal in normal_dataset:
    # 1. Add synthetic lesion
    x_syn, mask_syn = add_synthetic_lesion(x_normal)

    # 2. Forward
    x_recon, anomaly_map, _ = model(x_syn)

    # 3. Losses
    loss_recon = MSE(x_recon, x_normal)      # Perfect reconstruction
    loss_anomaly = BCE(anomaly_map, mask_syn) # Correct detection
    loss_identity = MSE(x_recon * (1-mask_syn),
                       x_syn * (1-mask_syn))  # Preserve normal

    loss = loss_recon + loss_anomaly + loss_identity
```

## Loss Functions

1. **Reconstruction Loss**: MSE(x_recon, x_normal)
   - 재구성 결과가 원본 정상과 같아야 함

2. **Anomaly Detection Loss**: BCE(anomaly_map, mask_lesion)
   - 병변 위치를 정확히 찾아야 함

3. **Identity Preservation Loss**: MSE(x_recon * (1-mask), x_input * (1-mask))
   - 정상 부분은 그대로 유지

4. **Perceptual Loss**: L1(gradient(x_recon), gradient(x_normal))
   - Edge와 texture 보존

## 성능 평가

```python
# Reconstruction quality
reconstruction_mse = MSE(x_recon, x_normal)

# Anomaly detection (IoU)
anomaly_iou = IoU(anomaly_map > 0.5, mask_lesion)
```

## 차별점

| 측면 | 기존 방법 | DSRN |
|------|----------|------|
| **학습** | 정상 + 비정상 또는 VAE | 정상만 |
| **정상 보존** | Blur 발생 | 거의 완벽 |
| **병변 제거** | 부분적 | 완전 재구성 |
| **속도** | 느림 (iterative) | 빠름 (one-shot) |
| **해석성** | 어려움 | 명확 (anomaly map) |

## Citation

```bibtex
@article{dsrn2025,
  title={DSRN: Dual-Stream Selective Reconstruction Network for Dental Lesion Normalization},
  author={Your Name},
  journal={arXiv preprint},
  year={2025}
}
```

## License

MIT License

## Contact

For questions or issues, please open an issue on GitHub.

---

**핵심**: 정상 데이터만으로 학습하여, 실제 병변을 자동으로 탐지하고 완전히 정상으로 복원!
