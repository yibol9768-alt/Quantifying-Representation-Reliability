# Experiment Results

## CIFAR-10 CLIP Single Model

### Configuration
- Dataset: CIFAR-10 (10 classes)
- Model: CLIP (ViT-B/32) + MLP classifier
- Feature dim: 512
- Trainable parameters: 268,810
- Training: 10 epochs, batch size 256, lr 1e-3

### Results

| Epoch | Loss | Val Acc | Test Acc |
|-------|------|---------|----------|
| 1 | 0.2336 | 95.09% | 94.43% |
| 2 | 0.1482 | 95.29% | 94.55% |
| 3 | 0.1284 | 95.30% | 94.70% |
| 4 | 0.1146 | 95.43% | 94.97% |
| 5 | 0.1037 | 95.25% | **95.00%** |
| 6 | 0.0921 | 95.39% | 94.90% |
| 7 | 0.0808 | **95.66%** | **95.00%** |
| 8 | 0.0742 | 95.61% | 94.82% |
| 9 | 0.0666 | 95.55% | 94.67% |
| 10 | 0.0604 | 95.37% | 94.91% |

### Summary
- **Best Test Accuracy: 95.00%**
- Best Validation Accuracy: 95.66%
- Model checkpoint: `outputs/checkpoints/cifar10_clip_single.pth`
- Features: `features/cifar10_clip_train.pt`, `features/cifar10_clip_test.pt`

---

## Pending Experiments

- [ ] CLIP + DINO fusion on CIFAR-10
- [ ] CLIP + DINO + MAE fusion on CIFAR-10
- [ ] All models on Stanford Cars
- [ ] All models on other datasets (CIFAR-100, Flowers-102, Pets, Food-101)
