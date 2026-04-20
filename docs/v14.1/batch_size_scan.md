# V14.1 Batch Size 显存扫描

RTX 5090 32GB, seg_len=65536, latent_dim=256, 55M params

## Phase 1: 纯重建 (无判别器)

| batch_size | 峰值显存 | 剩余 | 每样本 |
|------------|---------|------|--------|
| 1 | 3,911 MB | 28,857 MB | 3.9 GB |
| 2 | 7,382 MB | 25,386 MB | 3.7 GB |
| 4 | 14,316 MB | 18,452 MB | 3.6 GB |
| 6 | 21,270 MB | 11,498 MB | 3.5 GB |
| 8 | OOM | — | — |

## Phase 2: G+D step (加判别器)

| batch_size | 峰值显存 | 剩余 | 每样本 |
|------------|---------|------|--------|
| 1 | 4,982 MB | 27,786 MB | 5.0 GB |
| 2 | 9,451 MB | 23,317 MB | 4.7 GB |
| 3 | 13,948 MB | 18,820 MB | 4.6 GB |
| 4 | 18,354 MB | 14,414 MB | 4.6 GB |
| 5 | 22,865 MB | 9,903 MB | 4.6 GB |
| 6 | OOM | — | — |

## 推荐配置

统一 `batch_size=4, grad_accum=4` → 有效 batch=16

- Phase 1: 18.4 GB headroom，安全
- Phase 2: 14.4 GB headroom，安全
- 全程不需要切换配置
