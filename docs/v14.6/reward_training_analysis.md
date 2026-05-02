# V14.6 新型训练方法分析：Reward-Guided Decoder Training

## 背景

V14.4 best SNR = 11.36 dB，瓶颈在高频：

| 频段 | SNR |
|------|-----|
| 0-500 Hz | 25.4 dB |
| 500-1000 Hz | 26.8 dB |
| 1000-2000 Hz | 12.3 dB |
| 2000-4000 Hz | 6.1 dB |
| 4000-6000 Hz | 0.0 dB |
| 6000-8000 Hz | 4.1 dB |
| 8000-11025 Hz | -0.1 dB |

传统 GAN 判别器两次失败（V14.4 phase2, V14.6），原因是 121M 的 AE 碾压 8M 的判别器。

## 关键发现：VAE 退化为 AE

```
Latent std: mean=0.0006, range=[0.0001, 0.0258]
```

std 几乎为零，VAE 的随机采样完全不起作用。这意味着：
- 不能靠 VAE 自身的 reparameterization 产生有意义的多样性
- 要做多次采样，必须手动加噪声

Decoder 对 latent 扰动的敏感度：

| noise_scale | SNR | drop |
|-------------|-----|------|
| 0.001 | 13.21 | 0.00 |
| 0.01 | 13.21 | 0.00 |
| 0.05 | 13.12 | 0.09 |
| 0.10 | 12.87 | 0.34 |
| 0.20 | 12.03 | 1.18 |
| 0.50 | 8.11 | 5.10 |
| 1.00 | 2.08 | 11.13 |

noise_scale=0.1 时 SNR 下降 0.34 dB，各频段均匀下降 ~0.5 dB。
这是 GRPO 采样的合理工作区间。

---

## 方案 A：纯 GRPO (Policy Gradient)

### 原理
```
对每个 audio x:
  1. encoder 得到 z_mean
  2. 采样 K 个 z_k = z_mean + σ * ε_k,  ε_k ~ N(0,I)
  3. decoder 得到 K 个 x_hat_k
  4. 用不可微指标 R(x, x_hat_k) 给每个候选打分
  5. GRPO 更新: ∇θ ≈ Σ_k (R_k - R_baseline) * ∇θ log p(z_k | z_mean, σ)
```

### 优势
- reward 可以用任意不可微指标（band SNR、PESQ、DNSMOS）
- 不需要训练额外网络
- 天然支持频段加权

### 致命问题
1. **方差极大**：128 维连续空间的 policy gradient 方差远大于离散 token。
   LLM GRPO 的 action space 是 ~50K 离散 token，每个 token 的 log_prob 有明确意义。
   这里是 128 维连续高斯，log_prob = -0.5 * ||ε||² - 128/2 * log(2π)，
   梯度 ∇θ log p = ∇θ(-0.5 * ||(z-μ)/σ||²)，和 reward 的乘积方差极大。

2. **采样效率低**：noise_scale=0.1 时 SNR 只变化 0.34 dB，
   K=4 个样本之间的 reward 差异极小，信噪比太低。
   要得到有意义的梯度估计，可能需要 K=32-64，每 step 32-64 次 decoder forward，
   121M 参数的 decoder 在 5090 上一次 forward ~15ms，64 次 = 1s/step，太慢。

3. **梯度只流过 decoder**：encoder 的 z_mean 是 detach 的（采样在 z_mean 上加噪声），
   encoder 完全不更新。而高频问题可能部分在 encoder。

### 结论：不推荐。方差问题在连续空间几乎无解。

---

## 方案 B：可微 Reward Regression (推荐)

### 原理
```
训练一个轻量评分网络 S(real_stft, recon_stft) → per-band scores
用真实的 band SNR 作为回归目标监督 S
然后用 S 的输出作为可微 loss 反传到 decoder
```

### 具体设计

**评分网络 S (Spectral Quality Estimator, SQE)**：
```
输入: real_stft [B, F, T], recon_stft [B, F, T]
  → 拼接差值特征: [real_mag, recon_mag, |real-recon|_mag]  → [B, 3, F, T]
  → 几层 2D Conv (类似判别器但输出不是 logit)
  → per-band pooling (5 个频段)
  → 输出: [B, 5] 每个频段的质量分数
```

**训练流程 (交替)**：
```
Phase A - 更新 SQE:
  1. encoder(x) → z, decoder(z) → x_hat  (detach, 不更新 AE)
  2. 计算真实 band SNR: R_true = [snr_band0, ..., snr_band4]
  3. SQE(stft(x), stft(x_hat)) → R_pred
  4. L_sqe = MSE(R_pred, R_true)
  5. 更新 SQE 参数

Phase B - 更新 AE:
  1. encoder(x) → z, decoder(z) → x_hat
  2. SQE(stft(x), stft(x_hat)) → R_pred  (SQE frozen)
  3. L_reward = -Σ w_band * R_pred_band  (最大化质量分数)
  4. L_total = L_recon + λ * L_reward
  5. 更新 AE 参数
```

### 优势
1. **不会崩溃**：SQE 的目标是回归真实 SNR，不是二分类。
   即使 AE 变好了，SQE 只需要给更高的分数，不存在饱和问题。

2. **频段感知**：per-band 输出天然支持高频加权。
   可以给 4k-11kHz 的 band 更大的 w_band。

3. **可微**：梯度直接从 SQE 反传到 decoder，不需要 policy gradient。

4. **轻量**：SQE 只需要几层 2D Conv，~1-2M 参数，计算开销很小。

5. **自适应**：SQE 学到的是"感知距离"而不是固定公式。
   如果某个频段的 STFT L1 loss 和真实 SNR 不成正比（非线性关系），
   SQE 可以学到这个非线性映射。

### 风险
1. **SQE 过拟合**：如果 SQE 太小或训练数据太少，可能学到 shortcut。
   缓解：SQE 每 step 都用当前 AE 的输出重新计算 target，数据分布持续变化。

2. **梯度方向可能不准**：SQE 的梯度 ∂R/∂x_hat 不一定指向真正提升 SNR 的方向。
   缓解：SQE 和 recon loss 联合使用，recon loss 提供稳定基线，SQE 提供额外推力。

3. **reward hacking**：AE 可能找到让 SQE 给高分但实际质量没提升的 trick。
   缓解：SQE 持续用真实 SNR 重新训练，AE 的 trick 会被 SQE 学到并修正。

---

## 方案 C：对比排序 (Contrastive Ranking)

### 原理
```
对每个 audio x:
  1. 生成多个不同质量的重建: x_hat_good (小噪声), x_hat_bad (大噪声)
  2. 评分网络学习: S(x, x_hat_good) > S(x, x_hat_bad)
  3. 用 ranking loss (margin loss) 训练
```

### 问题
- 本质上还是在训练一个判别器，只是用 ranking 代替了二分类
- 当 AE 足够好时，x_hat_good 和 x_hat_bad 的差异极小（0.34 dB），
  ranking 信号太弱
- 不如方案 B 直接回归真实 SNR 来得直接

---

## 最终推荐：方案 B (可微 Reward Regression)

### 实现计划

1. **SpectralQualityEstimator (SQE)** — 新模块
   - 输入: (real_mag, recon_mag, diff_mag) 3 通道 STFT magnitude
   - 网络: 4 层 2D Conv + per-band avg pool
   - 输出: [B, N_bands] 质量分数
   - 参数量: ~1.5M

2. **训练集成**
   - 每个 step: 先更新 SQE (用真实 band SNR 做 target)，再更新 AE
   - SQE loss: MSE regression
   - AE 额外 loss: -Σ w_band * SQE_score
   - 权重: sqe_reward_weight 可调

3. **频段加权**
   - 低频 (0-1.5kHz): w=1.0 (已经很好)
   - 中频 (1.5-4kHz): w=2.0
   - 高频 (4-11kHz): w=4.0 (最弱，最需要推力)

### 和现有 loss 的关系
- L1/L2 time-domain: 提供波形级基线
- Multi-resolution STFT: 提供频谱级基线
- Multi-band STFT (高频加权): 提供频段级基线
- SQE reward: 提供可学习的、自适应的额外推力

SQE 不替代任何现有 loss，而是作为额外的 "learned loss" 叠加。
