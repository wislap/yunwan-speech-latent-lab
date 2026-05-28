# 实现计划: Generation-Friendly Latent

## 概述

将设计文档中的多阶段训练方案转化为可执行的编码任务。基于现有 V16.3 WavVAE 架构（不修改模型），新增 Masked Latent Predictor、Block Masker、Temporal Smoothness Loss、Phase Scheduler 和合成数据 Dataset，并修改训练脚本支持多阶段训练。

## Tasks

- [ ] 1. 数据准备: CosyVoice3 合成脚本与 SynthesizedAudioDataset
  - [ ] 1.1 创建单说话人 CosyVoice3 合成数据脚本
    - 新建 `autoencoder/scripts/synthesize_single_speaker.py`
    - 基于现有 `scripts/build_cosyvoice3_cross_speaker_dataset.py` 适配为单说话人版本
    - 输出 JSONL 格式: audio_path, text, phonemes, durations, speaker_id, sample_rate
    - 支持批量合成并保存 phoneme alignment 信息
    - _Requirements: 1.1_

  - [ ] 1.2 实现 SynthesizedAudioDataset
    - 新建 `autoencoder/datasets.py`
    - 实现 `SynthesizedAudioDataset(Dataset)` 类，加载 JSONL 数据
    - 实现低通滤波 (scipy butter filter, 可配置 cutoff_hz)
    - 返回 dict: `{'audio': [1, T], 'phoneme_ids': [T_ph], 'phoneme_durations': [T_ph]}`
    - 确保 audio 长度为 stride (256) 的整数倍
    - 确保 `sum(phoneme_durations) == audio_length / stride`
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

  - [ ]* 1.3 为 SynthesizedAudioDataset 编写属性测试
    - **Property 1: Low-pass filter energy attenuation**
    - **Property 2: Data pipeline output completeness**
    - **Validates: Requirements 1.2, 1.3, 1.5**

- [ ] 2. 核心组件: BlockMaskGenerator
  - [ ] 2.1 实现 BlockMaskGenerator
    - 新建 `autoencoder/masking.py`
    - 实现 `BlockMaskGenerator` 类: mask_ratio, min_block_frames, max_block_frames
    - `generate_mask(seq_len, batch_size) -> [B, T] bool tensor` (True=可见, False=遮蔽)
    - 保证遮蔽比例 >= 50%，每个序列至少有部分可见帧
    - 每个连续遮蔽块长度在 [min_block_frames, max_block_frames] 范围内
    - _Requirements: 3.1, 3.2, 3.3, 3.4_

  - [ ]* 2.2 为 BlockMaskGenerator 编写属性测试
    - **Property 5: Block masker invariants**
    - **Property 6: Block length bounds**
    - **Validates: Requirements 3.1, 3.2, 3.3, 3.4**

- [ ] 3. 核心组件: MaskedLatentPredictor
  - [ ] 3.1 实现 MaskedLatentPredictor
    - 新建 `autoencoder/models/masked_predictor.py`
    - 实现 Transformer-based predictor: latent_dim=384, hidden_dim=512, n_heads=8, n_layers=6
    - 音素 embedding + duration 展开到帧级别
    - 位置编码 (sinusoidal 或 learned)
    - 输入: z_visible, visible_mask, phoneme_ids, phoneme_durations
    - 输出: [B, T_masked, 384] 预测的被遮蔽 latent frames
    - 确保梯度可从 predictor loss 回传到 encoder
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

  - [ ]* 3.2 为 MaskedLatentPredictor 编写属性测试
    - **Property 3: Predictor output shape consistency**
    - **Property 4: Gradient flow from predictor to encoder**
    - **Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5**

- [ ] 4. 核心组件: TemporalSmoothnessLoss 与 PhaseScheduler
  - [ ] 4.1 实现 TemporalSmoothnessLoss
    - 新建 `autoencoder/losses_gen_friendly.py`
    - 计算 `mean(||z[:,:,t] - z[:,:,t-1]||²)`
    - 权重线性调度: initial_weight → max_weight (按 step/total_steps)
    - 常数序列返回 0，任意输入返回非负值
    - _Requirements: 4.1, 4.2, 4.3, 4.4_

  - [ ] 4.2 实现 PhaseScheduler
    - 在同一文件 `autoencoder/losses_gen_friendly.py` 中实现
    - 定义 `PhaseConfig` dataclass 和 `PhaseScheduler` 类
    - Phase 1: adv_enable=False, l1_time_weight=0, lpf_cutoff=4000
    - Phase 2: 逐步提高 lpf_cutoff, 增强 stft_weight
    - Phase 3: adv_enable=True, l1_time_weight>0, lpf_cutoff=None
    - `get_current_config(epoch) -> PhaseConfig`
    - `get_lpf_cutoff(epoch) -> float | None`
    - _Requirements: 5.1, 5.2, 5.3, 5.4_

  - [ ]* 4.3 为 TemporalSmoothnessLoss 和 PhaseScheduler 编写属性测试
    - **Property 7: Temporal smoothness loss correctness**
    - **Property 8: Smoothness weight linear scheduling**
    - **Property 9: Phase scheduler config invariants**
    - **Property 10: Phase 2 monotonic frequency increase**
    - **Validates: Requirements 4.1, 4.2, 4.3, 4.4, 5.1, 5.2, 5.3, 5.4**

- [ ] 5. Checkpoint - 确保所有组件单元测试通过
  - 确保所有测试通过，如有问题请询问用户。

- [ ] 6. 训练脚本: Phase 1 训练循环集成
  - [ ] 6.1 创建实验配置 YAML
    - 新建 `autoencoder/conf/experiment/v19_gen_friendly_phase1.yaml`
    - 配置: latent_dim=384, strides=[2,4,4,8], segment_length=65536
    - Phase 1 loss 权重: mel_weight, stft_mag_weight, predictor_weight, smooth_weight
    - 禁用判别器和时域 L1
    - Predictor 配置: hidden_dim=512, n_heads=8, n_layers=6
    - Masking 配置: mask_ratio=0.6, min_block_frames=400, max_block_frames=800
    - 数据路径指向合成数据 JSONL
    - _Requirements: 5.1, 5.2, 6.1, 6.2, 6.3_

  - [ ] 6.2 实现 Phase 1 训练脚本
    - 新建 `autoencoder/train_gen_friendly.py`
    - 从 V16.3 checkpoint 加载 Encoder/Decoder 权重
    - 随机初始化 MaskedLatentPredictor
    - 训练循环: encode → decode (mel+stft loss) + mask → predict (MSE loss) + smooth loss
    - 三个优化器组: encoder_lr, decoder_lr, predictor_lr
    - 梯度累积 (grad_accum_steps)
    - 定期评估: effective rank, predictor loss, mel loss
    - Checkpoint 保存/加载 (模型 + 优化器 + phase + epoch + global_step)
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 9.1, 9.2, 9.3, 9.4_

  - [ ] 6.3 实现训练稳定性保护逻辑
    - 在训练循环中添加 collapse 检测
    - Predictor collapse: effective_rank < 10 且 predictor_loss 极低 → 降低 predictor_weight
    - Representation collapse: latent 方差趋近零 → 检查重建 loss 梯度
    - 记录检测事件到日志
    - _Requirements: 7.1, 7.2, 7.3, 7.4_

- [ ] 7. 评估与诊断工具
  - [ ] 7.1 实现评估指标计算模块
    - 新建 `autoencoder/scripts/gen_friendly_diagnostics.py`
    - 计算 effective rank (95% 和 99% 方差解释比)
    - 计算 phoneme utility (有/无音素条件的 predictor loss 差值)
    - 计算帧间差分 L2 (时序平滑度)
    - 计算 mel loss 和 STFT magnitude loss
    - 支持从 checkpoint 加载模型并在 eval 数据上运行
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

  - [ ]* 7.2 为 effective rank 计算编写属性测试
    - **Property 13: Effective rank computation**
    - **Validates: Requirement 8.1**

- [ ] 8. Checkpoint - Phase 1 Smoke Test
  - 确保 Phase 1 训练脚本可以成功运行 1 个 epoch (小数据集)，loss 下降，梯度正常流动。如有问题请询问用户。

- [ ] 9. Phase 2 与 Phase 3 训练支持
  - [ ] 9.1 扩展训练脚本支持 Phase 2
    - 在 `train_gen_friendly.py` 中添加 Phase 2 逻辑
    - 从 Phase 1 checkpoint 继续训练
    - 逐步提高 LPF cutoff (4kHz → 8kHz → 11kHz)
    - 增强 STFT loss 权重
    - 继续 Masked Predictor 训练
    - 监控 effective rank 跳升并自动调整
    - _Requirements: 1.3, 5.3, 5.5, 7.3_

  - [ ] 9.2 扩展训练脚本支持 Phase 3
    - 添加 Phase 3 逻辑: 全频数据 + L1/L2 时域 loss + 判别器
    - 复用现有 `V14Discriminator` 或 `MultiScaleSubBandDiscriminator`
    - 从 Phase 2 checkpoint 继续训练
    - _Requirements: 1.4, 5.4, 5.5_

  - [ ]* 9.3 为 checkpoint save/load 编写属性测试
    - **Property 14: Checkpoint save/load round-trip**
    - **Validates: Requirements 9.3, 9.4**

- [ ] 10. Final Checkpoint - 确保所有测试通过
  - 确保所有测试通过，如有问题请询问用户。

## Notes

- 标记 `*` 的任务为可选，可跳过以加速 MVP
- 每个任务引用具体需求编号以确保可追溯性
- Checkpoint 任务确保增量验证
- 属性测试验证设计文档中定义的 Correctness Properties
- 单元测试验证具体示例和边界情况
- 训练脚本基于现有 `autoencoder/train.py` 模式，但独立为新文件以避免污染现有代码
