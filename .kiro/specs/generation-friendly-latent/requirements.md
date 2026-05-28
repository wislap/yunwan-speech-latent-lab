# 需求文档: Generation-Friendly Latent

## 简介

本文档定义了 Wav-VAE 多阶段训练流水线的需求，目标是通过信息论约束（数据清洁度 + 掩码潜表示预测）产出对下游 FM-DiT TTS 友好的连续潜表示。系统复用 V16.3 架构（256x stride, 384d latent），仅改变训练方案、数据和损失函数。

## 术语表

- **Encoder**: V16.3 Wav-VAE 编码器，将波形映射为 384 维潜表示序列
- **Decoder**: V16.3 Wav-VAE 解码器，将潜表示序列重建为波形
- **Masked_Latent_Predictor**: Transformer 模型，接收音素和可见 latent frames，预测被遮蔽的 latent frames
- **Block_Masker**: 生成连续大块遮蔽 mask 的组件
- **Phase_Scheduler**: 管理三阶段训练超参数调度的组件
- **Data_Pipeline**: 数据加载与预处理流水线，包含低通滤波
- **Temporal_Smoothness_Loss**: 帧间差分 L2 时序平滑正则化器
- **Effective_Rank**: 潜表示奇异值分布的有效维度度量（基于 95% 方差解释比）
- **Phoneme_Utility**: 有音素条件与无音素条件下 predictor loss 的差值
- **LPF**: 低通滤波器 (Low-Pass Filter)
- **Phase_1**: 低频 mel-only 训练阶段
- **Phase_2**: 频率扩展训练阶段
- **Phase_3**: 全质量训练阶段（含判别器）

## 需求

### 需求 1: 数据流水线

**用户故事:** 作为训练系统，我需要加载 CosyVoice3 合成的干净单说话人数据并施加低通滤波，以便控制输入信息上界。

#### 验收标准

1. THE Data_Pipeline SHALL 加载 CosyVoice3 合成的单说话人语音数据及其对应的音素对齐信息
2. WHEN Phase_1 训练时, THE Data_Pipeline SHALL 对音频施加 4kHz 截止频率的低通滤波
3. WHEN Phase_2 训练时, THE Data_Pipeline SHALL 根据 Phase_Scheduler 提供的当前截止频率施加低通滤波
4. WHEN Phase_3 训练时, THE Data_Pipeline SHALL 提供未经低通滤波的全频音频
5. THE Data_Pipeline SHALL 为每条音频返回波形张量、音素 ID 序列和音素帧级 duration

### 需求 2: 掩码潜表示预测器

**用户故事:** 作为训练系统，我需要一个 Transformer 预测器来预测被遮蔽的 latent frames，以便通过梯度回传迫使 encoder 编码可预测的低条件熵信息。

#### 验收标准

1. THE Masked_Latent_Predictor SHALL 接收可见 latent frames、可见 mask、音素 ID 和音素 duration 作为输入
2. THE Masked_Latent_Predictor SHALL 输出与被遮蔽位置数量一致的 384 维预测向量
3. WHEN 前向传播时, THE Masked_Latent_Predictor SHALL 将音素序列通过 duration 展开到帧级别作为位置条件
4. WHEN 计算预测损失时, THE Masked_Latent_Predictor SHALL 使用 MSE 损失比较预测值与目标 latent frames
5. THE Masked_Latent_Predictor SHALL 允许梯度从预测损失回传到 Encoder 参数（联合训练）

### 需求 3: 块遮蔽策略

**用户故事:** 作为训练系统，我需要生成大块连续遮蔽，以便迫使预测器依赖全局语义（音素）而非局部插值。

#### 验收标准

1. THE Block_Masker SHALL 生成遮蔽比例不低于 50% 的 mask
2. THE Block_Masker SHALL 生成连续块遮蔽，每个块长度在配置的最小值和最大值之间
3. THE Block_Masker SHALL 确保每个序列中至少存在部分可见帧供预测器参考
4. WHEN 生成 mask 时, THE Block_Masker SHALL 返回形状为 [B, T] 的布尔张量，True 表示可见，False 表示被遮蔽

### 需求 4: 时序平滑正则化

**用户故事:** 作为训练系统，我需要对 latent 施加时序平滑约束，以便产出时序连贯的潜表示。

#### 验收标准

1. THE Temporal_Smoothness_Loss SHALL 计算相邻帧差分的 L2 范数均值作为损失
2. WHEN 输入为常数序列时, THE Temporal_Smoothness_Loss SHALL 返回零损失
3. THE Temporal_Smoothness_Loss SHALL 根据当前训练步数从初始权重线性增长到最大权重
4. THE Temporal_Smoothness_Loss SHALL 对任意输入返回非负损失值

### 需求 5: 三阶段训练调度

**用户故事:** 作为训练系统，我需要管理三阶段训练的超参数切换，以便按计划从低频重建逐步过渡到全质量重建。

#### 验收标准

1. THE Phase_Scheduler SHALL 根据当前 epoch 返回对应阶段的完整配置（loss 权重、LPF cutoff、学习率等）
2. WHILE Phase_1 活跃时, THE Phase_Scheduler SHALL 设置判别器为禁用状态且时域 L1 权重为零
3. WHILE Phase_2 活跃时, THE Phase_Scheduler SHALL 逐步提高 LPF 截止频率并增强 STFT loss 权重
4. WHILE Phase_3 活跃时, THE Phase_Scheduler SHALL 启用判别器和时域 L1/L2 损失
5. WHEN 阶段切换发生时, THE Phase_Scheduler SHALL 保持模型权重连续（从上一阶段 checkpoint 继续）

### 需求 6: Phase 1 训练循环

**用户故事:** 作为训练系统，我需要执行 Phase 1 核心训练循环，以便建立低条件熵的潜空间结构。

#### 验收标准

1. WHEN Phase_1 训练时, THE Encoder SHALL 将低通滤波后的音频编码为 [B, 384, T/256] 的潜表示
2. WHEN Phase_1 训练时, THE Decoder SHALL 仅使用 Mel loss 和 Multi-Res STFT magnitude loss 进行重建（无时域 L1/L2，无相位 loss）
3. WHEN Phase_1 训练时, THE 训练循环 SHALL 同时计算重建损失、预测器损失和时序平滑损失的加权和
4. WHEN Phase_1 训练时, THE 训练循环 SHALL 同时更新 Encoder、Decoder 和 Masked_Latent_Predictor 的参数

### 需求 7: 训练稳定性保护

**用户故事:** 作为训练系统，我需要检测并应对训练异常，以便防止表示坍缩或梯度冲突导致训练失败。

#### 验收标准

1. IF Effective_Rank 降至 10 以下且 predictor loss 极低, THEN THE 训练系统 SHALL 判定为 predictor collapse 并降低 predictor_weight
2. IF latent 方差趋近于零, THEN THE 训练系统 SHALL 判定为 representation collapse 并确认重建 loss 梯度正常流动
3. IF Phase_2 频率扩展后 Effective_Rank 突然跳升超过阈值, THEN THE 训练系统 SHALL 放慢 cutoff 提升速度并增强 predictor_weight
4. IF 训练 loss 出现持续震荡, THEN THE 训练系统 SHALL 降低 predictor_weight 或对 encoder 使用更小学习率

### 需求 8: 评估指标计算

**用户故事:** 作为研究者，我需要定期评估潜表示质量和训练进展，以便判断训练是否朝正确方向发展。

#### 验收标准

1. THE 评估系统 SHALL 计算潜表示的 Effective_Rank（基于 95% 和 99% 方差解释比）
2. THE 评估系统 SHALL 分别计算有音素条件和无音素条件下的 predictor loss，并报告 Phoneme_Utility
3. THE 评估系统 SHALL 计算 mel loss 和 Multi-Res STFT magnitude loss 作为重建质量指标
4. THE 评估系统 SHALL 计算帧间差分 L2 范数作为时序平滑度指标
5. WHEN Phase_1 训练目标达成时, THE 评估系统 SHALL 报告 Effective_Rank 在 30-60 范围内（相比 V16.3 的 234）

### 需求 9: 模型初始化与 Checkpoint 管理

**用户故事:** 作为训练系统，我需要正确加载预训练权重和管理训练状态，以便支持多阶段训练的断点续训。

#### 验收标准

1. WHEN 训练开始时, THE 训练系统 SHALL 从 V16.3 预训练 checkpoint 加载 Encoder 和 Decoder 权重
2. WHEN 训练开始时, THE 训练系统 SHALL 随机初始化 Masked_Latent_Predictor 权重
3. WHEN 阶段切换时, THE 训练系统 SHALL 保存完整训练状态（模型权重、优化器状态、当前阶段、epoch、global_step）
4. WHEN 训练中断后恢复时, THE 训练系统 SHALL 从最近的 checkpoint 恢复所有训练状态并继续训练
