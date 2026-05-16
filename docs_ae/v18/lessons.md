# V18 Lessons: 对 V16.3 架构的一次压力测试

Date: 2026-05-16

## 立场

V18 不是失败的产品实验，而是对 V16.3 架构和 bottleneck 设计的**侧面探针**。
跑这一轮的目的是看：当我们在 V16.3 latent 上叠加一个外部监督信号（CTC
phoneme），现有的 bottleneck 和 loss recipe 是否还稳定。结果给出了几条
对后续工作（FM-DiT 端的真正问题）有用的判断。

不再继续 V18 ablation。下面是这次实验真正学到的东西，剥离掉具体的
phoneme 任务和数字。

## 一、V16.3 的 range-penalty bottleneck 是"recon-only friendly"的

观察：concat 后 384d latent 上套 `reg_weight=1e-4 / margin_mean=6 /
std_min=0.05 / std_max=3`，叠加 CTC 梯度后 rank95 从 V16.3 的 234
塌到 **10**（z_phoneme 用 8/128，z_residual 用 37/256）。

判断：

- range penalty 约束的是 latent 的"能量分布"，不是"使用维度数"
- 在纯 recon 下，decoder 梯度本身倾向于撑开维度，penalty 只起边界作用
- 一旦加入额外监督梯度（CTC 这种 token-level 信号），encoder 有强烈
  动机用低秩捷径同时满足两个目标，penalty 不仅不抗塌缩，还**鼓励**塌缩
  （因为低秩天然容易满足 mean/std 边界）
- 结论：V16.3 bottleneck 在多目标训练里需要重新设计。任何未来在 V16.3
  latent 上加新 loss 的方案，都要先想清楚 bottleneck 是不是还合适

可移植教训：**bottleneck 的稳态依赖训练目标**。换目标必须重新评估
bottleneck，不能默认沿用。

## 二、Latent split 架构本身工作

观察：

- z_residual 置零导致 SNR 下降 16 dB（说明 residual 真的承载信息）
- z_residual speaker probe 0.99（说明 speaker 信息确实落在 residual）
- 全 384d 在 frozen V16.3 decoder 上重建仍达 16 dB（说明拼接接口可用）

判断：

- "phoneme stream + residual stream + concat"这种最朴素的 split，在
  Wav-VAE 上是可行的架构，不需要更复杂的 fusion
- decoder 对 latent 的 channel 顺序不敏感，从 V16.3 直接热启动 OK
- 这条路线如果以后要重启（带更好的 loss 设计），底层架构不用动

可移植教训：**架构问题和 loss 问题要分清**。V18 真正的失败点不是
encoder 拆两条流，而是叠加在拼接结果上的 bottleneck 和 CTC 互相扭曲。

## 三、Frame-level CTC 在 AE 联合训练里很难驯服

观察：

- 训练时 non-blank fraction 14-19%，验证时只有 4.7%
- CTC loss 从 27 降到 ~3，看起来在收敛
- 但 z_phoneme speaker probe 75%，说明 phoneme channel 同时在编码
  非语音内容

判断：

- CTC 的 token-level 监督和 frame-level latent 之间存在 **reduction
  gap**：CTC 允许大量 blank，模型可以用极少帧承担分类，剩下的帧自由
  学其它信息（包括 speaker）
- 这种自由度让 CTC 无法天然地驱动 channel-level invariance
- 想用 CTC 强制语义解耦，必须配合 GRL / MI / cluster loss 等显式
  约束，**单一 CTC 不够**

可移植教训：**reduction-style 监督（CTC、connectionist 之类）不能
当作 channel-level 约束工具用**。要约束 channel 含义，就用直接约束
channel 含义的 loss。

## 四、V16.3 在多任务下的稳定性边界

合并以上三点：V16.3 (encoder + range-bottleneck + recon loss) 在
**纯 recon** 下稳定且优秀（20.7 dB / rank95=234），但当我们：

- 在 latent 上加额外监督
- 同时希望保持 latent 结构不变

它会在 rank 维度塌缩，并且 bottleneck 会**配合**塌缩而非阻止。

这个边界对 FM-DiT 那条线有直接含义：

- 如果以后想在 AE 端做"generation-aware"训练（比如直接用 flow-matching
  loss 参与 AE 训练），不能复用 V16.3 bottleneck，需要换成显式的
  rank-preserving 约束（比如 channel decorrelation、whitening、或者
  直接用更弱的 KL）
- 如果保持 V16.3 不动，所有 generation-aware 改造必须发生在 **下游**
  （FM-DiT 这边），而不是 AE 这边

## 后续不做什么

明确舍弃的方向：

- V18.0.1 ablation 全套（reg_weight=0、lambda_phoneme=0、CTC warmup、
  GRL）。边际信息量低于直接回主线
- V18.1 加 speaker encoder。在 V18 核心都没过 gate 的情况下，进一步
  扩展架构没意义
- V18 phoneme encoder + V18 latent 给 FM-DiT 跑下游对比。已知 latent
  rank 塌缩，FM-DiT 必然差，不需要花 GPU 验证

## 后续要做什么

回到 V17 报告里那张表：

```
t=0:     SNR=-0.44 dB, recon_cos=0.084, pred_std/target_std=0.463/1.103
t=0.025: SNR=10.19 dB, recon_cos=0.951
```

V16.3 latent 在 FM-DiT 里 t=0 端点不收敛是真问题。V18 已经告诉我们：
**不要在 AE 端折腾，去 FM-DiT 端解决**。下一步工作转向 fm_dit/。
