# InternVLA-M1 模型架构解析

本文档详细解析InternVLA-M1视觉-语言-动作(VLA)模型从输入到输出的完整处理流程。

## 🎯 模型概述

InternVLA-M1是一个空间引导的视觉-语言-动作框架，用于通用机器人策略。它采用双系统双监督的设计理念，集成了语言头和动作头，能够在统一框架下协同训练。

## 📊 整体架构

InternVLA-M1模型由四个核心子模块组成：

```
输入（多视图图像 + 指令文本）
         ↓
    ┌────────────────────────────────────────┐
    │  1. QWen2.5-VL 视觉语言模型            │
    │     - 处理图像和文本的融合表示          │
    │     - 输出多层隐藏状态                  │
    └────────────────────────────────────────┘
         ↓
    ┌────────────────────────────────────────┐
    │  2. DINO 视觉编码器                     │
    │     - 提取密集的空间视觉特征            │
    │     - 多视图并行处理                    │
    └────────────────────────────────────────┘
         ↓
    ┌────────────────────────────────────────┐
    │  3. Layer-wise QFormer 特征聚合器       │
    │     - 融合VLM和DINO的多层特征           │
    │     - 生成动作条件嵌入                  │
    └────────────────────────────────────────┘
         ↓
    ┌────────────────────────────────────────┐
    │  4. DiT 扩散动作头                      │
    │     - 基于条件特征预测未来动作序列       │
    │     - 使用扩散模型生成连续动作          │
    └────────────────────────────────────────┘
         ↓
输出（标准化的动作序列）
```

## 🔍 详细处理流程

### 步骤 1: 输入准备

**输入数据格式：**
- **多视图图像**：`List[List[PIL.Image]]` - 每个样本包含多个视角的RGB图像
- **指令文本**：`List[str]` - 自然语言任务描述
- **动作标签**（训练时）：`[B, T, action_dim]` - 机器人动作序列

**代码位置**：`InternVLA/model/framework/M1.py:103-105`

```python
batch_images = [example["image"] for example in examples]  # [B, [PIL.Image]]
instructions = [example["lang"] for example in examples]    # [B, str]
actions = [example["action"] for example in examples]       # [B, T, action_dim]
```

### 步骤 2: QWen2.5-VL 视觉语言编码

**功能**：将多模态输入（图像+文本）编码为融合的特征表示

**子模块**：`_QWen_VL_Interface`
- **位置**：`InternVLA/model/modules/vlm/QWen2_5.py`
- **基础模型**：Qwen2.5-VL-3B-Instruct（可配置）
- **实现**：Flash Attention 2加速

**处理流程**：
1. **输入构建**：将PIL图像和文本指令转换为QWen-VL的输入格式
2. **多模态融合**：Qwen2.5模型同时处理视觉和语言信息
3. **输出**：多层隐藏状态（hidden_states），shape: `[B, seq_len, hidden_dim]`

**关键代码**：
```python
# 构建QWen-VL输入格式
qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
    images=batch_images, 
    instructions=instructions
)

# 前向传播，保留所有层的隐藏状态
qwenvl_outputs = self.qwen_vl_interface(
    **qwen_inputs,
    output_hidden_states=True,  # 关键：获取多层特征
    return_dict=True,
)
```

**输出特征**：
- 隐藏状态列表：包含所有Transformer层的输出
- 每层形状：`[B, sequence_length, 2048]`（hidden_size取决于模型配置）

### 步骤 3: DINO 密集视觉编码

**功能**：提取密集的空间视觉特征，补充QWen-VL的视觉理解

**子模块**：`DINOv2BackBone`
- **位置**：`InternVLA/model/modules/dino_model/dino.py`
- **基础模型**：DINOv2-ViT（dinov2_vits14等变体）
- **特点**：专注于空间几何信息，适合机器人操作任务

**处理流程**：
1. **图像预处理**：
   - 调整大小：224×224
   - 标准化：ImageNet统计值
   - 并行处理多视图（使用ThreadPoolExecutor）

2. **特征提取**：
   - 通过DINOv2模型前向传播
   - 提取patch token特征（不使用CLS token）
   - 输出：`[B*num_views, num_patches, dino_dim]`

3. **特征重塑与投影**：
   - 重塑为：`[B, num_views * num_patches, dino_dim]`
   - 线性投影到VLM隐藏维度：`[B, num_views * num_patches, hidden_size]`

**关键代码**：
```python
# 准备DINO输入（多视图并行预处理）
image_tensors = self.dino_encoder.prepare_dino_input(batch_images)

# DINO前向传播
dino_features = self.dino_encoder(image_tensors)  
# 输出: [B*num_views, num_patches, 384]

# 重塑并投影到VLM维度
dino_encoded_features = dino_features.reshape(B, -1, dino_features.shape[-1])
dino_encoded_features = self.dino_pro(dino_encoded_features)  
# 输出: [B, num_views*num_patches, hidden_size]
```

**为什么需要DINO？**
- **空间精度**：提供更细粒度的空间几何信息
- **互补性**：DINO专注视觉，QWen-VL平衡视觉-语言
- **任务适配**：机器人操作需要精确的空间定位

### 步骤 4: Layer-wise QFormer 特征聚合

**功能**：融合QWen-VL多层特征和DINO视觉特征，生成紧凑的动作条件嵌入

**子模块**：`LayerwiseQFormer`
- **位置**：`InternVLA/model/modules/projector/QFormer.py`
- **设计理念**：类似BLIP-2的Q-Former，但采用逐层聚合策略

**架构组成**：
- **可学习查询token**：`[num_query_tokens, hidden_dim]`（默认64个）
- **跨注意力层**：与VLM层数相同（每层独立的CrossAttentionBlock）
- **线性投影**：将VLM维度投影到动作模型维度

**处理流程**：
1. **特征拼接**：
   - 对VLM的每一层隐藏状态，拼接DINO特征
   - `[B, seq_len, D] + [B, dino_tokens, D] → [B, seq_len+dino_tokens, D]`

2. **逐层交叉注意力**：
   - 初始化：扩展全局查询token到batch维度
   - 迭代：每层查询对应层的拼接特征
   - 注意力机制：Q来自查询token，K/V来自拼接特征

3. **输出**：压缩的动作条件特征 `[B, 64, action_hidden_dim]`

**关键代码**：
```python
# 获取指定层范围的隐藏状态
start_layer = self.config.framework.layer_qformer.qformer_start_layer
end_layer = self.config.framework.layer_qformer.qformer_end_layer
condition_features = qwenvl_outputs.hidden_states[start_layer:end_layer]

# 拼接每层的VLM特征与DINO特征
cat_conditions = []
for layer_features in condition_features:
    layer_features = torch.cat(
        [layer_features, dino_encoded_features], dim=1
    )  # [B, seq_len + dino_tokens, D]
    cat_conditions.append(layer_features)

# QFormer聚合
action_condition = self.layer_qformer(cat_conditions)  
# 输出: [B, 64, action_hidden_dim]
```

**设计优势**：
- **多层融合**：利用不同抽象层次的特征
- **高效压缩**：从长序列压缩到64个token
- **任务适配**：专门为动作预测优化

### 步骤 5: DiT 扩散动作预测

**功能**：基于条件特征，通过扩散模型生成未来动作序列

**子模块**：`ActionModel`（DiT变体）
- **位置**：`InternVLA/model/modules/action_model/DiTActionHeader.py`
- **架构**：Diffusion Transformer（DiT）
- **变体**：DiT-S/B/L（可配置深度和宽度）

**核心组件**：
1. **DiT Transformer**：
   - 时序建模backbone
   - 处理带噪声的动作序列
   - 条件embedding通过交叉注意力注入

2. **Gaussian Diffusion**：
   - 前向过程：逐步添加噪声
   - 反向过程：逐步去噪生成动作
   - 噪声调度：squaredcos_cap_v2（默认）

3. **DDIM Sampler**（推理时）：
   - 确定性采样
   - 可配置步数（默认5-10步）
   - 支持classifier-free guidance

**训练流程**：
1. **准备未来动作窗口**：
   - 提取动作序列的未来部分
   - Shape: `[B, future_window_size+1, action_dim]`

2. **数据增强**（训练加速技巧）：
   - 重复样本多次（repeated_diffusion_steps=4）
   - 相当于每个batch训练4遍
   - Shape变为：`[4*B, T, action_dim]`

3. **扩散训练**：
   - 随机采样时间步 t
   - 添加噪声：`noisy_action = sqrt(alpha_t) * action + sqrt(1-alpha_t) * noise`
   - 预测噪声：`noise_pred = DiT(noisy_action, t, condition)`
   - 损失：`MSE(noise_pred, noise)`

**关键代码（训练）**：
```python
# 提取未来动作窗口
actions_future = actions[:, -(self.future_action_window_size + 1):, :]

# 重复样本加速训练
repeated_diffusion_steps = 4
actions_repeated = actions_future.repeat(repeated_diffusion_steps, 1, 1)
action_condition = action_condition.repeat(repeated_diffusion_steps, 1, 1)

# DiT前向：添加噪声并预测
noise_pred, noise, timestep = self.action_model(actions_repeated, action_condition)

# 计算损失
action_loss = self.action_model.loss(noise_pred, noise)
```

**推理流程**：
1. **初始化随机噪声**：
   - Shape: `[B, future_window_size+1, action_dim]`
   - 标准正态分布采样

2. **Classifier-Free Guidance（可选）**：
   - 同时生成条件和无条件预测
   - 混合：`pred = uncond_pred + cfg_scale * (cond_pred - uncond_pred)`
   - 增强条件控制能力

3. **DDIM采样**：
   - 从纯噪声开始
   - 迭代去噪（5-10步）
   - 每步更新：`x_{t-1} = sqrt(alpha_{t-1}) * pred_x0 + sqrt(1-alpha_{t-1}) * noise`

4. **输出标准化动作**：
   - Shape: `[B, T, action_dim]`
   - 值域：[-1, 1]（需要后处理反标准化）

**关键代码（推理）**：
```python
# 初始化噪声
noise = torch.randn(
    B, self.future_action_window_size + 1, self.action_model.in_channels,
    device=action_condition_feature.device
)

# Classifier-Free Guidance设置
if cfg_scale > 1.0:
    noise = torch.cat([noise, noise], 0)
    uncondition = self.action_model.net.z_embedder.uncondition
    z = torch.cat([action_condition_feature, uncondition.expand(B, -1, -1)], 0)
    model_kwargs = dict(z=z, cfg_scale=cfg_scale)
    sample_fn = self.action_model.net.forward_with_cfg
else:
    model_kwargs = dict(z=action_condition_feature)
    sample_fn = self.action_model.net.forward

# DDIM采样
samples = self.action_model.ddim_diffusion.ddim_sample_loop(
    sample_fn, noise.shape, noise,
    clip_denoised=False, model_kwargs=model_kwargs,
    progress=False, device=device, eta=0.0,
)

# 提取条件预测（去除无条件部分）
if cfg_scale > 1.0:
    samples, _ = samples.chunk(2, dim=0)

normalized_actions = samples.cpu().numpy()  # [B, T, action_dim]
```

## 🎨 额外功能：Chat能力

除了动作预测，InternVLA-M1还支持标准的视觉-语言对话功能：

**功能**：`chat_with_M1(image, text)`
- **用途**：图像问答、空间定位（bounding box预测）
- **实现**：直接调用QWen2.5-VL的生成能力
- **代码位置**：`InternVLA/model/framework/M1.py:293-340`

**示例**：
```python
image = load_image("table.jpeg")
response = model.chat_with_M1(image, "Give the bounding box for the apple.")
# 输出: 苹果的边界框坐标
```

## 📈 关键超参数

| 参数 | 默认值 | 说明 |
|-----|--------|-----|
| `future_action_window_size` | 15 | 未来动作预测长度 |
| `past_action_window_size` | 0 | 过去动作上下文长度 |
| `qformer_start_layer` | -10 | QFormer起始层（负索引） |
| `qformer_end_layer` | -1 | QFormer结束层 |
| `num_query_tokens` | 64 | QFormer查询token数量 |
| `diffusion_steps` | 100 | 扩散总步数（训练） |
| `num_ddim_steps` | 5-10 | DDIM采样步数（推理） |
| `cfg_scale` | 1.5 | Classifier-free guidance强度 |
| `repeated_diffusion_steps` | 4 | 训练时样本重复次数 |

## 🔄 数据流总结

### 训练时
```
输入: [image_list, instruction, action_label]
  ↓
QWen2.5-VL: [B, seq, 2048] × N_layers
  ↓
DINO: [B, views*patches, 384] → [B, views*patches, 2048]
  ↓
Concat: [B, seq+dino_tokens, 2048] × N_layers
  ↓
QFormer: [B, 64, 768]
  ↓
Repeat 4×: [4B, 64, 768]
  ↓
DiT Diffusion: noise_pred vs noise
  ↓
Loss: MSE(noise_pred, noise)
```

### 推理时
```
输入: [image_list, instruction]
  ↓
QWen2.5-VL + DINO + QFormer: [B, 64, 768]
  ↓
Initial noise: [B, T, 7]
  ↓
DDIM sampling (10 steps with CFG)
  ↓
输出: normalized_actions [B, T, 7]
```

## 💡 设计亮点

1. **双视觉编码器**：
   - QWen-VL：语义理解 + 视觉推理
   - DINO：空间几何 + 密集特征
   - 互补融合，增强空间感知

2. **Layer-wise聚合**：
   - 利用VLM多层特征
   - 不同抽象层次信息融合
   - 提升动作预测质量

3. **扩散模型架构**：
   - 生成式建模，处理多模态分布
   - 迭代优化，更高质量输出
   - CFG技术增强条件控制

4. **训练加速技巧**：
   - 样本重复（4×）
   - Mixed precision (bfloat16)
   - Flash Attention 2

## 📚 相关文件索引

- **主框架**：`InternVLA/model/framework/M1.py`
- **QWen接口**：`InternVLA/model/modules/vlm/QWen2_5.py`
- **DINO编码器**：`InternVLA/model/modules/dino_model/dino.py`
- **QFormer**：`InternVLA/model/modules/projector/QFormer.py`
- **DiT动作头**：`InternVLA/model/modules/action_model/DiTActionHeader.py`
- **扩散模块**：`InternVLA/model/modules/action_model/DiT_modules/`

## 🔬 实验验证

模型在多个基准测试上取得SOTA结果：

| 基准 | InternVLA-M1 | 对比方法 |
|-----|-------------|---------|
| WindowX | **71.7%** | π₀: 27.1%, GR00t: 61.9% |
| Google Robot (VA) | **76.0%** | π₀: 54.8%, GR00t: 44.5% |
| Google Robot (VM) | **80.7%** | π₀: 58.8%, GR00t: 35.2% |
| LIBERO | **95.9%** | π₀: 94.2%, GR00t: 93.9% |

---

**文档更新日期**: 2026-02-06  
**对应版本**: InternVLA-M1 v1.0  
**作者**: InternVLA-M1 Contributors
