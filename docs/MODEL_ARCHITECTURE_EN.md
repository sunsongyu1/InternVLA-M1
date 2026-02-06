# InternVLA-M1 Model Architecture Analysis

This document provides a detailed analysis of the InternVLA-M1 Vision-Language-Action (VLA) model, explaining the complete processing flow from input to output.

## 🎯 Model Overview

InternVLA-M1 is a spatially guided vision-language-action framework designed for generalist robot policy. It adopts a dual-system and dual-supervision design philosophy, integrating both a language head and an action head for collaborative training within a unified framework.

## 📊 Overall Architecture

InternVLA-M1 consists of four core sub-modules:

```
Input (Multi-view Images + Text Instruction)
         ↓
    ┌────────────────────────────────────────┐
    │  1. QWen2.5-VL Vision-Language Model   │
    │     - Processes fused image-text repr. │
    │     - Outputs multi-layer hidden states│
    └────────────────────────────────────────┘
         ↓
    ┌────────────────────────────────────────┐
    │  2. DINO Visual Encoder                │
    │     - Extracts dense spatial features  │
    │     - Parallel multi-view processing   │
    └────────────────────────────────────────┘
         ↓
    ┌────────────────────────────────────────┐
    │  3. Layer-wise QFormer Aggregator      │
    │     - Fuses VLM and DINO multi-layer   │
    │     - Generates action condition embed │
    └────────────────────────────────────────┘
         ↓
    ┌────────────────────────────────────────┐
    │  4. DiT Diffusion Action Head          │
    │     - Predicts future action sequence  │
    │     - Generates continuous actions     │
    └────────────────────────────────────────┘
         ↓
Output (Normalized Action Sequence)
```

## 🔍 Detailed Processing Pipeline

### Step 1: Input Preparation

**Input Data Format:**
- **Multi-view Images**: `List[List[PIL.Image]]` - Each sample contains RGB images from multiple viewpoints
- **Instruction Text**: `List[str]` - Natural language task descriptions
- **Action Labels** (during training): `[B, T, action_dim]` - Robot action sequences

**Code Location**: `InternVLA/model/framework/M1.py:103-105`

```python
batch_images = [example["image"] for example in examples]  # [B, [PIL.Image]]
instructions = [example["lang"] for example in examples]    # [B, str]
actions = [example["action"] for example in examples]       # [B, T, action_dim]
```

### Step 2: QWen2.5-VL Vision-Language Encoding

**Function**: Encode multi-modal inputs (images + text) into fused feature representations

**Sub-module**: `_QWen_VL_Interface`
- **Location**: `InternVLA/model/modules/vlm/QWen2_5.py`
- **Base Model**: Qwen2.5-VL-3B-Instruct (configurable)
- **Implementation**: Flash Attention 2 acceleration

**Processing Flow**:
1. **Input Construction**: Convert PIL images and text instructions to QWen-VL input format
2. **Multi-modal Fusion**: Qwen2.5 model processes visual and language information simultaneously
3. **Output**: Multi-layer hidden states, shape: `[B, seq_len, hidden_dim]`

**Key Code**:
```python
# Build QWen-VL input format
qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
    images=batch_images, 
    instructions=instructions
)

# Forward pass, retaining hidden states from all layers
qwenvl_outputs = self.qwen_vl_interface(
    **qwen_inputs,
    output_hidden_states=True,  # Key: obtain multi-layer features
    return_dict=True,
)
```

**Output Features**:
- Hidden state list: Contains outputs from all Transformer layers
- Each layer shape: `[B, sequence_length, 2048]` (hidden_size depends on model config)

### Step 3: DINO Dense Visual Encoding

**Function**: Extract dense spatial visual features to complement QWen-VL's visual understanding

**Sub-module**: `DINOv2BackBone`
- **Location**: `InternVLA/model/modules/dino_model/dino.py`
- **Base Model**: DINOv2-ViT (dinov2_vits14 and other variants)
- **Characteristics**: Focuses on spatial geometric information, suitable for robotic manipulation tasks

**Processing Flow**:
1. **Image Preprocessing**:
   - Resize: 224×224
   - Normalize: ImageNet statistics
   - Parallel multi-view processing (using ThreadPoolExecutor)

2. **Feature Extraction**:
   - Forward pass through DINOv2 model
   - Extract patch token features (excluding CLS token)
   - Output: `[B*num_views, num_patches, dino_dim]`

3. **Feature Reshaping & Projection**:
   - Reshape to: `[B, num_views * num_patches, dino_dim]`
   - Linear projection to VLM hidden dimension: `[B, num_views * num_patches, hidden_size]`

**Key Code**:
```python
# Prepare DINO input (parallel multi-view preprocessing)
image_tensors = self.dino_encoder.prepare_dino_input(batch_images)

# DINO forward pass
dino_features = self.dino_encoder(image_tensors)  
# Output: [B*num_views, num_patches, 384]

# Reshape and project to VLM dimension
dino_encoded_features = dino_features.reshape(B, -1, dino_features.shape[-1])
dino_encoded_features = self.dino_pro(dino_encoded_features)  
# Output: [B, num_views*num_patches, hidden_size]
```

**Why DINO?**
- **Spatial Precision**: Provides finer-grained spatial geometric information
- **Complementarity**: DINO focuses on vision, QWen-VL balances vision-language
- **Task Adaptation**: Robotic manipulation requires precise spatial localization

### Step 4: Layer-wise QFormer Feature Aggregation

**Function**: Fuse QWen-VL multi-layer features and DINO visual features to generate compact action condition embeddings

**Sub-module**: `LayerwiseQFormer`
- **Location**: `InternVLA/model/modules/projector/QFormer.py`
- **Design Philosophy**: Similar to BLIP-2's Q-Former, but with layer-wise aggregation strategy

**Architecture Components**:
- **Learnable Query Tokens**: `[num_query_tokens, hidden_dim]` (default 64)
- **Cross-Attention Layers**: Same as VLM layer count (independent CrossAttentionBlock per layer)
- **Linear Projection**: Project VLM dimension to action model dimension

**Processing Flow**:
1. **Feature Concatenation**:
   - For each VLM hidden state layer, concatenate DINO features
   - `[B, seq_len, D] + [B, dino_tokens, D] → [B, seq_len+dino_tokens, D]`

2. **Layer-wise Cross-Attention**:
   - Initialize: Expand global query tokens to batch dimension
   - Iterate: Each layer queries corresponding layer's concatenated features
   - Attention mechanism: Q from query tokens, K/V from concatenated features

3. **Output**: Compressed action condition features `[B, 64, action_hidden_dim]`

**Key Code**:
```python
# Get hidden states from specified layer range
start_layer = self.config.framework.layer_qformer.qformer_start_layer
end_layer = self.config.framework.layer_qformer.qformer_end_layer
condition_features = qwenvl_outputs.hidden_states[start_layer:end_layer]

# Concatenate VLM features with DINO features for each layer
cat_conditions = []
for layer_features in condition_features:
    layer_features = torch.cat(
        [layer_features, dino_encoded_features], dim=1
    )  # [B, seq_len + dino_tokens, D]
    cat_conditions.append(layer_features)

# QFormer aggregation
action_condition = self.layer_qformer(cat_conditions)  
# Output: [B, 64, action_hidden_dim]
```

**Design Advantages**:
- **Multi-layer Fusion**: Leverages features from different abstraction levels
- **Efficient Compression**: Compresses from long sequences to 64 tokens
- **Task Adaptation**: Specifically optimized for action prediction

### Step 5: DiT Diffusion Action Prediction

**Function**: Generate future action sequences via diffusion model based on condition features

**Sub-module**: `ActionModel` (DiT variant)
- **Location**: `InternVLA/model/modules/action_model/DiTActionHeader.py`
- **Architecture**: Diffusion Transformer (DiT)
- **Variants**: DiT-S/B/L (configurable depth and width)

**Core Components**:
1. **DiT Transformer**:
   - Temporal modeling backbone
   - Processes noisy action sequences
   - Condition embeddings injected via cross-attention

2. **Gaussian Diffusion**:
   - Forward process: Progressively add noise
   - Reverse process: Progressively denoise to generate actions
   - Noise schedule: squaredcos_cap_v2 (default)

3. **DDIM Sampler** (during inference):
   - Deterministic sampling
   - Configurable steps (default 5-10 steps)
   - Supports classifier-free guidance

**Training Flow**:
1. **Prepare Future Action Window**:
   - Extract future portion of action sequence
   - Shape: `[B, future_window_size+1, action_dim]`

2. **Data Augmentation** (training acceleration trick):
   - Repeat samples multiple times (repeated_diffusion_steps=4)
   - Effectively trains each batch 4 times
   - Shape becomes: `[4*B, T, action_dim]`

3. **Diffusion Training**:
   - Random sample timestep t
   - Add noise: `noisy_action = sqrt(alpha_t) * action + sqrt(1-alpha_t) * noise`
   - Predict noise: `noise_pred = DiT(noisy_action, t, condition)`
   - Loss: `MSE(noise_pred, noise)`

**Key Code (Training)**:
```python
# Extract future action window
actions_future = actions[:, -(self.future_action_window_size + 1):, :]

# Repeat samples for accelerated training
repeated_diffusion_steps = 4
actions_repeated = actions_future.repeat(repeated_diffusion_steps, 1, 1)
action_condition = action_condition.repeat(repeated_diffusion_steps, 1, 1)

# DiT forward: add noise and predict
noise_pred, noise, timestep = self.action_model(actions_repeated, action_condition)

# Compute loss
action_loss = self.action_model.loss(noise_pred, noise)
```

**Inference Flow**:
1. **Initialize Random Noise**:
   - Shape: `[B, future_window_size+1, action_dim]`
   - Sampled from standard normal distribution

2. **Classifier-Free Guidance (optional)**:
   - Generate both conditional and unconditional predictions
   - Mix: `pred = uncond_pred + cfg_scale * (cond_pred - uncond_pred)`
   - Enhances conditional control capability

3. **DDIM Sampling**:
   - Start from pure noise
   - Iterative denoising (5-10 steps)
   - Each step updates: `x_{t-1} = sqrt(alpha_{t-1}) * pred_x0 + sqrt(1-alpha_{t-1}) * noise`

4. **Output Normalized Actions**:
   - Shape: `[B, T, action_dim]`
   - Value range: [-1, 1] (requires post-processing de-normalization)

**Key Code (Inference)**:
```python
# Initialize noise
noise = torch.randn(
    B, self.future_action_window_size + 1, self.action_model.in_channels,
    device=action_condition_feature.device
)

# Classifier-Free Guidance setup
if cfg_scale > 1.0:
    noise = torch.cat([noise, noise], 0)
    uncondition = self.action_model.net.z_embedder.uncondition
    z = torch.cat([action_condition_feature, uncondition.expand(B, -1, -1)], 0)
    model_kwargs = dict(z=z, cfg_scale=cfg_scale)
    sample_fn = self.action_model.net.forward_with_cfg
else:
    model_kwargs = dict(z=action_condition_feature)
    sample_fn = self.action_model.net.forward

# DDIM sampling
samples = self.action_model.ddim_diffusion.ddim_sample_loop(
    sample_fn, noise.shape, noise,
    clip_denoised=False, model_kwargs=model_kwargs,
    progress=False, device=device, eta=0.0,
)

# Extract conditional prediction (remove unconditional part)
if cfg_scale > 1.0:
    samples, _ = samples.chunk(2, dim=0)

normalized_actions = samples.cpu().numpy()  # [B, T, action_dim]
```

## 🎨 Additional Feature: Chat Capability

Beyond action prediction, InternVLA-M1 also supports standard vision-language dialogue:

**Function**: `chat_with_M1(image, text)`
- **Use Cases**: Image Q&A, spatial grounding (bounding box prediction)
- **Implementation**: Directly leverages QWen2.5-VL's generation capability
- **Code Location**: `InternVLA/model/framework/M1.py:293-340`

**Example**:
```python
image = load_image("table.jpeg")
response = model.chat_with_M1(image, "Give the bounding box for the apple.")
# Output: Bounding box coordinates of the apple
```

## 📈 Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `future_action_window_size` | 15 | Future action prediction length |
| `past_action_window_size` | 0 | Past action context length |
| `qformer_start_layer` | -10 | QFormer start layer (negative index) |
| `qformer_end_layer` | -1 | QFormer end layer |
| `num_query_tokens` | 64 | Number of QFormer query tokens |
| `diffusion_steps` | 100 | Total diffusion steps (training) |
| `num_ddim_steps` | 5-10 | DDIM sampling steps (inference) |
| `cfg_scale` | 1.5 | Classifier-free guidance strength |
| `repeated_diffusion_steps` | 4 | Sample repetition during training |

## 🔄 Data Flow Summary

### Training
```
Input: [image_list, instruction, action_label]
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

### Inference
```
Input: [image_list, instruction]
  ↓
QWen2.5-VL + DINO + QFormer: [B, 64, 768]
  ↓
Initial noise: [B, T, 7]
  ↓
DDIM sampling (10 steps with CFG)
  ↓
Output: normalized_actions [B, T, 7]
```

## 💡 Design Highlights

1. **Dual Visual Encoders**:
   - QWen-VL: Semantic understanding + visual reasoning
   - DINO: Spatial geometry + dense features
   - Complementary fusion enhances spatial perception

2. **Layer-wise Aggregation**:
   - Leverages VLM multi-layer features
   - Fuses information from different abstraction levels
   - Improves action prediction quality

3. **Diffusion Model Architecture**:
   - Generative modeling handles multi-modal distributions
   - Iterative optimization for higher quality outputs
   - CFG technique enhances conditional control

4. **Training Acceleration Techniques**:
   - Sample repetition (4×)
   - Mixed precision (bfloat16)
   - Flash Attention 2

## 📚 Related File Index

- **Main Framework**: `InternVLA/model/framework/M1.py`
- **QWen Interface**: `InternVLA/model/modules/vlm/QWen2_5.py`
- **DINO Encoder**: `InternVLA/model/modules/dino_model/dino.py`
- **QFormer**: `InternVLA/model/modules/projector/QFormer.py`
- **DiT Action Head**: `InternVLA/model/modules/action_model/DiTActionHeader.py`
- **Diffusion Modules**: `InternVLA/model/modules/action_model/DiT_modules/`

## 🔬 Experimental Validation

The model achieves SOTA results on multiple benchmarks:

| Benchmark | InternVLA-M1 | Comparison Methods |
|-----------|-------------|-------------------|
| WindowX | **71.7%** | π₀: 27.1%, GR00t: 61.9% |
| Google Robot (VA) | **76.0%** | π₀: 54.8%, GR00t: 44.5% |
| Google Robot (VM) | **80.7%** | π₀: 58.8%, GR00t: 35.2% |
| LIBERO | **95.9%** | π₀: 94.2%, GR00t: 93.9% |

---

**Document Update Date**: 2026-02-06  
**Corresponding Version**: InternVLA-M1 v1.0  
**Author**: InternVLA-M1 Contributors
