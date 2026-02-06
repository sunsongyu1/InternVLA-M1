# InternVLA-M1 Documentation

This directory contains comprehensive documentation for the InternVLA-M1 model architecture.

## 📄 Available Documents

### Model Architecture Analysis

Detailed explanation of the InternVLA-M1 architecture from input to output:

- **[中文版本 (Chinese)](MODEL_ARCHITECTURE_CN.md)** - 完整的模型架构解析，包含所有子模块详细说明
- **[English Version](MODEL_ARCHITECTURE_EN.md)** - Complete model architecture analysis with all sub-module details

### What's Covered

Both documents provide comprehensive coverage of:

1. **Model Overview** - High-level architecture and design philosophy
2. **Four Core Sub-modules**:
   - QWen2.5-VL Vision-Language Model
   - DINO Visual Encoder
   - Layer-wise QFormer Feature Aggregator
   - DiT Diffusion Action Head
3. **Processing Pipeline** - Step-by-step flow from input to output
4. **Code Examples** - Key code snippets with explanations
5. **Hyperparameters** - Complete parameter reference table
6. **Data Flow Diagrams** - Training and inference pipelines
7. **Design Highlights** - Key architectural innovations
8. **File Index** - Quick reference to implementation files

## 🎯 Quick Navigation

| Topic | Chinese | English |
|-------|---------|---------|
| Overall Architecture | [查看](MODEL_ARCHITECTURE_CN.md#-整体架构) | [View](MODEL_ARCHITECTURE_EN.md#-overall-architecture) |
| QWen2.5-VL Encoder | [查看](MODEL_ARCHITECTURE_CN.md#步骤-2-qwen25-vl-视觉语言编码) | [View](MODEL_ARCHITECTURE_EN.md#step-2-qwen25-vl-vision-language-encoding) |
| DINO Encoder | [查看](MODEL_ARCHITECTURE_CN.md#步骤-3-dino-密集视觉编码) | [View](MODEL_ARCHITECTURE_EN.md#step-3-dino-dense-visual-encoding) |
| QFormer Aggregation | [查看](MODEL_ARCHITECTURE_CN.md#步骤-4-layer-wise-qformer-特征聚合) | [View](MODEL_ARCHITECTURE_EN.md#step-4-layer-wise-qformer-feature-aggregation) |
| DiT Action Head | [查看](MODEL_ARCHITECTURE_CN.md#步骤-5-dit-扩散动作预测) | [View](MODEL_ARCHITECTURE_EN.md#step-5-dit-diffusion-action-prediction) |
| Hyperparameters | [查看](MODEL_ARCHITECTURE_CN.md#-关键超参数) | [View](MODEL_ARCHITECTURE_EN.md#-key-hyperparameters) |
| Data Flow | [查看](MODEL_ARCHITECTURE_CN.md#-数据流总结) | [View](MODEL_ARCHITECTURE_EN.md#-data-flow-summary) |

## 💡 Use Cases

These documents are helpful for:

- **Researchers** - Understanding the model architecture for research purposes
- **Developers** - Implementing custom modifications or extensions
- **Students** - Learning about vision-language-action models
- **Engineers** - Debugging and optimizing model performance
- **Contributors** - Contributing to the project with a clear understanding of the codebase

## 🔗 Related Resources

- **Main README**: [../README.md](../README.md)
- **Code Implementation**: [../InternVLA/model/framework/M1.py](../InternVLA/model/framework/M1.py)
- **Configuration Examples**: [../InternVLA/config/training/](../InternVLA/config/training/)
- **Examples**: [../examples/](../examples/)

## 📝 Document Updates

- **Version**: 1.0
- **Date**: 2026-02-06
- **Status**: Complete and verified against codebase

## 🤝 Contributing

If you find any errors or have suggestions for improving the documentation, please:

1. Open an issue describing the problem or suggestion
2. Submit a pull request with your proposed changes
3. Ensure your changes are consistent with the actual codebase

---

**Maintained by**: InternVLA-M1 Contributors
