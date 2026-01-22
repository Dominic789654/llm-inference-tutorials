# LLM 推理系统深度教学

> 从零掌握 vLLM、SGLang 等现代 LLM 推理框架的核心原理与实现

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

## 📚 简介

本仓库包含一系列深入浅出的 LLM 推理系统教学文档，旨在帮助学习者：

- ✅ 理解现代 LLM 推理框架的核心原理
- ✅ 掌握 PagedAttention、Radix Cache 等关键技术
- ✅ 学习 Chunked Prefill、Overlap Scheduling 等高级优化
- ✅ 对比不同框架的设计权衡
- ✅ 获得实际开发和优化能力

## 🎯 为什么学习 LLM 推理系统？

大语言模型（LLM）推理系统是当前 AI 领域的热点，掌握相关技术可以：

1. **提升推理性能**：从 50 tok/s 提升到 2000+ tok/s（40x 加速）
2. **降低硬件成本**：通过优化显著减少 GPU 需求
3. **构建生产服务**：搭建高并发、低延迟的在线服务
4. **深入 AI 系统**：理解分布式系统、CUDA 优化、调度算法等

## 📖 文档列表

### 核心教程

| 文档 | 内容 | 难度 | 时间 |
|------|------|------|------|
| **[nano-vLLM 教学指南](./VLLM_TUTORIAL.md)** | 从零理解 vLLM：PagedAttention、Scheduler、KV Cache 管理 | ⭐⭐ | 1-2 周 |
| **[mini-sglang 教学指南](./MINI_SGLANG_TUTORIAL.md)** | 深入 Radix Cache、Chunked Prefill、Overlap Scheduling | ⭐⭐⭐⭐ | 3-4 周 |
| **[框架对比分析](./NANO_VLLM_COMPARISON.md)** | nano-vLLM vs mini-sglang 全面对比 | ⭐⭐⭐ | 1 周 |

### 推荐学习路径

```
初学者路径（1-2 月）：
├─ 第 1-2 周：VLLM_TUTORIAL.md
│   ├─ PagedAttention 原理
│   ├─ Scheduler 调度策略
│   ├─ KV Cache 管理机制
│   └─ Continuous Batching
│
├─ 第 3 周：NANO_VLLM_COMPARISON.md
│   ├─ 理解不同架构选择
│   ├─ 对比性能优化技术
│   └─ 选择合适框架
│
└─ 第 4-8 周：MINI_SGLANG_TUTORIAL.md
    ├─ Radix Cache 实现
    ├─ Chunked Prefill 机制
    ├─ Overlap Scheduling
    ├─ 分布式架构
    └─ 自定义 CUDA Kernels

进阶路径（3-6 月）：
├─ 深入源码阅读
├─ 实现自定义优化
├─ 性能分析和调优
└─ 生产环境部署
```

## 🔑 核心技术要点

### 1. PagedAttention
- **问题**：传统 KV Cache 浪费严重
- **解决**：借鉴操作系统虚拟内存，分页管理
- **效果**：显存利用率从 30% 提升到 90%+

### 2. Radix Cache
- **问题**：哈希缓存只能匹配完整块
- **解决**：前缀树自动匹配，支持部分前缀
- **效果**：缓存命中率提升 1.5-2x

### 3. Chunked Prefill
- **问题**：长 prompt 需要 OOM 风险
- **解决**：分片处理长序列
- **效果**：峰值显存降低 25x

### 4. Overlap Scheduling
- **问题**：CPU 调度开销大
- **解决**：CPU/GPU 并行执行
- **效果**：吞吐量提升 1.5x

### 5. Continuous Batching
- **问题**：静态批处理等待慢请求
- **解决**：动态批处理，请求完成即移除
- **效果**：吞吐量提升 4x

## 💡 特色亮点

### 📊 丰富的示例代码

每个概念都配有详细的代码示例：

```python
# PagedAttention 块分配示例
seq = Sequence([1, 2, 3, ..., 1000])
manager.allocate(seq)
# seq.block_table = [0, 1, 2, 3]
```

### 🎨 可视化图示

使用 ASCII 图示展示复杂概念：

```
传统方式：连续分配
请求A: [████████████████████] 1000 tokens
请求B: [██████]              100 tokens
       ↑ 浪费：需要预分配

PagedAttention：分页管理
请求A: [██][██][██][██] 4个块
请求B: [██]             1个块
       ↑ 按需分配
```

### 🔢 实战练习

每章包含实战练习，巩固理解：

- 练习 1：手动画出 Radix 树结构
- 练习 2：模拟 Chunked Prefill 调度
- 练习 3：计算 Overhead 和性能提升
- 练习 4：对比缓存命中率
- 练习 5：计算分布式通信量

## 🛠️ 技术栈

### 教学项目
- **nano-vLLM**: [https://github.com/tzular/mini-vllm](https://github.com/tzular/mini-vllm)
  - 2000 行 Python
  - 适合学习核心概念

- **mini-sglang**: [https://github.com/sgl-project/mini-sglang](https://github.com/sgl-project/mini-sglang)
  - 5000 行 Python + CUDA
  - 生产级实现

### 关键依赖
- PyTorch
- FlashAttention / FlashInfer
- Triton / TVM
- ZeroMQ
- FastAPI

## 📈 性能对比

| 指标 | HuggingFace | vLLM | SGLang |
|------|-------------|------|--------|
| **吞吐量** | 50 tok/s | 2000 tok/s | 3000 tok/s |
| **显存利用率** | 30% | 90% | 95% |
| **并发能力** | 8 请求 | 256 请求 | 512 请求 |
| **前缀缓存** | 不支持 | 10-100x | 10-100x |

## 🚀 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/YOUR_USERNAME/llm-inference-tutorials.git
cd llm-inference-tutorials
```

### 2. 选择学习路径

```bash
# 初学者：从 nano-vLLM 开始
cat VLLM_TUTORIAL.md

# 有经验：直接看 mini-sglang
cat MINI_SGLANG_TUTORIAL.md

# 想对比：看对比文档
cat NANO_VLLM_COMPARISON.md
```

### 3. 动手实践

```bash
# 克隆教学项目
git clone https://github.com/tzular/mini-vllm.git
cd mini-vllm

# 运行示例
python example.py

# 开始修改和实验
vim nanovllm/engine/scheduler.py
```

## 📚 参考资源

### 论文
- **PagedAttention**: [vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- **RadixAttention**: [SGLang: Efficient Execution of Large Language Models with Structured Generation](https://arxiv.org/abs/2312.07157)
- **Chunked Prefill**: [Sarathi-Serve: Efficient LLM Serving over PCIe and NVLink](https://arxiv.org/abs/2403.02310)

### 项目
- [vLLM](https://github.com/vllm-project/vllm) - 生产级 LLM 推理服务
- [SGLang](https://github.com/sgl-project/sglang) - 结构化生成推理引擎
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) - NVIDIA 优化引擎

### 工具
- [FlashAttention](https://github.com/Dao-AILab/flash-attention) - 快速注意力实现
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer) - 高效 LLM 推理库
- [Triton](https://github.com/openai/triton) - Python GPU 编程

## 🤝 贡献

欢迎贡献！

### 如何贡献

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 开启 Pull Request

### 贡献方向

- 📝 修正错别字和表达
- ➕ 添加新的教学章节
- 🎨 改进示例代码
- 📊 补充性能测试
- 🌍 翻译成其他语言

## 📝 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 🌟 Star History

如果这个项目对你有帮助，请给个 Star ⭐️

## 📮 联系方式

- 提交 Issue：[GitHub Issues](https://github.com/YOUR_USERNAME/llm-inference-tutorials/issues)
- 邮件：your.email@example.com
- 微信：your_wechat_id

## 🙏 致谢

感谢以下开源项目的启发：

- [nano-vLLM](https://github.com/tzular/mini-vllm) - 简洁的教学实现
- [mini-sglang](https://github.com/sgl-project/mini-sglang) - 生产级参考实现
- [vLLM](https://github.com/vllm-project/vllm) - 开创性工作
- [SGLang](https://github.com/sgl-project/sglang) - 高级优化技术

---

<div align="center">

**开始你的 LLM 推理系统学习之旅！** 🚀

Made with ❤️ by AI Community

</div>
