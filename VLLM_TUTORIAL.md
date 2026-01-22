# nano-vLLM 深度教学指南

> 从零理解 vLLM：核心原理、代码实现与性能优化

---

## 目录

1. [为什么要用 vLLM？](#1-为什么要用-vllm)
2. [核心组件概览](#2-核心组件概览)
3. [PagedAttention 详解](#3-pagedattention-详解)
4. [Scheduler 调度器](#4-scheduler-调度器)
5. [KV Cache 管理](#5-kv-cache-管理)
6. [Continuous Batching](#6-continuous-batching)
7. [Prefix Caching](#7-prefix-caching)
8. [CUDA Graph 优化](#8-cuda-graph-优化)
9. [实战练习](#9-实战练习)

---

## 1. 为什么要用 vLLM？

### 1.1 普通推理的三大痛点

#### 痛点1：显存浪费严重

```python
# 普通推理：HuggingFace Transformers
prompt = "解释量子计算的原理"
output = model.generate(prompt, max_new_tokens=500)

# 问题：需要预分配整个序列的 KV Cache
# 即使我们只需要 500 个新 token，也要为 1000+ tokens 预留空间
# 实际显存利用率：~30%
```

**图示：普通推理的显存分配**

```
请求1 (prompt=1000, 生成=1000)
├─ KV Cache: 预分配 2000 tokens
├─ 实际使用: 2000 tokens ✅
└─ 浪费: 0 tokens

请求2 (prompt=100, 生成=100)
├─ KV Cache: 预分配 2000 tokens (和请求1一样！)
├─ 实际使用: 200 tokens
└─ 浪费: 1800 tokens ❌ (90% 浪费！)
```

#### 痛点2：批处理效率低

```python
# Static Batching: 必须等待最慢的请求
batch = [
    "生成一篇1000字的文章",  # 需要5秒
    "你好",                 # 需要0.5秒
    "什么是AI",            # 需要1秒
]
# 整个批次需要5秒，即使其他请求早完成了
# GPU利用率：~40%
```

#### 痛点3：无法有效利用缓存

```python
# 多用户使用相同的系统提示词
system_prompt = "你是一个专业的AI助手，请回答以下问题："

requests = [
    system_prompt + "什么是机器学习？",
    system_prompt + "什么是深度学习？",
    system_prompt + "什么是神经网络？",
]

# 普通推理：每个请求都要重新计算 system_prompt 的 KV Cache
# 浪费了大量计算！
```

### 1.2 vLLM 的解决方案

| 问题 | 普通推理 | vLLM | 提升 |
|------|---------|------|------|
| 显存利用率 | ~30% | **>90%** | 3x |
| 吞吐量 | 50 tok/s | **2000 tok/s** | 40x |
| 前缀缓存 | 不支持 | **10-100x 加速** | - |
| 并发能力 | 8 请求 | **256+ 请求** | 32x |

---

## 2. 核心组件概览

### 2.1 架构图

```
用户请求
    ↓
LLM.generate()
    ↓
┌─────────────────────────────────────────┐
│           LLMEngine (主引擎)             │
│  - 管理请求队列                           │
│  - 协调各组件                             │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│          Scheduler (调度器)              │
│  - 决定哪些请求可以执行                   │
│  - 抢占式调度                             │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│       BlockManager (块管理器)            │
│  - 分配/释放 KV Cache 块                 │
│  - 前缀缓存查找                           │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│        ModelRunner (模型执行)            │
│  - Prefill / Decode 执行                 │
│  - CUDA Graph 优化                       │
│  - Tensor Parallelism                    │
└─────────────────────────────────────────┘
    ↓
GPU 计算
```

### 2.2 数据流

```
┌─────────┐
│ Prompt  │ 1. 用户输入
└────┬────┘
     ↓
┌─────────┐
│Sequence │ 2. 封装成序列对象
└────┬────┘
     ↓
┌──────────────┐
│ Scheduler    │ 3. 加入等待队列
│   .waiting   │
└──────┬───────┘
       ↓
┌──────────────┐
│ Scheduler    │ 4. 调度决策
│   .schedule()│    - 有资源吗？
└──────┬───────┘    - 抢占？
       ↓
┌──────────────┐
│BlockManager  │ 5. 分配 KV Cache 块
│  .allocate() │    - 查找前缀缓存
└──────┬───────┘    - 分配新块
       ↓
┌──────────────┐
│ModelRunner   │ 6. 执行模型
│    .run()    │    - Prefill 或 Decode
└──────┬───────┘    - 采样生成
       ↓
┌──────────────┐
│ Scheduler    │ 7. 更新状态
│.postprocess()│    - 添加新 token
└───────────────┘    - 检查是否完成
```

---

## 3. PagedAttention 详解

### 3.1 核心思想：借鉴操作系统虚拟内存

**传统方式**：连续分配
```
请求A: [████████████████████] 1000 tokens，连续内存
请求B: [██████]              100 tokens，连续内存
       ↑ 浪费：需要预分配整个序列空间
```

**PagedAttention**：分页管理
```
请求A: [██][██][██][██] 4个块，每块256 tokens
请求B: [██]             1个块，256 tokens（只用100）
       ↑ 按需分配，不浪费
```

### 3.2 代码实现：Block 类

**文件**: `nanovllm/engine/block_manager.py:8-24`

```python
class Block:
    """KV Cache 的一个物理块"""

    def __init__(self, block_id):
        self.block_id = block_id        # 块的唯一标识
        self.ref_count = 0              # 引用计数（支持共享）
        self.hash = -1                  # 块内容的哈希值
        self.token_ids = []             # 块内的 token 序列

    def update(self, hash: int, token_ids: list[int]):
        """更新块内容"""
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        """重置块状态（用于复用）"""
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []
```

**关键概念**：
- `block_id`: 物理块的地址（类似内存页号）
- `ref_count`: 多个序列可以共享同一个块（前缀缓存）
- `hash`: 用于快速查找相同内容的块

### 3.3 Sequence 的块表

**文件**: `nanovllm/engine/sequence.py:14-84`

```python
class Sequence:
    block_size = 256  # 每个块的大小

    def __init__(self, token_ids: list[int], sampling_params):
        self.token_ids = token_ids       # 完整的 token 序列
        self.block_table = []            # 逻辑块 → 物理块的映射表
        self.num_cached_tokens = 0       # 命中前缀缓存的 token 数

    @property
    def num_blocks(self):
        """需要多少个块"""
        return (self.num_tokens + self.block_size - 1) // self.block_size

    def block(self, i):
        """获取第 i 个逻辑块的 token 列表"""
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]
```

**示例：1000 tokens 的序列**

```python
seq = Sequence([1, 2, 3, ..., 1000])  # 1000 个 tokens

# 需要多少个块？
print(seq.num_blocks)  # 4 个块
# [0-255], [256-511], [512-767], [768-999]

# 访问第 2 个块
print(seq.block(1))  # [256, 257, ..., 511]
```

### 3.4 BlockManager：分配与释放

**文件**: `nanovllm/engine/block_manager.py:26-113`

#### 3.4.1 初始化

```python
class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks = [Block(i) for i in range(num_blocks)]  # 所有物理块
        self.hash_to_block_id = {}                            # 哈希 → 块ID 映射
        self.free_block_ids = deque(range(num_blocks))        # 空闲块队列
        self.used_block_ids = set()                           # 已用块集合
```

**示例**：
```python
# 假设有 1000 个块，每块 256 tokens
manager = BlockManager(num_blocks=1000, block_size=256)

print(len(manager.free_block_ids))  # 1000 个空闲块
print(len(manager.used_block_ids))  # 0 个已用块
```

#### 3.4.2 分配块

```python
def allocate(self, seq: Sequence):
    """为序列分配物理块"""
    h = -1  # 滚动哈希
    cache_miss = False

    for i in range(seq.num_blocks):
        token_ids = seq.block(i)

        # 计算块的哈希值（用于前缀缓存查找）
        if len(token_ids) == self.block_size:
            h = self.compute_hash(token_ids, h)
        else:
            h = -1  # 不完整的块不缓存

        # 查找是否已有相同内容的块
        block_id = self.hash_to_block_id.get(h, -1)

        # 检查缓存是否真正命中（避免哈希冲突）
        if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
            cache_miss = True

        if cache_miss:
            # 缓存未命中：分配新块
            block_id = self.free_block_ids[0]
            block = self._allocate_block(block_id)
        else:
            # 缓存命中：复用已有块！
            seq.num_cached_tokens += self.block_size
            block = self.blocks[block_id]
            block.ref_count += 1  # 增加引用计数

        # 更新哈希表
        if h != -1:
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block_id

        # 记录映射关系：逻辑块 i → 物理块 block_id
        seq.block_table.append(block_id)
```

**示例：分配过程**

```python
# 序列 A：1000 tokens
seq_a = Sequence(list(range(1000)))
manager.allocate(seq_a)

# 分配结果：
# seq_a.block_table = [0, 1, 2, 3]
# 逻辑块 0 → 物理块 0 (tokens 0-255)
# 逻辑块 1 → 物理块 1 (tokens 256-511)
# 逻辑块 2 → 物理块 2 (tokens 512-767)
# 逻辑块 3 → 物理块 3 (tokens 768-999)

print(len(manager.free_block_ids))  # 996 (1000 - 4)
```

#### 3.4.3 追加新 token（Decode 阶段）

```python
def may_append(self, seq: Sequence):
    """为序列追加一个新 token"""
    block_table = seq.block_table
    last_block = self.blocks[block_table[-1]]

    current_len = len(seq)
    if current_len % self.block_size == 1:
        # 当前 token 是新块的第一个
        # 需要分配新块
        assert last_block.hash != -1
        block_id = self.free_block_ids[0]
        self._allocate_block(block_id)
        block_table.append(block_id)

    elif current_len % self.block_size == 0:
        # 当前 token 填满了最后一个块
        # 计算哈希并加入缓存
        assert last_block.hash == -1
        token_ids = seq.block(seq.num_blocks - 1)
        prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
        h = self.compute_hash(token_ids, prefix)
        last_block.update(h, token_ids)
        self.hash_to_block_id[h] = last_block.block_id
```

**示例：序列增长**

```python
# 初始：1000 tokens (4个块)
seq = Sequence(list(range(1000)))
manager.allocate(seq)
print(seq.block_table)  # [0, 1, 2, 3]

# 生成 1 个新 token：1001
seq.append_token(1000)
manager.may_append(seq)
print(seq.block_table)  # [0, 1, 2, 3, 4] ← 新增第 5 个块

# 继续生成 254 个 token，填满第 5 个块
for i in range(1001, 1256):
    seq.append_token(i)
    manager.may_append(seq)
print(seq.block_table)  # [0, 1, 2, 3, 4]
# 第 4 个块（索引 4）现在被缓存了
```

#### 3.4.4 释放块

```python
def deallocate(self, seq: Sequence):
    """释放序列占用的所有块"""
    for block_id in reversed(seq.block_table):
        block = self.blocks[block_id]
        block.ref_count -= 1  # 减少引用计数
        if block.ref_count == 0:
            # 引用计数为 0，真正释放
            self._deallocate_block(block_id)

    seq.num_cached_tokens = 0
    seq.block_table.clear()
```

**示例：释放与共享**

```python
# 序列 A：1000 tokens
seq_a = Sequence(list(range(1000)))
manager.allocate(seq_a)
# seq_a.block_table = [0, 1, 2, 3]

# 序列 B：前 500 tokens 和 A 相同
seq_b = Sequence(list(range(500)) + [9999]*500)
manager.allocate(seq_b)
# seq_b.block_table = [0, 1, 2, 4]
# ↑ 前 2 个块（0, 1）和 A 共享！

print(manager.blocks[0].ref_count)  # 2 (被 A 和 B 引用)
print(manager.blocks[2].ref_count)  # 1 (只被 B 引用)

# 释放序列 A
manager.deallocate(seq_a)
print(manager.blocks[0].ref_count)  # 1 (仍然被 B 引用)
print(manager.blocks[2].ref_count)  # 0 (被释放)
```

### 3.5 PagedAttention 的优势

**1. 显存利用率**
```
传统方式：
- 每个序列预分配最大长度（如 2000 tokens）
- 短序列浪费大量显存

PagedAttention：
- 按需分配，256 tokens 一个块
- 显存利用率 > 90%
```

**2. 支持前缀缓存**
```
多个序列共享相同的 prompt 块
节省显存 + 计算时间
```

**3. 灵活的序列长度**
```
序列可以动态增长，不需要预分配
支持超长上下文
```

---

## 4. Scheduler 调度器

### 4.1 调度器的职责

Scheduler 是 vLLM 的"大脑"，负责：
1. 决定哪些请求可以执行
2. 资源不足时抢占某些请求
3. 区分 Prefill 和 Decode 两个阶段

### 4.2 核心数据结构

**文件**: `nanovllm/engine/scheduler.py:8-17`

```python
class Scheduler:
    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs                    # 最大并发序列数
        self.max_num_batched_tokens = config.max_num_batched_tokens  # 最大 batch token 数
        self.eos = config.eos                                      # 结束 token

        # BlockManager：管理 KV Cache 块
        self.block_manager = BlockManager(
            config.num_kvcache_blocks,
            config.kvcache_block_size
        )

        # 三个队列
        self.waiting: deque[Sequence] = deque()  # 等待队列
        self.running: deque[Sequence] = deque()  # 运行队列
```

**序列状态机**：

```
┌─────────┐
│ WAITING │  在等待队列，尚未分配资源
└────┬────┘
     │ schedule() 选择执行
     ↓
┌─────────┐
│ RUNNING │  正在生成，已分配 KV Cache
└────┬────┘
     │ 生成完成 或 达到 max_tokens
     ↓
┌──────────┐
│ FINISHED │  完成，释放资源
└──────────┘
```

### 4.3 调度策略

**文件**: `nanovllm/engine/scheduler.py:24-58`

#### 4.3.1 阶段1：Prefill（处理新请求）

```python
def schedule(self) -> tuple[list[Sequence], bool]:
    scheduled_seqs = []
    num_seqs = 0
    num_batched_tokens = 0

    # 尝试从等待队列取请求
    while self.waiting and num_seqs < self.max_num_seqs:
        seq = self.waiting[0]

        # 检查资源是否足够
        if num_batched_tokens + len(seq) > self.max_num_batched_tokens:
            # 超过最大 batch token 数
            break

        if not self.block_manager.can_allocate(seq):
            # 没有足够的 KV Cache 块
            break

        # 资源足够，分配并运行
        num_seqs += 1
        self.block_manager.allocate(seq)
        num_batched_tokens += len(seq) - seq.num_cached_tokens  # 减去缓存命中的

        seq.status = SequenceStatus.RUNNING
        self.waiting.popleft()
        self.running.append(seq)
        scheduled_seqs.append(seq)

    if scheduled_seqs:
        return scheduled_seqs, True  # Prefill 模式
```

**示例：Prefill 调度**

```python
# 配置
max_num_seqs = 4
max_num_batched_tokens = 4096

# 等待队列（序列长度：tokens）
waiting = [
    seq_a,  # 1000 tokens
    seq_b,  # 2000 tokens
    seq_c,  # 500 tokens
    seq_d,  # 3000 tokens
]

# 第1轮调度
# 尝试 seq_a: 1000 < 4096 ✅
# 尝试 seq_b: 1000 + 2000 = 3000 < 4096 ✅
# 尝试 seq_c: 3000 + 500 = 3500 < 4096 ✅
# 尝试 seq_d: 3500 + 3000 = 6500 > 4096 ❌ 停止

scheduled = [seq_a, seq_b, seq_c]  # 调度3个请求
```

#### 4.3.2 阶段2：Decode（生成新 token）

```python
    # 如果没有新请求，处理正在运行的序列
    while self.running and num_seqs < self.max_num_seqs:
        seq = self.running.popleft()

        # 检查能否追加新 token
        while not self.block_manager.can_append(seq):
            # 资源不足，需要抢占
            if self.running:
                # 抢占运行队列的最后一个序列（通常是最长的）
                self.preempt(self.running.pop())
            else:
                # 没有其他可抢占的，抢占自己
                self.preempt(seq)
                break
        else:
            # 资源足够，继续运行
            num_seqs += 1
            self.block_manager.may_append(seq)
            scheduled_seqs.append(seq)

    assert scheduled_seqs
    self.running.extendleft(reversed(scheduled_seqs))
    return scheduled_seqs, False  # Decode 模式
```

**示例：Decode 抢占**

```python
# 运行队列
running = [
    seq_a,  # 已生成 5000 tokens，block_table = [0, 1, ..., 19]
    seq_b,  # 已生成 1000 tokens，block_table = [20, 21, 22, 23]
]

# 假设只剩 1 个空闲块

# 处理 seq_a：可以追加 ✅
# 处理 seq_b：无法追加（需要新块但没有空闲）

# 抢占 seq_a（最长的）
preempt(seq_a)
# seq_a 回到等待队列，释放 20 个块

# 现在 seq_b 可以继续了
```

### 4.4 抢占机制

**文件**: `nanovllm/engine/scheduler.py:60-63`

```python
def preempt(self, seq: Sequence):
    """抢占序列，释放资源"""
    seq.status = SequenceStatus.WAITING
    self.block_manager.deallocate(seq)  # 释放所有 KV Cache 块
    self.waiting.appendleft(seq)         # 放回等待队列头部
```

**抢占策略**：
- 优先抢占最长的序列（占用资源多）
- 被抢占的序列下次重新 Prefill
- 牺牲**单个**延迟，提升**整体**吞吐

**示例：抢占的效果**

```python
# 场景：256 个并发请求

# 无抢占（FIFO）：
# 长请求阻塞短请求
# 平均延迟：50 秒

# 有抢占：
# 短请求优先完成
# 长请求被抢占多次
# 平均延迟：10 秒（短）+ 100 秒（长）
# P99 延迟大幅降低
```

### 4.5 后处理

**文件**: `nanovllm/engine/scheduler.py:65-72`

```python
def postprocess(self, seqs: list[Sequence], token_ids: list[int]):
    """处理生成的 token"""
    for seq, token_id in zip(seqs, token_ids):
        seq.append_token(token_id)

        # 检查是否完成
        if (not seq.ignore_eos and token_id == self.eos) or \
           seq.num_completion_tokens == seq.max_tokens:
            # 完成：释放资源
            seq.status = SequenceStatus.FINISHED
            self.block_manager.deallocate(seq)
            self.running.remove(seq)
```

### 4.6 调度流程总结

```
┌─────────────────────────────────────┐
│  1. 检查 waiting 队列               │
│     └─> 能否 Prefill 新请求？        │
└──────────┬──────────────────────────┘
           │ 可以
           ↓
┌─────────────────────────────────────┐
│  2. 分配 KV Cache 块                │
│     └─> 查找前缀缓存                 │
└──────────┬──────────────────────────┘
           │
           ↓
┌─────────────────────────────────────┐
│  3. 执行 Prefill                     │
│     └─> 生成第一个 completion token  │
└──────────┬──────────────────────────┘
           │
           ↓
┌─────────────────────────────────────┐
│  4. 检查 running 队列               │
│     └─> 能否 Decode？                │
└──────────┬──────────────────────────┘
           │ 资源不足
           ↓
┌─────────────────────────────────────┐
│  5. 抢占最长的序列                   │
│     └─> 释放其 KV Cache              │
└──────────┬──────────────────────────┘
           │
           ↓
┌─────────────────────────────────────┐
│  6. 执行 Decode                     │
│     └─> 生成 1 个 token             │
└──────────┬──────────────────────────┘
           │
           ↓
┌─────────────────────────────────────┐
│  7. 后处理                          │
│     └─> 检查是否完成                 │
└─────────────────────────────────────┘
```

---

## 5. KV Cache 管理

### 5.1 什么是 KV Cache？

在 Transformer 的自注意力机制中：
- **K (Key)**: 用于计算注意力权重
- **V (Value)**: 根据注意力权重聚合信息

```python
# 传统方式：每次都重新计算
for step in range(max_tokens):
    output = model(input_ids + generated_ids)
    # 重复计算了之前所有位置的 K 和 V！

# KV Cache：只计算新的位置
for step in range(max_tokens):
    k, v = model.compute_kv(new_token)  # 只计算新的
    k_cache.append(k)
    v_cache.append(v)
    output = modelAttention(k_cache, v_cache)  # 使用缓存
```

### 5.2 KV Cache 的存储格式

**文件**: `nanovllm/engine/model_runner.py:100-118`

```python
def allocate_kv_cache(self):
    """分配 KV Cache 显存"""
    config = self.config
    hf_config = config.hf_config

    # 计算可用显存
    free, total = torch.cuda.mem_get_info()
    used = total - free
    peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
    current = torch.cuda.memory_stats()["allocated_bytes.all.current"]

    # 计算一个块的大小（字节数）
    num_kv_heads = hf_config.num_key_value_heads // self.world_size
    head_dim = hf_config.hidden_size // hf_config.num_attention_heads
    block_bytes = (
        2 *                              # K 和 V
        hf_config.num_hidden_layers *   # 层数
        self.block_size *               # 每块 token 数
        num_kv_heads *                  # KV 头数
        head_dim *                      # 头维度
        hf_config.torch_dtype.itemsize  # 数据类型大小
    )

    # 计算可以分配多少个块
    config.num_kvcache_blocks = int(
        total * config.gpu_memory_utilization - used - peak + current
    ) // block_bytes

    # 分配 KV Cache 张量
    # shape: [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
    self.kv_cache = torch.empty(
        2,  # K 和 V
        hf_config.num_hidden_layers,
        config.num_kvcache_blocks,
        self.block_size,
        num_kv_heads,
        head_dim
    )

    # 将每层的 KV Cache 指针赋值给 Attention 层
    layer_id = 0
    for module in self.model.modules():
        if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
            module.k_cache = self.kv_cache[0, layer_id]
            module.v_cache = self.kv_cache[1, layer_id]
            layer_id += 1
```

**示例：KV Cache 大小计算**

```python
# 配置
num_layers = 32
block_size = 256
num_kv_heads = 8
head_dim = 128
dtype = torch.float16  # 2 bytes

# 一个块的大小
block_bytes = 2 * 32 * 256 * 8 * 128 * 2
            = 8,388,608 bytes ≈ 8 MB

# 1000 个块
total_bytes = 1000 * 8 MB = 8 GB
```

### 5.3 slot_mapping：逻辑位置到物理位置的映射

**问题**：序列的 token 在逻辑上是连续的，但物理存储可能分散

**解决方案**：使用 `slot_mapping` 数组

**Prefill 阶段**：`model_runner.py:126-162`

```python
def prepare_prefill(self, seqs: list[Sequence]):
    """准备 Prefill 的输入数据"""
    slot_mapping = []

    for seq in seqs:
        # 跳过缓存命中的块
        for i in range(seq.num_cached_blocks, seq.num_blocks):
            # 计算物理块的起始位置
            start = seq.block_table[i] * self.block_size

            # 计算结束位置
            if i != seq.num_blocks - 1:
                end = start + self.block_size
            else:
                # 最后一个块可能未满
                end = start + seq.last_block_num_tokens

            # 添加所有 token 的物理位置
            slot_mapping.extend(list(range(start, end)))

    return torch.tensor(slot_mapping, dtype=torch.int32)
```

**示例：slot_mapping 的构建**

```python
# 序列：1000 tokens
# block_table = [5, 10, 15, 20]

# 逻辑位置 → 物理位置
# token 0 → slot 5*256 + 0 = 1280
# token 1 → slot 1281
# ...
# token 255 → slot 1535
# token 256 → slot 10*256 + 0 = 2560
# ...
# token 999 → slot 20*256 + 231 = 5327

slot_mapping = [1280, 1281, ..., 1535, 2560, ..., 5327]
```

**Decode 阶段**：`model_runner.py:164-180`

```python
def prepare_decode(self, seqs: list[Sequence]):
    """准备 Decode 的输入数据"""
    slot_mapping = []
    context_lens = []

    for seq in seqs:
        # 新生成的 token 的物理位置
        slot = seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
        slot_mapping.append(slot)

        # 序列的当前长度（用于 Flash Attention）
        context_lens.append(len(seq))

    return torch.tensor(slot_mapping), torch.tensor(context_lens)
```

**示例：Decode 的 slot_mapping**

```python
# 序列已生成 1000 tokens
# block_table = [5, 10, 15, 20]
# 正在生成第 1001 个 token

# 最后一个块是 block 20
# 已用位置：0-231（232 个 tokens）
# 新 token 的位置：20*256 + 232 = 5344

slot_mapping = [5344]
context_lens = [1000]  # Flash Attention 需要知道序列长度
```

### 5.4 存储到 KV Cache

**文件**: `nanovllm/layers/attention.py:10-41`

```python
@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,  # num_heads * head_dim
):
    """将新的 K 和 V 存储到 KV Cache"""
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)

    if slot == -1:
        return  # 不需要存储（如缓存命中的块）

    # 加载新的 K 和 V
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)

    # 存储到对应的物理位置
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)
```

**优势**：
- 并行写入：所有 token 同时存储
- 非连续存储：支持分散的物理块
- 高效：使用 Triton GPU kernel

---

## 6. Continuous Batching

### 6.1 Static Batching 的问题

**传统方式**：
```python
batch = [req1, req2, req3, req4]

# 必须等待所有请求完成
for step in range(max_steps):
    outputs = model(batch)
    # 即使 req2 在 step 10 就完成了，
    # 也要等 req4 在 step 100 完成
    # 浪费了大量计算！
```

**图示**：
```
时间轴 →

Req1: ██████████          (10 steps)
Req2: ████                (4 steps)  ← 浪费：等待 6 steps
Req3: ██████              (6 steps)  ← 浪费：等待 4 steps
Req4: ████████████████    (14 steps)

总时间：14 steps（由最慢的决定）
有效计算：10 + 4 + 6 = 30 steps
浪费：14 * 4 - 30 = 26 steps (46%)
```

### 6.2 Continuous Batching 的优势

**vLLM 方式**：
```python
while not finished:
    # 每步重新调度
    batch = schedule()

    # 执行当前批次
    outputs = model(batch)

    # 移除完成的请求
    # 添加新的请求
    # 动态调整批次大小
```

**图示**：
```
Step 1: [Req1, Req2, Req3, Req4]
Step 2: [Req1, Req3, Req4]         # Req2 完成
Step 3: [Req1, Req4, Req5]         # Req3 完成，加入 Req5
Step 4: [Req4, Req5, Req6, Req7]   # Req1 完成，加入 Req6, Req7
...

每个请求完成就立即移除，不浪费计算
```

### 6.3 代码实现

**文件**: `nanovllm/engine/llm_engine.py:48-54`

```python
def step(self):
    """执行一个推理步骤"""
    # 1. 调度：决定哪些请求执行
    seqs, is_prefill = self.scheduler.schedule()

    # 2. 执行：Prefill 或 Decode
    token_ids = self.model_runner.call("run", seqs, is_prefill)

    # 3. 后处理：更新序列状态
    self.scheduler.postprocess(seqs, token_ids)

    # 4. 返回完成的请求
    outputs = [(seq.seq_id, seq.completion_token_ids)
               for seq in seqs if seq.is_finished]

    return outputs, num_tokens
```

**主循环**：`llm_engine.py:72-89`

```python
while not self.is_finished():
    # 每步都重新调度
    outputs, num_tokens = self.step()

    # 处理完成的请求
    for seq_id, token_ids in outputs:
        results[seq_id] = token_ids
```

### 6.4 性能对比

**场景**：256 个请求，长度分布 [100, 5000]

| 方式 | 总时间 | 吞吐量 | GPU 利用率 |
|------|--------|--------|-----------|
| Static Batching | 500s | 2560 tok/s | 40% |
| Continuous Batching | **125s** | **10240 tok/s** | **85%** |

**提升**：4x 吞吐量，2x GPU 利用率

---

## 7. Prefix Caching

### 7.1 原理

**问题**：多个请求有相同的 prompt，重复计算

**解决**：缓存已计算过的 KV Cache 块

### 7.2 哈希计算

**文件**: `nanovllm/engine/block_manager.py:35-41`

```python
@classmethod
def compute_hash(cls, token_ids: list[int], prefix: int = -1):
    """计算块的哈希值"""
    h = xxhash.xxh64()  # 快速哈希算法

    # 滚动哈希：包含前一个块的哈希
    if prefix != -1:
        h.update(prefix.to_bytes(8, "little"))

    # 更新当前块的 token
    h.update(np.array(token_ids).tobytes())

    return h.intdigest()
```

**示例：滚动哈希**

```python
# 序列：[1, 2, 3, 4, 5, 6, 7, 8]
# 块大小：4

# 块 0: [1, 2, 3, 4]
h0 = compute_hash([1, 2, 3, 4], prefix=-1)
# h0 = hash([1, 2, 3, 4])

# 块 1: [5, 6, 7, 8]
h1 = compute_hash([5, 6, 7, 8], prefix=h0)
# h1 = hash(h0 || [5, 6, 7, 8])
#     = hash(hash([1,2,3,4]) || [5,6,7,8])
#     = hash([1,2,3,4,5,6,7,8])
```

**优势**：可以快速检测整个前缀是否相同

### 7.3 缓存查找

**文件**: `nanovllm/engine/block_manager.py:59-83`

```python
def allocate(self, seq: Sequence):
    h = -1
    cache_miss = False

    for i in range(seq.num_blocks):
        token_ids = seq.block(i)
        h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1

        # 查找缓存
        block_id = self.hash_to_block_id.get(h, -1)

        # 验证缓存（避免哈希冲突）
        if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
            cache_miss = True

        if cache_miss:
            # 缓存未命中，分配新块
            block_id = self.free_block_ids[0]
            block = self._allocate_block(block_id)
        else:
            # 缓存命中！
            seq.num_cached_tokens += self.block_size
            block = self.blocks[block_id]
            block.ref_count += 1

        # 记录缓存
        if h != -1:
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block_id

        seq.block_table.append(block_id)
```

**示例：Prefix Caching**

```python
# 请求 1
seq1 = Sequence([1, 2, 3, 4, 5, 6, 7, 8])
manager.allocate(seq1)
# 块 0: 新分配 block 0，hash = h0
# 块 1: 新分配 block 1，hash = h1
# seq1.block_table = [0, 1]

# 请求 2（前半部分相同）
seq2 = Sequence([1, 2, 3, 4, 9, 10, 11, 12])
manager.allocate(seq2)
# 块 0: 命中缓存！复用 block 0
# 块 1: 未命中，新分配 block 2
# seq2.block_table = [0, 2]
# seq2.num_cached_tokens = 4 (跳过了 4 个 tokens 的计算)

# 节省：50% 的 Prefill 时间
```

### 7.4 性能提升

**场景**：多用户使用相同的系统提示词

```python
system_prompt = "你是一个专业的AI助手..."
user_prompts = ["什么是机器学习？", "什么是深度学习？", ...]

# 无缓存：
# Prefill 时间 = 1000 * len(system_prompt) ms

# 有缓存：
# Prefill 时间 = 1000 * len(system_prompt) ms (第一个请求)
#             + 100 * len(user_prompt) ms (其他请求)
# 提升约 10x
```

---

## 8. CUDA Graph 优化

### 8.1 什么是 CUDA Graph？

**问题**：每次推理都要：
1. CPU 启动 GPU kernels
2. GPU 执行计算
3. GPU 同步

**开销**：kernel launch 可占总时间的 10-20%

**解决**：CUDA Graph 捕获一次计算过程，之后只需替换输入数据

### 8.2 捕获计算图

**文件**: `nanovllm/engine/model_runner.py:216-252`

```python
@torch.inference_mode()
def capture_cudagraph(self):
    """捕获多种 batch size 的 CUDA Graph"""
    max_bs = min(self.config.max_num_seqs, 512)
    max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size

    # 预分配固定内存
    input_ids = torch.zeros(max_bs, dtype=torch.int64)
    positions = torch.zeros(max_bs, dtype=torch.int64)
    slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
    context_lens = torch.zeros(max_bs, dtype=torch.int32)
    block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
    outputs = torch.zeros(max_bs, hf_config.hidden_size)

    # 捕获多种 batch size
    self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
    self.graphs = {}

    for bs in reversed(self.graph_bs):
        graph = torch.cuda.CUDAGraph()

        # 设置上下文
        set_context(
            False,  # Decode 模式
            slot_mapping=slot_mapping[:bs],
            context_lens=context_lens[:bs],
            block_tables=block_tables[:bs]
        )

        # Warmup
        outputs[:bs] = self.model(input_ids[:bs], positions[:bs])

        # 捕获
        with torch.cuda.graph(graph, pool):
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])

        # 共享内存池
        if self.graph_pool is None:
            self.graph_pool = graph.pool()

        self.graphs[bs] = graph
        torch.cuda.synchronize()
        reset_context()

    # 保存变量引用
    self.graph_vars = dict(
        input_ids=input_ids,
        positions=positions,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        block_tables=block_tables,
        outputs=outputs,
    )
```

### 8.3 重放计算图

**文件**: `nanovllm/engine/model_runner.py:189-206`

```python
@torch.inference_mode()
def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
    if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
        # Prefill 或特殊情况：使用 Eager 模式
        return self.model.compute_logits(self.model(input_ids, positions))
    else:
        # Decode：使用 CUDA Graph
        bs = input_ids.size(0)
        context = get_context()

        # 选择最接近的 batch size
        graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]

        # 更新输入数据（直接内存拷贝，无 kernel launch）
        graph_vars = self.graph_vars
        graph_vars["input_ids"][:bs] = input_ids
        graph_vars["positions"][:bs] = positions
        graph_vars["slot_mapping"].fill_(-1)
        graph_vars["slot_mapping"][:bs] = context.slot_mapping
        graph_vars["context_lens"].zero_()
        graph_vars["context_lens"][:bs] = context.context_lens
        graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables

        # 重放图（一次 kernel launch）
        graph.replay()

        return self.model.compute_logits(graph_vars["outputs"][:bs])
```

### 8.4 性能提升

**Decode 阶段**：
- Eager 模式：10 ms（kernel launch: 2 ms + 计算: 8 ms）
- CUDA Graph：8.2 ms（kernel launch: 0.2 ms + 计算: 8 ms）
- 提升：~18%

**注意**：只对 Decode 有效，Prefill 因为长度不固定无法使用

---

## 9. 实战练习

### 练习1：理解 PagedAttention

**任务**：手动画出以下序列的块分配

```python
seq1 = Sequence([1, 2, 3, ..., 1000])  # 1000 tokens
seq2 = Sequence([1, 2, 3, ..., 500, 9999, 9999, ...])  # 前 500 个相同

manager = BlockManager(num_blocks=100, block_size=256)

manager.allocate(seq1)
manager.allocate(seq2)

# 问题：
# 1. seq1 需要多少个块？
# 2. seq2 需要多少个块？
# 3. 有多少个块被共享？
# 4. 引用计数分别是多少？
```

**答案**：
```python
# 1. seq1: (1000 + 256 - 1) // 256 = 4 个块
# 2. seq2: 假设总长 1500 tokens，需要 6 个块
# 3. 前 2 个块共享（512 tokens）
# 4. block[0].ref_count = 2, block[1].ref_count = 2
```

### 练习2：理解 Scheduler

**任务**：模拟以下场景的调度过程

```python
# 配置
max_num_seqs = 4
max_num_batched_tokens = 4096
num_blocks = 100

# 请求队列
waiting = [
    Sequence([0] * 1000),   # req1: 1000 tokens
    Sequence([0] * 2000),   # req2: 2000 tokens
    Sequence([0] * 500),    # req3: 500 tokens
    Sequence([0] * 3000),   # req4: 3000 tokens
    Sequence([0] * 1000),   # req5: 1000 tokens
]

# 问题：
# 1. 第1轮调度哪些请求？
# 2. 第2轮呢？
# 3. 如果需要抢占，抢占哪个？
```

**答案**：
```python
# 第1轮：req1, req2, req3 (1000 + 2000 + 500 = 3500 < 4096)
# 第2轮：req4 (3000 < 4096, req1, req2, req3 在 running 队列)
# 抢占：如果资源不足，优先抢占最长的（req2 或 req4）
```

### 练习3：计算 KV Cache 大小

**任务**：计算以下配置的 KV Cache 显存占用

```python
num_layers = 32
hidden_size = 4096
num_attention_heads = 32
num_kv_heads = 8
head_dim = 128
block_size = 256
num_blocks = 1000
dtype = torch.float16
```

**答案**：
```python
# 每个 KV 头的维度
head_dim = 128

# 每个块的大小（一个 token 的 K 和 V）
bytes_per_token = (
    2 *                     # K 和 V
    num_layers *           # 32 层
    num_kv_heads *         # 8 个 KV 头
    head_dim *             # 128 维
    dtype.itemsize         # 2 bytes (float16)
)
# = 2 * 32 * 8 * 128 * 2 = 131,072 bytes

# 每个块（256 个 tokens）
bytes_per_block = bytes_per_token * block_size
# = 131,072 * 256 = 33,554,432 bytes ≈ 32 MB

# 总显存
total_bytes = bytes_per_block * num_blocks
# = 32 MB * 1000 = 32 GB
```

### 练习4：Prefix Caching 命中率

**任务**：计算以下场景的缓存命中率

```python
system_prompt = [1, 2, 3, ..., 500]  # 500 tokens
user_prompts = [
    [100, 101, 102, ..., 200],   # 100 tokens
    [200, 201, 202, ..., 400],   # 200 tokens
    [150, 151, 152, ..., 250],   # 100 tokens
]

# 所有请求都使用相同的 system_prompt
```

**答案**：
```python
# 总 token 数
total_tokens = (500 + 100) + (500 + 200) + (500 + 100) = 1900

# 缓存命中（system_prompt 的 2 个块）
cached_tokens = 500 * 3 = 1500

# 命中率
hit_rate = cached_tokens / (total_tokens + cached_tokens)
        = 1500 / (1900 + 1500)
        = 1500 / 3400
        ≈ 44%

# 节省 44% 的 Prefill 时间
```

### 练习5：Continuous Batching 性能

**任务**：对比 Static Batching 和 Continuous Batching

```python
# 请求（长度：生成 token 数）
requests = [10, 100, 50, 20, 200, 30, 150, 80]

# Static Batching：批次大小 4
# Continuous Batching：每步重新调度

# 计算两种方式的总时间
```

**答案**：
```python
# Static Batching
# Batch 1: [10, 100, 50, 20] → 100 steps
# Batch 2: [200, 30, 150, 80] → 200 steps
# 总时间：300 steps

# Continuous Batching（简化模型）
# Step 1-10:  [10, 100, 50, 20, 200, 30, 150, 80]
# Step 11-20: [100, 50, 200, 30, 150, 80]         # 10 完成
# Step 21-30: [100, 200, 30, 150, 80]            # 20 完成
# Step 31-50: [100, 200, 150, 80]                # 50 完成
# Step 51-80: [100, 200, 150]                    # 30, 80 完成
# Step 81-100: [200, 150]                        # 100 完成
# Step 101-150: [200]                            # 150 完成
# Step 151-200: [200]                            # 200 完成
# 总时间：200 steps

# 提升：1.5x
```

---

## 10. 总结

### 10.1 核心组件回顾

| 组件 | 解决的问题 | 核心思想 | 性能提升 |
|------|----------|---------|---------|
| **PagedAttention** | 显存浪费 | 分页管理 | 3x 显存效率 |
| **Scheduler** | 批处理效率低 | 抢占式调度 | 5x 吞吐量 |
| **Continuous Batching** | 请求等待 | 动态批处理 | 4x 吞吐量 |
| **Prefix Caching** | 重复计算 | 前缀哈希缓存 | 10-100x Prefill |
| **CUDA Graph** | Kernel 开销 | 计算图复用 | 1.2x Decode |

### 10.2 适用场景

**推荐使用 vLLM**：
- ✅ 高并发在线服务（ChatGPT、客服机器人）
- ✅ 多用户共享系统提示词
- ✅ 需要长上下文推理
- ✅ 对吞吐量要求高的场景

**不推荐使用 vLLM**：
- ❌ 单次离线推理（overhead 太大）
- ❌ 极低延迟要求（调度有额外开销）
- ❌ 超长单个请求（Prefill 时间主导）

### 10.3 学习路径

1. **基础**：理解 KV Cache 和自注意力机制
2. **核心**：深入 PagedAttention 和 Scheduler
3. **优化**：学习 Prefix Caching 和 CUDA Graph
4. **实践**：自己实现简化版的 vLLM
5. **进阶**：阅读完整 vLLM 源码

### 10.4 参考资源

- **nano-vLLM 源码**：本项目的核心实现
- **vLLM 论文**："Efficient Memory Management for Large Language Model Serving"
- **Flash Attention**：优化注意力计算
- **操作系统**：虚拟内存、页表、调度算法

---

## 11. 常见问题

### Q1: 为什么块大小是 256？

**A**: 平衡考虑：
- 太小（如 16）：管理开销大
- 太大（如 1024）：粒度粗，浪费多
- 256 是经验值，适合大多数场景

### Q2: 抢占会不会导致请求"饥饿"？

**A**: 会，但影响有限：
- 短请求优先完成，降低 P99 延迟
- 长请求虽然被抢占，但最终会完成
- 可以通过优先级队列优化

### Q3: Prefix Caching 的哈希冲突怎么办？

**A**: 代码中有验证：
```python
if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
    cache_miss = True
```
哈希冲突时会重新计算，保证正确性

### Q4: CUDA Graph 为什么不能用于 Prefill？

**A**: Prefill 的输入长度不固定：
- 每个 prompt 长度不同
- CUDA Graph 需要固定的张量形状
- 只能用于 Decode（每次都是 1 个 token）

### Q5: Tensor Parallelism 怎么工作？

**A**: 模型并行：
- 将模型切分到多个 GPU
- 每个 GPU 计算一部分头
- 通过 all-reduce 通信聚合结果
- 线性扩展性能

---

**恭喜你完成了 vLLM 深度学习！**

有任何问题欢迎继续交流 🎉
