# nano-vLLM vs mini-sglang 深度对比

> 两个轻量级 LLM 推理框架的全方位比较

---

## 目录

1. [项目概览对比](#1-项目概览对比)
2. [架构设计对比](#2-架构设计对比)
3. [核心组件对比](#3-核心组件对比)
4. [缓存机制对比](#4-缓存机制对比)
5. [调度策略对比](#5-调度策略对比)
6. [性能优化对比](#6-性能优化对比)
7. [代码复杂度对比](#7-代码复杂度对比)
8. [适用场景对比](#8-适用场景对比)
9. [学习路径建议](#9-学习路径建议)

---

## 1. 项目概览对比

### 1.1 基本信息

| 维度 | nano-vLLM | mini-sglang |
|------|-----------|-------------|
| **代码行数** | ~2,000 行 | ~5,000 行 |
| **主要语言** | Python | Python + C++ CUDA |
| **设计目标** | 教学性质，简化实现 | 生产就绪，高性能 |
| **支持模型** | Qwen2/3 | Llama3, Qwen3 |
| **多GPU支持** | Tensor Parallelism | Tensor Parallelism |
| **API服务** | 无 | OpenAI-compatible API |
| **部署模式** | 单进程脚本 | 多进程分布式系统 |

### 1.2 项目定位

**nano-vLLM**：
```
教学工具
├─ 核心理解 vLLM 工作原理
├─ 代码简洁，易于修改
├─ 适合快速原型开发
└─ 最佳学习项目
```

**mini-sglang**：
```
生产系统
├─ 完整的在线服务能力
├─ 高性能优化
├─ 分布式架构
└─ 可直接用于生产环境
```

---

## 2. 架构设计对比

### 2.1 进程模型

**nano-vLLM：单进程 + Tensor Parallelism**

```
┌─────────────────────────────────────┐
│         LLM.generate()              │
│      (用户直接调用 Python API)       │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│         LLMEngine                   │
│    (主进程，管理所有逻辑)            │
└──────────────┬──────────────────────┘
               ↓
       ┌───────┴────────┐
       │  Multiprocessing│
       └───────┬────────┘
               ↓
    ┌──────────┼──────────┐
    ↓          ↓          ↓
┌──────┐  ┌──────┐  ┌──────┐
│Rank 0│  │Rank 1│  │Rank 2│  (Tensor Parallel)
└──────┘  └──────┘  └──────┘
```

**mini-sglang：多进程分布式系统**

```
┌──────────────┐
│ API Server   │ ← FastAPI，OpenAI接口
└──────┬───────┘
       │ ZeroMQ
       ↓
┌──────────────────────────────┐
│  Tokenizer Worker Process    │
│  - 文本 → Token              │
└──────┬───────────────────────┘
       │ ZeroMQ
       ↓
┌──────────────────────────────┐
│  Scheduler Worker (Rank 0)   │
│  - 接收请求                   │
│  - 调度决策                   │
│  - 广播到其他 Rank            │
└──────┬───────────────────────┘
       │ NCCL + ZeroMQ
       ↓
┌──────────────────────────────┐
│  Scheduler Workers (Rank 1-N)│
│  - 每个 GPU 一个进程          │
│  - 本地 Engine 执行           │
└──────┬───────────────────────┘
       │ ZeroMQ
       ↓
┌──────────────────────────────┐
│  Detokenizer Worker Process  │
│  - Token → 文本               │
└──────┬───────────────────────┘
       │
       ↓
┌──────────────┐
│ API Server   │ ← 流式返回结果
└──────────────┘
```

**关键差异**：
- **nano-vLLM**：所有组件在一个进程，简单但扩展性有限
- **mini-sglang**：进程隔离，组件独立部署，可水平扩展

### 2.2 通信机制

| 组件 | nano-vLLM | mini-sglang |
|------|-----------|-------------|
| **控制消息** | Python 方法调用 | ZeroMQ |
| **张量数据** | Shared Memory | NCCL + ZeroMQ |
| **进程间通信** | multiprocessing.Queue | ZMQ Socket |
| **流式返回** | 无 | 支持流式输出 |

**示例：nano-vLLM 的简单通信**

```python
# model_runner.py:68-74
def read_shm(self):
    assert self.world_size > 1 and self.rank > 0
    self.event.wait()  # 简单的事件同步
    n = int.from_bytes(self.shm.buf[0:4], "little")
    method_name, *args = pickle.loads(self.shm.buf[4:n+4])
    self.event.clear()
    return method_name, args
```

**示例：mini-sglang 的消息系统**

```python
# message/backend.py
@dataclass
class BatchBackendMsg(BaseBackendMsg):
    """批次请求消息，支持序列化"""
    uids: List[int]
    req_ids: List[int]
    input_ids: List[int]
    ...
```

---

## 3. 核心组件对比

### 3.1 代码结构

**nano-vLLM（~2,000 行）**：
```
nanovllm/
├── llm.py                # 100 行：LLM API
├── config.py             # 50 行：配置
├── sampling_params.py    # 30 行：采样参数
├── engine/
│   ├── llm_engine.py     # 100 行：主引擎
│   ├── scheduler.py      # 80 行：调度器
│   ├── model_runner.py   # 250 行：模型执行
│   ├── block_manager.py  # 120 行：块管理
│   └── sequence.py       # 90 行：序列状态
├── layers/               # 500 行：神经网络层
├── models/               # 400 行：模型实现
└── utils/                # 200 行：工具函数
```

**mini-sglang（~5,000 行）**：
```
minisgl/
├── server/
│   ├── api_server.py     # 300 行：FastAPI 服务器
│   ├── launch.py         # 200 行：进程启动
│   └── args.py           # 150 行：CLI 参数
├── scheduler/
│   ├── scheduler.py      # 500 行：调度器
│   ├── prefill.py        # 200 行：Prefill 管理
│   ├── decode.py         # 150 行：Decode 管理
│   ├── table.py          # 200 行：页表管理
│   ├── cache.py          # 100 行：缓存接口
│   └── utils.py          # 100 行：工具函数
├── kvcache/
│   ├── radix_manager.py  # 400 行：Radix Cache
│   ├── naive_manager.py  # 150 行：朴素缓存
│   ├── base.py           # 100 行：缓存接口
│   └── mha_pool.py       # 200 行：MHA 池
├── engine/
│   ├── engine.py         # 400 行：执行引擎
│   ├── graph.py          # 150 行：CUDA Graph
│   └── sample.py         # 100 行：采样
├── attention/            # 300 行：注意力后端
├── layers/               # 600 行：神经网络层
├── models/               # 500 行：模型实现
├── kernel/               # 800 行：CUDA Kernels
├── message/              # 400 行：消息定义
└── tokenizer/            # 300 行：分词器
```

### 3.2 核心类对比

#### 3.2.1 序列/请求表示

**nano-vLLM：Sequence 类**

```python
# sequence.py:14-84
class Sequence:
    """简单的序列状态"""
    block_size = 256

    def __init__(self, token_ids: list[int], sampling_params):
        self.token_ids = token_ids          # 完整 token 序列
        self.block_table = []               # 块表
        self.num_cached_tokens = 0          # 缓存命中数
        self.status = SequenceStatus.WAITING
```

**mini-sglang：Req 类**

```python
# core.py
@dataclass
class Req:
    """更丰富的请求状态"""
    uid: int                               # 唯一 ID
    req_id: int                            # 请求 ID
    input_ids: torch.Tensor                # 输入（在 GPU 上）
    parent: Optional[Req] = None           # 父请求（用于 Chunked Prefill）
    filling_status: Literal["prefill", "decode"] = "prefill"

    # 采样参数
    sampling_params: SamplingParams = field(default_factory=SamplingParams)

    # 状态管理
    decoded_tokens: List[int] = field(default_factory=list)
    output_ids: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    completion_tokens_wo_eos: int = 0

    # KV Cache 引用
    cache_handle: Optional[BaseCacheHandle] = None
```

**关键差异**：
- **nano-vLLM**：数据在 CPU，序列化时传输
- **mini-sglang**：数据在 GPU，避免频繁传输

#### 3.2.2 缓存管理

**nano-vLLM：BlockManager**

```python
# block_manager.py:26-113
class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        self.blocks = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id = {}          # 哈希 → 块ID
        self.free_block_ids = deque(range(num_blocks))

    def allocate(self, seq: Sequence):
        """分配块，支持前缀缓存"""
        ...
```

**mini-sglang：RadixCacheManager**

```python
# kvcache/radix_manager.py:87-100
class RadixCacheManager(BaseCacheManager):
    def __init__(self, device: torch.device):
        self.root_node = RadixTreeNode()    # Radix 树根节点
        self.evictable_size = 0             # 可驱逐大小
        self.protected_size = 0             # 受保护大小

    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False):
        """锁定/解锁缓存块"""
        ...
```

**关键差异**：
- **nano-vLLM**：哈希表查找，O(1) 复杂度
- **mini-sglang**：Radix 树，支持最长前缀匹配，更灵活

---

## 4. 缓存机制对比

### 4.1 nano-vLLM：PagedAttention + 哈希缓存

**核心思想**：
1. 将 KV Cache 分成固定大小的块（256 tokens）
2. 使用哈希表记录已计算的块
3. 新请求查找哈希表，复用相同块

**代码示例**：

```python
# block_manager.py:59-83
def allocate(self, seq: Sequence):
    h = -1
    cache_miss = False

    for i in range(seq.num_blocks):
        token_ids = seq.block(i)

        # 计算哈希（滚动哈希）
        if len(token_ids) == self.block_size:
            h = self.compute_hash(token_ids, h)
        else:
            h = -1  # 不完整的块不缓存

        # 查找缓存
        block_id = self.hash_to_block_id.get(h, -1)

        # 验证（避免哈希冲突）
        if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
            cache_miss = True

        if cache_miss:
            # 分配新块
            block_id = self.free_block_ids[0]
            block = self._allocate_block(block_id)
        else:
            # 命中缓存！
            seq.num_cached_tokens += self.block_size
            block = self.blocks[block_id]
            block.ref_count += 1
```

**特点**：
- ✅ 简单高效，O(1) 查找
- ✅ 引用计数支持共享
- ❌ 只支持完整块匹配
- ❌ 哈希冲突需要验证

### 4.2 mini-sglang：Radix Cache

**核心思想**：
1. 将所有请求的 KV Cache 组织成 Radix 树
2. 树的每个节点代表一个 token 前缀
3. 支持部分匹配和自动分裂

**代码示例**：

```python
# kvcache/radix_manager.py:13-80
class RadixTreeNode:
    def __init__(self, tic: int | None = None):
        self.children: Dict[int, RadixTreeNode] = {}  # 子节点
        self._parent: RadixTreeNode | None = None
        self.ref_count: int = 0

        # KV Cache 数据
        self._key: torch.Tensor      # Key cache
        self._value: torch.Tensor    # Value cache
        self._length: int            # 节点长度

    def get_match_len(self, input_ids: torch.Tensor) -> int:
        """计算与输入的匹配长度"""
        from minisgl.kernel import fast_compare_key
        return fast_compare_key(self._key, input_ids)

    def _split_at(self, pos: int) -> RadixTreeNode:
        """在位置 pos 分裂节点"""
        assert 0 < pos < self.length
        parent = self.parent

        # 创建新节点（前半部分）
        new_node = RadixTreeNode(self.timestamp)
        new_node.set_key_value(self._key[:pos], self._value[:pos])
        new_node.set_parent(parent)
        new_node.ref_count = self.ref_count

        # 当前节点保留后半部分
        self.set_key_value(self._key[pos:], self._value[pos:])
        self.set_parent(new_node)

        return new_node
```

**使用示例**：

```python
# 场景：两个请求共享前缀
req1_tokens = [1, 2, 3, 4, 5, 6]
req2_tokens = [1, 2, 3, 7, 8, 9]

# Radix 树结构：
# Root
#  └─ [1,2,3]           ← 共享节点，ref_count=2
#      ├─ [4,5,6]       ← req1 的独有部分
#      └─ [7,8,9]       ← req2 的独有部分
```

**特点**：
- ✅ 支持部分前缀匹配（更灵活）
- ✅ 自动分裂和合并
- ✅ 内存利用率更高
- ❌ 实现复杂度高
- ❌ 查找 O(tree_depth) 复杂度

### 4.3 性能对比

| 场景 | nano-vLLM | mini-sglang |
|------|-----------|-------------|
| **完全匹配** | O(1)，极快 | O(depth)，较快 |
| **部分匹配** | 不支持 | O(depth)，自动处理 |
| **内存开销** | 哈希表 | Radix 树 |
| **缓存命中率** | 中等 | 高（更细粒度） |

**示例：部分匹配**

```python
# nano-vLLM：无法利用部分匹配
seq1 = [1, 2, 3, 4, 5, 6, 7, 8]  # 块 0: [1-8]
seq2 = [1, 2, 3, 9, 10, 11, 12]  # 块 0: [1-7] 不匹配
# 结果：缓存未命中，完全重新计算

# mini-sglang：自动匹配最长前缀
req1 = [1, 2, 3, 4, 5, 6, 7, 8]
req2 = [1, 2, 3, 9, 10, 11, 12]
# 结果：自动匹配 [1,2,3]，只需计算 [4,5,6,7,8] 和 [9,10,11,12]
```

---

## 5. 调度策略对比

### 5.1 nano-vLLM：简单抢占调度

**代码**：`scheduler.py:24-58`

```python
def schedule(self) -> tuple[list[Sequence], bool]:
    # 阶段1：Prefill（优先处理新请求）
    while self.waiting and num_seqs < self.max_num_seqs:
        seq = self.waiting[0]

        # 检查资源
        if num_batched_tokens + len(seq) > self.max_num_batched_tokens:
            break
        if not self.block_manager.can_allocate(seq):
            break

        # 调度
        self.block_manager.allocate(seq)
        scheduled_seqs.append(seq)

    if scheduled_seqs:
        return scheduled_seqs, True  # Prefill 模式

    # 阶段2：Decode（生成新 token）
    while self.running and num_seqs < self.max_num_seqs:
        seq = self.running.popleft()

        # 抢占逻辑
        while not self.block_manager.can_append(seq):
            if self.running:
                self.preempt(self.running.pop())  # 抢占最长的
            else:
                self.preempt(seq)
                break
        else:
            scheduled_seqs.append(seq)

    return scheduled_seqs, False  # Decode 模式
```

**特点**：
- ✅ 简单直观
- ✅ 优先短请求
- ❌ Prefill 和 Decode 分离，效率损失
- ❌ 无法处理超长 prompt

### 5.2 mini-sglang：Chunked Prefill + 混合调度

**代码**：`scheduler/prefill.py`

```python
class PrefillManager:
    def __init__(self, cache_manager, table_manager, decode_manager):
        self.cache_manager = cache_manager
        self.table_manager = table_manager
        self.decode_manager = decode_manager

    def schedule(self, pending_reqs: List[Req], budget: int):
        """Chunked Prefill 调度"""
        ready_to_decode = []

        for req in pending_reqs:
            # 检查 Radix Cache 命中
            cache_handle = self.cache_manager.query(req.input_ids)

            if cache_hit_length >= len(req.input_ids):
                # 完全命中，直接进入 Decode
                req.filling_status = "decode"
                ready_to_decode.append(req)
            else:
                # 部分命中，Chunked Prefill
                remaining_budget = budget - used_tokens
                if remaining_budget <= 0:
                    break  # 预算用完

                # 计算这次 prefill 多少
                chunk_size = min(remaining_budget, max_prefill_length)
                new_chunk_end = cache_hit_length + chunk_size

                # 创建子请求（chunk）
                chunked_req = ChunkedReq(
                    parent=req,
                    chunk_start=cache_hit_length,
                    chunk_end=new_chunk_end
                )
                # 执行 prefill...
```

**特点**：
- ✅ **Chunked Prefill**：长 prompt 分片处理
- ✅ **混合调度**：Prefill 和 Decode 可以在同一批次
- ✅ **自适应**：根据预算动态调整 chunk 大小
- ❌ 复杂度高

### 5.3 Overlap Scheduling（mini-sglang 独有）

**原理**：CPU 调度与 GPU 计算重叠

```python
# scheduler.py:75-100
def _process_last_data(
    self, last_data: ForwardData | None, ongoing_data: ForwardData | None
) -> None:
    if last_data is None:
        return

    batch, (_, next_tokens_cpu, copy_done) = last_data[0].batch, last_data[1]
    copy_done.synchronize()  # 等待 GPU 完成

    # 在 GPU 计算的同时，CPU 处理结果
    for i, req in enumerate(batch.reqs):
        next_token_id = next_tokens_cpu[i]
        req.append_host(next_token_id.unsqueeze(0))

        # 准备下一轮的元数据（CPU 工作）
        self.table_manager.update(req)
        ...
```

**时间线对比**：

```
nano-vLLM：
CPU调度 → GPU计算 → CPU处理 → CPU调度 → GPU计算 ...
↑ 2ms    ↑ 8ms     ↑ 1ms    ↑ 2ms    ↑ 8ms
总延迟：11ms/step

mini-sglang (Overlap)：
CPU调度 → GPU计算
           ↑ 8ms           CPU处理 → CPU调度
                          ↑ 1ms    ↑ 2ms (与GPU并行)
总延迟：8ms/step (隐藏了3ms的CPU开销)
```

**性能提升**：~20-30% (CPU密集型场景)

---

## 6. 性能优化对比

### 6.1 Attention 后端

| 框架 | Prefill | Decode | 备注 |
|------|---------|--------|------|
| **nano-vLLM** | FlashAttention2 | FlashAttn KV Cache | 单一后端 |
| **mini-sglang** | FlashAttention2/3 | FlashInfer | 可配置不同后端 |

**mini-sglang 的灵活性**：

```bash
# Prefill 用 FA3，Decode 用 FlashInfer（H100 最优）
python -m minisgl --model "Qwen3-32B" --attn fa,fi

# 都用 FlashAttention2
python -m minisgl --model "Qwen3-32B" --attn fa

# 都用 FlashInfer
python -m minisgl --model "Qwen3-32B" --attn fi
```

**原因**：
- **FlashAttention3** (H100)：Prefill 极快，但 Decode 一般
- **FlashInfer**：Decode 优化好，Prefill 也快
- 组合使用达到最优性能

### 6.2 CUDA Graph

**nano-vLLM**：基础实现

```python
# model_runner.py:216-252
def capture_cudagraph(self):
    for bs in [1, 2, 4, 8, 16, 32, ...]:
        graph = torch.cuda.CUDAGraph()
        outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # warmup
        with torch.cuda.graph(graph):
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # capture
        self.graphs[bs] = graph
```

**mini-sglang**：更精细的控制

```python
# engine/graph.py
class CudaGraphRunner:
    def __init__(self, max_batch_size: int, capture_sizes: List[int]):
        self.max_batch_size = max_batch_size
        self.capture_sizes = capture_sizes

        # 预分配更大的内存池（支持动态大小）
        self.graph_pool = torch.cuda.graph_pool()

        # 多个 graph，每个对应不同的 batch size
        self.graphs = {}

    def replay(self, batch_size: int, *args):
        # 选择最接近的 graph
        graph_size = next(s for s in self.capture_sizes if s >= batch_size)
        graph = self.graphs[graph_size]

        # 只更新有效部分
        with torch.cuda.graph(graph):
            update_inputs(batch_size, args)
```

**差异**：
- **nano-vLLM**：固定大小，浪费内存
- **mini-sglang**：动态大小，内存效率高

### 6.3 自定义 CUDA Kernels

**nano-vLLM**：无自定义 kernel，完全依赖第三方库

**mini-sglang**：包含多个优化 kernel

```cpp
// kernel/csrc/jit/radix_cache.cu
// 自定义 Radix Cache 比较 kernel
__global__ void fast_compare_key_kernel(
    const int* key,
    const int* input_ids,
    int* match_len,
    int key_len,
    int input_len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 1) return;

    int len = min(key_len, input_len);
    for (int i = 0; i < len; i++) {
        if (key[i] != input_ids[i]) {
            *match_len = i;
            return;
        }
    }
    *match_len = len;
}
```

**优势**：
- GPU 原生实现，避免 CPU-GPU 传输
- 针对性优化，比通用实现快 2-5x

---

## 7. 代码复杂度对比

### 7.1 学习曲线

```
nano-vLLM:
└─ 入门时间：1-2 天
   ├─ 理解 PagedAttention：4 小时
   ├─ 理解 Scheduler：2 小时
   ├─ 理解整体流程：2 小时
   └─ 实验修改：1 天

mini-sglang:
└─ 入门时间：1-2 周
   ├─ 理解 Radix Cache：1 天
   ├─ 理解 Chunked Prefill：1 天
   ├─ 理解 Overlap Scheduling：1 天
   ├─ 理解多进程通信：2 天
   ├─ 理解 CUDA Graph：1 天
   └─ 实验修改：3-5 天
```

### 7.2 可维护性

| 维度 | nano-vLLM | mini-sglang |
|------|-----------|-------------|
| **模块数** | ~15 个 | ~40 个 |
| **依赖数** | ~10 个 | ~30 个 |
| **代码行数** | ~2,000 | ~5,000 |
| **注释率** | 低 | 高 |
| **类型标注** | 部分 | 完整 |

### 7.3 扩展性

**nano-vLLM**：
```python
# 添加新模型相对简单
# 1. 复制 models/qwen3.py
# 2. 修改模型结构
# 3. 实现 packed_modules_mapping
# 4. 完成！
```

**mini-sglang**：
```python
# 添加新模型需要考虑更多
# 1. 实现 models/xxx.py
# 2. 实现特定的 attention 配置
# 3. 处理分布式同步
# 4. 更新文档和测试
# 5. 性能调优（CUDA kernels）
```

---

## 8. 适用场景对比

### 8.1 nano-vLLM 最佳场景

✅ **强烈推荐**：

1. **学习 vLLM 原理**
   - 代码简洁，易于理解
   - 核心思想完整保留
   - 适合教学和研究

2. **快速原型开发**
   - 验证新算法
   - 实验新调度策略
   - 测试新优化技术

3. **小规模部署**
   - 单 GPU 或少量 GPU
   - 低并发场景
   - 对延迟要求不极端

4. **自定义需求**
   - 需要深度定制
   - 不需要完整 API 服务
   - 集成到现有系统

❌ **不推荐**：

- 生产环境高并发服务
- 需要流式输出
- 多用户在线服务
- 超长上下文（>32k）

### 8.2 mini-sglang 最佳场景

✅ **强烈推荐**：

1. **生产环境部署**
   - OpenAI 兼容 API
   - 高并发在线服务
   - 多用户 Chatbot

2. **高性能需求**
   - 对吞吐量要求高
   - 对延迟敏感
   - 需要极致优化

3. **复杂场景**
   - 超长上下文（>128k）
   - 多用户共享系统提示
   - 混合长度请求

4. **分布式部署**
   - 多 GPU 服务器
   - 需要水平扩展
   - 需要容错和监控

❌ **不推荐**：

- 快速学习和原型（过于复杂）
- 单机低并发（资源浪费）
- 预算受限（需要多 GPU）

---

## 9. 学习路径建议

### 9.1 初学者路径（1-2 周）

```
Week 1: nano-vLLM 深度学习
├─ Day 1-2: 理解整体架构
│   └─ 阅读 VLLM_TUTORIAL.md
├─ Day 3-4: 深入 PagedAttention
│   └─ 修改 block_size，观察效果
├─ Day 5-6: 理解 Scheduler
│   └─ 实现简单的 FIFO 调度
├─ Day 7: 运行和调试
│   └─ 使用 example.py，添加日志

Week 2: mini-sglang 对比学习
├─ Day 8-9: 理解 Radix Cache
│   └─ 对比 BlockManager
├─ Day 10-11: 理解 Chunked Prefill
│   └─ 测试不同 chunk size
├─ Day 12-13: 运行 benchmark
│   └─ 对比两个框架的性能
└─ Day 14: 总结和项目实践
```

### 9.2 进阶路径（1-2 个月）

```
Month 1: 深入优化
├─ Week 1-2: 性能分析
│   ├─ 使用 Nsight 分析 GPU 性能
│   ├─ 找到瓶颈并优化
│   └─ 对比不同 attention 后端
├─ Week 3-4: 实现新特性
│   ├─ Speculative Decoding
│   ├─ Quantization (INT8/FP8)
│   └─ LoRA 推理支持

Month 2: 生产实践
├─ Week 5-6: 部署优化
│   ├─ Docker 容器化
│   ├─ Kubernetes 编排
│   └─ 监控和日志
└─ Week 7-8: 自定义开发
    ├─ 添加新模型支持
    ├─ 实现自定义调度策略
    └─ 性能调优和 benchmark
```

### 9.3 实战项目建议

**项目1：对比不同缓存策略**

```python
# 在 nano-vLLM 中实现三种缓存
# 1. 无缓存（baseline）
# 2. 简单哈希缓存（当前实现）
# 3. Radix Cache（参考 mini-sglang）

# 测试场景：
# - 100 个请求，前缀长度 0-1000 随机
# - 测量吞吐量和缓存命中率
# - 绘制性能对比图
```

**项目2：实现 Chunked Prefill**

```python
# 在 nano-vLLM 中添加 Chunked Prefill
# 参考 mini-sglang 的实现

# 关键点：
# 1. 在 Scheduler 中添加 budget 管理
# 2. 将长 prompt 分成多个 chunk
# 3. 每个 Chunk 作为一个独立的 Sequence

# 测试：
# - 32k 长度的 prompt
# - 对比完整 prefill vs chunked prefill
# - 测量峰值内存和吞吐量
```

**项目3：添加 Speculative Decoding**

```python
# 在两个框架中都实现 Speculative Decoding
# 使用小模型（如 Qwen3-0.5B）作为 draft model

# 实现步骤：
# 1. draft model 生成 k 个候选 tokens
# 2. 验证模型并行验证
# 3. 接受/拒绝机制

# 对比：
# - 不同场景下的加速比
# - 不同 draft model 大小的影响
```

---

## 10. 总结

### 10.1 核心差异总结

| 维度 | nano-vLLM | mini-sglang |
|------|-----------|-------------|
| **学习曲线** | ⭐⭐ 低 | ⭐⭐⭐⭐ 高 |
| **代码复杂度** | ⭐⭐ 低 | ⭐⭐⭐⭐⭐ 高 |
| **性能** | ⭐⭐⭐ 好 | ⭐⭐⭐⭐⭐ 优秀 |
| **扩展性** | ⭐⭐ 中 | ⭐⭐⭐⭐⭐ 强 |
| **生产就绪** | ⭐⭐ 中 | ⭐⭐⭐⭐⭐ 强 |
| **社区支持** | ⭐ 小 | ⭐⭐⭐⭐ 大 |
| **文档完善度** | ⭐⭐ 中 | ⭐⭐⭐⭐ 好 |

### 10.2 选择建议

**选择 nano-vLLM 如果你**：
- 正在学习 LLM 推理系统
- 需要快速验证想法
- 部署规模小（单机/少用户）
- 希望完全理解和控制代码

**选择 mini-sglang 如果你**：
- 需要生产级部署
- 追求极致性能
- 需要完整 API 服务
- 有多 GPU 资源

### 10.3 最佳实践

**学习路径**：
```
1. 从 nano-vLLM 开始（1-2周）
   └─ 理解核心概念

2. 对比 mini-sglang（1-2周）
   └─ 学习高级优化

3. 动手实践（持续）
   └─ 实现自己的优化

4. 生产部署（按需）
   └─ 选择合适的框架
```

**开发建议**：
- 先用 nano-vLLM 验证想法
- 再在 mini-sglang 中实现生产版本
- 参考两者代码，取长补短

---

## 11. 参考资源

### 11.1 论文

- **PagedAttention**: "Efficient Memory Management for Large Language Model Serving with PagedAttention" (vLLM)
- **Radix Cache**: "Efficient LLM Inference with Radix Attention" (SGLang)
- **Chunked Prefill**: "Sarathi-Serve: Efficient LLM Serving over PCIe and NVLink Networks using Token-Chopping"
- **Overlap Scheduling**: "NanoFlow: A Microkernel-Based Inference System for Large Language Models"

### 11.2 项目链接

- **nano-vLLM**: https://github.com/tzular/mini-vllm
- **mini-sglang**: https://github.com/sgl-project/mini-sglang
- **vLLM**: https://github.com/vllm-project/vllm
- **SGLang**: https://github.com/sgl-project/sglang

### 11.3 工具

- **FlashAttention**: https://github.com/Dao-AILab/flash-attention
- **FlashInfer**: https://github.com/flashinfer-ai/flashinfer
- **Triton**: https://github.com/openai/triton

---

**祝学习顺利！🎉**

有任何问题欢迎继续交流。
