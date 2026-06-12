# cm_sdpa_vlen 优化计划（Xe3/PTL）

基线：`seq=6864, sub_seq=3432, 16h/16kvh/d64` → **~9.5 ms**

平台：PTL 4xe（Xe3，等价于 Xe2 —— 存在 `CM_HAS_LSC_UNTYPED_2D`，
`CM_GRF_WIDTH=512`，`REG_N=16`，`REG_K=16`，`REG_M=8`）。

当前激活的内核路径：`sdpa_kernel_lsc_prefetch`（通过 `USE_LSC_PREFETCH=1`）。

---

## 当前 prefetch 路径的工作方式

每次 kv 迭代中，对于 `d=64`（`padded_head_size/REG_K = 4` 个分块）：

**K 阶段**（内层循环 `ri=0..3`）：
```
ri=0: cm_prefetch K_next[row=wg_local_id, col=0 ],  cm_load<Normal> K_curr[col=0 ], DPAS
ri=1: cm_prefetch K_next[row=wg_local_id, col=16],  cm_load<Normal> K_curr[col=16], DPAS
ri=2: cm_prefetch K_next[row=wg_local_id, col=32],  cm_load<Normal> K_curr[col=32], DPAS
ri=3: cm_prefetch K_next[row=wg_local_id, col=48],  cm_load<Normal> K_curr[col=48], DPAS
```
每个线程预取下一块 K tile 的 1 行（`kv_step/wg_local_size = 1`）。
K_next 的预取分布在整个 K 阶段，提前量较合理。

**Softmax + 转置**（位于 K 阶段和 V 阶段之间）

**V 阶段**（内层循环 `k=0,16,32,48`）：
```
k=0:  cm_prefetch V_next[row=wg_local_id, col=0 ],  cm_load<VNNI> V_curr[col=0 ], scale rO, DPAS
k=16: cm_prefetch V_next[row=wg_local_id, col=16],  cm_load<VNNI> V_curr[col=16], scale rO, DPAS
...
```
V 的 chunk k 预取紧挨着该 chunk 的 `cm_load<VNNI>` 发出 ——
**tile 内提前量为 0**。下一次迭代的 chunk 0 在上一次 V 循环末尾已预取；
而 chunk 1–3 在循环内仍是冷数据。

**关于 K 冗余**：`b2dK` 会把完整的 `[kv_step × REG_K]` tile 读入每个线程的寄存器。
16 个线程都在加载同样的 512 字节 —— 硬件会把带宽去重到 1×，
但会消耗 16× 的 LSC load 指令槽位。

---

## 优化项

### 1. 用预计算映射数组替换 `need_wg_mapping`

**问题：** `need_wg_mapping=1` 分支会在每个 WG 的每个线程里，对 `cu_seqlens` 运行 while 循环 —— O(num_sequences) 开销被重复执行，并且每次迭代都做 SVM 读。

**修复：** 采用 `pa_multi_token.cm` 的 `blocked_q_starts_and_subseq_mapping` 模式。主机端一次性预计算一个扁平 `int32[2 × wg_count]` 数组：

```python
mapping = []
for i, (seq_start, seq_end) in enumerate(zip(cu_seqlens, cu_seqlens[1:])):
    seq_len = seq_end - seq_start
    for k in range((seq_len + wg_seq_len - 1) // wg_seq_len):
        mapping += [int(seq_start) + k * wg_seq_len, i]
wg_count = len(mapping) // 2
```

内核读取 2 个标量即可 O(1) 推导全部信息，无需分支：

```cpp
int block_start_pos = mapping[wg_id * 2];
int seq_id          = mapping[wg_id * 2 + 1];
int kv_start        = cu_seqlens[seq_id];
int kv_seq_len      = cu_seqlens[seq_id + 1] - kv_start;
int q_start         = block_start_pos + wg_local_id * q_step;
```

这也会移除 `need_wg_mapping` 内核参数、死代码保护（`wg_base > wg_id` 不可能为真），以及 Python 里的 `wg_count` 张量算术错误。

**文件：** `cm_sdpa_vlen.cm`（内核分发块）、`cmfla.py`（`__call__` 方法）。

---

### 2. 将 V prefetch 前移到 K 阶段（更早提前量）

**问题：** 当前迭代 chunk k 的 `prefetch_V` 紧挨 `cm_load<VNNI>`，没有提前量。tile 内 chunk 1–3 仍是冷数据。

**修复：** 在 K 内层循环（`ri=0..3`）中，把当前 `kv_pos` 的 V prefetch 与 `kv_pos+kv_step` 的 K prefetch 交织：

```cpp
for ri in 0..3:
    cm_prefetch K_next[ri*REG_K]     // 已有
    cm_prefetch V_curr[ri*REG_N]     // 从 V 循环挪到这里
    cm_load<Normal> K_curr[ri*REG_K]
    DPAS
```

随后 V 循环只保留 `cm_load<VNNI>` + scale + DPAS。借助完整 K 阶段 + softmax + 转置的延迟，V 在进入 V 阶段前已被 L1 预热。

**文件：** `cm_sdpa_common.hpp`，`sdpa_kernel_lsc_prefetch`。

---

### 3. 在 K/V 内层循环中做软件流水（load→DPAS，隐藏 L1 停顿）

**问题：** `cm_load K[ri]` 后紧跟 `DPAS(Kmat)`，会产生 load→use 相关停顿。V 循环同理。

**修复：** 对 `Kmat`（和 `Vmat`）做双缓冲，让 `ri+1` 的 load 与 `ri` 的 DPAS 重叠：

```cpp
matrix<half, num_K, REG_M*REG_K> Kmat_a, Kmat_b;
cm_load K[0] → Kmat_a;
for ri = 1..3:
    cm_load K[ri] → Kmat_b       // 提前发出，与 Kmat_a 的 DPAS 重叠
    DPAS(Kmat_a)
    swap(Kmat_a, Kmat_b)
DPAS(Kmat_a)                     // 最后一个 tile
```

V 循环同理（在第 2 项使 V 进入 L1 后，L1→寄存器延迟在 Xe2/Xe3 上仍约 20 cycles）。

代价：每线程多一个 `Kmat`/`Vmat` 寄存器缓冲。

**文件：** `cm_sdpa_common.hpp`，`sdpa_kernel_lsc_prefetch`。

---

### 4. 循环前预热 prefetch（`kv_pos=0` 冷启动）

**问题：** 循环前没有 V prefetch。第一轮迭代的 V 完全冷。

**修复：** 进入循环前，预取 `kv_pos=0` 的所有 V chunks，以及 `kv_pos=kv_step`（下一轮）的所有 K chunks：

```cpp
// 循环前预热
for ri in 0..3:
    cm_prefetch V[kv_pos=0, chunk=ri]
    cm_prefetch K[kv_pos=kv_step, chunk=ri]
```

结合第 2 项，可让 `kv_pos=0` 的 V 获得完整 K 阶段的提前量。

**文件：** `cm_sdpa_common.hpp`，`sdpa_kernel_lsc_prefetch`。

---

### 5. 将 `kv_pos=0` 从主循环剥离（去掉热路径分支）

**问题：** 循环内 `if (kv_pos == 0) ugemm_PV0 / else ugemm_PV1` 每轮都有分支，并阻碍编译器对 PV1 路径（除首轮外占 100%）做充分调度。

**修复：** 在循环前先执行 `kv_pos=0` 的 `ugemm_PV0`；主循环从 `kv_pos=kv_step` 开始并无条件走 `ugemm_PV1`。

```cpp
// 循环前处理 kv_pos=0（ugemm_PV0，无 rescale）
// 主循环 kv_pos = kv_step .. kv_stop（恒定 ugemm_PV1）
```

**文件：** `cm_sdpa_common.hpp`，`sdpa_kernel_lsc_prefetch`。

---

## GRF 占用分析（已修正）

该内核在线程大 GRF 模式下约使用 163 GRF/线程（每线程 256 GRF bank）。

**Xe3 有两种 GRF 模式**（实验确认）：
- `-Qxcm_register_file_size=256`：每 EU 4 个线程上下文，每线程 256 GRF（16 KB）→ 总计 64 KB
- `-Qxcm_register_file_size=512`：每 EU 2 个线程上下文，每线程 512 GRF（32 KB）→ 总计 64 KB

在 256-GRF 模式下，每线程始终占满一个 16 KB bank，与实际 GRF 使用量无关 —— `floor(256/163)` 不适用。163/256 GRF 只会浪费 93 GRF，不会驱逐其他线程上下文。

**仅切到 512-GRF 模式（q1，同每线程工作量）是中性的**：测得 9.390 ms 对比基线 9.386 ms。上下文数减半正好抵消潜在收益 —— 与“内核并非依赖上下文切换隐藏延迟”的判断一致。

GRF 预算（信息性）：
- `rO`：`float[4 × 2 × 8 × 16]` = 4096 bytes = **128 GRFs**（FP32）
- `rQ`：`half[4 × 16 × 16]` = 2048 bytes = **64 GRFs**（FP16）
- 其他（cur_max、cur_sum、St、P、Kmat、Vmat、描述符）：约 22 GRFs

**结论：** 第 6、7 项（head-split、按需重载 rQ）基于错误的占用模型，预期不会带来收益。第 6 项已尝试并回退 —— 在无占用收益下将 kv 循环成本翻倍，导致约 71% 回退。

---

### ~~6. 拆分 head 维度~~（已过时——前提错误）

已尝试并回退。kv 循环成本翻倍，且无占用收益（大 GRF 模式下每 EU 始终 4 线程上下文，与单线程 GRF 使用量无关）。

---

### ~~7. 按需重载 rQ~~（已过时——前提错误）

已放弃。原因同第 6 项。

---

## 状态与优先级

| # | 变更 | 状态 | 结果 |
|---|--------|--------|--------|
| 1 | 映射数组替换 `need_wg_mapping` | 完成 | GPU 性能约 0%，代码更简洁 |
| 2 | V prefetch 前移到 K 阶段 | 完成 | −1.7% 到 −6.2% |
| 3 | Load/DPAS 双缓冲（K） | 试了 2 次，已回退 | 两次都回退——4 线程上下文已能隐藏 L1 停顿 |
| 4 | 循环前预热 prefetch | 完成（与 2 一起） | 计入 2+4+5 的结果 |
| 5 | 剥离 kv_pos=0 / 统一 PV 路径 | 完成（与 2 一起） | 计入 2+4+5 的结果 |
| 6 | 拆分 head 维度（2 passes） | 尝试后回退 | +71% 回退——kv 成本翻倍、无占用收益 |
| 7 | 按需重载 rQ | 放弃 | 前提错误（占用不是瓶颈） |
| 8 | K 两步超前 prefetch + 额外预热 tile | 尝试后回退 | ~0%——并非 prefetch 受限 |
| 9 | `q_step` 提升到 32（每线程 2 行 Q） | 试了 2 次，已回退 | +16% 回退——256 GRF（spill）与 512 GRF（无 spill，2 ctx/EU）结论相同；exp 随 Q 行数线性增长，无摊销 |
| 10 | kv_step=32 | 放弃 | 数学比例相同——exp 也随 kv_step 线性增长；循环开销节省被 ALU 流水淹没 |
| 11 | FP16 softmax | 拒绝 | 精度损失不可接受 |
| 12 | 消除 St→P 转置（约 66 mov/iter） | 已测上界 | 2-seq 上潜力 −14%；转置在关键路径上；彻底消除受 softmax 归约布局限制 |
| 13 | 将 log2e 融入 Q 预缩放（`qscale = scale_factor * log2e`） | 完成 | 从 softmax 关键路径移除每 tile 16 次 mul；St 落在 log2 域，`cm_exp` 无需逐元素 ×log2e |
| 14 | KV 分块（`KV_BLK=2`）：将 rO rescale 在 2 个 tile 上摊销 | 完成 | rO rescale（64 mul）从每 tile 一次变为每 2 tile 一次；不同于 Q 行或 kv_step 翻倍，exp 次数不变（仍按 token） |
| 15 | `online_softmax_update_tree` 树形归约 | 完成 | BLK_ROWS=32 时 max/sum 归约深度从线性 31 降至 log₂(32)=5；循环携带依赖链更短 |
| 16 | `transpose_St_to_P_half`：shuffle 前先做 float→half 窄化 | 完成 | 先转 half，再做 4-pass GRF shuffle（16-bit）；每次 select 的数据通路宽度减半（约 32 次等效数据搬运 vs ~64） |
| 13–16 合并 | 第 13–16 项（commit 0621405） | **完成 —— 9.25 ms → 7.654 ms（−17%）** | 2-seq × 3432 目标；当前最优改进 |

## ASM 分析（每 kv 迭代，d=64）

主循环指令画像：
- **12 dpas**（K 阶段 8 + V 阶段 4），约 32 cy XMX-pipe（由 4 上下文隐藏）
- **17 exp**（`online_softmax_update` 中 SIMD16 串行链），约 43 cy MATH-pipe
- **66 mov** 用于 `Transpose2DMatrix(St→P)`，约 66 cy ALU（与 math 流水）
- **16 mul** 用于 rO rescale，约 16 cy ALU

MATH 管线（exp）与 XMX 基本平衡，均约 40–50 cy。

**为何 `wg_size=32` 无效**：`kv_step/wg_local_size = 16/32 = 0` → prefetch 描述符无效。且总线程数不变。

---

## 为什么“摊销 softmax”行不通 —— 修正后的分析

第 9 项（2 行 Q）和 batch-2-kv 思路都假设 softmax 是每次 kv 迭代的*固定*成本，这个前提错误。

`online_softmax_update(St[rows=kv_step, cols=q_step], ...)` 会在 SIMD-`q_step` 向量上执行 `kv_step` 次 exp。无论如何分块，exp 元素总数始终是 `kv_step × q_step`：

| 方案 | DPAS ops/iter | exp instr/iter | 比例 |
|--------|---------------|----------------|-------|
| 基线（q_step=16） | 12 × SIMD16 | 17 × SIMD16 | 1.0 |
| 2 行 Q（q_step→32） | 24 × SIMD16 | 34 × SIMD16 | 1.0 |
| batch-2-kv | 24 × SIMD16 | 34 × SIMD16 | 1.0 |

比例不变 —— **不存在摊销**。exp 与 DPAS 同步线性增长。第 9 项的回退（+14%）纯粹来自寄存器 spill；batch-2-kv 即便 GRF 更干净也仍无增益。

**`-Qxcm_register_file_size=512` + 第 9 项也已验证无效**（实测）：
Xe3 上 512-GRF 模式确实存在（2 contexts/EU，32 KB/thread），因此第 9 项可无 spill 编译。
结果：10.890 ms 对比基线 9.386 ms（+16%），与 256-GRF spill 结果（10.873 ms）一致。回退并非 spill 导致，而是数学比例不变。上下文从 4→2 的减半也无法带来隐藏延迟收益，因为内核受 math-pipe 限制，而非内存延迟限制。

---

## 真实瓶颈与剩余机会

每次 kv 迭代内核执行：
- **MATH**：17 次 exp × SIMD16 FP32 —— math pipe 上每条指令处理 1 组 SIMD16 exp
- **XMX**：12 次 dpas —— 由 4 上下文重叠隐藏
- **ALU**：66 mov（转置）+ 16 mul（rescale）—— 与 math 流水

要提升性能，只剩改变比例的杠杆：

### ~~10. kv_step=32~~（不值得尝试——数学比例相同）

**思路**：KV tile 加倍，使主循环迭代减半，用更多 DPAS 工作摊薄每轮循环开销。

**为何无效**：`online_softmax_update(St[rows=kv_step, cols=q_step])` 在 SIMD-`q_step` 向量上执行 `kv_step` 次 exp。**exp 与 kv_step 线性增长**：

| kv_step | exp/iter | loop iters（seq=3432） | exp total/seq |
|---------|----------|-------------------------|---------------|
| 16 | 17 | 215 | 3655 |
| 32 | 33 | 108 | 3564 |

按 KV token 的比例不变 —— 与第 9 项及 batch-2-kv 同样是死路。唯一节省是约 107 轮迭代 × ~25 条 ALU 指令（描述符设置、计数器、分支），但在 4 上下文模型下 ALU 与 MATH 流水重叠，几乎不可见。

**寄存器成本**：`St` 从 16→32 GRF，`Kmat` 从 4→8 GRF（总计约 +20 GRF，仍可放进 256）。实现也不简单：`kv_step` 需与 `REG_K` 解耦，转置需新建 `[32×16]→[16×32]` 重载，V 阶段的 P 矩阵 `[16,32]` 需拆成两个 `[16,16]` DPAS tile。

**结论**：无需实验直接放弃——数学比例分析已足够确定。

---

### ~~11. FP16 softmax~~（拒绝——精度损失不可接受）

可把 exp 指令数减半（每条指令 SIMD32 FP16，对比 SIMD16 FP32）。
已拒绝：生产负载中 FP16 softmax 中间值可能下溢/上溢，精度损失不可接受。

---

### 12. 消除显式 St→P 转置（移除约 66 mov/iter）

`Transpose2DMatrix(St, P)` 将 `float[16,16] St` 转为 `half[16,16] P`（转置 + fp32→fp16），每次 kv 迭代约生成 66 条 SIMD16 mov。
该步骤严格位于 `online_softmax_update` 之后（不能与 exp 流水重叠），在关键路径上增加串行 ALU 延迟。

**上界探针（实测）**：将 `Transpose2DMatrix` 替换为直接 float→half cast（结果错误，仅测性能）：

| 配置 | 基线 | 跳过转置 | Δ |
|--------|----------|----------------|---|
| 2 seqs × 3432 | 9.389 ms | 8.084 ms | **−14%** |
| 16 seqs × 512 | 1.801 ms | 1.605 ms | −11% |
| 128 seqs × 64 | 0.715 ms | 0.708 ms | −1% |
| 15 seqs × 3840 | 85.403 ms | 74.134 ms | **−13%** |

转置确实在关键路径上（并未被 4 上下文隐藏 —— 当 exp 与 mov 串行执行时，同一线程上 ALU 与 MATH 会竞争）。

**为何不能简单消除**：现有 `online_softmax_update` 是沿 kv 维度、在 `St[kv=16, q=16]` 上做 SIMD16 行归约。
若改写 K-DPAS 直接产出 `St_new[q=16, kv=16]`，则需按列对每个 q-row 归约 —— 16 条长度 16 的标量链，明显劣于 16 条 SIMD16 行操作。交换 DPAS 操作数相当于用 16 倍更长的 softmax exp 链，去换掉 66 条 ALU mov，净效果大概率回退。

**剩余可行路径**：寻找更快的转置实现。`Transpose_16x16` 当前是 4 passes × 16 SIMD16 mov = 64 mov。理论上若有合适 GRF select 模式，16×16 float→half 转置可接近 ~16–32 mov。可继续评估是否能借助 XMX 管线或其它 select 模式把 mov 数压到 32 以下，同时保留 float→half downcast。

**文件：** `cm_sdpa_common.hpp`（`Transpose2DMatrix` 调用）、`cm_attention_common.hpp`（`Transpose_16x16` 实现）。

---

### 13. 将 log2e 融入 Q 预缩放

**问题：** `online_softmax_update` 计算 `St[r] = cm_exp((St[r] - new_max) * log2e)`。
其中 `* log2e` 是每个 kv 行 16 次 SIMD16 FP32 乘法，在减法后串行落在 ALU 上，位于 softmax 关键路径。

**修复：** 在加载 Q 时预先乘 `qscale = scale_factor * log2e`（编译期常量）。
此时 `St = K @ Q^T` 已是 log2 缩放的点积，softmax 可改为 `cm_exp(St[r] - new_max)`，不再需要逐元素乘法。数学等价，仅是把常量折叠进 Q。

**收益：** 每 tile 从 softmax 关键路径移除 16 次 mul（每个 kv tile 实打实减少，不是摊销）。

**文件：** `cm_sdpa_common.hpp`（`sdpa_kernel_lsc_prefetch`，Q 加载段）。

---

### 14. KV 分块（`KV_BLK=2`）：将 rO rescale 在多个 tile 上摊销

**问题：** rO rescale（`rO[t] *= max_comp`）每个 kv tile 执行一次 —— 8（tiles）× 8（`REG_M` 行）= 每 tile 64 次 SIMD16 乘法，串行落在 ALU 上。

**为何这次可行而 9/10 不行**：第 9、10 项通过增加 Q 行或 kv_step 尝试摊销，但两者都会让 exp 数随 tile 大小线性增长，比例不变。这里把 `BLK_ROWS = KV_BLK × kv_step` 放大后，exp 数确实同比增长（每块 32 行 vs 16 行），但 rO rescale 是**每块一次**，与 `KV_BLK` 无关。rO rescale 不是每 KV token 成本，而是每次 online-softmax-update 成本。块大小翻倍意味着每次 rescale 覆盖更多 token，从而把单位 token 的 rescale 成本减半。

**收益：** 每 2 tile 执行 64 次 mul → 折算每 tile 32 次 mul。等价于每 tile 节省 32 次 mul，且位于串行 ALU 路径上。

**文件：** `cm_sdpa_common.hpp`（外层循环重构为 `kv_base += BLK_ROWS`）。

---

### 15. 在 `online_softmax_update_tree` 中使用树形归约

**问题：** 当 `BLK_ROWS=32` 时，`online_softmax_update` 中 max/sum 归约是 32 次 `cm_max`/`cm_add` 的线性链。依赖深度为 31，后条指令必须等待前条结果。

**修复：** `online_softmax_update_tree` 把成对元素先折叠到临时缓冲，再做平衡二叉树归约：深度 = log₂(32) = 5。编译器可在各独立 pair 内自由调度。

**约束：** `rows` 必须是 2 的幂（由 `static_assert` 保证）。
`BLK_ROWS = KV_BLK × kv_step` 且 `KV_BLK` 为 2 的幂时，对实用值（1、2、4）都满足。

**文件：** `cm_sdpa_common.hpp`（新增函数 `online_softmax_update_tree`）。

---

### 16. `transpose_St_to_P_half`：GRF shuffle 前先做 float→half 窄化

**问题：** 通用 `Transpose_16x16<float,half>` 在 32-bit float 数据上执行 4-pass `select<2,1,8,2>` shuffle。每次 select 搬运 16 元素 × 4 字节 = 512 字节/行，因此 4 个 pass 全程都以 32-bit 宽度跑 ALU 数据通路（总计 64 条 SIMD16 mov），最后才转 half。

**修复：** `transpose_St_to_P_half` 先做 float→half（16 次 SIMD16 窄化转换，约 16 cycles），再调用 `Transpose_16x16<half,half>`。此时每次 select 仅搬运 16 元素 × 2 字节 = 256 字节/行，数据通路压力减半。总成本约：~16（cast）+ 64（半宽 shuffle）≈ 80 个“半宽等效操作”，对比 64 个“全宽操作”；指令数近似持平，但更窄的数据搬运在 GRF bank 冲突下可能降低 ALU 停顿。

**说明：** 单独看该项的净收益不确定，需要实测；当前它与第 13–15 项一起包含在 commit 0621405。

**文件：** `cm_sdpa_common.hpp`（新增函数 `transpose_St_to_P_half`）。

---

## Roofline（15 seqs × 3840, d=64, 16h, PTL 4xe）

- 计算峰值：约 20 TFLOPS FP16 XMX
- 实测：约 10.6 TFLOPS（峰值 53%）
- 内存：3.77 GB，68 GB/s → 若纯带宽受限约 56 ms，实测为 85 ms
- 算术强度：240 FLOP/byte；拐点：约 295 FLOP/byte
- EU 始终 4 线程上下文（大 GRF 模式，硬件固定）
- **完成第 13–16 项后**（commit 0621405）：2-seq × 3432 从 9.25 ms → **7.654 ms（−17%）**；
  这是本轮优化中单次提交效果最佳的一次
