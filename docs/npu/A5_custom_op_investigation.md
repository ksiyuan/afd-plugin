# A5（Ascend 950）自定义算子 507035 定位：混合组拿不到远端窗口资源

状态：**进行中的后续跟进** · 最后更新：2026-09-03

> 先读 [`A5_ADAPTATION.md`](A5_ADAPTATION.md)。那份文档记录了前一轮调查
> （分支 tag `backup/a5-debug-507035`、`backup/a5-full-record`），结论是
> a2e/e2a 自定义算子在 A5 上按现有实现不可行，并有驱动日志佐证。本文档在此
> 基础上把根因定位到更精确的层面，并给出"自定义算子是否还有救 + B2 怎么优化"
> 的判断，因为 Route B2（HCCL p2p）的吞吐达不到生产要求。

---

## 结论（2026-09-03）

**A5 的 `HcclAllocComResourceByTiling` 不会给 AFD 的"attention + FFN 混合组"
分配任何跨卡窗口资源。** 这不是 kernel bug，kernel 侧改不动。

- PR #276 / 不加 `SetCommEngine`：直接 `ret=5 (HCCL_E_NOT_SUPPORT)`
- PR #295 `SetCommEngine(3)`：不报错了，但**什么资源也没分配**
  （`remoteResNum == 0`）→ kernel 一访问对端窗口就 507035（MTE 越界）
- 纯 EP 组在 A5 上能正常分配（前一轮 probe 验证过 native MC2 可用）
- vllm-ascend 的 `dispatch_ffn_combine` 用**完全相同**的
  `Mc2CcTilingConfig(group, 8, "AlltoAll=...")`、**不加** `SetCommEngine`，
  在 A5 能跑

→ 卡点是**"组里混了非专家 rank"这个拓扑形状**，与算子实现、kernel 无关。

---

## 定位过程与证据

### dump 方法

`a2e.h` / `e2a.h` 里 `AFD_A5_DUMP_WINDOWS` 编译开关：在 block 0 上同时按
**两种解读**打印 HCCL 上下文——

- 树形（A3 结构，`HcclOpResParam`）：`localUsrRankId` / `rankSize` /
  `winSize` / `localWindowsIn` / `localWindowsOut` / `remoteResNum`，以及每个
  peer 的 `remoteRes[i].nextDevicePtr` 和 `windowsIn`
- 扁平（PR #276/#295 假设的 `HcclCombinOpParam`）：`windowsIn[i]` /
  `windowsOut[i]`

### dump 输出（A5，2-rank，1 Attention + 1 FFN）

```
a2e[rank=1] tree: localUsrRankId=1 rankSize=2 winSize=4296015872
                  localWindowsIn=403440000000 localWindowsOut=1200c0000000 remoteResNum=0
a2e[rank=1] tree: remoteRes[0].nextDevicePtr=0 windowsIn=0
a2e[rank=1] tree: remoteRes[1].nextDevicePtr=0 windowsIn=0
a2e[rank=1] flat: windowsIn[0]=403440000000  windowsOut[0]=4034c0080000
a2e[rank=1] flat: windowsIn[1]=1200c0000000  windowsOut[1]=120140080000

a2e[rank=0] tree: localUsrRankId=0 rankSize=2 winSize=4296015872
                  localWindowsIn=120b00000000 localWindowsOut=124000000000 remoteResNum=0
a2e[rank=0] tree: remoteRes[0..1].nextDevicePtr=0 windowsIn=0
a2e[rank=0] flat: windowsIn[0]=120b00000000  windowsIn[1]=124000000000
```

### 读数结论

1. **结构体没读错。** 前缀字段全部合理；vllm-ascend 的 MC2 在 A5 上用的是
   同一个 `HcclOpResParam` 布局且能跑通。`remoteResNum == 0` 是真实读数。
2. **扁平 ABI 是假的。** 每个 rank 的 `flat.windowsIn[1]` 正好等于该 rank
   自己的 `localWindowsOut`（偏移 40 撞偏移 40）。它从来就不是对端地址。
   `flat.windowsIn[0]` 恰好撞上 `localWindowsIn`（偏移 32），所以"本地能用、
   远端全崩"——正是把树形结构当扁平数组读的指纹。
3. **`remoteRes` 树是空的。** `remoteResNum == 0`，所有
   `remoteRes[*].nextDevicePtr == 0`。混合组根本没建立任何对端 transport。
4. 驱动日志：`A2e_..._mix_aiv+0x....`，8 个 AIV core 全 MTE 越界，与前一轮
   记录的签名一致。

---

## 自定义算子还剩的路

| 方案 | 成本 | 概率 | 说明 |
| --- | --- | --- | --- |
| **a. comm engine 扫值** | ~1h | 低 | 已加 `AFD_A5_COMM_ENGINE` 环境变量（tiling 里读 `std::getenv`，**不用重编**）。跑 `AFD_A5_COMM_ENGINE=0/1/2/4 AFD_A5_DUMP_WINDOWS=1 ...`，看有没有哪个值让 `remoteResNum > 0`。 |
| **b. SHMEM / 对称内存改造** | 几周，要 NPU | 中 | vllm-ascend 的非 `HCCL_COMM` MC2 路径用 `shmem_ptr(symmetricPtr, rank)`（CANN `shmem_api.h`）访问对端，**不走 HCCL 窗口树**。host 侧分配一块跨 rank 对称内存、作为算子输入传进去；kernel 里把 `winBaseOf` 换成 `shmem_ptr`。与 `HcclAllocComResourceByTiling` 无关。 |
| **c. 拆成 A5 认可的组** | 大 | ? | 让 a2e/e2a 跑在一个均匀子拓扑上。不清楚能否映射到 AFD 的 A/F 切分。 |
| **d. 找 CANN / HCCL** | 阻塞 | — | 问法很具体了：ascend950 上，一个混合了 MC2 专家 rank 和非专家 rank 的组，`HcclAllocComResourceByTiling` 不给 `remoteRes`，该用哪个 comm engine / tiling 配置才能分配跨卡窗口？附本文档的 dump。 |
| **e. 转而优化 B2** | 几天，要 NPU | 高 | 2-ubatch overlap 是主杠杆，见下文。 |

**当前判断**：华为联系不上、SHMEM 改造是几周的活，务实路线是 **e（优化 B2）**，
先花 1 小时做 **a（comm engine 扫值）** 当个便宜的侧注。

---

## B2 性能分析（代码走读，2026-09-03）

`RemoteFFNProxy.forward`（`model_executor/models/deepseek_v2.py`）**每个 MoE
层调一次**——DeepSeek-V2-Lite 26 层，V3 58 层。每次调用：

1. attention rank：`send_attn_output` → **阻塞 `dist.send`** 给对应 FFN rank
2. `maybe_apply_dbo_yield`
3. attention rank：`recv_ffn_output` → **阻塞 `dist.recv`**

FFN 侧对称：`recv_attn_output`（`attn_size > ffn_size` 时是**逐 peer 阻塞
`dist.recv` 的循环**）→ `torch.cat` → MoE → `send_ffn_output`（逐 peer 阻塞
`dist.send`）。

**瓶颈是串行，不是 HCCL 带宽。** 每个 decode step，attention rank 在整个 FFN
的 26~58 层计算期间完全空转，反之亦然——吞吐 ≈ `1/(t_attn + t_ffn + 2·t_comm)`，
而不该是 `1/max(t_attn, t_ffn)`。910C 自定义算子逻辑上做同样的往返，但走 AIV
core 轮询 IPC 窗口（无 host HCCL launch、无全设备 sync、可流水）。

### 快速优化（低风险，仍需 NPU 验证）

- `p2p_recv` 每次调用都 `torch.empty(shape)` → 按 `(shape, dtype, peer)`
  **预分配 / 缓存 recv buffer**。
- FFN 侧逐 peer recv 循环 → **`dist.batch_isend_irecv`**（1 次 launch 代替 N 次）。
  `vllm_ascend/eplb/eplb_updator.py` 已有这个用法。
- 阻塞 `send`/`recv` → **`isend`/`irecv` + 显式 `.wait()`**，让第 L 层的 send
  与第 L 层 recv buffer 的准备重叠。
- 确认 `p2p_send` 里的 `.contiguous()` 对正常路径是 no-op（hidden_states 一般
  已连续）。

### 主杠杆：2-ubatch overlap（DBO-lite）

FFN 算第 L 层 ubatch A 的时候，attention 算第 L 层 ubatch B。把 comm 延迟藏
起来，消掉大部分跨角色空转。原生 DBO 是已知的 A5 gap，但**连接器里做一个
B2 专用的两 ubatch 乒乓**可能比原生 DBO 简单得多（不需要自定义 yield 算子，
只是交替用已经按 ubatch 建好的 `afd` / `afd1` 组）。这是最有价值、且不依赖
任何外部输入的工作。设计时对着 `attention_model_runner` / `ffn_model_runner`
的 ubatch 处理。

### 先做基准（需要 NPU）

B2 目前只验证过 2-rank eager。优化前先在 A5 上量：

- decode TPOT / 吞吐：B2 vs 910C 自定义算子的数（同模型、同 A/F 切分）——
  差 10% 还是差 2 倍？
- 拆分 profile：每步 comm 字节数、`send`/`recv` 次数、sync 停顿、comm 是否
  与 FFN 计算串行（是——确认量级）。

harness：扩 `tests/e2e/runner.py`（`--device-backend npu`，加一个 A5 2-rank
场景）加一段 decode-only 计时循环，或照 `tools/npu_a5_p2p_probe.py` 写个独立
脚本。

---

## 本分支（`a5-custom-op-research`）的改动

已推送 `fork/a5-custom-op-research`：

```
66c291b  feat(npu): make A5 MC2 comm engine configurable via AFD_A5_COMM_ENGINE
4f26513  feat(npu): default A5 a2e/e2a to the A3 remoteRes-tree window ABI
cbf0562  docs(npu): restore A5 investigation record and p2p probe
b089f3c  [NPU] add A5/ascend950 support for A2E and E2A ops   （原 PR #295）
```

三个环境变量开关（910C 路径完全不受影响，`AFD_ARCH_A5` 未定义时全部失效）：

| 环境变量 | 作用 | 生效时机 |
| --- | --- | --- |
| *(不设)* | A5 走树形窗口路径（`remoteRes[peer].nextDevicePtr->windowsIn`），和 910C 一样。默认。 | 编译期 |
| `AFD_A5_FLAT_WINDOW_ABI=1` | 恢复旧的扁平 `HcclCombinOpParam.windowsIn[]` 解读（PR #295 行为）。 | 编译期 |
| `AFD_A5_DUMP_WINDOWS=1` | block 0 上 `printf` 两种解读的实际地址值。 | 编译期 |
| `AFD_A5_COMM_ENGINE=<n>` | 覆盖 A5 tiling 里 `SetCommEngine` 的值（默认 3）。 | **运行时**，不用重编 |

---

## 下一步具体命令（A5 节点）

### a. comm engine 扫值（先做这个，便宜）

先按默认（树形 + dump）编一次：

```bash
AFD_A5_DUMP_WINDOWS=1 AFD_BUILD_ASCEND_OPS=1 SOC_VERSION=<A5 SOC 名> \
  python -m pip install -v --no-build-isolation --no-deps -e .
```

然后**不重编**，逐个试 comm engine 值，各跑一次 2-rank 冒烟，看日志里的
`remoteResNum`：

```bash
for e in 0 1 2 4 5; do
  AFD_A5_COMM_ENGINE=$e <启动 2-rank 1A+1F 冒烟的命令>
  # grep 'remoteResNum' 看有没有 > 0
done
```

- 某个值让 `remoteResNum > 0` 且对端 `windowsIn` 是合理地址 → 自定义算子有救，
  把默认 comm engine 改成那个值
- 全都是 0 → comm engine 这条路封死，转 b 或 e

### e. B2 优化（主线）

1. 写基准脚本（我来），你在 A5 上跑 B2 2-rank eager decode 计时，拿到"差多少"
2. 快速优化（预分配 buffer + `batch_isend_irecv`）开分支改好，你验
3. 有基准数后评估 2-ubatch overlap 是否值得（大概率值）

`<A5 SOC 名>`：`build_aclnn.sh` 接受 `950` / `ascend950*` / `Ascend950*`，
以你们之前编 A5 算子用的值为准（950PR 一般 `ascend950`，950DT 是
`ascend950dt_9582`，`npu-smi info` 确认）。

---

## 参考

- [`A5_ADAPTATION.md`](A5_ADAPTATION.md) —— 前一轮调查，含驱动日志证据和时间线
- tag `backup/a5-debug-507035`（kernel 二分的调试提交）、
  `backup/a5-full-record`（B2 + 完整文档）
- `tools/npu_a5_probe.py`（在 `backup/a5-*` / commit `26da97a`）—— stage 2 纯
  EP vs stage 3 混合组，native `dispatch_v2` / `combine_v2`
- `tools/npu_a5_p2p_probe.py` —— B2 载体探针（2-rank 往返）
- vllm-ascend `csrc/mc2/dispatch_ffn_combine/op_kernel/utils/hccl_shmem.hpp`
  —— 树形窗口访问 + 非 `HCCL_COMM` 的 `shmem_ptr` 路径
- PR #295（`b089f3c`）—— A5 构建 + 扁平窗口 ABI；PR #303 —— Route B2
- A5 节点：只有 device 0/1 健康；重跑前先清残留 vLLM 进程
