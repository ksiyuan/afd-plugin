# A5（Ascend 950）自定义算子 507035 定位：混合组拿不到远端窗口资源

状态：**HCCL 窗口路径已封死；主线 = SHMEM 对称内存重写（preflight P0–P6），B2 降为备用** · 最后更新：2026-09-03

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

### comm engine 扫值结果（2026-09-03，方案 a 已执行）

`AFD_A5_COMM_ENGINE=0..7` 全部实测，**没有任何一个值让 `remoteResNum > 0`**：

- 一部分值：`HcclAllocComResourceByTiling` 直接返回失败（tiling 阶段就报错，
  与不加 `SetCommEngine` 时的 `ret=5 HCCL_E_NOT_SUPPORT` 同类）。
- 另一部分值（含默认 3）：tiling 调用成功但**静默不分配**，kernel 一访问对端
  窗口即 507035，dump 里 `remoteResNum == 0` 不变。

→ **comm engine 这条路彻底封死。** 混合组在 A5 上通过
`HcclAllocComResourceByTiling` 拿跨卡窗口资源，无论哪个 engine 值都不成立。
自定义算子若要在 A5 复活，只剩**方案 b（SHMEM / 对称内存，绕开 HCCL 窗口树）**
一条路，且需要专门几周投入 + NPU。

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

| 方案 | 成本 | 概率 | 状态 |
| --- | --- | --- | --- |
| ~~a. comm engine 扫值~~ | ~1h | — | **已做，全灭**（0..7 无一分配，见上节）。路封死。 |
| **b. SHMEM / 对称内存改造** | 几周，要 NPU | 中 | **主线（2026-09-03 晚定）**。蓝本：CANN SHMEM 仓库（gitcode.com/cann/shmem）的 topo-aware device RMA（`aclshmem_putmem/getmem` 内部按 `topo_list` 选 SDMA→UDMA→MTE→ROCE），跨端由 UDMA/RDMA 引擎代劳、AI core 不寻址对端。不是 `shmem_ptr` 直读（A5 会崩）。启动前置见 [`A5_shmem_preflight.md`](A5_shmem_preflight.md)。 |
| c. 拆成 A5 认可的组 | 大 | 低 | 让 a2e/e2a 跑在一个均匀子拓扑上。混合组形状是根因，但 AFD 的 A/F 切分本质上就是非均匀，映射代价不明，暂缓。 |
| **d. 找 CANN / HCCL** | 阻塞 | — | 问法现在更硬：ascend950 上混合了 MC2 专家 rank 与非专家 rank 的组，`HcclAllocComResourceByTiling` **在 comm engine 0..7 全部取值下**都不给 `remoteRes`，是否有支持的配置？附本文档 dump + 扫值结果。**并行推进，不阻塞 b。** |
| e. 优化 B2 | 几天～，要 NPU | 高 | **降级为备用方案**。SHMEM 若在 preflight 门禁被判死（P3 只有 MTE 位 / P4 崩）再回到这条。2-ubatch overlap 仍是这条路的主杠杆。 |

**当前判断（2026-09-03 晚更新）**：**主线锁定 b（SHMEM 重写）**，B2 降为备用。
理由：B2 是串行往返，吞吐上限 `1/(t_attn+t_ffn+2·t_comm)`，优化只能压常数、
拿不回 `1/max(t_attn,t_ffn)` 的量级；SHMEM 走 AI core 轮询 IPC 窗口、可流水，
是唯一能追平 910C 自定义算子形态的路。d（找 CANN/HCCL）并行发出。
**下一步：执行 [`A5_shmem_preflight.md`](A5_shmem_preflight.md) 的 P0–P6。**

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

## SHMEM 重写的新证据（2026-09-03 晚）— cann/shmem 仓库分析

原方案 b（冷冻）的蓝本只有 vllm-ascend 里 `hccl_shmem.hpp` 的 `shmem_ptr`
片段。当天把真正的 **CANN SHMEM 仓库**（`gitcode.com/cann/shmem`，
OpenSHMEM 风格对称内存 RMA 库，master = v1.6.0，2026-07）扒下来做了分析：

- **git clone 在本机连 gitcode 会挂死**（智能协议的 pack 传输不通，ls-remote
  偶尔通）；改用 gitcode v5 API（`/api/v5/repos/cann/shmem/git/trees/<sha>`）
  + raw 直链（`raw.gitcode.com/cann/shmem/files/master/<path>`）拿到 **59 个
  核心文件**，本地快照在工作区根 `shmem-src/`（非仓库目录）。远端还有未合入
  master 的 `feat/ascend950-relay-barrier` 分支（A5 relay/barrier 仍在演进）。
- **device 侧 API**（可直接编进自定义算子，`#include "shmem.h"`）：
  `aclshmem_putmem/getmem`、逐类型 put/get、`_nbi` 异步、strided
  `iput/iget`、GlobalTensor 重载、单元素 `_p/_g`、atomic 与 signal。
- **引擎分发**（`src/device/gm2gm/shmem_device_rma.hpp`，按
  `device_state->topo_list[pe]` 位掩码**逐对端**选）：SDMA → UDMA → MTE →
  ROCE(RDMA)：
  - **MTE 路径 = 910C 老模式**：`aclshmem_ptr(src,pe)` =
    `p2p_device_heap_base[pe] + offset`，AI core 直接 DataCopy 读远端地址——
    与我们的 a2e/e2a 同样依赖"内核可寻址的对端基址"，**A5 上一样会 507035**。
    所以拿 `hccl_shmem.hpp` 的 `shmem_ptr`（MTE 直读）当 A5 蓝本是**错的**，
    那是 A3/910C 模式。
  - **UDMA 引擎（A5 新增）= 正确蓝本**：AI core 只把 WQE 塞进 SQ ring
    （`aclshmemx_udma_put/get_nbi`，PIPE_MTE3 暂存或 PIPE_S 标量），**跨端
    访问由 UDMA 引擎完成**，AI core 不碰远端地址；`quiet(pe)` 等完成。还带
    relay put/get（`relay_pe` 多链路分流，需 `ACLSHMEM_RELAY_SUPPORT=ON`）、
    `put_signal_nbi`（WRITE_WITH_NOTIFY，数据+旗标一条 WQE）、按 QP
    （`*_qp_*`，多 QP）、UDMA 原子操作。
  - **RDMA（A5 新增）**：backends = `xscale` / `hns_1825` / `in_die`
    （`src/device/gm2gm/engine/rdma_backends/`）；host 侧 QP/传输管理在
    `src/host/transport/device_rdma/`（fixed/dynamic_ranks_qp_manager）。
  - host 拓扑已支持 950：`aclshmemi_product_strategy.cpp` 走
    `driver/topo/950` + `atlas_950_1.json` + DCMI URMA EID（MESH/CLOS/ROCE
    平面），内存注册带远端 R/W 权限位 + MemoryKey（RDMA 式 MR）。
- **官方 MoE dispatch/combine 参考实现就在仓库里**：
  `examples/dispatch/dispatch_doubleplane/` 与
  `examples/combine/combine_doubleplane/` —— 用对称 `shmem_window`
  （payload + assist + ready + count 区）+ per-AIV-per-peer 循环 +
  `aclshmemx_signal_op` 旗标，功能上就是 a2e/e2a 的翻版。
  **勘误（读本地快照后）**：这两个 doubleplane 例子的数据面**硬编码
  `aclshmemx_sdma_put_nbi` + `aclshmemx_mte_put_nbi`**（doubleplane = SDMA 平面
  + MTE-direct 平面按字节阈值二选一），**含 MTE 直读平面 → A5 混合组上大概率
  仍 507035**。可抄的是协议骨架（四区窗口 + 信号三元组 + Stage 分段），数据面
  必须换成 topo-aware 的 `aclshmem_putmem/getmem`（内部按 `topo_list` 选
  SDMA→UDMA→MTE→ROCE）。这正是 preflight P3 要先测的。

**对方案 b 的意义**：
- 可行性显著上升：A5 上"AI core 参与、低延迟、自定义 dispatch/combine"的
  正确形态 = **SHMEM 对称内存 + UDMA/RDMA 引擎**，Huawei 自己给了示例实现。
- 改造方向修正：不是把 a2e/e2a 窗口读写换成 `shmem_ptr` 直读（A5 会崩），
  而是换成 **SHMEM device RMA（对称 offset + 引擎代劳）**，flag/count 协议可
  直接抄官方双平面示例。
- 前置仍需验证：CANN 9.1.0 是否随带 SHMEM 运行时（`source set_env.sh` +
  `-DSHMEM_RDMA=ON` 构建）、A5 上 `topo_list[pe]` 实际给 UDMA 还是 RDMA。
  拆成可执行清单 [`A5_shmem_preflight.md`](A5_shmem_preflight.md)。

## 下一步方案（2026-09-03 晚起 —— SHMEM 转主线）

### 主线 — b. SHMEM 重写

执行 [`A5_shmem_preflight.md`](A5_shmem_preflight.md) 的 P0–P6。分工：

**A5 机时（你）— preflight P1–P5，按顺序做，遇红即停**

- P1：CANN 9.1.0 是否随带 SHMEM 运行时 + 头（一条 `find` 命令，见清单）。
- P2：不带则从完整 cann/shmem 源码构建，确认 `ACLSHMEM_UDMA_SUPPORTED=1`。
- **P3（门禁）**：2-rank 1A+1F 跑 topo 探针，读 `state->topo_list[peer]` ——
  含 UDMA/ROCE 位 → 绿；只有 MTE 位 / 全 0 → 红，SHMEM 死，回退 e。
- **P4（门禁）**：最小 put+signal 往返，payload 校验通过且无 507035。
- P5：官方 doubleplane 示例跑一遍（参考，非门禁）。
- 探针的 host + kernel 代码我在 Windows 侧写好给你，你只 build + run。

**Windows 侧（我）— 不依赖 NPU 的准备**

- P0：尽量用 gitcode v5 API 补全 `shmem-src/` 快照（host/init 头 + 构建系统）。
- P3/P4 探针代码：host launch + `ShmemTopoProbe` / `ShmemPutSignalProbe` kernel。
- **P6：a2e/e2a 集成面盘点** —— `winBaseOf` / `shareAddrs[]` /
  `waitFlagWithScalar` / `sendX` / `sendExpertIds` / `sendExpertScales` /
  `sendBatchSize` / `recvWithMte` / `recvExpertIdsWithMte` 逐个映射到 SHMEM
  对称 offset + `aclshmem_putmem/getmem` + `aclshmemx_signal_op`；team 切分
  （A/F 混合组 → `aclshmem_team_split_strided`）；窗口尺寸对称化。
- 构建集成：`csrc/npu/CMakeLists.txt` + `build_aclnn.sh` 加 SHMEM 开关。

### 并行 — d. 升级问 CANN / HCCL（不阻塞 b）

带上本文档的 dump + "comm engine 0..7 全灭"结果，问：ascend950 上一个混了
MC2 专家 rank 和非专家 rank 的组，`HcclAllocComResourceByTiling` 在所有 comm
engine 取值下都不分配 `remoteRes`，有没有支持的 tiling / 建组配置？渠道：
CANN issue / HCCL 对接人 / 内部 Ascend 群。**顺带问 SHMEM 在这个混合拓扑上
`topo_list` 会给什么引擎**（等于让对方先回答 P3）。

### 备用 — e. B2 优化

仅当 SHMEM 在 P3/P4 门禁被判死才启动。三步（基准计时 → recv buffer 缓存 +
`batch_isend_irecv` + isend/irecv 重叠 → 连接器内 2-ubatch 乒乓 DBO-lite）
详见 git 历史里本节的旧版本（commit `0efec9c` 之前）。B2 吞吐上限是
`1/(t_attn+t_ffn+2·t_comm)`，优化只压常数。

### 构建参数备忘

`<A5 SOC 名>`：`build_aclnn.sh` 接受 `950` / `ascend950*` / `Ascend950*`，
以你们之前编 A5 算子用的值为准（950PR 一般 `ascend950`，950DT 是
`ascend950dt_9582`，`npu-smi info` 确认）。默认（树形 + dump）编译：

```bash
AFD_A5_DUMP_WINDOWS=1 AFD_BUILD_ASCEND_OPS=1 SOC_VERSION=<A5 SOC 名> \
  python -m pip install -v --no-build-isolation --no-deps -e .
```

---

## 参考

- [`A5_ADAPTATION.md`](A5_ADAPTATION.md) —— 前一轮调查，含驱动日志证据和时间线
- tag `backup/a5-debug-507035`（kernel 二分的调试提交）、
  `backup/a5-full-record`（B2 + 完整文档）
- `tools/npu_a5_probe.py`（在 `backup/a5-*` / commit `26da97a`）—— stage 2 纯
  EP vs stage 3 混合组，native `dispatch_v2` / `combine_v2`
- `tools/npu_a5_p2p_probe.py` —— B2 载体探针（2-rank 往返）
- vllm-ascend `csrc/mc2/dispatch_ffn_combine/op_kernel/utils/hccl_shmem.hpp`
  —— 树形窗口访问 + 非 `HCCL_COMM` 的 `shmem_ptr` 路径（注意：`shmem_ptr` MTE 直读是 A3/910C 模式，A5 蓝本应改用 SHMEM 的 UDMA/RDMA 引擎）
- CANN SHMEM（对称内存 RMA 库）`gitcode.com/cann/shmem`：master = v1.6.0；本地快照在工作区根 `shmem-src/`（59 文件）；device API `include/device/gm2gm/shmem_device_rma.h`；UDMA 引擎 `include/device/gm2gm/engine/shmem_device_udma.h`；官方 MoE 示例 `examples/dispatch/dispatch_doubleplane/`、`examples/combine/combine_doubleplane/`；未合入 master 的 `feat/ascend950-relay-barrier` 分支
- PR #295（`b089f3c`）—— A5 构建 + 扁平窗口 ABI；PR #303 —— Route B2
- A5 节点：只有 device 0/1 健康；重跑前先清残留 vLLM 进程
