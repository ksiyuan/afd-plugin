# A5 SHMEM 重写 —— a2e/e2a 集成面盘点（preflight P6）

状态：**设计输入，随 preflight 推进更新** · 最后更新：2026-09-03

> 上游：[`A5_shmem_preflight.md`](A5_shmem_preflight.md) 的 P6。本文件把
> a2e/e2a 现在「直接算对端 window 地址 + MTE 搬运 + 裸 flag」的每一处，逐个映射
> 到 SHMEM「对称 offset + 引擎代劳 RMA + signal」。**不改代码，只盘点**，等
> P3/P4 门禁绿了照这张表动手。所有行号针对分支 `a5-custom-op-research`。

---

## 0. 现在的窗口是怎么来的（要整体替换的东西）

| 层 | 现状 | SHMEM 后 |
| --- | --- | --- |
| host tiling | [`a2e_tiling.cpp`](../../csrc/npu/a2e/op_host/a2e_tiling.cpp) `Mc2CcTilingConfig(groupEp, ALL_TO_ALL, "AlltoAll=...")` → `GetTiling(mc2InitTiling/mc2CcTiling1)`；A5 上还 `SetCommEngine(3)` | **删掉整个 MC2 tiling**。不再要 HCCL comm resource。 |
| host 建资源 | HCCL 运行时 `HcclAllocComResourceByTiling`（就是这步在 A5 混合组不给 `remoteRes`） | `aclshmemx_init_attr` + `aclshmem_malloc`（对称堆），一次性，连接器 init 时做 |
| kernel 取 window | `AscendC::GetHcclContext<HCCL_GROUP_ID_0>()` → `HcclOpResParam*`（树）或 `HcclCombinOpParam*`（扁平），`winBaseOf(rankId)` 从 `remoteRes[rankId].nextDevicePtr->windowsIn` 取对端基址 | op 新增一个输入参数 `GM_ADDR shmem_window`（`aclshmem_malloc` 的返回值，host 传入）；对端地址用**对称 offset**（`shmem_window + peer_stride` 或直接把对称指针交给 `aclshmem_putmem` 的 `pe` 参数）。**`winBaseOf` / `epWinContext_` / `epWinContextA5_` 整个删掉。** |
| 引擎选择 | 写死 MTE（`CpGM2GMPingPong` / `DataCopyPad` GM↔GM，AI core 直接寻址对端） | `aclshmem_putmem/getmem` 内部按 `state->topo_list[pe]` 选 SDMA→UDMA→MTE→ROCE。**A5 上期望走 UDMA/RDMA 引擎。** |

现在的三区窗口布局（`comm_args.h:50-51`，两 op 共用）：

```
[0,                 IPC_DATA_OFFSET=2MB)   flag 区
    · 每 rank 一段，stride OPT_RANK_OFFSET=512B（shareAddrs[i] = winBaseOf(i) + i*512）
    · magic tensor 在 IPC_DATA_OFFSET - blockNum*32 处（每 block 8×int32）
[IPC_DATA_OFFSET,    +100MB)                data 区（IPC_BUFF_MAX_SIZE=100MB）
    · a2e computeGate=1: [int32 batchSize][expertIds][expertScales][x]
    · a2e computeGate=0 / e2a: [x] 从 IPC_DATA_OFFSET 起
```

SHMEM 后这块布局大体保留（payload / flag 分区的思路和官方 doubleplane 例子的
payload+assist+ready+count 四区一致），但：

- flag 不再是「写对端 window 某 offset 的裸 uint64」，而是 `aclshmemx_signal_op`
  的 signal 变量（SHMEM 单独管理，`aclshmem_signal_wait_until` 配套）。
- 每 rank stride `OPT_RANK_OFFSET` 的手工分段可以去掉 —— SHMEM 对称堆天然
  「本 rank malloc + heap_size = 下一 rank 同一 buffer 的地址」（见
  `shmem-src/docs/principles_en.md`），peer 侧地址 = 本地对称指针 + `pe` 参数。

---

## 1. a2e 数据流（[`a2e.h`](../../csrc/npu/a2e/op_kernel/a2e.h)）

角色：`rank < expertRankSize` = FFN rank（收）；`rank >= expertRankSize` = attention rank（发）。
`attnToMoeRatio` = 一个 FFN rank 服务几个 attention rank。

### computeGate == 0（只搬 x）

| 步 | attention rank（send） | FFN rank（recv） |
| --- | --- | --- |
| 1 | `sendX`：`copyGmToGmWithBlocks(shareXGt, xGt, ...)` 把 x 写进**自己** window 的 data 区（`shareAddrs[rank] + IPC_DATA_OFFSET + xOffset`） | — |
| 2 | `SyncAll()` | — |
| 3 | `camCpUB2GM(shareFlagGt, flagLt, ...)` 把 `magic<<32` 写进**对端 FFN rank**（`shareAddrs[rank % expertRankSize]`）flag 区的某 offset | — |
| 4 | — | `waitFlagWithScalar(offset, magic)`：自旋 `DATA_FLUSH` + 读**自己** window flag == magic |
| 5 | — | `copyGmToGmWithBlocks(expandXGt, shareXGt, ...)`：从**对端 attention rank** window data 区（`shareAddrs[sendRank] + IPC_DATA_OFFSET + xOffset`）读 x → 输出 **← 这步 507035** |

### computeGate == 1（x + expertIds + expertScales + batchSize）

`sendWithMte` 按 `blockIdx` 分工：最后一个 block 发 expertIds，倒数第二发
expertScales，其余发 x；`sendExpertIds` / `sendExpertScales` / `sendBatchSize`
各自「写自己 data 区 → 写对端 flag → `SyncAll`」。`recvWithMte` 对称地
`waitFlagWithScalar` 三个 flag、从对端 data 区读三段。`recvExpertIdsWithMte`
额外走 UB 中转（`camCpGM2UB` 对端 → UB → `camCpUB2GM` 输出）。

---

## 2. e2a 数据流（[`e2a.h`](../../csrc/npu/e2a/op_kernel/e2a.h)）

角色反过来：`rank < expertRankSize` = FFN rank（发结果）；`rank >= expertRankSize` = attention rank（收结果）。

| 步 | FFN rank（send） | attention rank（recv） |
| --- | --- | --- |
| 1 | `sendWithMte`：对每个 attention rank，`copyGmToGmWithBlocks(shareXGt, expandXGt[off], ...)` 写**对端 attention rank** data 区（`shareAddrs[sendRank] + IPC_DATA_OFFSET`）**← 这步 507035** |
| 2 | `SyncAll()` | — |
| 3 | `camCpUB2GM(shareFlagGt, flagLt, ...)` 写**对端** flag 区起始（`shareAddrs[sendRank]`），值 `magic<<32` | — |
| 4 | — | `recvWithMte`：自旋 `camCpGM2UB` 读**自己** flag == `magic<<32` |
| 5 | — | `copyGmToGmWithBlocks(xGt, shareXGt, ...)`：从**自己** data 区读到输出 |

注意 e2a 的 507035 在 **send 侧**（FFN rank 写对端），a2e 在 **recv 侧**（FFN rank 读对端）—— 方向不同但都是「非本 rank 的 window 直接寻址」。

---

## 3. 替换表（逐原语）

| 现在 | 位置 | SHMEM 替代 |
| --- | --- | --- |
| `winBaseOf(rankId)` + `epWinContext_` / `epWinContextA5_` | a2e.h:160 / e2a.h:120 | **删除**。对称堆基址 = `aclshmem_malloc` 返回值（host 传进来的 `shmem_window`）。 |
| `shareAddrs[i] = winBaseOf(i) + i*OPT_RANK_OFFSET` | a2e.h:101-112 | **删除**。本地写 `shmem_window + local_off`；对端用 `pe` 参数寻址，不手工算地址。 |
| `copyGmToGmWithBlocks(shareXGt_own, xGt, ...)` 写自己 data 区 | a2e.h `sendX` | 本地 `DataCopy` 到 `shmem_window` 的 payload 区（本地操作，不变） |
| `copyGmToGmWithBlocks(out, shareXGt_remote, ...)` 读对端 data 区 | a2e.h `recvWithMte` | `aclshmem_getmem(local_dst, shmem_window + payload_off, bytes, src_pe)` + `aclshmemx_udma_quiet(src_pe)`（或让发送方 `aclshmem_putmem` 推过来，见下「推 vs 拉」） |
| `copyGmToGmWithBlocks(shareXGt_remote, expandXGt, ...)` 写对端 data 区 | e2a.h `sendWithMte` | `aclshmem_putmem(shmem_window + payload_off, local_src, bytes, dst_pe)` + `aclshmemx_udma_quiet(dst_pe)` |
| `camCpUB2GM(shareFlagGt_remote, flagLt=magic<<32, ...)` 写对端裸 flag | a2e.h:231 / e2a.h:198 | `aclshmemx_signal_op(sig_addr, magic, ACLSHMEM_SIGNAL_SET, dst_pe)` |
| `waitFlagWithScalar` / `recvWithMte` 自旋读自己 flag | a2e.h:206 / e2a.h:203 | `aclshmem_signal_wait_until(sig_addr, ACLSHMEM_CMP_EQ, magic)` |
| `DATA_FLUSH` / `DataCacheCleanAndInvalid` + `dsb` 手工 flush | a2e.h:20 | SHMEM signal/quiet 内部管一致性；大概率可删，P4 验证 |
| `SyncAll()`（block 间） | 多处 | 保留（block 内同步，与 transport 无关）；跨 rank 用 `aclshmem_barrier(team)` |
| `magicTensor_` 原子自增拿 magic | a2e.h:86-98 | 保留思路（每次调用换 magic 避免 ABA），或用 SHMEM signal 的 ADD 模式累计 |
| `IPC_DATA_OFFSET` / `OPT_RANK_OFFSET` / `shareAddrs[]` 布局常量 | comm_args.h:50-76 | 重定义为对称窗口内的分区 offset（payload / signal 两区起点） |
| 数据面「推」还是「拉」 | a2e 拉 / e2a 推 | **统一成推**（`put` + `put_signal`），对上官方 doubleplane 例子的 `put_signal_nbi`（数据+旗标一条 WQE），少一次 round-trip |

---

## 4. host 侧改动

### op 定义 / 原型（`op_host/a2e_def.cpp`、`a2e_proto.cpp`、`aclnn_a2e.{cpp,h}`）

- [ ] 新增输入 tensor `shmem_window`（`GM_ADDR`，`aclshmem_malloc` 返回的 device 指针）。
- [ ] 新增 attr：`shmem_pe`（本 rank 在 SHMEM global team 里的 mype）、
      `shmem_team_id`（若用子 team）。
- [ ] 现有 attr `group_ep` / `aiv_num` 保留；`rank` / `expertRankSize` /
      `attentionRankSize` 保留（kernel 里的角色判断还要用）。

### tiling（`a2e_tiling.cpp` / `e2a_tiling.cpp`）

- [ ] 删 `Mc2CcTilingConfig` / `SetCommEngine` / `mc2InitTiling` / `mc2CcTiling1`
      （连同 `AFD_TILING_HAS_COMM_ENGINE` 和 `AFD_A5_COMM_ENGINE` env）。
- [ ] `A2ETilingData` 去掉 `mc2InitTiling` / `mc2CcTiling1` 字段。

### 算子编译（`csrc/npu/CMakeLists.txt`、`build_aclnn.sh`）

- [ ] 加 SHMEM 头（`-I<shmem>/include`）和库（`-laclshmem` 或静态）。
- [ ] 新开关 `AFD_BUILD_SHMEM_OPS` / `AFD_A5_SHMEM=1`，与现有 `AFD_A5_*` 编译期
      开关并列；910C 路径完全不受影响（`AFD_A5_SHMEM` 未定义时旧 MC2 路径不变）。

### 连接器（[`camp2p.py`](../../afd_plugin/connectors/npu/camp2p.py)）

- [ ] `initialize()` 里、建完 `afd` / `afd{i}` PG 之后（camp2p.py:319-333），
      调 `aclshmemx_init_attr`。bootstrap 用**已建好的 `afd_pg`** 做 UID
      广播（rank0 的 ip/port/magic）—— 不要再起一套 TCP。
- [ ] `aclshmem_malloc(window_bytes)` 一次，所有 rank 同步同尺寸。返回指针存进
      `custom_states`，传给 `torch.ops.afd_ascend.a2e/e2a`。
- [ ] `close()` 里 `aclshmem_free` + `aclshmem_finalize`。
- [ ] `a2e()` / `e2a()` 调用点（camp2p.py:552 / 605、847...）：参数表去掉
      `hccl_comm_name2/3`（DBO 的 afd1/afd2 组），加 `shmem_window` +
      `shmem_pe`。**2-ubatch DBO 怎么和单一对称堆共存要单独设计**（见开放问题）。

---

## 5. team 映射（关键，容易错）

a2e/e2a 的 rank 语义（`a2e_tiling.cpp:69`）：`0 <= rank < expertRankSize + attentionRankSize`，
前 `expertRankSize` 个是 FFN，后 `attentionRankSize` 个是 attention。
`rank % expertRankSize` 把 attention rank 映到它对应的 FFN rank。

SHMEM global team（`ACLSHMEM_TEAM_WORLD`）：`mype` = 0..npes-1。
如果 `afd` PG 的 rank 编号和 a2e 的 `rank` 一致（都来自 `world_rank`），那
**global mype == a2e rank**，直接用，不需要子 team。

- [ ] 确认 `camp2p.py` 里 `self.world_rank` 的编号规则 == a2e 传入的 `rank`
      （`ATTR_ENUM_RANK`）。看 `afd_plugin/v1/worker/` 里 a2e 调用处怎么填 rank。
- [ ] 若一致 → SHMEM 用 global team，`src_pe` / `dst_pe` 直接是 `rank + k*expertRankSize`
      / `rank % expertRankSize`，零转换。
- [ ] 若不一致 → `aclshmem_team_split_strided` 建子 team，或维护一张
      `a2e_rank → global_mype` 映射表传进 kernel。

---

## 6. 窗口对称化

- attention rank 需要：x 的 payload（`batchSize * hiddenSize * sizeof(T)`）
  ± expertIds/scales。
- FFN rank 需要：`recvBatchSize * hiddenSize` × `attnToMoeRatio` 段。
- 两者不等，但 `aclshmem_malloc` 要求所有 PE 同尺寸 → **按 max 分配**，
  多出的部分闲置。现有上界有两个：`comm_args.h` 的 `IPC_BUFF_MAX_SIZE = 100MB`
  和 a2e/e2a kernel 类常量 `IPC_BUFF_MAX_SIZE_MUL_EXP = 800MB`（kernel 实际按
  这个上界寻址）。对称 window 取 800MB（+ 2MB flag）沿用，对 A5 HBM 无压力；
  实际用量按 `batchSize/hiddenSize/attnToMoeRatio` 算，够了可调小。
- signal 区：每个 (peer, 用途) 一个 int32 signal slot。a2e computeGate=1 每
  peer 4 个 flag（batchSize/expertIds/expertScales/x）× `attnToMoeRatio` peer。
  按 `CAM_MAX_RANK_SIZE` 上界预留，和现在 2MB flag 区一个量级。

---

## 7. 开放问题（P3/P4 之后定）

1. **2-ubatch DBO 与单对称堆**：现在 afd / afd1 / afd2 是 3 个 HCCL 组轮流用。
   SHMEM 对称堆是一份。方案：一个 window 内切 `PING_PONG_SIZE` 份、按 ubatch
   轮转 signal namespace；还是 `aclshmem_malloc` 分配 2× 大小、ubatch A/B 用不同
   半区。倾向后者，简单。
2. **推 vs 拉 + put_signal**：a2e 现在是「拉」（FFN rank 主动读）。改「推」
   （attention rank `put` + `put_signal`）能少一跳，但要 attention rank 知道
   FFN rank 的接收布局。官方 doubleplane 例子是推。
3. **relay 要不要**：`aclshmemx_udma_relay_put_nbi` 多链路分流提带宽，需
   `ACLSHMEM_RELAY_SUPPORT=ON` + `feat/ascend950-relay-barrier` 分支。先不用，
   P4 单链路够了再说。
4. **`computeGate` 门控 gate 在 attention 侧**（`README` 已知 gap）—— SHMEM 重写
   不改这个，维持 FFN 侧算 gate。
5. **quiet 粒度**：`aclshmemx_udma_quiet(pe)` per-peer，还是攒一批再 quiet。
   影响流水，P4 量一下。

---

## 参考

- [`A5_shmem_preflight.md`](A5_shmem_preflight.md) —— P0–P6 门禁与分工
- [`A5_custom_op_investigation.md`](A5_custom_op_investigation.md) —— 根因、SHMEM 证据
- `shmem-src/src/device/gm2gm/shmem_device_rma.hpp` —— `aclshmem_putmem/getmem` 引擎分发
- `shmem-src/src/device/gm2gm/shmem_device_so.hpp` —— `aclshmemi_udma_put_signal_nbi`
- `shmem-src/examples/{dispatch,combine}/*_doubleplane/` —— 四区窗口 + signal 协议骨架
- `shmem-src/docs/principles_en.md` —— init、对称 malloc、`aclshmem_team_split_strided`
