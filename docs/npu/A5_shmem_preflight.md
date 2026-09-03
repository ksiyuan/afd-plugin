# A5 SHMEM 重写 —— 启动前置验证清单

状态：**P0/P1/P2 ✅ 绿（SHMEM 1.7.0 built，UDMA 编入，transport 依赖就位）；P3/P4 待跑** · 最后更新：2026-09-03

> 上游文档：[`A5_custom_op_investigation.md`](A5_custom_op_investigation.md)。
> SHMEM 重写是当前主线（B2 降为备用）。本清单把启动前置拆成可在 A5 节点逐条
> 执行的动作，每条给出**明确的通过/失败判据**。P3/P4 是可行性门禁：双绿则
> 正式进入重写实现；任一红则 SHMEM 在当前 A5 软件栈上不成立，回退备用方案
> e（B2 优化），并把红的读数写回本文件。
>
> 本地快照 [`shmem-src/`](../../../shmem-src/)（工作区根，59 文件）是 device 侧 +
> transport 侧的**子集**：**不含** `include/host/`(init/team/mem 头)、
> `scripts/`、`CMakeLists`、`examples/*/` 的 host 侧 launch 代码。写真正的 probe
> 需要完整仓（见 P0）。

---

## 结论先行：这次要回答的唯一问题

a2e/e2a 在 A5 上 507035 的根因是 **FFN rank 用 MTE 直接读 attention rank 的
window**（[`csrc/npu/a2e/op_kernel/a2e.h`](../../csrc/npu/a2e/op_kernel/a2e.h)
`recvWithMte` → `copyGmToGmWithBlocks(..., shareAddrs[sendRank] + ...)`），而混合组
的 `remoteRes` 是空的，`shareAddrs[remote]` 是野指针。

SHMEM 能不能救，取决于**在 A5 的这套混合拓扑上，SHMEM 的引擎选择
`state->topo_list[peer]` 会不会给出一个非 MTE 的引擎位（UDMA / ROCE）**。

- `shmem-src/src/device/gm2gm/shmem_device_rma.hpp`：`aclshmem_putmem/getmem` 按
  `topo_list[pe]` 位掩码依次尝试 **SDMA → UDMA → MTE → ROCE**。
- 若 A5 混合组只给 `ACLSHMEM_TRANSPORT_MTE` 位 → SHMEM 高层 API 内部还是走
  「AI core 直接寻址对端」→ **和 a2e/e2a 同样 507035，SHMEM 不解决问题**。
- 若给 `ACLSHMEM_TRANSPORT_UDMA` 或 `ACLSHMEM_TRANSPORT_ROCE` 位 → 跨端访问交给
  UDMA/RDMA 引擎，AI core 不碰远端地址 → **重写可行**。

**P3 就是这个判据的直接测量。P1/P2 是让 P3 能跑起来的前提。**

---

## P0 —— 准备完整 SHMEM 源码 ✅ 完成（2026-09-03，同事在 A5 节点）

- [x] A5 节点完整 clone + 构建 `gitcode.com/cann/shmem`。**版本 SHMEM 1.7.0
      (master)**，aarch64 / Ascend950 / CANN 9.1.0。
- [x] host init 头、构建系统、`.run` 安装包齐全。

判据：达成。

---

## P1 / P2 —— SHMEM 运行时 ✅ 绿（2026-09-03，同事在 A5 节点，从源码构建）

CANN 9.1.0 不随带 → 走 P2 从源码构建，产物齐全：

| 项 | 值 |
| --- | --- |
| 主库 | `/home/k00930897/shmem/install/shmem/lib/libshmem.so` |
| 伴随库 | 同目录 `libshmem_utils.so` + 2 个 bootstrap so —— **运行时要 `LD_LIBRARY_PATH` 指到 `install/shmem/lib/`**，否则 `libshmem_utils.so => not found` |
| 头文件 | `/home/k00930897/shmem/install/shmem/include/`（`shmem.h` / `shmem_host_init.h` / device+host 头齐全） |
| `.run` 安装包 | `/home/k00930897/shmem/ci/release/aarch64/SHMEM_1.7.0_linux-aarch64.run` |
| P3 需要的 host API 符号（`nm` 确认在 `libshmem.so` 里） | `aclshmemx_init_attr`、`aclshmemx_init_attr_with_buffers`、`aclshmemx_get_uniqueid`（UID bootstrap）、`aclshmem_malloc`、`aclshmem_my_pe`、`aclshmem_n_pes`、`aclshmem_finalize` |
| `ACLSHMEM_UDMA_SUPPORTED` | ✅ 已编入（`nm` 符号确认） |
| host transport 依赖 | ✅ `libhcomm.so`（CANN lib64）、`liburma.so.0` / `liburma_common.so.0`（`/usr/lib64/`）、`libdcmi.so` 全部 `ldd` 已解析 |

判据：**达成**。`libshmem.so` 可链接、UDMA 编入、transport 依赖就位 —— 这正是 P3
「跨端走 UDMA/RDMA 引擎」的依赖基础。

> ⚠️ P3/P4 探针仍需在 **A5 NPU 节点**跑（2-rank device 0/1）。"给 Windows 侧" 指
> 的是：把 `install/shmem/include/` 的 host 头拷给我，我照真实 API 写 probe 的
> host+kernel 代码，同事在 A5 build+run。

<details><summary>原 P1/P2 检查步骤（存档）</summary>

在 A5 节点执行：

```bash
# 1. CANN 版本
cat "$ASCEND_HOME_PATH/version.info" 2>/dev/null || \
  cat /usr/local/Ascend/ascend-toolkit/latest/version.info

# 2. 头文件
find "$ASCEND_HOME_PATH" /usr/local/Ascend -name "shmem.h" \
     -o -name "shmem_host_init.h" 2>/dev/null

# 3. 运行时 so（命名未定，全扫）
find "$ASCEND_HOME_PATH" /usr/local/Ascend \
     \( -name "*shmem*.so" -o -name "*aclshmem*.so" \) 2>/dev/null

# 4. 环境脚本里有没有 shmem 段
grep -ril shmem "$ASCEND_HOME_PATH"/../set_env.sh \
     "$ASCEND_HOME_PATH"/set_env.sh 2>/dev/null

# 5. pip 侧（有的 CANN 版本把 shmem 放 site-packages）
python -c "import site,glob,os; [print(p) for s in site.getsitepackages() for p in glob.glob(os.path.join(s,'**','*shmem*'),recursive=True)]"
```

判据：

- **绿**：找到 `shmem.h` + 一个 `*shmem*.so` + 头里的 `aclshmemx_init_attr` /
  `aclshmem_malloc` 声明齐全 → 跳过 P2，直接 P3。
- **黄**：只有头没有 so，或版本 < 仓库 v1.6.0 → 需要 P2（从源码构建）。
- **红**：CANN 9.1.0 完全不带、且 P2 也构建不出（缺 `hcomm` / `urma` /
  `dcmi` 依赖）→ SHMEM 路径在当前 A5 软件栈上不成立，写回 investigation 文档。

---

## P2 —— 从源码构建 SHMEM 运行时（仅当 P1 为黄）

依据 investigation 文档 + `docs/compilation_build_guide*.md`：

```bash
cd shmem/           # P0 的完整仓
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash scripts/build.sh -DSHMEM_RDMA=ON        # 具体 flag 名以 build 脚本为准
# A5 relay/barrier 若 P4 需要：额外 -DACLSHMEM_RELAY_SUPPORT=ON
```

关注点（构建日志里逐一确认）：

- [ ] SOC / arch：A5 是 `__NPU_ARCH__ == 3510`
      （`shmem_device_so.hpp:19` 用它 gate `ACLSHMEM_RDMA_V2_SUPPORTED`）。
      build 脚本的 SOC 名取值同 `build_aclnn.sh`（`ascend950` / `ascend950dt_9582`，
      `npu-smi info` 确认）。
- [ ] `ACLSHMEM_UDMA_SUPPORTED` 是否被打开（`shmem_device_rma.hpp:22` 默认 0）。
      **这个宏关着的话 P3 里 UDMA 分支根本不会编进去** —— 是 P2 的头号检查项。
- [ ] host 侧 transport 依赖：`hcomm` (`dl_hcomm_def.h`)、URMA/DCMI
      (`aclshmemi_product_strategy.cpp` 走 `driver/topo/950` + `atlas_950_1.json`
      + DCMI URMA EID)。缺任一 → 记录缺什么。
- [ ] `driver/topo/950/atlas_950_1.json` 在 A5 的 driver 安装路径下存在
      (`hal.get_driver_install_path() + "/driver/topo/950/"`)。

判据：产出可链接的 `libaclshmem*.so` + `ACLSHMEM_UDMA_SUPPORTED=1`。

</details>

---

## P3 —— 混合拓扑的引擎选择探针（决定性测试）

**这是可行性门禁的核心。** 2-rank，device 只用 0/1，1 Attention + 1 FFN，
复现 AFD 的混合组形状。

### probe 要做的事

host 侧（参照完整仓 `examples/dispatch_gmm_combine/` 的 host launch）：

1. 用 PyTorch 起 2 进程（bootstrap：rank0 广播 UID(ip/port/magic)，同 AFD 连接器
   建 `afd` 组的方式），各绑 device 0 / 1。
2. `aclshmemx_init_attr(bootstrap_flags, &attr)`，`attr.local_mem_size` 两 rank
   一致。
3. `aclshmem_malloc(window_bytes)` —— 两 rank **同步、同大小**（对称性前提，
   见 `principles_en.md`：malloc 必须所有 PE 同步同尺寸）。
4. launch 一个只做打印的 kernel（下）。

kernel 侧：

```cpp
#include "shmem.h"
extern "C" __global__ __aicore__ void ShmemTopoProbe(GM_ADDR win) {
    if (AscendC::GetBlockIdx() != 0) return;
    __gm__ aclshmem_device_host_state_t* st = aclshmemi_get_state();
    int me = aclshmem_my_pe();
    int npes = aclshmem_n_pes();
    for (int pe = 0; pe < npes; ++pe) {
        AscendC::printf("SHMEM probe[me=%d] topo_list[%d]=0x%02x  "
            "(SDMA_sup=%d UDMA_sup=%d)\n",
            me, pe, (unsigned)st->topo_list[pe],
            ACLSHMEM_TRANSPORT_SDMA_SUPPORTED, ACLSHMEM_UDMA_SUPPORTED);
    }
    // p2p_device_heap_base[pe] 是否非空（MTE 直读路径是否「看起来可用但会崩」）
    for (int pe = 0; pe < npes; ++pe) {
        AscendC::printf("SHMEM probe[me=%d] p2p_heap_base[%d]=%p rdma_heap_base[%d]=%p\n",
            me, pe, st->p2p_device_heap_base ? st->p2p_device_heap_base[pe] : (void*)0,
            pe, st->rdma_device_heap_base ? st->rdma_device_heap_base[pe] : (void*)0);
    }
}
```

`ACLSHMEM_TRANSPORT_*` 位常量定义在快照没拉到的头里（`shmem_common_types` 附近
或 `host/shmem_host_def.h`）；完整仓拿到后把实际数值填到下表。

### 读数与判据

| `topo_list[peer]`（对端 rank，非自己） | 含义 | 门禁 |
| --- | --- | --- |
| 含 `UDMA` 位 | 跨端走 UDMA 引擎，AI core 不寻址对端 | **绿。** 进 P4 |
| 只含 `ROCE` 位 | 跨端走 RDMA 引擎 | **绿（次选）。** 进 P4，注意延迟 |
| 含 `SDMA` 位 | 同 die/同 super-node 的 SDMA | 绿，但确认 A/F 跨节点时还成立 |
| **只含 `MTE` 位** | AI core 直接寻址对端 heap | **红 —— 与 a2e/e2a 同根因，SHMEM 不解决。** 写回读数，回退备用方案 e |
| 全 0 / peer 不可达 | 混合组没建立 transport | **红。** 等价于 `remoteResNum==0`，SHMEM init 没为这个拓扑建链 |

附加观察：

- `p2p_device_heap_base[peer]` 非空但 `topo_list` 只有 MTE 位 → 正是「本地看着能
  用、远端一访问就 507035」的指纹（同 investigation 文档 flat-ABI 分析）。
- `rdma_device_heap_base[peer]` 非空 → RDMA 平面已注册，P4 可试 ROCE 路径。
- 若 init 本身就失败（返回非 0）→ 记录错误码，多半是 `product_strategy` 没识别到
  A5 mainboard_id（`aclshmemi_product_strategy_t::create` 返回 nullptr）。

---

## P4 —— 最小 put + signal 往返（a2e/e2a 数据面的缩影）

P3 绿之后做。验证「对称窗口 + 引擎搬运 + 旗标同步」这条链在 A5 混合组上真的通。

### 要覆盖的原语（对应 a2e 的哪一步）

| a2e/e2a 现在的做法 | SHMEM 替代 | probe 动作 |
| --- | --- | --- |
| attn rank `copyGmToGmWithBlocks` 写自己 window | 本地写 `aclshmem_malloc` 的 buffer | rank0 填 payload |
| attn rank `camCpUB2GM` 写 FFN rank window 的 flag | `aclshmemx_signal_op(sig, val, SET, peer)` | rank0 发信号给 rank1 |
| FFN rank `waitFlagWithScalar` 自旋读自己 window | `aclshmem_signal_wait_until(sig, EQ, val)` | rank1 等信号 |
| FFN rank `copyGmToGmWithBlocks` 读 attn rank window | `aclshmem_getmem(local, remote_sym, n, peer)` 或对端 `aclshmem_putmem` | rank1 校验 payload |

推荐直接用 **topo-aware 高层 API**（`aclshmem_putmem` / `aclshmem_getmem` /
`aclshmemx_signal_op` / `aclshmem_signal_wait_until`），让 SHMEM 内部按 P3 测出的
`topo_list` 选引擎 —— **不要**直接调 `aclshmemx_mte_put_nbi`（官方
`dispatch_doubleplane` / `combine_doubleplane` 例子就是硬编码 SDMA+MTE，
**不是 A5 跨节点的正确蓝本**，见下）。

### 判据

- **绿**：rank1 校验 payload 全对，无 507035，`quiet` 正常返回。**→ 重写基本 de-risk。**
- **红**：507035 / 校验错位 / `signal_wait_until` 永久 hang → 记录在哪个原语上崩，
  这决定重写是「换 API 就行」还是「连协议都要重设计」。

---

## P5 —— 官方 dispatch/combine 示例（参考，非门禁）

快照里的 `examples/dispatch/dispatch_doubleplane/dispatch_doubleplane_kernel.cpp`
和 `examples/combine/combine_doubleplane/combine_doubleplane_kernel.cpp`：

- 用 `aclshmemx_sdma_put_nbi` + `aclshmemx_mte_put_nbi` +
  `aclshmemx_signal_op` + `aclshmem_signal_wait_until` + `aclshmem_quiet`。
- **doubleplane = SDMA 平面 + MTE-direct 平面**，按字节数阈值二选一。
- 因为含 MTE-direct 平面，**在 A5 混合组上大概率复现 507035**。

所以：

- [ ] 先按原样跑一遍，确认失败点与 P4 一致（增强信心，不是门禁）。
- [ ] 真正要抄的是**协议骨架**（payload / assist / ready / count 四区窗口布局、
      per-AIV-per-peer 循环、`aclshmemx_signal_op` 三元组、Stage1/2/3 的
      `aclshmemi_sync_core_soft` 分段），把数据面搬运换成 P4 验证过的 topo-aware
      高层 API。
- [ ] `examples/dispatch_gmm_combine/`（完整仓）可能是更贴近 AFD 的版本，P0 拿到后比对。

---

## P6 —— AFD 集成面盘点（Windows 侧，无需 NPU）

**详细盘点见 [`A5_shmem_integration_map.md`](A5_shmem_integration_map.md)**（a2e/e2a
数据流、逐原语替换表、host 侧改动、team 映射、开放问题）。以下是 checklist 摘要：

与 P1–P5 并行，为重写实现做准备：

- [x] 盘点文档 `A5_shmem_integration_map.md` 初稿完成（2026-09-03）
- [ ] **init 时机 / bootstrap**：SHMEM `aclshmemx_init_attr` 需要 rank0 UID 广播。
      对上 [`afd_plugin/connectors/npu/camp2p.py`](../../afd_plugin/connectors/npu/camp2p.py)
      建 `afd` / `afd1` 组的地方 —— 复用同一个 torch PG 做 bootstrap。
- [ ] **team 切分**：AFD 的 A/F 混合组是非均匀拓扑。SHMEM 用
      `aclshmem_team_split_strided(ACLSHMEM_TEAM_WORLD, pe_start, pe_stride,
      pe_size, &team)`（`principles_en.md`）。厘清 attention rank / FFN rank 在
      global team 里的 `mype`，以及 a2e 的 `rank % expertRankSize` 映射到 SHMEM
      的哪个 team-level `mype`。
- [ ] **窗口尺寸**：a2e 现在 `IPC_BUFF_MAX_SIZE_MUL_EXP = 800MB`。SHMEM
      `aclshmem_malloc` 必须所有 PE 同步同尺寸 —— 混合组里 attention rank 和 FFN
      rank 需要的 window 大小不同，要按 max 对齐分配（对称性要求）。
- [ ] **替换点清单**（[`csrc/npu/a2e/op_kernel/a2e.h`](../../csrc/npu/a2e/op_kernel/a2e.h)）：
      `winBaseOf()` / `shareAddrs[]` / `waitFlagWithScalar()` / `sendX` / `sendExpertIds`
      / `sendExpertScales` / `sendBatchSize` / `recvWithMte` / `recvExpertIdsWithMte`
      —— 全是「直接算对端 window 地址 + MTE 搬运 + 裸 flag」，逐个映射到
      SHMEM 对称 offset + 高层 RMA + `signal_op`。e2a 同构。
- [ ] **构建集成**：`csrc/npu/CMakeLists.txt` 加 SHMEM 头/库路径，
      `build_aclnn.sh` 加开关（比照现有 `AFD_A5_*` 编译期开关的做法）。

---

## 门禁汇总

| 步骤 | 在哪做 | 通过判据 | 状态 |
| --- | --- | --- | --- |
| P0 完整源码 | A5 clone | 完整仓 + 构建脚本就位 | ✅ SHMEM 1.7.0 master |
| P1 运行时自带？ | A5 | `shmem.h` + `*shmem*.so` + init 声明 | ✅ 不随带 → P2 |
| P2 源码构建 | A5 | `libshmem.so` + `ACLSHMEM_UDMA_SUPPORTED=1` + transport 依赖 | ✅ 绿（2026-09-03） |
| **P3 引擎选择探针** | **A5 2-rank** | **`topo_list[peer]` 含 UDMA / ROCE / SDMA 位** | ⏳ 待跑（probe 代码待写） |
| P4 put+signal 往返 | A5 2-rank | payload 校验通过，无 507035 | ⏳ 待跑 |
| P5 官方示例 | A5 2-rank | （参考）失败点与 P4 一致 | ⏳ |
| P6 集成面盘点 | Windows | 替换点清单 + team 映射清楚 | ✅ [`A5_shmem_integration_map.md`](A5_shmem_integration_map.md) |

失败动作：P3 只有 MTE 位或全 0 / P4 崩 → 回退备用方案 e（B2 优化），写回读数。

**P3 + P4 双绿 = 正式进入 SHMEM 重写实现。** 否则回退备用方案 e（B2 优化）
+ 并行 d（问 CANN/HCCL）。

---

## 参考

- [`A5_custom_op_investigation.md`](A5_custom_op_investigation.md) —— 根因定位、
  comm engine 扫值、SHMEM 新证据
- `shmem-src/docs/principles_en.md` —— init 流程、对称 malloc、team 切分
- `shmem-src/src/device/gm2gm/shmem_device_rma.hpp` —— `topo_list` 引擎分发
  （SDMA→UDMA→MTE→ROCE）
- `shmem-src/src/device/gm2gm/shmem_device_so.hpp` —— `aclshmemi_udma_put_signal_nbi`
  （数据+旗标一条链）
- `shmem-src/src/host/transport/topo/rootinfo/aclshmemi_product_strategy.cpp` ——
  950 走 pod product（mesh + clos 双层），`atlas_950_1.json` + DCMI URMA EID
- `shmem-src/include/device/gm2gm/engine/shmem_device_udma.h` —— UDMA WQE/QP/quiet/
  relay API 全表
- `shmem-src/examples/{dispatch,combine}/*_doubleplane/` —— 协议骨架（数据面需改）
- A5 节点：只有 device 0/1 健康；重跑前先清残留 vLLM 进程
