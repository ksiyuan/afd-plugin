# AFD 在 Atlas A5 (Ascend950) 上的适配清单

> **当前结论（2026-09-01，路线已定）**：a2e/e2a **自定义算子路线在 A5 上确认不可行**。
> 根因：A5 HCCL 暴露的跨卡窗口地址（`HcclCombinOpParam.windowsIn[远端rank]`）在本卡
> **不可达**，内核读写即触发 **vector core exception（507035，MTE out-of-range）**，
> 已由驱动日志（`MTE_ERROR_T0_0=0x80000000`）铁证确认。
> **备选（原生 MC2 算子）也已排除**：torch_npu 原生 `npu_moe_distribute_dispatch_v2` /
> `npu_moe_distribute_combine_v2` 在 A5 上可用、参数校验通过，但 AFD 混合组
> （Attention rank 不持专家、收 0 token）在 dispatch kernel 上再次触发 **507035** ——
> 原生算子只支持“组内全为专家 rank”的标准 EP 配置，非对称混合组是设计盲区。
> **最终路线（B2）**：A5 连接器改用 **HCCL p2p 数据搬运**（移植 GPU 侧
> `P2pNcclAFDConnector` 模式）：Attention↔FFN 用 HCCL p2p 传 hidden states
> （路由在 FFN 侧算，`compute_gate_on_attention=false`），FFN 侧内部跑标准 EP MoE
> （原生算子只用于 FFN 组内标准配置，A5 已验证可用），见 ②。
> **状态（2026-09-01）**：B2 已实现并在 A5 **2-rank eager 跑通**（1 Attention + 1 FFN，
> DeepSeek-V2-Lite 本地模型 + completion 冒烟通过）。⏳ 待办：多 rank、ACL graph、DBO、910C 回归。

## 背景

- 目标硬件：**Atlas A5**，芯片 **Ascend 950PR**（CANN SOC 名 `ascend950`，架构 DAV_3510，
  arch 值 3510）；另一变体 950DT 的 SOC 名为 `ascend950dt_9582`。
- 现状基线（910C）：vLLM `0.26.0` + vLLM-Ascend commit `80d8c194f` + 配套 CANN/torch_npu
  （vLLM-Ascend v0.26.0 对应 **CANN 9.1.0 + PTA 2.10.0.post4**）。
  **验证记录只覆盖 Ascend 910C / Atlas A3**，但代码层面 `80d8c194f` 已含 ascend950/A5 支持（见 ①）。
- 分支：私有 fork `ksiyuan/afd-plugin` 的 `a5-adapt`；当前基于干净上游基线（PR #295 已回退）。

## 时间线（为什么自定义算子不可行）

1. **PR #276**（`SetCommEngine(MTE)` + A5 窗口 ABI）：`HcclAllocComResourceByTiling ret=5 (NOT_SUPPORT)`；
2. **B 修改**（移除 `SetCommEngine(MTE)`）：仍 `ret=5` → **comm engine 不是根因**，已排除；
3. **PR #295**（门控 `SetCommEngine` + `ascend950` 构建 + 扁平 `windowsIn[]` ABI）：
   `ret=5` 消失、**资源分配通过**，但 kernel 执行报 **507035**；
4. **内核二分**：本地窗口一切操作正常（magic / 8MB 拷贝 / flag 写），
   唯独**对远端窗口 `windowsIn[对端]` 的读写崩**（A 写崩 / F 读崩，多台机器复现）；
5. **驱动日志铁证**：`MTE_ERROR_T0_0=0x80000000`（MTE out-of-range），
   出错指令 `A2e_..._mix_aiv+0x26d4`，8 个 AIV core 全部 aivec 异常；
6. **结论**：a2e/e2a 的 910C 协议（写/读对端窗口做 ping-pong）在 A5 上不可行。

## 适配清单（按优先级）

### ① vLLM-Ascend 版本基线（代码已支持 A5，缺的是验证）

**已确认：commit `80d8c194f`（2026-08-01）本身已包含 ascend950/A5 支持**：

- 构建层：`CMakeLists.txt` 有 `ascend950` 分支，`Dockerfile.a5` / `Dockerfile.a5.openEuler` 存在，
  `csrc/scripts/util/const_var.py` 含 `"ascend950": "Ascend950"`；
- 运行时层：`vllm_ascend/attention/dsa_v1.py`、`vllm_ascend/device/mxfp_compat.py` 均有 Ascend950 分支；
- A5 支持由 `#7151`（Add support for Ascend950 chip）与 `#9271`（A5 custom op build）引入，均早于 `80d8c194f`。

| 项 | 现状 | 需要做的事 |
| --- | --- | --- |
| vLLM-Ascend 基线 | `80d8c194f` 代码层已支持 ascend950，AFD 补丁也锚定此版本 | **无需换版本**；在该版本上完整验证 A5 链路（正在做） |
| AFD 兼容补丁 | `compat/patches/npu/` 按 80d8c194f 内部接口编写 | 在 80d8c194f 上应用无问题；仅当升级 vLLM-Ascend 时逐条复核 |
| CANN / torch_npu | 实测 OPP 含 `ascend950`；目标配对 **CANN 9.1.0 + PTA 2.10.0.post4** | 用与 80d8c194f A5 路径互配的版本，并确认 `version.info` |

### ② Attention↔FFN 通信（A5 最终路线：HCCL p2p，即 B2）

**结论（已定）**：a2e/e2a 自定义算子依赖的“对端窗口直读写”协议在 A5 上不可行
（见时间线第 5 点，驱动日志 MTE out-of-range 铁证）。曾尝试改用 torch_npu 原生
`npu_moe_distribute_dispatch_v2` / `npu_moe_distribute_combine_v2`（vLLM-Ascend
`TokenDispatcherWithMC2` 的 A5 用法），但 AFD 混合组在 dispatch kernel 上再次触发
**507035**（见下“原生 MC2 混合组”记录）——原生算子不支持“组内含非专家 rank”的非对称
配置。**最终改为 HCCL p2p 数据搬运**：移植 GPU 侧 `P2pNcclAFDConnector` 模式，
Attention→FFN 用 HCCL p2p 传 hidden states + router_logits，FFN 内部用 `ffn_pg` 跑
标准 EP MoE（原生算子只出现在标准配置中），FFN→Attention 再 p2p 传回结果。

| 项 | 状态 | 备注 |
| --- | --- | --- |
| PR #295（`SetCommEngine` 门控 + 扁平窗口 ABI） | ✅ 已应用 → **已回退** | 自定义算子不再用于 A5，`csrc/npu` 回到 910C 上游状态 |
| **A5 连接器改用 HCCL p2p（B2）** | ✅ **已实现并验证** | 实现：`connectors/npu/camp2p.py` 4 个 A5 分支 + 新模块 `connectors/npu/camp2p_a5.py`（`is_a5` / rank 映射 / per-peer 计数 / p2p 原语）；`init_afd_connector` 在 A5 跳过自定义算子加载 |
| p2p 载体验证 | ✅ 已验证 | `tools/npu_a5_p2p_probe.py`：2 rank 往返 `MATCH=True`；`dist.send/recv` over 现有 `afd_pg` HCCL 组可用 |
| 2-rank eager 验证 | ✅ 已跑通 | 1A+1F，DeepSeek-V2-Lite 本地模型，completion 冒烟通过（A5 950PR） |
| FFN 内部 MoE | ✅ 走标准 EP | 原生算子只用于标准 EP 配置（A5 已验证可用），无混合组问题 |
| 910C 回归 | 自定义算子未改动 | ⏳ 待回归；910C 继续走 a2e/e2a 自定义算子，A5 分支走 p2p |

**记录：原生 MC2 混合组为何也失败（探针，已弃用）**
`tools/npu_a5_probe.py`（已删除，见 git 历史）曾在 A5 上用 `dispatch_v2`/`combine_v2`
模拟混合组：参数校验（H∈[1024,8192]、每 rank token∈[1,512]、`global_bs=max_bs*ep_world_size`）
全部通过，但 dispatch kernel 执行时触发 **507035**（全 64 AIV core 同一 PC、同一 para base
的 MTE 越界，与自定义算子失败签名一致）。溢出（rank 收 9 > 容量 8）与 0 接收两个假设
未最终定论，但足以判定：**原生算子的混合组路径风险高、不构成可靠方案**，故改走 p2p。

### ③ CAM 异步连接器（`CAMAsyncAFDConnector`）

| 项 | 现状 |
| --- | --- |
| CAM vendor 包 | 仓库仅含 `CAM_ascend910_93_openEuler_aarch64.run`（910C 专属），**A5 无对应包** |
| `umdk_cam_op_lib` wheel | `209.0.0b1`，是否支持 A5 未确认 |

**结论：`CAMAsyncAFDConnector` 在 A5 上不可用，除非拿到 ascend950 的 CAM 包。**
CAMP2p 是 A5 上唯一可行路径；其 A5 通信改用 HCCL p2p（见 ②）。

### ④ Python 层平台假设（基本无需改动）

- `validation.py` 仅拒绝 310P/XLite worker，接受标准 `NPUWorker` —— A5 用标准 NPUWorker，无需改。
- NPU 连接器 / model runner 依赖的 vLLM-Ascend 接口（`get_hccl_comm_name`、forward_context 等）
  在固定基线 `80d8c194f` 上无需改动；仅当升级 vLLM-Ascend 时随新版本复核。

### ⑤ 构建系统

| 项 | 现状 |
| --- | --- |
| `setup.py` | 默认 `SOC_VERSION=910c`；B2 路线 A5 不构建自定义算子，无需 `ascend950` 构建 |
| `csrc/npu/build_aclnn.sh` | PR #295 已回退；A5 走 p2p，无需自定义算子构建 |
| CANN 9.1 OPP 命名 | 实测 OPP config 只有 `ascend950`（无 `ascend910_95`） |

### ⑥ 文档与验证

- README / `docs/npu/*` / `recipe/npu/*` 基于 Ascend 910C / Atlas A3，需补充 A5 内容。
- ✅ A5 上 CAMP2p B2 **eager 已跑通**（2 rank）。⏳ 待做：多 rank、`FULL_DECODE_ONLY` ACL graph
  （p2p 暂不支持 graph 捕获，需后续专门处理）、DBO 两 ubatch，并记录结果。

## 排障记录：`HcclAllocComResourceByTiling ret=5`（PR #276/B 修改阶段，已闭环）

- **现象**：A5 (950PR) 上 `aclnnA2e` 失败，HCCL 对 tiling 通信资源分配返回 `ret=5 (NOT_SUPPORT)`。
- **已确认**：950PR 8 卡健康；OPP 只有 `ascend950`；`GetCurNpuArch()==3510`。
- **结论**：B 修改（去 `SetCommEngine`）后仍 `ret=5` → 显式 comm engine 不是根因；
  PR #295 用 `ascend950` 构建 + 门控 `SetCommEngine(3)` 后资源分配通过。

## 排障记录：`507035 vector core exception`（根因，已铁证闭环）

- **现象**：PR #295 下 a2e/e2a 执行崩，`torch.npu.synchronize()` 报 `507035`。
- **驱动日志**（`plog-17568_*.log`）：
  - `aivec error exception`, `error code=95`, 8 个 AIV core 全部异常；
  - **`MTE_ERROR_T0_0=0x80000000`**（MTE 越界访问）;
  - 出错指令 `A2e_..._mix_aiv+0x26d4`（module=MTE）;
  - fault kernel `A2e_...`, `device_id=1`, `stream_id=61`, `task_id=167`, `blockDim=8`;
  - dump: `/home/.../extra-info/data-dump/0/A2e_....o` 等。
- **内核二分结论**：本地窗口一切操作正常；**对远端窗口 `windowsIn[对端]` 的读写即崩**
  （机器 1 A 写崩、机器 2 F 读崩、干净代码也崩）。
- **根因**：A5 上 `windowsIn[]` 中对端条目**不是可访问的跨卡映射地址**
  （与 vllm-ascend 头文件注释相反），MTE 访问越界 → vector core exception。
- **决策**：放弃自定义算子 A5 路线，改用 torch_npu 原生 v2 算子（见 ②）。

## 新路线执行步骤（B2：HCCL p2p）

1. ✅ **p2p 载体探针**：2 rank 验证 `dist.send/recv` over 现有 `afd_pg` HCCL 组可搬运 tensor
   （`tools/npu_a5_p2p_probe.py`，往返 `MATCH=True`）；
2. ✅ 改 `connectors/npu/camp2p.py`：A5（`is_a5()`，即 `get_ascend_device_type()==A5`）分支把
   4 个 a2e/e2a 调用点换成 p2p —— Attention→FFN 传 hidden states，FFN→Attention 传回结果；
   per-peer 计数沿用现有 Gloo 控制面（DP metadata），rank 映射与 `_num_tokens_for_ffn_rank` 一致；
3. ✅ FFN 内部 MoE 走标准 EP（原生算子只出现在标准配置）；
4. ✅ 2 rank（1 FFN + 1 Attn）eager 验证语义已跑通 → ⏳ 多 rank → ⏳ 910C 回归（自定义算子路径不动）；
5. ⏳ `export HCCL_DEBUG_INFO=1` + `ASCEND_GLOBAL_LOG_LEVEL=1` 观察通信细节（按需）。

## 相关链接

- A5 路线决策与排障记录：本文档（自定义算子 507035 铁证 → 原生 MC2 混合组同样 507035 → B2 p2p）
- 已判不可行的自定义算子 PR：#276、#295（已回退；结论：A5 跨卡窗口地址不可达，MTE out-of-range）
- A5 p2p 载体探针：[`tools/npu_a5_p2p_probe.py`](../../tools/npu_a5_p2p_probe.py)
- A5 p2p 实现（新模块）：[`connectors/npu/camp2p_a5.py`](../../afd_plugin/connectors/npu/camp2p_a5.py)
- GPU 侧 p2p 参照实现：[`connectors/gpu/p2p.py`](../../afd_plugin/connectors/gpu/p2p.py)
- vLLM-Ascend A5 支持：`vllm-ascend` main（`CMakeLists.txt`、`Dockerfile.a5`、`docs/source/installation.md`）
- NPU 排障：[`TROUBLESHOOTING.md`](TROUBLESHOOTING.md)
- CAMP2p 用户指南：[`CAM_P2P_CONNECTOR_USER_GUIDE.md`](CAM_P2P_CONNECTOR_USER_GUIDE.md)

