# AFD 在 Atlas A5 (Ascend950) 上的适配清单

> 本文记录 AFD（Attention-FFN Disaggregation）NPU 路径迁移到 Atlas A5
> （Ascend 950PR / 950DT）所需的适配工作、当前状态与排障记录。
> 基于 `afd-plugin` 代码与 `csrc/npu` A5 适配（PR #276 + B 修改）整理。

## 背景

- 目标硬件：**Atlas A5**，芯片 **Ascend 950PR**（CANN SOC 名 `ascend950`，架构 DAV_3510，
  arch 值 3510）；另一变体 950DT 的 SOC 名为 `ascend950dt_9582`。
- 现状基线（910C）：vLLM `0.26.0` + vLLM-Ascend commit `80d8c194f` + 配套 CANN/torch_npu，
  仅验证过 Ascend 910C / Atlas A3。
- 本分支（`a5-adapt`）：私有 fork `ksiyuan/afd-plugin` 的 main + PR #276（A5 算子适配）+
  B 修改（移除 `SetCommEngine(MTE)`）。

## 适配清单（按优先级）

### ① vLLM-Ascend 版本基线（最高优先级，工作量最大）

| 项 | 现状 | 需要做的事 |
| --- | --- | --- |
| vLLM-Ascend 版本 | AFD 锁定 commit `80d8c194f`（910C 时代），该快照**无 ascend950 支持** | 换成支持 ascend950 的 vLLM-Ascend 版本（当前 main 已有） |
| AFD 兼容补丁 | `compat/patches/npu/` 全部按 80d8c194f 内部接口编写 | 逐一对齐新版本：`ascend_platform.py`、`force_load_balance.py`、`mla_graph.py`、`config_validation.py` |
| CANN / torch_npu | 未确认（需 `version.info`） | 用与 vLLM-Ascend A5 快照互配的 CANN（CANN 9.1 命名为 `ascend950`）+ torch_npu |

> 补丁锚点示例：`compat/patches/npu/ascend_platform.py` 注释
> "Upstream source: `vllm_ascend/platform.py` at commit `80d8c194f`"。
> 换版本后必须逐条核对补丁是否仍命中，否则会悄悄失效。

### ② a2e/e2a 自定义算子（当前进行中）

| 项 | 状态 | 备注 |
| --- | --- | --- |
| SOC 注册（`ascend950` / `ascend910_95`） | ✅ 已加（PR #276） | `a2e_def.cpp` / `e2a_def.cpp` 的 `AddConfig` |
| A5 窗口 ABI kernel | ✅ 已加（PR #276） | `moe_distribute_base.h` 的 `HcclA5OpResParam` / `GetHcclLocalWindowsIn` |
| comm engine 设置 | ⚠️ **B 修改验证中** | 移除 `SetCommEngine(MTE)`，改用默认引擎（对齐 vLLM-Ascend MC2 用法） |
| SOC 解析尊重环境变量（review P1-1） | ❌ 未修 | `build_aclnn.sh` 需支持 `ASCEND_OPP_PATH` / `ASCEND_TOOLKIT_HOME` |
| 64 窗口 rank 上限守卫（review P1-2） | ❌ 未修 | `expertRankSize + attentionRankSize > 64` 时必须拒绝（A2E + E2A） |
| A3（910C）回归 | ❌ 未做 | review 要求，改动共用 tiling/kernel，必须回归 |

### ③ CAM 异步连接器（`CAMAsyncAFDConnector`）

| 项 | 现状 |
| --- | --- |
| CAM vendor 包 | 仓库仅含 `CAM_ascend910_93_openEuler_aarch64.run`（910C 专属），**A5 无对应包** |
| `umdk_cam_op_lib` wheel | `209.0.0b1`，是否支持 A5 未确认 |

**结论：`CAMAsyncAFDConnector` 在 A5 上不可用，除非拿到 ascend950 的 CAM 包。**
CAMP2p 使用插件自带的 a2e/e2a 算子，不依赖 CAM 包，是 A5 上唯一可行路径。

### ④ Python 层平台假设（基本无需改动）

- `validation.py` 仅拒绝 310P/XLite worker，接受标准 `NPUWorker` —— A5 用标准 NPUWorker，无需改。
- NPU 连接器 / model runner 依赖的 vLLM-Ascend 接口（`get_hccl_comm_name`、forward_context 等）
  随 ① 的版本变化需复核。

### ⑤ 构建系统

| 项 | 现状 |
| --- | --- |
| `setup.py` | 默认 `SOC_VERSION=910c`；A5 构建需显式 `SOC_VERSION=ascend950`（PR 已支持） |
| `csrc/npu/build_aclnn.sh` | PR 已支持 `950/a5/ascend950*/ascend910_95*`；review P1-1 未修 |
| `csrc/npu/build.sh` | 帮助文案仍是 910b 默认值（无碍） |

### ⑥ 文档与验证

- README / `docs/npu/*` / `recipe/npu/*` 全部基于 Ascend 910C / Atlas A3，需补充 A5 内容。
- A5 上需重新走 CAMP2p 验证：eager、`FULL_DECODE_ONLY` ACL graph、DBO 两 ubatch，并记录结果。

## 当前排障记录：`HcclAllocComResourceByTiling ret=5`

- **现象**：A5 (950PR) 上 `aclnnA2e` 执行失败，HCCL 对 tiling 通信资源分配返回
  `ret=5`（`HCCL_E_NOT_SUPPORT`），`comm = 0x...`。
- **已确认**：
  - 芯片 Ascend 950PR，8 卡，HBM 112GB，健康；
  - CANN OPP 配置目录存在 `ascend950`（无 `ascend910_95`）；
  - 运行时 `GetCurNpuArch() == 3510`（`IsDav3510` 命中，PR 的 arch 判断正确）。
- **怀疑点**（按优先级）：
  1. `HCCL_COMM_ENGINE_MTE = 3` 的枚举值在目标 CANN 中是否真的是 MTE（`grep HCCL_COMM_ENGINE <CANN include>`）；
  2. 是否根本不该显式 `SetCommEngine` —— **B 修改验证中**（vLLM-Ascend 自己的 MC2 算子在 ascend950 上不设 comm engine 也能跑）；
  3. `AlltoAll=level0:fullmesh;level1:pairwise` 算法配置在 950 互联拓扑上是否被支持；
  4. CANN 版本是否确实为 9.1（`version.info`）且与构建/产物一致；
  5. HCCL comm 创建方式（torch_npu 版本）是否与 tiling 分配兼容。
- **下一步**：
  1. 在 A5 上用 B 修改版本重编译 a2e/e2a 重跑，观察 `ret=5` 是否消失；
  2. `export HCCL_DEBUG_INFO=1` + `ASCEND_GLOBAL_LOG_LEVEL=1` 看分配失败的具体原因；
  3. 核对 `HCCL_COMM_ENGINE_*` 枚举值；
  4. 必要时做最小化单测（独立建 HCCL 组 + 调 a2e），与全量部署隔离。

## 相关链接

- A5 算子适配 PR：[vllm-project/afd-plugin#276](https://github.com/vllm-project/afd-plugin/pull/276)
- vLLM-Ascend A5 支持：`vllm-ascend` main（`CMakeLists.txt`、`Dockerfile.a5`、`docs/source/installation.md`）
- NPU 排障：[`TROUBLESHOOTING.md`](TROUBLESHOOTING.md)
- CAMP2p 用户指南：[`CAM_P2P_CONNECTOR_USER_GUIDE.md`](CAM_P2P_CONNECTOR_USER_GUIDE.md)
