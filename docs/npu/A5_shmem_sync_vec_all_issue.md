# CANN SHMEM issue 草稿 —— `aclshmemx_sync_vec_all` 在 Ascend950 上触发 507035

> 发给 CANN / SHMEM 对接人的问题单草稿。可直接复制。最小 repro 见文末。

---

## 标题

SHMEM 1.7.0：`aclshmemx_sync_vec_all()` 在 Ascend950 (A5) 上触发 507035 vector core exception（数据已正确传输）

## 环境

| 项 | 值 |
| --- | --- |
| 芯片 | Ascend950 (A5)，`__NPU_ARCH__ == 3510` / `DAV_C310` |
| CANN | 9.1.0 |
| SHMEM | 1.7.0（`gitcode.com/cann/shmem` master），从源码构建，`-DUSE_EXAMPLES=ON`，`data_op_engine_type = ACLSHMEM_DATA_OP_UDMA` |
| 规模 | 2 PE，单机 device 0 / 1 |
| bootstrap | `test_set_attr(pe, n_pes, mem, "tcp://127.0.0.1:<port>", uid, &attr)` + `aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attr)`，`rc = 0` |

## 现象

一个**单 AIV block**（`<<<1, nullptr, stream>>>`，`[[bisheng::core_ratio(0, 1)]]`）
的 kernel：

- 只做 `aclshmemx_udma_put_signal_nbi(...)` + `aclshmemx_udma_quiet(pe)` —— **干净**，
  `aclrtSynchronizeStream` 返回 0，数据 + signal 都正确到达对端。
- 在末尾**加一句 `aclshmemx_sync_vec_all()`** —— `aclrtSynchronizeStream` 返回
  **507035**，尽管数据 + signal 仍然正确到达（校验 pass）。

即：`aclshmemx_sync_vec_all()` 是唯一触发点。

## 错误信息

```
[ERROR] Notify: notify stream_id=<n>, task_id=3, retcode:507035
[ERROR] Vector Core kernel execution failed, retCode=0x31
[ERROR] AI Core kernel execution failed ... fault kernel_name=<用户 kernel>
[ERROR] rtStreamSynchronize: ErrCode=507035, desc=[vector core exception], InnerCode=0x715005e
```

（同一台机器上，`udma_demo` 官方示例也复现 507035 —— 它同时调了
`aclshmemx_sync_vec_all()` 和 `aclshmemx_report_exception()`；经二分，
`aclshmemx_sync_vec_all()` 是主因，`aclshmemx_report_exception()` 的
`aclshmemi_udma_exception_report_read_entry_kernel` 是次要触发点。）

## bisect（同一二进制，env 切 mode）

| mode | kernel 内容 | `aclrtSynchronizeStream` rc | 数据校验 |
| --- | --- | --- | --- |
| 1 | `udma_put_signal_nbi` + `udma_quiet` | **0** | pass |
| 2 | mode 1 + `aclshmemx_sync_vec_all()` | **507035** | pass |
| 3 | `udma_put_signal_nbi` only | **0** | pass |

## 问题

1. `aclshmemx_sync_vec_all()` 在 Ascend950 / CANN 9.1.0 / SHMEM 1.7.0 上是否已知不可用？
2. 是否需要额外的前置配置（FFTS / `SetSyncBaseAddr` / 特定 block_dim / AIC:AIV 比例）？
   —— `udma_demo` 的 `udma_put_signal_kernel` 里直接调，没有额外配置。
3. `feat/ascend950-relay-barrier` 分支或更新的 SHMEM tag 是否修复？
4. A5 上跨 rank 同步的推荐做法是什么？（我们的场景可以用 signal-based 自旋绕开，
   但想确认这是不是官方建议。）

## 影响

我们在做一个 vLLM 的 Attention-FFN 解耦插件（AFD），需要在 A5 上用 SHMEM UDMA
做 attention rank ↔ FFN rank 的数据搬运。UDMA 数据面本身可用，只是 `sync_vec_all`
拦住了直接照搬 `udma_demo` 的写法。当前绕过方案 = signal-based 同步。

## 最小 repro

`vllm-project/afd-plugin` 分支 `a5-custom-op-research`，`tools/shmem_probe/`：

- `afd_probe_kernel.cpp` —— `ShmemUdmaPutSignalProbe`，`mode` 参数切 1/2/3
- `main.cpp` —— init + launch + 校验
- `RUNBOOK.md` —— 作为 `$SHMEM_SRC/examples/afd_probe/` 树内 example 构建
- `run_probe.sh` —— `PROBE_P4_MODE=1|2|3 ./run_probe.sh <bin>`
