# Bubble Time 与 V1

Last updated: 2026-10-23

适配基线为上游 `release/v0.9.0`。PPO 主循环仍由上游 V1 trainer 执行；
本仓库仅提供生命周期 hook、独立池 worker、数据采集和发布协议。
V1 rollout 后端仍限于 vLLM。

## 配置

使用 `--config-name=speco_v1_trainer`，沿用以下配置：

```text
trainer.v1.trainer_mode=sync|colocate_async|separate_async
actor_rollout_ref.rollout.drafter.training.scheduler.execution.strategy=rollout_idle_worker
actor_rollout_ref.rollout.drafter.training.scheduler.training_quota.enable=true
actor_rollout_ref.rollout.drafter.training.scheduler.training_quota.allow_critical_path_fallback=false
```

启用 ROI 时，设置
`actor_rollout_ref.rollout.drafter.training.drafter_convergence_freeze` 下的
`enabled=true`、`economics.enabled=true` 和 `probe.enabled=true`。
`mode=shadow` 只观察，`mode=active` 才实际冻结。原有窗口、覆盖率和漂移恢复阈值继续生效。

`separate_async` 固定将 drafter 训练 worker 放到独立 rollout placement groups，
不新增部署模式。分配 GPU 时复用已有资源池，不申请额外 GPU。
worker 在训练 RPC 返回、资源租约释放之前完成 offload；optimizer writer 仍保留。

## 边界

| 模式 | Bubble 可运行阶段 | PPO/同步边界 |
| --- | --- | --- |
| sync | rollout 的已确认空闲窗口 | sample end 停线程、回收训练组，再进入上游 sleep/old-logprob |
| colocate_async | 预取期间的已确认空闲窗口 | 先关闭 admission，再回收、abort/sleep；发布完成后恢复 |
| separate_async | 独立 rollout 池的已确认空闲窗口，可与 actor 更新重叠 | target 同步前回收冲突训练资源；有发布、探测、保存时进入受控暂停 |

独立池的训练 rank 与 rollout replica rank 分开处理。初始化根据 node、实际设备和
collective 元数据校验归属。每个独立 replica 的发布 ACK 使用独立命名空间，
避免不同 replica 的本地 rank 0 相互覆盖。休眠 hybrid replica 重新加入 balancer 前，
先补齐已确认的 drafter 权重及 serving revision。

old-logprob 特征继续使用每份采集计划独立、重试稳定的样本 ID。
separate_async 在上游旧策略 CPU-slot 恢复上下文内完成不可变 head 导出，
随后才恢复当前 actor。head 的 CPU 分发、版本 ACK 和 quota 保护继续使用共享协议。

ROI 事务顺序为：关闭普通请求 → 固定 actor target → before-arm →
drafter 发布并等待完整 ACK → 更新 serving revision → after-arm → 恢复请求。
两臂直接调用 serving RPC，不进入 PPO、reward 或 TransferQueue。
探测缺少 revision、混合 revision、执行失败或覆盖不足时，不产生冻结证据。

成本跨空闲窗口保留，按计划 ID 去重。组内准备和训练阶段取最大 rank 耗时，
有时间区间的阶段计算区间并集，避免 head、训练及传输重叠重复计费。
worker 时间区间使用集群 wall clock；验收机器应保持时钟同步。
日志分别提供更新成本、probe 耗时、暂停时间和 step-end 关键路径耗时。

未完成 quota 或未确认发布会延迟 checkpoint。发布和 ROI 状态提交后，
在关闭 admission 的边界使用实际 RL step 保存联合恢复点。
异步 Bubble 保存要求 TransferQueue 支持 snapshot/reissue；不支持时明确报错。
最后一步仍有 quota 且未允许关键路径补齐时明确失败，不保存不完整结果。

联合保存要求 actor 和已启用 critic 的 `checkpoint.async_save=false`，
确保模型保存 RPC 返回时组件已完成。这与 V1 的异步 rollout 模式相互独立。
上游 driver 的 latest 写入及提前清理在保存事务内暂缓；worker 仍按原配置同步保存。
模型、冻结状态、TQ、Feature Store 和联合 manifest 均完成后，才原子提交 latest，
随后在各 owning worker 上执行原有 actor/critic 保留数量策略。
组件保存或 latest 提交失败时，旧指针和旧模型 checkpoint 均保留，允许重试。

generation 启动事件使用注册的真实 replica 分组，不再根据 rollout DP 数量推断 rank。
separate_async 的事件总线只管理独立池，hybrid 请求与同号训练 rank 的租约相互隔离。
head 传输成本在同步返回或预取轮询收到完整版本 ACK 后统一记账；
同一传输重复完成不重复计费，同一版本再次分发则按新增传输计费。

## CPU 验证

专用 CI job `V1 Bubble contracts (verl 0.9)` 安装真实上游 V1 和 TransferQueue，
设置 `SPECO_REQUIRE_V1=1`。缺少 V1 导入依赖将直接失败，不能整组跳过。

初次适配在干净 Python 3.12 / CPU Torch 环境验证：真实 0.9 V1 测试 99 项通过；
legacy 基线全量 unit/integration/V1 回归 2539 项通过、16 项因可选 GPU/后端依赖跳过。
2026-10-23 的事件映射、联合提交与 head ACK 计费修复，使用真实 0.9 V1
及事件总线集成测试验证，共 168 项通过。

```bash
PYTHONPATH="$PWD:$UPSTREAM_VERL" SPECO_REQUIRE_V1=1 \
  python -m pytest tests/v1 -q -ra

PYTHONPATH="$PWD:$LEGACY_VERL" python -m pytest \
  tests/unit/test_bubble* tests/unit/test_rollout_idle_worker_scheduler.py \
  tests/unit/test_idle_window_readmission.py -q -ra
```

## GPU 验收记录

CPU 合约通过不代表 GPU 功能验收完成。当前开发环境无可用 CUDA GPU，以下项目仍须实测。

| 负载 | sync | colocate_async | separate_async |
| --- | --- | --- | --- |
| DSpark：Bubble / 普通 drafter 等工作量对照 | 待测 | 待测 | 待测 |
| DFlash：Bubble / 普通 drafter 等工作量对照 | 待测 | 待测 | 待测 |
| ROI 探测、冻结、漂移恢复、checkpoint 恢复 | 待测 | 待测 | 待测 |

每组必须记录 optimizer 工作量、发布次数、未偿 quota、probe 暂停和端到端耗时，
并核对共享 GPU 无越界重叠、完整 quota 只执行一次、发布和恢复版本一致。
separate_async 还须覆盖跨节点 CPU/Ray 传输及 hybrid replica 重新入池。
减少训练或遗留 quota 的结果不能作为等工作量加速证据。
