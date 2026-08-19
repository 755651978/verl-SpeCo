# Drafter Scheduler 实现与策略接入指南

Last updated: 08/19/2026

本文描述 SPECO 在线草稿模型（drafter）的调度边界、当前同步调用链，以及新增调度策略时应修改的接口。它面向需要增加训练、收集或发布策略的开发者。

## 设计目标

`verl_speco.trainer.scheduler` 是数据收集、草稿模型训练和权重发布的唯一决策入口。外层 Trainer 只提供生命周期事件和运行时 RPC 适配；Worker 只校验并执行已生成的计划。

这样可以保证同步训练和后续 Bubble Time / rollout idle worker 训练共享同一套触发条件、数据版本校验和发布语义。新增策略只替换“何时执行、在哪里执行、最多执行多少”，不应复制或改变训练语义。

边界如下：

```text
SpecoRayPPOTrainer
  ├─ 构造上下文、绑定 Ray WorkerGroup RPC
  └─ 调用 Scheduler 生命周期事件
            │
            ▼
DrafterScheduler（唯一决策 Facade）
  ├─ CollectionPlan / TrainingPlan / PublishPlan
  ├─ Trigger / Budget / Strategy
  └─ Executor ports
            │
            ▼
SpecoWorker / rollout runtime
  └─ 校验 plan 并执行 collection、training、publish RPC
```

外部代码应只从 `verl_speco.trainer.scheduler` 包入口导入公共类型，例如：

```python
from verl_speco.trainer.scheduler import (
    DrafterScheduler,
    DrafterScheduleConfig,
    TrainingPlan,
)
```

不要让 Trainer、Worker 或新业务代码直接依赖 `training_trigger.py`、`training_budget.py`、`execution_strategy.py` 等 Scheduler 内部实现。

## 当前同步路径

### 初始化和基础设施绑定

`SpecoRayPPOTrainer.attach_speco_worker_group()` 创建并绑定三个 Executor：

| Executor | 作用 | 适配的现有 RPC |
| --- | --- | --- |
| `DrafterWorkerExecutor` | 数据状态、训练准备、preflight、训练提交 | `SpecoWorkerGroup` |
| `DrafterCollectionExecutor` | collection 的 stage / commit / rollback / finalize | `SpecoWorkerGroup` |
| `DrafterPublishExecutor` | 获取训练快照并热更新 rollout drafter | drafter WorkerGroup + rollout runtime |

Scheduler 只依赖这些 Protocol，不依赖 Ray。Ray ObjectRef 的提交、等待和结果解析都由 `Callback*Executor` 适配器处理。

### 一次在线训练的完整时序

```text
rollout / compute_old_log_prob 生成特征
  │
  ├─ Trainer: plan_collection(context, config)
  │     └─ CollectionPlan：是否收集、来源、采样和窗口预算、collection_id
  │
  ├─ Trainer: prepare_collection_payload(...)
  │     └─ SGLang adapter 按 replica_rank 分桶
  │        old-logprob adapter 按显式 owner 分桶
  │
  └─ Scheduler: on_collection_ready(plan, payload)
        └─ SyncCollectionStrategy
             ├─ set_global_step
             ├─ stage
             ├─ 校验 collection_id、路由、版本和 Worker 身份
             ├─ commit
             └─ finalize；失败则 abort / rollback

PPO actor update 前
  │
  └─ Scheduler: on_before_actor_update(context)
        ├─ prepare_training_plan()
        │    ├─ 对 collect_only / pending / interval 等低成本跳过条件直接生成 skip plan
        │    ├─ 仅在可能训练时查询全部 Worker 的 TrainingDataStatus
        │    ├─ TrainingTrigger：判断是否应训练
        │    ├─ TrainingBudget：计算 max_batches / min_batches 等预算
        │    └─ TrainingPlan：冻结本次训练的版本、预算和执行策略
        └─ prepare_training_execution(plan)
             ├─ 设置 Worker global_step
             └─ use_logits=False 时同步 target lm_head

PPO actor update
  │
  └─ 保持原 PPO 更新逻辑

PPO actor update 后
  │
  └─ Scheduler: on_after_actor_update(plan, runtime_state)
        └─ SyncExecutionStrategy
             ├─ 所有 Worker preflight
             │    └─ 校验 plan_id、step、buffer/data/target version、incarnation、batch
             ├─ 任一 Worker 未就绪：全部 abort，不提交分布式训练
             ├─ 全部就绪：提交并等待 train_drafter
             └─ TrainingOutcome 聚合结果；不一致则 trained=false，禁止发布

主模型权重更新后的 rollout-safe point
  │
  └─ Scheduler: on_after_weight_update(context)
        ├─ plan_publish()
        └─ PublishExecutionStrategy
             ├─ 等待上一轮异步 publish（如有）
             ├─ 获取唯一 publish leader 的训练快照
             └─ 同步或异步更新 rollout drafter 权重
```

### 当前同步策略的规则

- `TrainingTrigger` 的默认实现是 `IntervalAndBufferTrigger`：必须满足训练 interval、没有 pending training、所有 Worker 数据版本一致，并且 Worker 聚合后至少有 `min_trainable_batches` 个可训练 batch。
- `SyncTrainingBudgetPolicy` 使用配置中的 `training.step` 作为 `max_batches`。数据是否存在只决定“能不能启动”，不再把实际可区分 batch 数静默缩短为更少的训练步数。
- `min_batches` 是启动前的最低可训练 batch 要求；若 `max_batches < min_batches`，Scheduler 返回不启动的 `insufficient_training_budget` 计划。
- `use_logits=False` 时，`required_target_version` 等于本次主训练 `global_step`，所有 Worker 必须使用该 target 版本。
- `use_logits=True` 时，训练目标已在数据中，`required_target_version=None` 表示 **不约束** Worker 当前 target version，而不是要求其为 `None`。
- 发布只在 `TrainingOutcome.trained=True` 且 `TrainingPlan.publish_after_success=True` 时进行；发布快照只要求唯一 publish leader 可用。

## 核心对象与职责

| 对象 | 职责 | 不应承担的职责 |
| --- | --- | --- |
| `DrafterScheduleContext` | 当前 step、模式、pending 状态等只读调度事实 | 训练预算或 Worker 执行细节 |
| `TrainingDataStatus` | 全部 Worker 数据状态的聚合快照 | 触发训练或调用训练 RPC |
| `TrainingPlan` | 一次不可变训练决定：是否启动、策略、预算、版本、plan_id | 在 Worker 端重新计算触发条件 |
| `CollectionPlan` | 一次收集决定和静态采样预算 | 解析来源特有的数据格式 |
| `PublishPlan` | 一次发布决定 | 获取或传输权重 |
| `TrainingOutcome` | 聚合多 Worker 的训练结果与一致性，决定是否可发布 | 再次决定训练预算 |
| Executor | 连接 Scheduler 与 Ray/Worker/rollout runtime | 训练触发、预算、发布间隔等策略 |

`plan_id`、`collection_id`、`data_version`、`buffer_version`、`worker_incarnation` 用于防止跨进程状态漂移：计划生成后数据变化、Worker 重启或消息错配时，应 fail closed，而不是让部分 Worker 进入分布式训练。

## 新增策略：修改位置

选择下面最小的改动范围。不要为了新增策略在 `SpecoRayPPOTrainer` 或 `SpecoWorker` 再写一套 interval、buffer 或 publish 判断。

### 1. 只改变“是否触发训练”

适用例子：按接受率、loss、样本年龄或外部 SFT 事件决定是否训练。

1. 在 `scheduler/training_trigger.py` 实现 `TrainingTriggerPolicy.should_train(...)`。
2. 返回 `TriggerDecision(should_train, reason)`；为新 reason 在 `TrainingPlan._REASON_CODES` 增加稳定数值编码。
3. 在 `DrafterScheduler` 中注入或选择该 trigger policy。
4. 保持 `prepare_training_plan()` 的“cheap skip 在查询 Worker 前返回”行为，避免无效 Worker RPC。
5. 为 trigger 的各分支编写纯单元测试。

此类策略不应修改 `SyncExecutionStrategy`、Worker 的 preflight 或发布调用链。

### 2. 只改变“训练多少步 / 何时停止”

适用例子：根据 buffer 大小、token 预算、deadline 或 SFT 配额计算训练步数。

1. 在 `scheduler/training_budget.py` 实现 `TrainingBudgetPolicy.make_budget(...)`。
2. 返回 `TrainingBudget(max_batches, min_batches, deadline_ts, require_full_batch, sample_last_n_steps, reason)`。
3. 保证 `max_batches >= min_batches`；否则让 `DrafterScheduler.plan_training()` 生成 `insufficient_training_budget` 的 skip plan。
4. 若增加新的预算字段，先扩展 `TrainingBudget` 和 `TrainingPlan`，再更新 `TrainingPlan.to_worker_payload()`，最后让 Worker 仅消费该字段。
5. 为 0 预算、最小预算、上限和 deadline 分支写测试。

### 3. 新增一种训练执行策略

适用例子：Bubble Time、rollout idle worker、异步队列或 SFT co-train。当前枚举中预留了 `ROLLOUT_IDLE_WORKER`，但尚未实现。

1. 在 `schedule_types.py` 的 `DrafterExecutionStrategy` 增加或启用策略枚举值，并更新 `TrainingPlan.metrics()` 的 strategy code。
2. 在 `execution_strategy.py` 实现 `DrafterTrainingExecutionStrategy.execute(plan, executor, runtime_state)`。
3. 在 `DrafterScheduler.plan_training()` 中选择该执行策略；触发和预算仍复用统一 policy，除非确实需要新的 policy。
4. 在 `DrafterScheduler.execute_training_plan()` 中路由到新 strategy。未知策略必须显式报错，不能静默回退到同步训练。
5. 若策略需要新的资源查询、非阻塞 poll、任务取消或 deadline 执行，向 `DrafterWorkerExecutor` 增加明确方法，并由 `CallbackDrafterWorkerExecutor` 将其适配到 Trainer/Ray RPC。
6. 保持 preflight、版本校验、结果聚合和发布门控。异步策略可以改变等待时间，但不能绕过这些一致性约束。

Bubble Time 策略通常只应改变“何时提交训练、是否等待结果、使用哪些空闲资源”。`TrainingPlan` 仍是执行的唯一输入；Worker 不应自行决定是否训练。

### 4. 新增收集来源或收集策略

适用例子：SFT 数据、第三种 rollout engine，或新的 feature 表示。

1. 在 `DrafterCollectionSource` 增加来源枚举，并在 `CollectionPlan.metrics()` 中增加 source code。
2. 实现 `DrafterCollectionAdapter`，负责把来源特有样本转为统一 `CollectionPayload` 并完成 owner 分桶。
3. 通过 `DrafterScheduler.register_collection_adapter()` 注册 adapter；不要在 Trainer 中手写第二套 payload 分桶和 bucket 对齐逻辑。
4. 若收集事务语义不同，实现新的 `DrafterCollectionStrategy`；否则复用 `SyncCollectionStrategy` 的 stage / commit / rollback / finalize 流程。
5. 若需要新的外部 RPC，在 `DrafterCollectionExecutor` 中扩展，并同时实现 callback 适配器与 Worker 端处理。
6. 保持同一个 `collection_id` 贯穿 stage、commit、rollback 和 finalize；成功 collection 必须更新可验证的数据版本。

### 5. 改变发布方式

适用例子：不同 rollout engine、异步热更新、版本化权重仓库。

1. 优先复用 `PublishPlan` 和 `PublishExecutionStrategy`。
2. 仅在 RPC 边界不同的情况下实现或替换 `DrafterPublishExecutor`。
3. 若新增发布决策规则，集中在 `DrafterScheduler.plan_publish()`；不要在 Trainer 和 Worker 同时判断 publish interval。
4. 发布输入必须来自训练成功后唯一 publish leader 的快照。

## SFT Co-Train 的推荐接入方式

SFT co-train 不需要复制在线 Scheduler。推荐将 SFT 数据视为新的 collection source，并将其训练触发/预算实现为 policy：

```text
SFT dataloader / producer
  → SFT CollectionAdapter
  → CollectionPayload
  → Scheduler.on_collection_ready()
  → SFT trigger / budget policy 生成 TrainingPlan
  → 复用统一 preflight、训练执行、TrainingOutcome 和发布路径
```

若 SFT 与在线 RL 数据必须隔离，应在 payload 和 Worker buffer 中加入明确的数据域或 source 标识，并让 `TrainingDataStatus` 聚合、trigger 和 budget 显式选择可混合或不可混合的域。不要仅依赖“当前 global step”来区分两类数据。

## 开发检查清单

新增策略或 Executor 后，至少确认：

- 外部 Trainer 仍只调用 `DrafterScheduler` 的公共 Facade，而非内部 policy。
- Worker 只执行 `TrainingPlan.to_worker_payload()`，没有新的 trigger、interval 或发布判断。
- 所有参与 Worker 在训练前通过 preflight；任一失败会使全部 Worker 跳过。
- `data_version` 必须一致；仅当 `required_target_version` 非空时才校验 target version。
- 异步策略仍有明确的 pending、完成、失败和安全发布状态，不会在未完成训练时发布。
- collection 的失败走 abort 或 rollback；不会把半提交数据当成可训练数据。
- 新 reason/source/strategy 的指标编码稳定，且计划和 outcome 都有可观测日志。
- 添加单元测试：正常路径、跳过路径、版本不一致、重复/缺失 Worker 结果、异常清理路径。

现有测试入口可参考：

```bash
python -m pytest -q tests/unit tests/integration/test_drafter_runtime_control_contract.py
python -m ruff check verl_speco/trainer/scheduler tests/unit
```
