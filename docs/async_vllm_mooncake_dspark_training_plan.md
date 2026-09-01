# verl-SpeCo 异步 vLLM → TransferQueue（Mooncake 后端）→ DSpark 流式训练方案

Last updated: 08/21/2026

## 1. 文档范围

本文只讨论当前 `verl-SpeCo-ls` 项目的独立草稿模型训练入口：

```text
examples/run_qwen3-8b_drafter_separate_training.sh
  → python -m verl_speco.standalone_tq_training_launcher
       ├─→ TransferQueue owner
       ├─→ vLLM hidden-state producer
       └─→ TransferQueue consumer
            → python -m verl_speco.draft_train_launcher
            → torch.distributed.run
            → python -m verl_speco.draft_train
            → run_standalone_draft_training()
```

目标是训练 `mode=train/offline` 的草稿模型，不是 RL 与草稿模型一起训练，也不是先生成全量 hidden states 再长期保存到磁盘。

输入既可以是已有 response 的 replay 文件，也可以是 verl prompt-only Parquet。新流水线需要：

1. Producer 读取 prompt；缺少 response 时由 target vLLM 生成；
2. Producer 并行请求一个或多个 vLLM endpoint 做 prefill；
3. hidden states 写入 TransferQueue（简称 TQ），TQ 的数据后端使用 Mooncake；
4. DSpark 的各训练 rank 从 TQ/Mooncake 并行读取自己的 local batch；
5. 所有 rank 完成同一个 optimizer step 后，清理这一批 TQ 数据；
6. 不使用自研 Stream Coordinator；第一版复用 PR #48 已验证的 TQ KV 传输模式，由 TQ 保存 payload 和 tags，rank 0 通过 `kv_list` 发现 ready key 并组成 global batch。

本文不照搬 AngelSpec 的进程组织。实现依据是旁边只读参考仓库 `verl-SpeCo` 的 PR #48 分支，尤其是：

```text
verl_speco/integration/transferqueue_bridge.py
verl_speco/integration/sglang_runtime.py
verl_speco/integration/oldlogprob_runtime.py
verl_speco/workers/speco_worker.py
verl_speco/integration/task_runner.py
```

参考仓库只用于理解和复用设计，不修改其代码。真正实现仍放在本项目 `verl-SpeCo-ls`，服务于 standalone drafter training。

建议按下面顺序阅读：

1. 第一部分先介绍 verl/SpeCo 中的进程、`TokenOutput`、`DataProto`、WorkerGroup、replica 和 owner rank；
2. 再完整解释 PR #48 的 SGLang TQ 路径；
3. 再解释 PR #48 的 old-logprob TQ 路径；
4. 再解释 hidden 恢复后如何进入 online buffer 和 optimizer step；
5. 第二部分才讨论如何把相同 TQ 传输能力改造成独立 Producer/DSpark Consumer。

## 第一部分：PR #48 原始 TQ 流程

这一部分只解释只读参考仓库 `../verl-SpeCo` 的 PR #48。这里的 Producer、Consumer、Ray driver、数据结构和生命周期全部是 PR #48 当前代码的行为，不是本文为 standalone 设计的行为。

### 1A. 阅读 PR #48 前必须知道的项目对象

#### SGLang server

SGLang server 是 rollout 推理进程。它接收 prompt，执行 target model 推理并生成 response。SpeCo 的 SGLang patch 还会在推理过程中收集指定层 hidden states。

它不是 drafter trainer，也不执行 optimizer step。在 PR #48 中它是 hidden-state Producer。

#### TokenOutput

`TokenOutput` 是一次 rollout request 的返回对象。核心字段可以概括为：

```python
TokenOutput(
    token_ids=list[int],
    log_probs=...,
    routed_experts=...,
    extra_fields={
        "global_steps": int,
        "drafter_sample": dict | None,
    },
)
```

`token_ids` 是生成结果；`extra_fields` 是 SpeCo 添加的旁路字段。`drafter_sample` 不影响正常 response 返回，它用于把草稿训练所需信息从 rollout 侧带回训练控制层。

#### DataProto 和 non_tensor_batch

verl 将一批 rollout 结果整理成 `DataProto`。它通常分为：

```python
DataProto(
    batch=TensorDict(...),
    non_tensor_batch={...},
    meta_info={...},
)
```

- `batch`：规则的批量 tensor，例如 prompts、responses、attention mask；
- `non_tensor_batch`：不能直接组成规则 dense batch 的 Python 对象或 object array；
- `meta_info`：批次级配置和指标。

每个 request 的 `TokenOutput.extra_fields["drafter_sample"]` 最终会被 rollout/agent-loop 聚合到：

```python
gen_batch_output.non_tensor_batch["drafter_sample"]
```

所以 SGLang 侧写的是单 request `TokenOutput`，driver 侧拿到的是批量 `DataProto`。

#### RayPPOTrainer driver

`SpecoRayPPOTrainer` 所在进程是控制进程，简称 driver。它负责按训练 step 依次调用 rollout、old-logprob、actor update，以及向各 WorkerGroup 发 RPC。

driver 不执行 drafter 模型 forward。PR #48 之前，它会接触包含 hidden tensor 的 `drafter_sample`；PR #48 之后，它只处理中小 tensor、标量 metadata 和 TQ key。

#### WorkerGroup

WorkerGroup 是 verl 对一组 Ray worker 的调用封装。代码：

```python
self.drafter_wg.collect_rollout_features(buckets)
```

不是普通本地函数调用，而是根据注册的 dispatch rule，将参数分发到多张卡上的 `SpecoWorker.collect_rollout_features()`。

#### Rollout replica

rollout replica 是一组共同承载一个 rollout 模型副本的进程/GPU。一个任务可能存在多个 data-parallel rollout replica。`replica_rank` 标识当前 sample 是哪个 rollout replica 生成的：

```text
replica_rank = 0, 1, 2, ...
```

#### Drafter training replica、DP rank 和 SP rank

drafter 训练也可能按 data parallel 和 sequence parallel 组织：

```text
drafter replica / DP rank 0
  ├─ SP rank 0
  └─ SP rank 1

drafter replica / DP rank 1
  ├─ SP rank 0
  └─ SP rank 1
```

同一个 drafter DP replica 内的 SP ranks 共同执行一个模型副本的训练。`replica_rank` 用来把 rollout replica 产生的数据路由到对应 drafter DP replica。

#### Owner rank

`collect_rollout_features` 注册了：

```python
@register(
    dispatch_mode=make_nd_compute_dispatch_fn(
        mesh_name="drafter_owner_route"
    )
)
```

每个 drafter DP replica 的 `SP rank 0` 被标记为 collect leader/owner。driver 传入的是按 replica 分好的 bucket，dispatch 层负责把 bucket 发到对应训练组。一个 replica 内可能有多个 rank 需要共同训练，但只有指定 leader 负责汇总 RPC 返回。

这里的 owner route 是 PR #48 为什么不能简单让任意 worker 随机拿 sample 的原因：sample 必须进入与当前 drafter device mesh 一致的训练组。

## 2. PR #48 改造前的 online 特征流程

PR #48 改造的是 SpeCo 的 online drafter feature transport。hidden states 有两条主要来源。

### 2.1 SGLang rollout hidden 路径

改造前：

```text
SGLang server
  → 生成 drafter_sample，其中直接包含 hidden_states CPU tensor
  → TokenOutput.extra_fields
  → RayPPOTrainer driver 收集 drafter_sample
  → driver 按 drafter replica/owner 分桶
  → Ray dispatch / object store
  → SpecoWorker.collect_rollout_features(samples)
  → _store_rollout_sample()
  → online drafter buffer/train
```

此时 `drafter_sample` 类似：

```python
{
    "input_ids": int64[1, total_len],
    "prompts": int64[1, prompt_len],
    "responses": int64[1, response_len],
    "hidden_states": bf16[1, hidden_rows, hidden_dim],
    "hidden_positions": int64[1, hidden_rows],
    "target_logprobs": tensor | None,
    "global_step": 42,
    "replica_rank": 1,
}
```

问题是整个字典经过 driver，而 `hidden_states` 是其中最大的字段。driver 的 host memory 和 Ray object store 都会承载这些 tensor。

### 2.2 old-logprob hook hidden 路径

另一条路径在 actor old-logprob forward 中捕获 hidden states。改造前，大 chunk 通过：

```python
chunk_ref = ray.put(hidden_chunk)
```

sample 不直接带 tensor，而是带：

```python
{
    "hidden_states_ref_chunks": [
        {
            "ref": ray_object_ref,
            "start": 0,
            "length": 512,
        },
    ],
}
```

drafter worker 再 `ray.get(ref)`，根据 `start/length` 切出每条 sample 所需行。

### 2.3 PR #48 要改变的边界

PR #48 没有改变：

- rollout 什么时候产生 sample；
- driver 如何触发 drafter worker；
- drafter worker 如何调用 `_store_rollout_sample()`；
- drafter model 的训练逻辑；
- drafter 权重发布。

它只改变大 tensor 的跨进程介质：

```text
改造前：Producer → Ray driver/object store → Consumer
改造后：Producer → TQ storage → Consumer
                   key 仍走原 Ray 控制路径
```

## 3. PR #48 改造后的完整 TQ 流程

### 3.0 总览

PR #48 的目标不是让 TQ 自己产生训练 batch，而是把原来经过 Ray driver/Ray object store 的大 hidden tensor 攁到 TQ。原来的控制路径继续存在，只是控制路径上从“大 tensor”变成“小 key”。

整体结构是：

```text
                           原 Ray 控制路径
                    drafter_sample / chunk ref
Producer ───────────────────── key ───────────────────▶ Consumer
   │                                                       │
   │ kv_put(large tensor)                                  │ kv_batch_get(key)
   ▼                                                       ▼
TransferQueue storage ─────────────────────────────────────┘
```

因此 PR #48 同时保留两条通道：

```text
控制通道：Producer → Ray driver → drafter worker
数据通道：Producer → TQ storage → drafter worker
```

控制通道负责告诉 Consumer “本次应该处理哪个样本、对应哪个 TQ key”；数据通道负责传输 hidden states 等大 tensor。

#### 3.0.1 配置放在哪里

PR #48 在 drafter training 配置下增加：

```yaml
actor_rollout_ref:
  rollout:
    drafter:
      training:
        transfer_queue:
          enable: false
          backend:
            storage_backend: SimpleStorage
            SimpleStorage:
              total_storage_size: 100000
              num_data_storage_units: 8
```

对应参考代码：

```text
../verl-SpeCo/verl_speco/config/speco_base.yaml
```

这里的 `transfer_queue` 是 SpeCo drafter 自己的配置，不是 standalone `feature_store.type`，也不是简单把上游 verl 的 `transfer_queue.enable` 打开。

#### 3.0.2 TaskRunner 创建整套 TQ

RL 任务启动时，`SpecoTaskRunner.run()` 在创建 workers 之前调用：

```python
from verl_speco.integration.transferqueue_bridge import (
    close_transfer_queue,
    init_transfer_queue,
)

transfer_queue_started = init_transfer_queue(config)
try:
    trainer.init_workers()
    trainer.fit()
finally:
    if transfer_queue_started:
        close_transfer_queue()
```

对应参考代码：

```text
../verl-SpeCo/verl_speco/integration/task_runner.py:319
```

`init_transfer_queue(config)` 内部读取：

```python
config.actor_rollout_ref.rollout.drafter.training.transfer_queue
```

然后执行：

```python
tq.init(_to_plain_dict(tq_cfg))
```

并记录：

```python
_state["initialized"] = True
_state["owner"] = True
```

这里的 owner 是“创建/拥有 TQ 生命周期的进程”。只有 owner 在任务结束时执行 `tq.close()`。

关键顺序是：

```text
SpecoTaskRunner
→ tq.init(完整配置)
→ trainer.init_workers()
→ Ray workers 启动
```

也就是说，PR #48 假设 TQ Controller/storage 已经由 TaskRunner 在 Ray 集群环境中建立，后启动的 worker 只需要连接。

#### 3.0.3 每个 Producer/Consumer 进程怎么连接 TQ

bridge 中的 `_ensure_initialized()` 是进程级懒初始化：

```python
def _ensure_initialized():
    if _state["initialized"]:
        return

    with _state_lock:
        if _state["initialized"]:
            return

        tq.init()
        _state["initialized"] = True
```

注意这里是：

```python
tq.init()
```

不是：

```python
tq.init(config)
```

无参初始化的含义是连接 TaskRunner 已创建的同一套 TQ。它依赖 PR #48 所处的 Ray 运行环境完成服务发现。

因此 PR #48 不是“每个 worker 各创建一套 TQ”，而是：

```text
TaskRunner：tq.init(config)，创建一次
SGLang producer：tq.init()，连接
actor producer：tq.init()，连接
drafter consumer：tq.init()，连接
```

#### 3.0.4 SGLang Producer 怎么写 hidden states

SGLang 已经完成 rollout，并组装出 `drafter_sample` 后，PR #48 执行：

```python
configure_transfer_queue(training_cfg)

if is_transfer_queue_enabled():
    tq_key = make_sample_key(
        collection_global_steps,
        self.replica_rank,
        request_id,
    )

    tq_payload = {
        "hidden_states": hidden_states.unsqueeze(0).cpu(),
    }

    if target_logprobs is not None:
        tq_payload["target_logprobs"] = (
            target_logprobs.unsqueeze(0).cpu()
        )

    put_sample(
        tq_key,
        tq_payload,
        tag={
            "global_step": collection_global_steps,
            "replica_rank": self.replica_rank,
        },
    )

    drafter_sample["hidden_states_tq_key"] = tq_key
    drafter_sample["hidden_states"] = None
```

对应参考代码：

```text
../verl-SpeCo/verl_speco/integration/sglang_runtime.py:2020
```

这里发生了两条不同的数据流：

```text
大 tensor：SGLang → TQ
小字典/key：SGLang → 原 Ray/driver 路径 → drafter worker
```

写进 TQ 后将：

```python
drafter_sample["hidden_states"] = None
```

是为了避免 hidden states 继续经过 driver/Ray object store。driver 仍收到 sample，但大 tensor 已替换成：

```python
drafter_sample["hidden_states_tq_key"]
```

#### 3.0.5 `put_sample()` 实际怎么写

bridge 中：

```python
def put_sample(key, tensor_dict, *, tag=None):
    payload = {
        k: v
        for k, v in tensor_dict.items()
        if torch.is_tensor(v)
    }

    _ensure_initialized()

    tq.kv_put(
        key=key,
        partition_id="speco_drafter_features",
        fields=payload,
        tag=tag or {},
    )
```

对应参考代码：

```text
../verl-SpeCo/verl_speco/integration/transferqueue_bridge.py:206
```

这里可以明确看到：

- PR #48 使用 TQ 高层 KV API；
- 一个 key 对应一个 sample；
- `fields` 是 tensor 字典；
- `tag` 是小 metadata；
- partition 当前写死为 `speco_drafter_features`；
- 写入前 tensor 已 `.cpu()`；
- 写入失败直接抛异常，不静默回退。

key 的生成代码是：

```python
def make_sample_key(global_step, replica_rank, request_id):
    return f"speco:{global_step}:{replica_rank}:{request_id}"
```

这个 key 对 RL rollout 是合理的，因为它用 step、rollout replica 和 request ID 标识一次在线采样。

#### 3.0.5.1 PR #48 中“数据”和“元数据”分别长什么样

PR #48 一条 SGLang 样本在写 TQ 之前，`drafter_sample` 同时包含大 tensor 和控制信息。简化后类似：

```python
drafter_sample = {
    # 普通训练输入，仍走原 sample/Ray 控制路径
    "input_ids": int64[1, prompt_len + response_len],
    "prompts": int64[1, prompt_len],
    "responses": int64[1, response_len],

    # 大 tensor，开启 TQ 后从这个字典移除
    "hidden_states": bf16[1, hidden_rows, hidden_dim],
    "target_logprobs": fp32[1, rows, topk_or_vocab] | None,

    # hidden 与 token 对齐所需的小字段
    "hidden_positions": int64[1, hidden_rows] | None,
    "hidden_position_start": int,
    "hidden_position_end": int,
    "hidden_window_start": int,
    "hidden_window_end": int,

    # 控制信息
    "global_step": int,
    "replica_rank": int,
}
```

执行 `put_sample()` 时，并不是把整个 `drafter_sample` 放进 TQ。PR #48 只抽取占用大的 tensor：

```python
tq_payload = {
    "hidden_states": bf16[1, hidden_rows, hidden_dim],
    "target_logprobs": fp32[1, rows, ...],       # 可选
    "hidden_raw_target_logprobs": ...,           # 可选
    "hidden_raw_target_logprobs_positions": ..., # 可选
}
```

这就是 TQ 的 data payload。它被传给：

```python
tq.kv_put(fields=tq_payload)
```

另外还有 TQ tag：

```python
tag = {
    "global_step": 42,
    "replica_rank": 1,
}
```

tag 是 TQ 侧轻量 metadata，用于描述/检索对象，不承载 hidden tensor。

写入完成后，仍经 Ray 传递的轻量 `drafter_sample` 变成：

```python
drafter_sample = {
    "input_ids": int64[1, total_len],
    "prompts": int64[1, prompt_len],
    "responses": int64[1, response_len],

    "hidden_states": None,
    "target_logprobs": None,
    "hidden_states_tq_key": "speco:42:1:req-007",

    "hidden_positions": int64[1, hidden_rows] | None,
    "hidden_position_start": 128,
    "hidden_position_end": 640,
    "global_step": 42,
    "replica_rank": 1,
}
```

因此 PR #48 实际存在三类对象：

| 对象 | 内容 | 传输路径 | 作用 |
|---|---|---|---|
| TQ fields/payload | `hidden_states` 等大 tensor | Producer → TQ storage → Consumer | 避免 driver 搬运大 tensor |
| TQ tag | `global_step`、`replica_rank` | TQ control/index metadata | 描述该 key |
| 轻量 `drafter_sample` | tokens、位置、标量 metadata、`hidden_states_tq_key` | 原 Ray driver 路径 | 告诉 Consumer 取哪个 key，以及如何解释取回的 tensor |

代码实现解耦的关键不是“所有内容都进 TQ”，而是：

```python
drafter_sample["hidden_states_tq_key"] = tq_key
drafter_sample["hidden_states"] = None
```

第一行给 Consumer 留下寻址信息；第二行阻止大 tensor 继续沿旧路径传输。

#### 3.0.5.2 Consumer 如何把两部分重新合成一个训练样本

Consumer 最初拿到的是轻量 sample：

```python
sample["hidden_states"] is None
sample["hidden_states_tq_key"] == "speco:42:1:req-007"
```

它执行：

```python
payload = get_sample(sample["hidden_states_tq_key"])
sample["hidden_states"] = payload["hidden_states"]
```

合并后：

```python
sample = {
    "input_ids": ...,
    "prompts": ...,
    "responses": ...,
    "hidden_positions": ...,
    "hidden_states": bf16[1, hidden_rows, hidden_dim],
    "hidden_states_tq_key": "speco:42:1:req-007",
    ...
}
```

后面的 `_store_rollout_sample()` 看到的结构与关闭 TQ 时基本一致，所以训练主体不需要增加 TQ 分支。TQ bridge 只改变“大 tensor 从哪里恢复”，不改变 drafter trainer 的输入语义。

#### 3.0.6 old-logprob Producer 怎么写 chunk

PR #48 的另一条 Producer 路径来自 actor old-logprob hidden hook。原来是：

```python
chunk_ref = ray.put(hidden_chunk)
```

开启 TQ 后改成：

```python
tq_key = make_sample_key(
    global_step,
    owner,
    f"chunk{len(chunk_refs)}",
)

put_sample(
    tq_key,
    {"hidden": hidden_chunk},
    tag={
        "global_step": global_step,
        "owner": owner,
    },
)

chunk_ref = tq_key
```

对应参考代码：

```text
../verl-SpeCo/verl_speco/integration/oldlogprob_runtime.py:533
```

后面的 driver 不需要知道 ref 是 Ray ObjectRef 还是 TQ string key，它只把 ref 当不透明 token 继续传递。

#### 3.0.7 Consumer 怎么根据 key 读取

drafter worker 收到原来的 sample 小字典后：

```python
tq_key = sample.get("hidden_states_tq_key")

if tq_key is not None and self._speco_tq_enabled:
    payload = get_sample(tq_key)

    for field in (
        "hidden_states",
        "target_logprobs",
        "hidden_raw_target_logprobs",
        "hidden_raw_target_logprobs_positions",
    ):
        if payload.get(field) is not None:
            sample[field] = payload[field]

    if sample.get("hidden_states") is None:
        raise RuntimeError(
            "TQ key exists but hidden_states payload is missing"
        )
```

对应参考代码：

```text
../verl-SpeCo/verl_speco/workers/speco_worker.py:848
```

恢复 tensor 后，后面的逻辑仍使用原 `sample`/`batch`，drafter trainer 不需要知道 tensor 来自 Ray 还是 TQ。

#### 3.0.8 `get_sample()` 实际怎么读

```python
def get_sample(key):
    _ensure_initialized()

    result = tq.kv_batch_get(
        keys=[key],
        partition_id="speco_drafter_features",
    )

    value = _extract_value(result, key)
    return _tensordict_to_dict(value)
```

对应参考代码：

```text
../verl-SpeCo/verl_speco/integration/transferqueue_bridge.py:237
```

`_extract_value()` 兼容三种返回形态：

```python
if isinstance(result, dict):
    return result.get(key)
if isinstance(result, (list, tuple)):
    return result[0]
return result
```

这是因为不同 TQ 版本/后端返回包装可能不同。

#### 3.0.9 为什么需要 `_densify_tq_tensor()`

PR #48 后续修复发现，TQ 把单样本 tensor 放进 TensorDict 后，`kv_batch_get` 可能返回 NestedTensor，并额外带 batch 维。旧代码要执行：

```python
tensor[start:start + length]
```

但 NestedTensor 不支持在 jagged dim 直接 slice。因此加入：

```python
def _densify_tq_tensor(tensor):
    if tensor.is_nested:
        parts = [
            part
            for part in tensor.unbind()
            if part.numel() > 0
        ]
        tensor = torch.cat(parts, dim=0)

    if tensor.dim() == 3:
        tensor = tensor.squeeze(0)
    elif tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)

    return tensor.contiguous()
```

对应参考代码：

```text
../verl-SpeCo/verl_speco/workers/speco_worker.py:72
```

standalone Consumer 同样必须做这个转换，不能假定 `kv_batch_get` 返回普通 dense `[seq, hidden]`。

#### 3.0.10 为什么需要 per-step cache

old-logprob 路径里，一个 owner hidden chunk 可能被约 16 个 sample 共同引用。如果每个 sample 都：

```python
get_sample(same_tq_key)
```

就会重复传输同一个数百 MB chunk。PR #48 在每次 `collect_rollout_features()` 开始时创建：

```python
self._tq_chunk_cache = {}
```

解析 ref 时：

```python
cache_key = ref if isinstance(ref, str) else id(ref)

if cache_key not in cache:
    cache[cache_key] = _resolve_tq_or_ray_ref(ref)

tensor = cache[cache_key]
```

对应参考代码：

```text
../verl-SpeCo/verl_speco/workers/speco_worker.py:98
../verl-SpeCo/verl_speco/workers/speco_worker.py:854
```

独立训练如果每个 sample 都是独立 TQ key，主要使用 `kv_batch_get(keys=[...])` 批量读取，不一定需要跨 sample chunk cache；但 prefetch 重试或共享 packed object 时仍应保留 key cache。

#### 3.0.11 PR #48 什么时候删除数据

PR #48 没有在 `get_sample()` 后删除。其注释明确说明：同一 drafter replica 的多个 TP/SP rank 可能读取同一 key，第一次读取后立刻删除会让剩余 rank 失败。

当前策略是任务结束时由 owner：

```python
tq.close()
```

统一结束 TQ 生命周期。也就是说，PR #48 当前没有实现精细的逐 step `kv_clear`。

#### 3.0.12 PR #48 的完整时序

```text
SpecoTaskRunner
  → tq.init(config)
  → 启动 Ray workers

SGLang/actor Producer process
  → configure_transfer_queue()
  → 第一次 put 时 tq.init()
  → kv_put(key, tensor fields, tag)
  → 把 key 塞回原 sample/ref

Ray driver
  → 只中转小 sample/key

drafter worker Consumer process
  → 第一次 get 时 tq.init()
  → kv_batch_get([key])
  → 解包 TensorDict/NestedTensor
  → 恢复 sample["hidden_states"]
  → 原 drafter collect/train 逻辑

任务结束
  → TaskRunner owner tq.close()
```

### 3.1 已经实现的可复用能力

PR #48 新增 `verl_speco/integration/transferqueue_bridge.py`，锁定思路是把 TQ 当成独立传输库，不修改上游 verl。它提供：

```python
configure_transfer_queue(training_cfg)
init_transfer_queue(config)
make_sample_key(global_step, replica_rank, request_id)
put_sample(key, tensor_dict, tag=...)
get_sample(key)
close_transfer_queue()
```

实际写入调用是：

```python
tq.kv_put(
    key=key,
    partition_id="speco_drafter_features",
    fields=payload,
    tag=tag,
)
```

实际读取调用是：

```python
result = tq.kv_batch_get(
    keys=[key],
    partition_id="speco_drafter_features",
)
```

另外，PR #48 已经处理了多项 standalone 方案也需要的问题：

1. TQ 返回值可能是 direct value、`{key: value}` 或 list，需要统一解包；
2. TQ 返回的 tensor 可能是 NestedTensor，必须通过 `unbind + cat` 恢复为生产端写入的 dense tensor；
3. 多个样本引用同一 hidden chunk 时，需要 per-step cache，避免重复 `kv_batch_get` 同一个大对象；
4. TQ key 存在但 payload 缺失时 fail loud，不能静默丢样本；
5. `enable=false` 时保留原传输路径。

这些逻辑应直接作为本项目 TQ adapter 的参考。

### 3.2 PR #48 的数据流

PR #48 优化的是 RL online 路径：

```text
SGLang/actor worker
  → kv_put(hidden states)
  → 把 hidden_states_tq_key 塞进原 drafter_sample
  → 原 Ray driver 继续传递小 sample/key
  → drafter worker collect_rollout_features()
  → kv_batch_get(key)
```

它没有让 consumer 自己从 TQ 发现下一批 key；key 仍沿原来的 Ray 控制路径到达 drafter worker。

### 3.3 PR #48 没有提供的 standalone 能力

PR #48 当前没有实现：

- 从预生成 response 文件读取数据的独立 Producer；
- Producer 并行请求外部 vLLM endpoint；
- standalone DSpark trainer 主动发现 ready key；
- global batch 到各 torchrun rank 的分片；
- 每个 optimizer step 后精确 `kv_clear`；
- EOS；
- standalone 无 Ray 的 TQ bootstrap；
- MooncakeStore 的实际运行验证。

PR #48 当前配置是：

```yaml
transfer_queue:
  enable: false
  backend:
    storage_backend: SimpleStorage
    SimpleStorage:
      total_storage_size: 100000
      num_data_storage_units: 8
```

并且 bridge 注释针对 `TransferQueue==0.1.7`。旁边当前 verl 主线已使用 `0.1.8` 文案并包含 `MooncakeStore` 配置。当前机器没有安装 `transfer_queue` 包，因此正式实现前必须锁定版本并实机验证 API 签名，不能把 0.1.7 和 0.1.8 混用。

### 3.4 standalone 方案对 PR #48 的扩展

不再自行实现 `publish_ready/claim/lease/ack` HTTP 服务。第一版在 PR #48 KV 模式上补四个操作：

```python
tq.kv_batch_put(...)   # Producer 批量写
tq.kv_list(...)        # rank 0 列出 key + tag
tq.kv_batch_get(...)   # 各 rank 并行读
tq.kv_clear(...)       # optimizer step 成功后删
```

第一版不依赖 TQ `Sampler/StreamingDataLoader/get_meta`，因为 PR #48 并未使用或验证这些接口。等 KV 流程稳定后再升级为 TQ StreamingDataLoader。

### 3.5 PR #48 与 standalone 独立训练逐项映射

| PR #48 online RL | standalone drafter training |
|---|---|
| `SpecoTaskRunner` 调用 `tq.init(config)` | 新增独立 TQ owner/bootstrap 进程，或在确认安全后由 Producer owner 调用 `tq.init(config)` |
| SGLang/actor worker 是 Producer | `feature_producer.py` 是独立 Producer |
| rollout 过程中已经得到 hidden states | Producer 读取预生成 response，再请求外部 vLLM prefill |
| `make_sample_key(global_step, replica_rank, request_id)` | `make_replay_sample_key(dataset,row,tokens,target fingerprint)` |
| `put_sample()` / `kv_put()` | 继续复用同一写入模式，可扩展为 `kv_batch_put()` |
| key 通过 Ray driver/sample 传给 consumer | 没有 driver；rank 0 用 `kv_list()` 主动发现 ready keys |
| drafter worker `get_sample(key)` | 每个 torchrun rank `kv_batch_get(local_keys)` |
| `collect_rollout_features()` 恢复 sample | 转换为 `DraftFeatureSample` 后调用现有 `prepare_training_batch_from_samples()` |
| 多 TP/SP rank 可能读同一 key，所以不立即删除 | data-parallel rank 读取互不重叠的 key；全 rank step 成功后统一 `kv_clear(global_keys)` |
| TaskRunner 结束时 `tq.close()` | 每 step clear；输入 drain 完成后 owner 最后 `tq.close()` |

standalone 需要新增的控制流是：

```text
Producer                            DSpark rank 0                  其他 ranks
   │                                     │                            │
   │ kv_put(sample key, fields, tag)     │                            │
   ├────────────────────────────────────▶│                            │
   │                                     │ kv_list READY keys         │
   │                                     │                            │
   │                                     │ broadcast selected_keys ──▶│
   │                                     │                            │
   │                                     │ kv_batch_get(local keys)   │ kv_batch_get(local keys)
   │                                     │                            │
   │                                     ├──── DSpark synchronized step ────┤
   │                                     │                            │
   │                                     │ kv_clear(global keys)      │
```

这个映射中，TQ 同时承担：

- 大 tensor 存储/传输；
- key、tag 和 partition 的轻量索引。

但第一版 global batch 的选择仍由单个 DSpark job 的 rank 0 完成。这样最接近 PR #48 的 KV API，避免在同一次改造中再引入未经该 PR 验证的 Sampler/StreamingDataLoader。

### 3.6 PR #48 的 SGLang 路径：逐函数、逐对象完整流程

下面从一次生成请求开始，不省略中间层。

#### 阶段 1：SGLang完成生成并收集 hidden states

执行进程：SGLang rollout server。

输入是一次 request 对应的 prompt 和生成配置。生成结束时，代码已经持有：

```python
prompt_tensor: int64[prompt_len]
response_tensor: int64[response_len]
hidden_states: bf16[hidden_rows, hidden_dim]
hidden_positions: int64[hidden_rows] | None
target_logprobs: tensor | None
request_id: str
collection_global_steps: int
self.replica_rank: int
```

这些变量的语义：

- `prompt_tensor`：输入 prompt token IDs；
- `response_tensor`：SGLang生成的 response token IDs；
- `hidden_states`：target model 指定层在部分 token positions 上的输出；
- `hidden_positions`：每个 hidden row 对应完整 `prompt+response` 序列中的哪个 token position；
- `target_logprobs`：可选的目标概率监督；
- `request_id`：当前 rollout request 标识；
- `replica_rank`：执行该 request 的 rollout replica。

SGLang 先构造完整 sample：

```python
drafter_sample = {
    "input_ids": torch.cat(
        [prompt_tensor, response_tensor], dim=0
    ).unsqueeze(0),
    "prompts": prompt_tensor.unsqueeze(0),
    "responses": response_tensor.unsqueeze(0),
    "hidden_states": hidden_states.unsqueeze(0).cpu(),
    "hidden_positions": hidden_positions.unsqueeze(0).cpu(),
    "target_logprobs": (
        target_logprobs.unsqueeze(0).cpu()
        if target_logprobs is not None
        else None
    ),
    "global_step": collection_global_steps,
    "replica_rank": self.replica_rank,
    # 还有 hidden window/alignment metadata
}
```

前导维 `1` 表示这是一个单样本 batch。`.cpu()` 表示跨进程传输前把大 tensor 放到 CPU 内存。

#### 阶段 2：PR #48 将大 fields 写入 TQ

同一个 SGLang进程执行：

```python
tq_key = make_sample_key(
    collection_global_steps,
    self.replica_rank,
    request_id,
)

tq_payload = {
    "hidden_states": hidden_states.unsqueeze(0).cpu(),
}

put_sample(
    tq_key,
    tq_payload,
    tag={
        "global_step": collection_global_steps,
        "replica_rank": self.replica_rank,
    },
)
```

调用展开后是：

```python
tq.init()  # 当前进程第一次使用时
tq.kv_put(
    key=tq_key,
    partition_id="speco_drafter_features",
    fields=tq_payload,
    tag=tag,
)
```

效果是 TQ 中增加一行：

```text
partition = speco_drafter_features
key       = speco:42:1:req-007
fields    = {hidden_states: bf16[1, H, D], ...}
tag       = {global_step: 42, replica_rank: 1}
```

`kv_put` 返回后，SGLang侧将旧 sample 改成：

```python
drafter_sample["hidden_states_tq_key"] = tq_key
drafter_sample["hidden_states"] = None
```

此后该 Python 字典不再携带 hidden tensor，只携带定位它的 key。

#### 阶段 3：把 drafter_sample 放进 TokenOutput.extra_fields

SGLang返回：

```python
TokenOutput(
    token_ids=token_ids,
    log_probs=log_probs,
    routed_experts=routed_experts,
    extra_fields={
        "global_steps": collection_global_steps,
        "drafter_sample": drafter_sample,
    },
)
```

此时 `TokenOutput` 中有两类输出：

- 正常 rollout 输出：`token_ids/log_probs`；
- SpeCo 训练旁路输出：`extra_fields.drafter_sample`。

TQ payload 不在 `TokenOutput` 中，只有 `hidden_states_tq_key` 在其中。

#### 阶段 4：多个 TokenOutput 聚合成 gen_batch_output

rollout/agent-loop 层将多个 request 的输出合并为批量 `DataProto`：

```python
gen_batch_output.non_tensor_batch["drafter_sample"]
```

可能是 object array：

```python
array([
    {"hidden_states_tq_key": "speco:42:0:req-A", ...},
    {"hidden_states_tq_key": "speco:42:1:req-B", ...},
], dtype=object)
```

之所以进入 `non_tensor_batch`，是因为每条 sample 的 hidden window、Python metadata 和可选字段不一定具有统一 dense shape。

#### 阶段 5：driver 从 DataProto 取出 drafter samples

`generate_sequences_with_speco()` 包装原 rollout 调用：

```python
gen_batch_output = original_generate_sequences(...)
collected = self._speco_collect_generation_samples(gen_batch_output)
```

`_speco_collect_generation_samples()` 调用：

```python
samples = pop_drafter_samples(gen_batch_output)
```

`pop_drafter_samples()` 实际执行：

```python
non_tensor_batch = gen_batch_output.non_tensor_batch
samples_array = non_tensor_batch.pop("drafter_sample", None)
samples = normalize_drafter_samples(samples_array)
```

这里 `pop` 有两个作用：

1. 取得 SpeCo drafter side-channel samples；
2. 从正常 PPO 的 `gen_batch_output` 中移除该旁路字段，避免后续 PPO batch 继续携带它。

`normalize_drafter_samples()` 将 dict、object array 或 list 统一成：

```python
samples: list[dict]
```

#### 阶段 6：driver 按 replica_rank 分桶

假设有两个 rollout/drafter replicas，收到：

```python
samples = [
    {"replica_rank": 1, "hidden_states_tq_key": "k1", ...},
    {"replica_rank": 0, "hidden_states_tq_key": "k2", ...},
    {"replica_rank": 1, "hidden_states_tq_key": "k3", ...},
]
```

执行：

```python
buckets = bucket_drafter_samples_by_replica(
    samples,
    num_replicas=2,
)
```

结果：

```python
buckets = [
    [sample_k2],             # bucket 0
    [sample_k1, sample_k3],  # bucket 1
]
```

分桶依据只有：

```python
owner_rank = int(sample["replica_rank"])
buckets[owner_rank].append(sample)
```

这一步没有读取 TQ，也没有处理 hidden tensor；只对小字典做路由。

#### 阶段 7：driver 通过 WorkerGroup RPC 分发 buckets

driver 调用：

```python
self._speco_set_drafter_global_step()
self._speco_collect_rollout_features_rpc(
    "rollout",
    buckets,
)
```

RPC 内部调用：

```python
self.drafter_wg.collect_rollout_features(buckets)
```

因为 worker 方法注册了 `drafter_owner_route` dispatch，WorkerGroup 将 `buckets[0]` 发给 drafter DP replica 0，将 `buckets[1]` 发给 drafter DP replica 1。一个训练 replica 内的 SP ranks 根据 mesh dispatch 规则参与对应调用。

这里传输的对象仍是：

```python
list[dict]
```

其中包含 tokens、position metadata 和 TQ key，不包含被置空的 hidden states。

#### 阶段 8：SpecoWorker 根据 key 从 TQ 恢复 tensor

目标 worker 执行：

```python
def collect_rollout_features(self, samples):
    for sample in samples:
        tq_key = sample.get("hidden_states_tq_key")
        payload = get_sample(tq_key)
        sample["hidden_states"] = payload["hidden_states"]
```

`get_sample()` 展开为：

```python
tq.init()  # 此 Consumer 进程第一次使用时
result = tq.kv_batch_get(
    keys=[tq_key],
    partition_id="speco_drafter_features",
)
payload = _extract_value(result, tq_key)
payload = _tensordict_to_dict(payload)
```

现在 `sample` 再次包含：

```python
{
    "input_ids": ...,
    "hidden_positions": ...,
    "hidden_states": bf16[1, hidden_rows, hidden_dim],
    "hidden_states_tq_key": "...",
}
```

这与关闭 TQ 时 worker 收到的逻辑内容一致。

#### 阶段 9：worker 构造 DrafterBaseTrainer 所需 batch dict

worker 先保留 token fields：

```python
batch = {
    "input_ids": sample["input_ids"],
    "prompts": sample["prompts"],
    "responses": sample["responses"],
}
```

再复制 hidden alignment metadata，例如：

```python
batch["hidden_positions"]
batch["hidden_position_start"]
batch["hidden_position_end"]
batch["hidden_states_layout"]
batch["global_step"]
```

hidden tensor 单独作为参数：

```python
self._store_rollout_sample(
    batch=batch,
    hidden_states=hidden,
    target_logprobs=target_logprobs,
)
```

#### 阶段 10：样本进入在线 buffer 或落盘

`_store_rollout_sample()` 根据 training mode 分支：

```python
if mode == "collect_only":
    self._write_rollout_feature_sample(
        batch,
        hidden_states,
        target_logprobs,
    )
else:
    self.trainer.collect_online_data(
        batch,
        hidden_states,
        target_logprobs,
    )
```

`collect_only` 会转换成 `DraftFeatureSample` 并写 `TorchShardFeatureStore`。online 模式则进入 `DrafterBaseTrainer.collect_online_data()`。

`collect_online_data()` 做：

1. 将 `input_ids/hidden_states/positions/logprobs` 规范化到 CPU；
2. 按 batch 维拆成逐样本；
3. 根据 `hidden_positions` 校验 hidden row 与 token position；
4. 截取可训练窗口；
5. 构造内部 training item；
6. 保存到当前 step 的 `collected_data`，或在启用 data buffer 时保存到跨 step buffer。

因此 TQ get 完成不代表立即 optimizer step。它先恢复 online training sample，再进入现有数据准备逻辑。

#### 阶段 11：driver 在 actor update 周期触发 drafter 训练

driver 包装了 `update_actor()`：

```python
should_train_drafter = (
    self._speco_should_attempt_drafter_train_this_step()
)

actor_output = original_update_actor(...)

if should_train_drafter:
    drafter_trained, metrics = self._speco_train_drafter()
```

`_speco_train_drafter()` 再向 WorkerGroup 发：

```python
self.drafter_wg.train_drafter()
```

每个 `SpecoWorker.train_drafter()`：

1. 检查是否属于 drafter training group；
2. 检查 `training_interval_steps`；
3. 激活 drafter training model；
4. 循环 `train_steps_per_trigger` 次；
5. 每次调用 `self.trainer.training_step(global_step)`；
6. 成功时准备需要发布的 drafter state dict；
7. 清理训练期间临时状态。

`training_step()` 从刚才的 online `collected_data/DataBuffer` 组成 batch，执行 drafter forward、loss、backward 和 optimizer step。

所以 SGLang TQ 路径的最终效果是：

```text
TQ 只替换 hidden tensor 跨进程传输
→ sample 收集逻辑不变
→ online buffer 不变
→ drafter training trigger 不变
→ loss/optimizer 不变
```

### 3.7 PR #48 old-logprob 路径的完整差异

old-logprob 路径没有 `TokenOutput.extra_fields`。它从 PPO 的 actor old-logprob forward 开始。

#### 阶段 1：driver 构造 collect plan

driver 根据 batch、collect interval 和 drafter owner 数量决定：

```python
collect_mask: bool[batch]
hidden_positions: list/tensor per sample
owner_rank: int64[batch]
prompt_lens: int64[batch]
response_lens: int64[batch]
```

并把 `global_step` 等控制字段放入 old-logprob micro-batch。

#### 阶段 2：actor forward hook 选择 hidden rows

actor worker 在 old-logprob forward 中捕获指定层 hidden states，根据 `collect_mask/position_mask` 只保留需要训练的样本和 token rows。

输出可以是 dense selected tensor，也可以是 sparse rows。随后 `_put_oldlogprob_hidden_refs()` 将同一个 owner 的多条 sample rows 拼成一个较大的 `hidden_chunk`。

#### 阶段 3：hidden chunk 写入 TQ

改造前：

```python
chunk_ref = ray.put(hidden_chunk)
```

PR #48：

```python
tq_key = make_sample_key(
    global_step,
    owner,
    f"chunk{chunk_index}",
)

put_sample(
    tq_key,
    {"hidden": hidden_chunk},
    tag={"global_step": global_step, "owner": owner},
)

chunk_ref = tq_key
```

TQ fields：

```python
{"hidden": bf16[total_owner_rows, hidden_dim]}
```

控制路径中的 chunk metadata：

```python
chunk_info = {
    "sample_indices": [0, 3, 5],
    "starts": [0, 128, 384],
    "lengths": [128, 256, 96],
    "row_indices": [...],
    "dtype": "bfloat16",
    "shape": [480, hidden_dim],
}
```

`starts/lengths` 描述每条 sample 在共享 chunk 中对应的行区间。

#### 阶段 4：driver 将 chunk ref 还原为逐样本引用

driver 的 `_speco_collect_oldlogprob_features()` 读取：

```python
chunk_refs = ["speco:42:0:chunk0", ...]
chunk_meta = [chunk_info, ...]
```

然后为每个 batch sample 构造：

```python
sample["hidden_states_ref_chunks"] = [
    {
        "ref": "speco:42:0:chunk0",
        "chunk_start": 128,
        "chunk_length": 256,
        "chunk_row_indices": ...,
        "dtype": "bfloat16",
        "shape": [480, hidden_dim],
    }
]
```

同时构造该 sample 的：

```python
input_ids
prompts
responses
hidden_positions
hidden_states_layout
replica_rank=owner
```

再按 owner 放入 `buckets[owner]`，通过同一个 `collect_rollout_features()` RPC 发给 drafter worker。

#### 阶段 5：Consumer 获取共享 chunk 并切片

drafter worker 发现：

```python
sample.get("hidden_states") is None
sample.get("hidden_states_ref_chunks") is not None
```

于是调用 `_resolve_hidden_state_chunks()`。对字符串 ref：

```python
if ref.startswith("speco:"):
    full_chunk = get_sample(ref)["hidden"]
    full_chunk = _densify_tq_tensor(full_chunk)
```

然后按 sample metadata 取行：

```python
sample_hidden = full_chunk[
    chunk_start : chunk_start + chunk_length
]
```

同一个 chunk 被多个 sample 复用，所以使用：

```python
self._tq_chunk_cache[ref] = full_chunk
```

保证一次 `collect_rollout_features()` 中同一个 TQ key 只 get 一次。

得到逐样本 hidden 后，后续 `_store_rollout_sample → collect_online_data → train_drafter` 与 SGLang 路径相同。

### 3.8 PR #48 数据生命周期和清理

PR #48 的 TQ row 生命周期是：

```text
TaskRunner tq.init(config)
→ Producer kv_put
→ key 经 Ray 控制路径传递
→ 一个或多个 drafter TP/SP rank kv_batch_get
→ online drafter 收集/训练继续执行
→ 整个 trainer.fit() 结束
→ TaskRunner finally 调用 tq.close()
```

当前没有：

```python
tq.kv_clear(key)
```

原因是同一 sample/key 可能被一个 drafter replica 的多个 rank 读取。若第一个 rank get 后删除，其余 rank 可能 get 失败。

因此 PR #48 采用任务级生命周期，而不是样本级确认和回收。这简化了并发正确性，但意味着长任务中的 TQ storage 占用可能持续增长；代码注释也把“leader 在 barrier 后精细 clear”留作后续工作。

### 3.9 PR #48 开启与关闭时的行为差异

`configure_transfer_queue()` 返回：

```python
enabled_in_config and transfer_queue_importable
```

关闭时：

```text
SGLang drafter_sample 继续内联 hidden_states
old-logprob 继续 ray.put(hidden_chunk)
Consumer 继续 ray.get/ref resolve
```

开启时：

```text
SGLang hidden fields → TQ，sample 只带 key
old-logprob hidden chunk → TQ，ref 变成字符串 key
Consumer 根据 key 类型走 TQ get
```

如果配置开启但 `transfer_queue` 包未安装，bridge 会记录 warning，并让 `is_transfer_queue_enabled()` 返回 false，保留旧 Ray 路径。若已经进入 `put_sample/get_sample` 却发生 TQ 错误，则抛异常，不静默丢 hidden data。

## 第二部分：基于 PR #48 的 standalone drafter training 适配

从这一部分开始才讨论 `verl-SpeCo-ls` 的独立训练。下面的 `kv_list`、rank 0 选 global keys、逐 step `kv_clear`、独立 vLLM Producer 都是需要在本项目新增的逻辑，不是 PR #48 已有逻辑。

### 当前 standalone 基线

当前独立训练是：

```text
draft_train_launcher
→ torch.distributed.run
→ 每个 rank 创建 DraftFeatureDataLoader
→ 每个 rank 在训练循环内同步 TargetFeatureReplayer.materialize()
→ vLLM/file hidden payload
→ prepare_training_batch_from_samples()
→ training_step_from_batch()
```

新方案要把 `TargetFeatureReplayer.materialize()` 从训练 rank 的同步取数路径移到独立 Producer，同时保持后两步训练接口不变。

## 4. TQ metadata 到底记录什么

### 4.1 Partition

一次训练运行使用一个独立 partition：

```python
partition_id = f"speco:{run_id}:dspark_train"
```

partition 用来隔离：

- 不同训练 run；
- train 和 validation；
- 不同 target checkpoint 生成的 hidden states。

不能让两个 target 模型共用同一 partition，否则训练侧可能消费错误的 hidden states。

### 4.2 Sample key

每条输入样本使用稳定 key：

```python
sample_key = sha256(
    dataset_id
    + row_id
    + prompt_token_ids
    + response_token_ids
    + tokenizer_fingerprint
    + target_model_fingerprint
    + target_layer_ids
    + hidden_states_layout
).hexdigest()
```

稳定 key 用于：

- vLLM HTTP 请求重试时不生成不同对象；
- Producer 重启后识别相同样本；
- 检查 hidden states 是否属于正确模型和正确层；
- TQ/Mooncake 清理时准确定位对象。

### 4.3 Fields 与 READY 约定

每个样本包含固定字段：

```python
{
    "input_ids": int64[seq],
    "loss_mask": float32[seq],
    "position_ids": int64[seq],
    "hidden_states": bf16[seq, aux_hidden_dim or aux_hidden_dim+hidden_size],
}
```

这里必须保持当前 `DraftFeatureSample` 的契约：`TargetFeatureReplayer._feature_from_vllm_payload()` 会把多个 aux layers flatten；当 layout 是 `dflash_aux_plus_last` 时，还会把 final hidden 拼到同一个 `hidden_states` tensor 尾部。`DSparkTrainerBackend.preprocess_individual_items()` 再根据 metadata 中的 `hidden_states_layout` 将 final hidden 切出来，生成训练 batch 的 `target_last_hidden_states`。

因此 TQ 不需要新增一个独立 `target_last_hidden_states` field。开启 DSpark L1 时，要求：

```text
metadata.hidden_states_layout = dflash_aux_plus_last
hidden_states.shape[-1] = num_context_layers * hidden_size + hidden_size
```

完整 TQ native metadata API 可以做字段级 ready 判定，但 PR #48 当前走的是高层 KV API：一次 `kv_put` 把一个样本的多个 tensor fields 一起写入。因此第一版采用更直接的约定：

```python
required_fields = [
    "input_ids",
    "loss_mask",
    "position_ids",
    "hidden_states",
]
```

Producer 先在内存中验证所有必需字段，再进行一次 `kv_put`。只有 `kv_put` 成功返回，key 才会出现在 `kv_list` 结果中，并带有：

```python
tag={
    "status": "ready",
    "run_id": run_id,
    "sample_id": sample_key,
}
```

Consumer 只选择 `status=ready` 且 `run_id` 匹配的 key。不要先 put `input_ids`、再单独 put `hidden_states`，否则 Consumer 可能观察到半成品。

### 4.4 Tags

tags 是轻量 metadata，不放大 tensor：

```python
tags = {
    "sample_id": sample_key,
    "source_row": row_id,
    "seq_len": seq_len,
    "payload_bytes": payload_bytes,
    "target_model_fp": target_model_fingerprint,
    "target_layers": "8,16,24",
    "hidden_layout": "dflash_aux_plus_last",
    "producer_status": "success",
}
```

tags 可用于过滤、监控、背压统计和错误排查。它不能替代 tensor shape/dtype 校验。

### 4.5 Run ID，而不是先依赖 task_name

PR #48 的 `kv_put/kv_batch_get` 路径没有使用 `task_name` 或 Sampler consumption history。第一版按它的已验证接口，在 partition 和 tags 中放 `run_id`：

```python
partition_id = "speco_drafter_features"
tag = {
    "run_id": run_id,
    "status": "ready",
}
```

不同 run 最好直接使用不同 partition：

```python
partition_id = f"speco_drafter_features_{run_id}"
```

这样 job 重启和清理更简单。`task_name=dspark_train` 留到后续迁移 TQ native metadata/Sampler 时再使用。

### 4.6 standalone 中一条样本的完整对象形态

standalone 没有 PR #48 的轻量 `drafter_sample → Ray driver` 路径，因此需要让 TQ tag 承担“如何找到和解释 payload”的 metadata 作用。

#### Producer 读到的原始记录

```python
source_record = {
    "dataset_id": "math-train",
    "row_id": 12345,
    "prompt": "...",
    "response": "已经提前生成的 response",
}
```

#### Token replay 样本

分词和对齐后：

```python
replay_sample = DraftReplaySample(
    input_ids=int64[full_seq],
    loss_mask=float32[full_seq],
    position_ids=int64[full_seq],
    feature_positions=int64[feature_rows],
    draft_position_ids=int64[feature_rows],
    metadata={
        "dataset_id": "math-train",
        "row_id": 12345,
    },
)
```

这里的 `full_seq` 是 prompt 与预生成 response 拼接后的长度；`feature_positions` 指明哪些 token 位置最终进入 drafter 监督样本。

#### vLLM 返回的原始 hidden payload

当前文件协议要求 safetensors 至少包含：

```python
vllm_payload = {
    "token_ids": int64[prefill_rows],
    "hidden_states": bf16[prefill_rows, returned_layers, hidden_size],
}
```

这还不能直接给 DSpark。Producer 应复用当前：

```python
TargetFeatureReplayer._feature_from_vllm_payload(...)
```

完成 token 校验、position 对齐、选层和 flatten。

#### Producer 最终得到的 DraftFeatureSample

```python
feature = DraftFeatureSample(
    algorithm="DSpark",
    input_ids=int64[feature_rows],
    loss_mask=float32[feature_rows],
    position_ids=int64[feature_rows],
    hidden_states=bf16[feature_rows, feature_hidden_dim],
    metadata={
        "hidden_states_layout": "dflash_aux_plus_last",
        "target_layer_ids": [8, 16, 24],
        "target_model_path": "...",
        "target_config_fingerprint": "...",
        "feature_start": 128,
        "feature_end": 640,
        "sequence_length": 512,
    },
)
```

若有 3 个 aux layer、target hidden size 为 4096，并包含 final hidden：

```text
feature_hidden_dim = 3 * 4096 + 4096 = 16384
hidden_states.shape = [feature_rows, 16384]
```

前 `12288` 维是 aux/context hidden，最后 `4096` 维是 DSpark L1 所需 final hidden。现有 DSpark backend 会根据 `dflash_aux_plus_last` 自动拆分。

#### 写入 TQ 的 data fields

第一版建议一个 key 对应一个已经规范化完成的 `DraftFeatureSample`：

```python
tq_fields = {
    "input_ids": feature.input_ids.cpu(),
    "loss_mask": feature.loss_mask.cpu(),
    "position_ids": feature.position_ids.cpu(),
    "hidden_states": feature.hidden_states.cpu(),
}
```

这里只放 tensor，因为 PR #48 的 `put_sample()` 会过滤非 tensor：

```python
payload = {
    key: value
    for key, value in tensor_dict.items()
    if torch.is_tensor(value)
}
```

#### 写入 TQ 的 tag metadata

```python
tq_tag = {
    "run_id": "run-20260818-001",
    "status": "ready",
    "sample_id": sample_key,
    "sequence_no": 12345,
    "algorithm": "DSpark",
    "hidden_states_layout": "dflash_aux_plus_last",
    "target_model_fingerprint": "sha256:...",
    "target_layer_ids": "8,16,24",
    "feature_rows": 512,
    "hidden_dim": 16384,
    "payload_bytes": 16777216,
}
```

tag 中只放 TQ 版本支持序列化的小标量/字符串。列表等复杂对象可以编码为稳定字符串或 JSON。`payload_bytes` 用于背压统计。

#### TQ 中逻辑上保存的 row

```text
partition: speco_drafter_features_run-20260818-001
key:       86a4...ef2

fields:
  input_ids       → int64[512]
  loss_mask       → float32[512]
  position_ids    → int64[512]
  hidden_states   → bf16[512, 16384]

tag:
  status          → ready
  sequence_no     → 12345
  hidden_layout   → dflash_aux_plus_last
  target_model_fp → sha256:...
```

#### Consumer 恢复出的对象

rank 0 通过 `kv_list` 同时取得 key 和 tag；每个 rank 用 local keys 调用 `kv_batch_get` 取得 fields，然后组合：

```python
feature = DraftFeatureSample(
    algorithm=tag["algorithm"],
    input_ids=densify(fields["input_ids"]).reshape(-1),
    loss_mask=densify(fields["loss_mask"]).reshape(-1),
    position_ids=densify(fields["position_ids"]).reshape(-1),
    hidden_states=densify(fields["hidden_states"]),
    metadata={
        "hidden_states_layout": tag["hidden_states_layout"],
        "target_model_fingerprint": tag["target_model_fingerprint"],
    },
)

feature.validate(strict=True)
```

这样传给：

```python
trainer.prepare_training_batch_from_samples([feature, ...])
```

的数据结构，与现有文件 feature store 读出的 `DraftFeatureSample` 一致。也就是说，TQ 替换的是存取介质和调度方式，不改变 DSpark backend 的样本契约。

## 5. 新的整体架构

```text
                              小 metadata
                    ┌──────────────────────────┐
                    │ TransferQueueController  │
                    │ KV metadata / key / tags │
                    │ partition / storage map  │
                    └────────────┬─────────────┘
                                 │
JSONL/token replay               │
        │                        │
        ▼                        │
Feature Producer                 │
  ├─ tokenizer/window            │
  ├─ asyncio bounded concurrency │
  ├─ vLLM endpoint pool          │
  ├─ validate/pack               │
  └─ TQ put ─────────────────────┤
                                 ▼
                      TQ Mooncake backend
                       hidden-state tensors
                                 │
             ┌───────────────────┼───────────────────┐
             ▼                   ▼                   ▼
       DSpark rank 0       DSpark rank 1       DSpark rank N
          TQ get              TQ get              TQ get
             └───────────────────┼───────────────────┘
                                 ▼
                     synchronized optimizer step
                                 │
                                 ▼
                      TQ clear after success
```

大 tensor 的路径是：

```text
vLLM/Producer memory → TQ Mooncake backend → each training rank
```

不会走：

```text
Mooncake → rank 0 → rank 1/2/3
```

rank 0 最多只广播 `kv_list` 得到的 key 字符串列表；tags 只在选 batch 时由 rank 0 使用。

### 5.1 standalone 每一步为什么能实现推理和训练异步

#### 步骤 A：Producer 独立推进输入 cursor

Producer 自己维护：

```python
reader_cursor = 12346
```

它不等待 Trainer 请求样本。只要 TQ ready bytes 没超过背压上限，就继续读取文件并创建 vLLM task。

效果是 Producer 的执行进度与 `optimizer_step` 解耦：

```text
Producer sequence_no: 1200,1201,1202,...
Trainer optimizer_step: 87
```

两者通过 TQ 中的 ready rows 衔接，不互相直接调用。

#### 步骤 B：并发 vLLM task 完成顺序可以乱序

例如 Producer 同时提交：

```text
sequence_no 100 → endpoint 0
sequence_no 101 → endpoint 1
sequence_no 102 → endpoint 0
```

完成顺序可能是：

```text
101 → 100 → 102
```

每个 task 完成后独立执行 `kv_put`，所以慢请求不会阻塞已经完成的请求写入。tag 中的 `sequence_no` 保留原数据顺序。

#### 步骤 C：`kv_put` 成功是 READY 可见性的边界

Producer 在调用前已经得到完整 `DraftFeatureSample`。一次 `kv_put` 写入该 sample 的全部 tensor fields，并在 tag 中标记 `status=ready`。

因此 Consumer 的判断规则是：

```text
kv_list 能列出该 key
且 tag.run_id 匹配
且 tag.status == ready
→ 可以尝试 kv_batch_get
```

Consumer 仍需对取回 fields 做完整性校验；tag 是调度 metadata，不是正确性证明。

#### 步骤 D：rank 0 只负责选 key

rank 0 执行：

```python
entries = list_ready_keys()
selected = sorted(entries, key=sequence_no)[:global_batch_size]
```

这一步处理的数据只是：

```python
[
    {"key": "k100", "sequence_no": 100, ...},
    {"key": "k101", "sequence_no": 101, ...},
]
```

不包含 `[seq, hidden_dim]` hidden tensor，所以 rank 0 不成为大数据中转瓶颈。

#### 步骤 E：广播保证所有 rank 对同一个 global step 达成一致

所有 rank 调用同一次：

```python
dist.broadcast_object_list(holder, src=0)
```

广播结束后，每个 rank 看到完全相同的 global key list。然后按确定性区间切分：

```text
rank 0: keys[0:per_rank]
rank 1: keys[per_rank:2*per_rank]
...
```

这样不会出现 rank 0 训练 batch A、rank 1 训练 batch B 数量不同，或者某个 rank 没进入 backward 的情况。

#### 步骤 F：各 rank 直接读取 Mooncake 后端

每个 rank 执行：

```python
tq.kv_batch_get(keys=local_keys, partition_id=partition_id)
```

TQ 根据 key 找到 storage backend 中的数据。使用 MooncakeStore 时，大 tensor 数据路径是 Mooncake → 本 rank；rank 0 不读取其他 rank 的 local payload。

因此：

```text
控制面：rank 0 → broadcast small keys
数据面：Mooncake → each rank directly
```

#### 步骤 G：恢复现有 DraftFeatureSample 契约

每个 rank 将 TQ fields 与 tag 合并、densify、校验，得到 `list[DraftFeatureSample]`。从这里开始继续执行项目现有代码：

```python
batch = trainer.prepare_training_batch_from_samples(
    materialized_samples,
    step=optimizer_step,
)

ok = await trainer.training_step_from_batch(
    batch,
    optimizer_step,
)
```

所以 TQ 不进入 DSpark model/backend 内部，训练数学逻辑不变。

#### 步骤 H：全 rank 成功以后才能清理

每个 rank 的 `ok` 通过现有 `_all_ranks_true()` 聚合：

```text
rank 0 ok = true
rank 1 ok = true
rank 2 ok = true
rank 3 ok = true
→ global_ok = true
```

只有此时 rank 0 执行：

```python
tq.kv_clear(keys=global_keys, partition_id=partition_id)
```

这样保证被删除的数据已经参与完成的 optimizer step。若任何 rank get/OOM/backward 失败，不执行 clear，便于作业失败后的诊断或恢复。

#### 步骤 I：异步重叠如何形成

时间线上：

```text
时间 ─────────────────────────────────────────▶

Producer:  vLLM(batch N+1) ─ put ─ vLLM(batch N+2) ─ put
Trainer:       get(batch N) ─ train(batch N) ─ get/train(batch N+1)
```

Producer 和 Trainer 是不同进程，互相不调用；TQ ready rows 是缓冲区。因此 vLLM prefill、网络传输和 DSpark GPU 训练可以重叠。背压只在缓冲区达到容量上限时暂停 Producer。

## 6. Producer：读取预生成 response 并并行请求 vLLM

### 6.1 输入处理

Producer 从现有 JSONL/token replay 数据源读取：

```python
sample = {
    "row_id": "12345",
    "prompt": "...",
    "response": "提前生成好的文本",
}
```

构造：

```python
prompt_ids = tokenizer.encode(sample["prompt"])
response_ids = tokenizer.encode(sample["response"])
input_ids = prompt_ids + response_ids
```

同时产生：

```python
loss_mask
position_ids
feature_positions
sample_key
```

### 6.2 有界并发

不能按样本串行请求：

```python
for sample in samples:
    result = request_vllm(sample)
```

改成：

```python
async def run_producer(samples):
    semaphore = asyncio.Semaphore(max_inflight_requests)

    async def run_one(sample):
        async with semaphore:
            result = await vllm_pool.prefill(sample)
            feature = validate_and_pack(sample, result)
            await tq_transport.put(feature)

    async with asyncio.TaskGroup() as group:
        for sample in samples:
            group.create_task(run_one(sample))
```

`max_inflight_requests` 是 Producer 同时未完成的请求数量，不是 vLLM batch size。vLLM 服务端仍会对同时到达的请求做自己的 continuous batching。

### 6.3 多 endpoint

多个 endpoint 例如：

```yaml
vllm_endpoints:
  - http://node0:8000/v1
  - http://node1:8000/v1
  - http://node2:8000/v1
```

调度器维护每个 endpoint 的 inflight 数：

```python
endpoint = min(
    endpoints,
    key=lambda item: item.inflight,
)
```

请求前 `inflight += 1`，在 `finally` 中 `inflight -= 1`。失败只重试对应样本，不阻塞全部 Producer。

### 6.4 当前 vLLM 文件桥接

当前客户端协议期望：

```python
response.kv_transfer_params["hidden_states_path"]
```

所以第一阶段仍然是：

```text
vLLM 写临时 safetensors
→ Producer load_file
→ 校验 token_ids/hidden_states
→ TQ put 到 Mooncake backend
→ TQ put 成功后删除临时文件
```

删除必须发生在 TQ put 成功之后：

```python
path = request_vllm_hidden_file(sample)
try:
    feature = load_and_validate(path)
    await tq_transport.put(feature)
finally:
    if put_succeeded:
        Path(path).unlink(missing_ok=True)
```

### 6.5 目标版本：vLLM 直接写 TQ/Mooncake

目标响应可改成：

```json
{
  "kv_transfer_params": {
    "backend": "transfer_queue",
    "partition_id": "speco:run-1:dspark_train",
    "sample_key": "abc123"
  }
}
```

服务端顺序必须是：

```text
prefill
→ 捕获指定层 hidden states
→ TQ/Mooncake put 完成
→ 返回 HTTP success 和 sample key
```

这需要定制 vLLM exporter；当前 `verl-SpeCo-ls` 中没有服务端 writer 实现。

## 7. 按 PR #48 扩展 TQ bridge

不要重新发明一套 transport。将 PR #48 的 bridge 设计移植到本项目并增加 standalone 所需方法：

```python
class StandaloneTQTransport:
    def put_sample(self, key, tensor_dict, tag): ...
    def list_ready_keys(self, run_id): ...
    def get_samples(self, keys, fields=None): ...
    def clear_samples(self, keys): ...
    def put_control(self, key, tag): ...
    def close(self): ...
```

写入延续 PR #48 的真实形式：

```python
tq.kv_put(
    key=key,
    partition_id=partition_id,
    fields={
        "input_ids": feature.input_ids.cpu(),
        "loss_mask": feature.loss_mask.cpu(),
        "position_ids": feature.position_ids.cpu(),
        "hidden_states": feature.hidden_states.cpu(),
    },
    tag={
        "run_id": run_id,
        "status": "ready",
        "sequence_no": sequence_no,
        "payload_bytes": payload_bytes,
    },
)
```

批量读取延续 PR #48 的 `kv_batch_get`：

```python
result = tq.kv_batch_get(
    keys=keys,
    partition_id=partition_id,
    fields=required_fields,  # 0.1.7 是否支持该参数需实机确认
)
```

新增发现和清理：

```python
items = tq.kv_list(partition_id=partition_id)
tq.kv_clear(keys=keys, partition_id=partition_id)
```

这里的 `kv_list/kv_clear` 参数名必须根据锁定的 TQ 版本验证。当前参考仓库只实际调用了 `kv_put/kv_batch_get`，没有为这两个接口提供运行证据。

读取结果继续复用 PR #48 的两个适配函数：

```python
value = _extract_value(result, key)
row = _tensordict_to_dict(value)
row["hidden_states"] = _densify_tq_tensor(row["hidden_states"])
```

## 8. DSpark 多 rank 如何消费

### 8.1 第一版：rank 0 用 kv_list 发现 READY keys

PR #48 中 key 由 Ray driver 传给 drafter worker；standalone 没有这条控制路径，所以 rank 0 需要主动列出 key：

```python
rank = dist.get_rank()
world_size = dist.get_world_size()
global_batch_size = batch_size_per_gpu * world_size

if rank == 0:
    entries = tq_transport.list_ready_keys(run_id=run_id)
    entries.sort(key=lambda x: (x.tag["sequence_no"], x.key))
    selected_keys = [x.key for x in entries[:global_batch_size]]
else:
    selected_keys = None

holder = [selected_keys]
dist.broadcast_object_list(holder, src=0)
selected_keys = holder[0]
```

`sequence_no` 是输入文件顺序。它让多个 Producer 并发完成顺序不同的情况下，Trainer 仍能确定性地组成 batch。

rank 0 广播的是字符串 key 列表，不是 hidden-state tensor。

### 8.2 各 rank 切自己的 keys

例如 global batch keys：

```text
[s0, s1, s2, s3, s4, s5, s6, s7]
```

world size 为 4、每卡 batch size 为 2：

```text
rank 0 → [s0, s1]
rank 1 → [s2, s3]
rank 2 → [s4, s5]
rank 3 → [s6, s7]
```

代码：

```python
def shard_keys(keys, rank, world_size):
    assert len(keys) % world_size == 0
    per_rank = len(keys) // world_size
    start = rank * per_rank
    end = start + per_rank
    return keys[start:end]
```

### 8.3 每个 rank 并行 get

所有进程执行：

```python
local_keys = shard_keys(
    selected_keys,
    rank=rank,
    world_size=world_size,
)

local_payloads = tq_transport.get_samples(local_keys)
```

数据路径：

```text
rank 0 ← Mooncake(s0,s1)
rank 1 ← Mooncake(s2,s3)
rank 2 ← Mooncake(s4,s5)
rank 3 ← Mooncake(s6,s7)
```

不是 rank 0 get 全部后再 scatter。

### 8.4 转成当前训练格式

TQ 返回的数据需要先按 PR #48 的规则解包、densify，再转换成现有 `DraftFeatureSample`：

```python
def tq_row_to_feature(row, tag):
    return DraftFeatureSample(
        algorithm="DSpark",
        input_ids=row["input_ids"],
        loss_mask=row["loss_mask"],
        position_ids=row["position_ids"],
        hidden_states=row["hidden_states"],
        metadata={
            "hidden_states_layout": tag["hidden_states_layout"],
            **row.get("metadata", {}),
        },
    )
```

`hidden_states_layout` 不能只放在无法恢复的临时 Python 对象里；应随 TQ tag 保存。rank 0 从 `kv_list` 取得 key/tag 后，需要把每个 local key 对应的 tag 一起交给本 rank 的转换逻辑。Consumer 必须把 layout 放回 `DraftFeatureSample.metadata`。DSpark L1 所需的 `target_last_hidden_states` 随后由现有 backend 从拼接的 `hidden_states` 中切出，transport 层不计算 loss，也不拆 layout。

## 9. 修改当前训练循环

在 `run_standalone_draft_training()` 中增加数据源分支：

```python
feature_store_type = str(feature_store_cfg.get("type", "torch_shard"))

if feature_store_type == "transfer_queue":
    tq_stream = build_transfer_queue_stream(
        config=config,
        rank=rank,
        world_size=world_size,
    )
    store = None
    loader = None
    feature_replayer = None
else:
    store = build_feature_store_from_config(
        feature_store_cfg,
        read_only=True,
    )
    loader = DraftFeatureDataLoader(...)
```

流式训练循环：

```python
while successful_steps < max_steps:
    global_keys, materialized_samples = tq_stream.next_local_batch()

    batch = trainer.prepare_training_batch_from_samples(
        materialized_samples,
        step=optimizer_step,
    )

    has_batch = batch is not None
    if not _all_ranks_true(has_batch, trainer.runtime_device):
        raise RuntimeError("at least one rank failed to fetch its TQ batch")

    ok = await trainer.training_step_from_batch(
        batch,
        optimizer_step,
    )

    if not _all_ranks_true(ok, trainer.runtime_device):
        raise RuntimeError("DSpark step failed on at least one rank")

    dist.barrier()
    if rank == 0:
        tq_stream.clear_global_batch(global_keys)
    dist.barrier()
```

TQ 接在数据输入层，不放进 `DSparkTrainerBackend`。Backend 只负责模型、forward、loss、backward 和 optimizer。

## 10. READY key、inflight key 和训练提交

### 10.1 Ready

在本方案中 ready 表示：

> `dspark_train` 所需的所有 fields 已经成功写入 TQ storage，训练侧可以读取。

第一版不是通过 PR #48 尚未验证的 Sampler 自动判定，而是 Producer 完整校验 payload 后一次 `kv_put`，并设置 `tag.status=ready`。rank 0 的 `kv_list` 只选择这些 key。

### 10.2 Inflight key

rank 0 选出一个 global batch 后，在本地保存：

```python
inflight_global_keys = selected_keys
```

其他 step 不应再次选中这批 key。因为 standalone 只有一个同步 DSpark job，第一版由 rank 0 进程内 `set` 排除 inflight key 即可：

```python
ready = [x for x in listed if x.key not in inflight_keys]
```

若以后允许多个独立 Trainer job 同时消费同一 partition，进程内 set 就不够，届时必须使用 TQ Controller/Sampler 的原子消费分配能力。

### 10.3 Optimizer committed

optimizer committed 表示所有 DSpark rank 已经完成：

```text
forward → backward → gradient synchronization → optimizer.step
```

它比 `kv_list` 发现 key、甚至 `kv_batch_get` 取出 tensor 都更晚。

第一版推荐简单语义：

```text
TQ 负责 key/tag 和 tensor 传输
rank 0 负责单 Trainer job 的 batch 选择和 inflight set
训练失败 → 整个作业 fail-fast
训练成功 → kv_clear payload，并从 inflight set 移除
恢复 → 从最近 checkpoint + 输入 cursor 重新启动
```

这样不需要重新实现 lease/ack 状态机，也没有夸大 PR #48 当前尚未使用的 Sampler 能力。

## 11. 为什么训练完一个 step 才清理

不能在 `kv_batch_get()` 后立即 clear：

```text
get 成功
→ clear
→ forward OOM
→ 数据已不存在，无法重试
```

正确顺序：

```text
rank 0..N get
→ 所有 rank 确认 batch 有效
→ training_step_from_batch
→ _all_ranks_true(ok)
→ rank 0 kv_clear global keys
```

当前 `_all_ranks_true()` 已经是项目现有的跨 rank 同步工具，可以继续复用。

## 12. 背压

背压表示 Producer 生成速度高于 Trainer 消费速度时，Producer 必须暂停继续提交，以免 Mooncake 内存无限增长。

建议限制：

```yaml
max_vllm_inflight_requests: 32
max_pending_put_bytes: 8589934592
max_tq_ready_samples: 256
max_tq_ready_bytes: 68719476736
```

Producer 在 tags 中写：

```python
{"payload_bytes": payload_bytes}
```

周期性通过 TQ list/metadata 统计当前 partition 尚未消费的数据量：

```python
while ready_bytes >= max_tq_ready_bytes:
    await asyncio.sleep(backpressure_poll_interval)
```

如果锁定的 TQ 版本提供容量/ready 统计接口，应直接使用，避免全量扫描 keys。

## 13. Stable ID、幂等和孤儿数据

### 13.1 幂等

幂等表示同一操作重复执行，最终逻辑结果仍只有一份。

Producer 对同一样本重试时必须使用相同 `sample_key`：

```python
await tq.put(key="abc123", ...)
await tq.put(key="abc123", ...)
```

不能每次生成随机 key：

```text
abc123-retry-1
abc123-retry-2
```

否则一个输入可能训练多次并持续占用 Mooncake。

### 13.2 孤儿数据

孤儿数据表示 tensor 已经写进 storage，但由于 Producer 崩溃或 metadata 更新失败，没有进入正常消费路径。

使用 TQ 后，metadata 和 storage 由同一套系统管理，可以减少“Mooncake有对象、自研 Coordinator 没记录”的双系统窗口，但仍要配置 TTL/partition cleanup：

```text
训练正常结束 → clear partition
训练异常退出 → 下次启动检查旧 partition
超过 TTL → 清理未消费数据
```

## 14. EOS 和 drop-last

EOS 表示 Producer 已经读完输入，并且所有 vLLM/TQ put 都已完成。

TQ 中需要一种结束条件，具体使用 Controller API、特殊 metadata 或单独的运行状态记录取决于锁定版本。不能把“当前暂时没有 ready sample”当成 EOS，因为 Producer 可能仍在请求 vLLM。

最后不足一个 global batch 时：

```python
global_batch_size = batch_size_per_gpu * world_size
```

第一版使用 `drop_last=true`，避免某些 rank 有数据、某些 rank 没数据导致分布式训练不同步。

结束条件：

```text
producer_done == true
and ready_samples < global_batch_size
and inflight_requests == 0
and pending_puts == 0
```

## 15. 双缓冲预取

训练 batch N 时，CPU 后台线程预取 batch N+1：

```python
next_future = executor.submit(tq_stream.next_local_batch)

current_batch = first_batch
while current_batch is not None:
    next_batch = next_future.result()
    next_future = executor.submit(tq_stream.next_local_batch)

    train(current_batch)
    current_batch = next_batch
```

实际顺序应调整为避免等待 future 后才训练。推荐：

```python
current = tq_stream.next_local_batch()

while current is not None:
    future = executor.submit(tq_stream.next_local_batch)
    train_and_clear(current)
    current = future.result()
```

第一版只预取一个 global batch，避免 Trainer 崩溃时大量数据已被采样但未训练。

如果 Mooncake/TQ Python get 是阻塞函数，使用专用 `ThreadPoolExecutor`，不要阻塞 Producer 的 asyncio event loop。

## 16. 建议代码结构

```text
verl_speco/
  trainer/
    tq_transport.py          # TQ client、put/get/meta/clear 封装
    tq_feature_stream.py     # kv_list、global keys、rank shard、decode/prefetch
    feature_producer.py      # JSONL → 并发 vLLM → TQ
    draft_training_loop.py   # 增加 transfer_queue 数据源分支
    target_feature_replay.py # 复用 vLLM payload 校验/转换逻辑
```

不要新增：

```text
coordinator.py
coordinator_client.py
```

建议抽象：

```python
class StreamingFeatureSource(Protocol):
    def next_local_batch(self) -> tuple[list[str], list[DraftFeatureSample]]: ...
    def clear_global_batch(self, keys: list[str]) -> None: ...
    def close(self) -> None: ...
```

这样训练循环不依赖 TQ 的具体类型。

## 17. 配置草案

```yaml
actor_rollout_ref:
  rollout:
    drafter:
      training:
        mode: offline
        backend: dspark
        batch_size_per_gpu: 2
        max_steps: 1000

        feature_store:
          type: transfer_queue
          partition_id: speco_drafter_features_${run_id}
          drop_last: true
          prefetch_steps: 1

          transfer_queue:
            # 与 PR #48 的配置层级和 init 方式保持一致。
            enable: true
            package_version: 0.1.8  # 最终以实测版本为准
            backend:
              storage_backend: MooncakeStore
              MooncakeStore:
                auto_init: false
                metadata_server: localhost:50123
                master_server_address: localhost:50124
                local_hostname: localhost
                protocol: tcp
                global_segment_size: 4294967296
                local_buffer_size: 1073741824
                device_name: ""

          required_fields:
            - input_ids
            - loss_mask
            - position_ids
            - hidden_states

        producer:
          input_path: /path/to/generated_responses.jsonl
          vllm_endpoints:
            - http://node0:8000/v1
            - http://node1:8000/v1
          max_inflight_requests: 32
          max_pending_put_bytes: 8589934592
          max_ready_samples: 256
          max_ready_bytes: 68719476736
```

当前 examples 中的：

```bash
transfer_queue.enable=False
```

属于 verl RL 主入口配置，当前 standalone `draft_train_launcher` 不读取它。不能只改成 `True`；必须实现上述 `feature_store.type=transfer_queue` 分支。

## 18. 启动顺序

逻辑顺序：

```text
1. 启动 Mooncake metadata/master 服务；
2. 启动 standalone TQ owner 进程，调用一次 `tq.init(tq_config)` 创建/连接 TQ Controller 和 MooncakeStore backend；
3. 启动一个或多个定制 vLLM server
4. 启动 Feature Producer
5. Producer 和所有 Trainer rank 调用无参 `tq.init()`，连接 owner 创建的同一套 TQ；
6. 启动 verl_speco.draft_train_launcher
7. torchrun 启动所有 DSpark rank
8. 各 rank 连接 TQ
9. rank 0 用 `kv_list` 选择 global keys，各 rank 并行 `kv_batch_get`/train；
10. 输入耗尽后 Producer 发布 done 状态
11. Trainer drain 完整 global batches 后退出
12. 清理 partition，停止 Producer、vLLM、TQ、Mooncake
```

PR #48 的 `init_transfer_queue()` 在 Ray `SpecoTaskRunner` 中调用 `tq.init(config)`，worker 的无参 `tq.init()`依靠同一个 Ray 集群发现 named Controller；这部分不能原样复制到 standalone torchrun。

本项目要求一开始不使用 Ray，因此 Phase 0 必须先证明锁定的 TQ 版本支持独立 owner/controller 进程，以及 Producer/torchrun rank 如何获得连接信息。若 TQ 0.1.7/0.1.8 实际只能通过 Ray named actor 完成发现，那么有两个选择：

1. 接受仅用 Ray 承载 TQ 控制面的最小方案；
2. 给 TQ 增加或使用其已有的独立 ZMQ/server-info bootstrap。

在这项验证完成前，文档不能声称 PR #48 已经提供“无 Ray TQ standalone 启动”。

## 19. 故障处理

### vLLM 请求失败

- 对单个 sample 按稳定 key 重试；
- 指数退避；
- 超过次数记录失败，并根据配置 fail-fast 或跳过；
- 不写不完整 TQ fields。

### vLLM 文件读取成功，但 TQ put 失败

- 暂时保留临时文件；
- 重试 TQ put；
- put 成功后再删除；
- 不把样本视为 ready。

### 某个训练 rank get 失败

- 该 rank 报告 `local_ok=false`；
- `_all_ranks_true()` 使全部 rank 得到一致失败结果；
- 第一版整个训练 fail-fast；
- 不 clear global batch。

### OOM/optimizer step 失败

- 不 clear；
- 所有 rank 一致退出；
- 从最近训练 checkpoint 恢复；
- 根据 TQ 消费提交语义决定是否重放当前 batch。

### clear 失败

- optimizer 已成功，不能再次训练这批；
- 将 batch keys 写入本地小型 `gc_pending` 日志；
- 后台重试 clear；
- checkpoint 保存最近 committed sample IDs，避免恢复时重复消费。

## 20. 观测指标

Producer：

```text
producer/vllm_inflight
producer/vllm_requests_per_sec
producer/vllm_prefill_tokens_per_sec
producer/vllm_p50_latency
producer/vllm_p95_latency
producer/tq_put_bytes_per_sec
producer/tq_put_failures
producer/pending_put_bytes
```

TQ/Mooncake：

```text
tq/ready_samples
tq/ready_bytes
tq/consumed_samples
tq/storage_bytes
tq/clear_failures
mooncake/put_bandwidth
mooncake/get_bandwidth
```

Trainer：

```text
trainer/tq_wait_seconds
trainer/tq_get_seconds
trainer/tq_get_bytes_per_sec
trainer/decode_seconds
trainer/h2d_seconds
trainer/step_seconds
trainer/data_stall_ratio
trainer/successful_steps
```

## 21. 实施阶段

### Phase 0：锁定依赖和契约

- 从 PR #48 的 `TransferQueue==0.1.7` 起验证，同时对比当前 verl 文档使用的 0.1.8；
- 实测 `tq.init(config)`、worker `tq.init()`、`kv_put`、`kv_batch_get`、`kv_list`、`kv_clear`；
- 验证该版本是否支持无 Ray owner/controller 部署以及连接信息传递；
- 先用 PR #48 已配置的 `SimpleStorage` 做最小闭环；
- 再把 backend 换成 `MooncakeStore`，验证 tcp，最后再验证 rdma；
- 固定 fields、tags、partition 和 `run_id/sequence_no/status`；
- 写 fake TQ 单元测试。

### Phase 1：文件桥接 + TQ KV 模式

- 新增独立 Producer；
- 32 个有界并发 vLLM 请求；
- 读取 vLLM 临时 safetensors；
- TQ put 成功后删除文件；
- standalone trainer 由 rank 0 `kv_list` 并广播 global keys；
- 各 rank 并行 `kv_batch_get`；
- 复用 PR #48 的返回值解包、NestedTensor densify 和 per-step cache；
- optimizer 成功后 `kv_clear`。

验收：连续训练 1000 step，临时文件数量、TQ ready bytes 和 Mooncake占用均保持有界。

### Phase 2：双缓冲与多 endpoint

- 增加多 endpoint 最少 inflight 调度；
- 增加一个 global batch 预取；
- 动态背压；
- 注入单 rank get 失败，确认所有 rank 一致退出而非死锁。

### Phase 3：vLLM 直接写 TQ/Mooncake

- 修改外部定制 vLLM exporter；
- 去掉 `hidden_states_path` 临时文件；
- HTTP 响应返回 partition/sample key；
- 验证 HTTP 重试的幂等性。

### Phase 4：可选升级到 TQ StreamingDataLoader

- 在当前保守方案稳定后再引入 RankAwareSampler；
- 让每个 rank 自动取得 local micro-batch；
- 去掉 rank 0 手工 key-list 广播；
- 验证与 torchrun/DSpark 的 global step 对齐。

## 22. 最终推荐

针对当前 `verl-SpeCo-ls`，推荐的第一版不是 AngelSpec 的完整架构，也不是单独写 Coordinator，而是：

```text
当前预生成 response 文件
→ 独立 asyncio Producer
→ 并行访问多个 vLLM endpoint
→ 读取并校验临时 hidden-state 文件
→ TransferQueue put
→ Mooncake storage backend
→ rank 0 kv_list 获取 READY global keys
→ broadcast key list
→ 各 DSpark rank 并行 kv_batch_get
→ 现有 prepare_training_batch_from_samples()
→ 现有 training_step_from_batch()
→ 全 rank 成功
→ TQ clear
```

这套方案保留当前 standalone DSpark 训练主体，只替换 `DraftFeatureDataLoader + TargetFeatureReplayer.materialize()` 所在的数据输入路径。它真正建立在 PR #48 已实现的 KV transport 之上，而不是假设 PR #48 已经实现了 standalone Sampler/StreamingDataLoader。

## 23. 参考

- verl TransferQueue：<https://github.com/verl-project/verl/blob/main/docs/data/transfer_queue.md>
- TransferQueue：<https://github.com/Ascend/TransferQueue>
- Mooncake Store：<https://github.com/kvcache-ai/Mooncake>
- 本地只读参考：`../verl-SpeCo/verl_speco/integration/transferqueue_bridge.py`（PR #48）
- 本地只读参考：`../verl-SpeCo/docs/transferqueue_integration_plan.md`（PR #48）
