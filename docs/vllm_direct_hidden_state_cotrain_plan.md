# Co-train 复用 Producer 直接从 vLLM 获取 Hidden State 的实施方案

## 1. 目标与边界

本文方案面向 **RL 与草稿模型共同训练（co-train）**，目标是把 target hidden state 的来源从 actor 的 old-logprob 前向切换为 vLLM：

```text
原方案：vLLM 生成 response
        -> actor 为 PPO 计算 old_log_probs
        -> 在 actor forward 内通过 hook/output_hidden_states 抓 hidden state
        -> 草稿模型训练

新方案：vLLM 生成 response
        -> 使用 prompt + response 再向 hidden-state vLLM 发起 prefill 请求
        -> vLLM 返回 hidden-state 文件位置，客户端读取并归一化
        -> 现有 scheduler -> SpecoWorker -> drafter buffer -> 草稿模型训练
```

需要特别区分两个动作：

- actor 的 `compute_log_prob()` **仍然保留**，因为 PPO 更新 actor 需要 `old_log_probs`。
- 删除的是 old-logprob 前向中专为草稿训练增加的 hidden-state 捕获、拼接、CPU copy 和 Ray put 逻辑。

第一版不修改独立训练的 TQ producer/consumer 行为，不让 co-train 使用独立训练的文件读取、EOS、TQ owner 和离线 dataloader。复用的是 producer 中已经验证过的 **vLLM 请求、并发、重试、文件读取、token 对齐和样本归一化能力**。

## 2. 当前 co-train 全流程

### 2.1 运行时角色

| 角色 | 所在位置 | 当前职责 |
|---|---|---|
| `RayPPOTrainer` driver | `verl_speco/trainer/speco_ray_trainer.py` | 驱动 rollout、old-logprob、actor update，并通过 drafter scheduler 决定采样和训练时机 |
| vLLM rollout workers | verl rollout worker group | 根据 prompt 生成 response；当前 vLLM 路径不直接给 drafter hidden state |
| actor workers | `actor_rollout_wg` | 计算 PPO 所需 old log-prob；当前还承担 hidden-state 捕获 |
| drafter workers | `drafter_wg` 中的 `SpecoWorker` | 接收按 owner 分桶的样本，写入在线 buffer，执行 drafter train/publish/checkpoint |
| `DrafterScheduler` | driver 进程内普通 Python 对象 | 决定本 step 是否采集、是否训练，并组织 collection 的 stage/commit/finalize |

这里的 scheduler 不是另一个服务，也不读取数据；它只负责控制顺序和事务状态。

### 2.2 rollout 生成的数据

rollout 后，driver 持有 `DataProto batch`。与本方案直接相关的 tensor 通常为：

```python
batch.batch["prompts"]        # [B, Pmax]，左 padding
batch.batch["responses"]      # [B, Rmax]，右 padding
batch.batch["attention_mask"] # [B, Pmax + Rmax]
batch.batch["response_mask"]  # [B, Rmax]，若存在则表示有效 response token
```

单个有效样本会还原为：

```python
prompt_ids:   Tensor[P]
response_ids: Tensor[R]
input_ids = torch.cat([prompt_ids, response_ids])  # Tensor[P + R]
```

padding token 不能发给 hidden-state vLLM；必须通过 mask 去掉。

### 2.3 当前 old-logprob hidden-state 路径

入口位于 `SpecoRayPPOTrainer._speco_online_fit_hooks()` 安装的 `compute_old_log_prob_with_speco()` 包装函数：

1. `_speco_plan_drafter_collection(OLD_LOGPROB)` 调 scheduler，决定当前 `global_step` 是否采集。
2. `_speco_build_oldlogprob_collect_plan(batch)` 选择样本、hidden positions 和 owner rank。
3. driver 把以下控制数据写进 `batch_td`：

   ```python
   OLD_LOGPROB_COLLECT_MASK_KEY
   OLD_LOGPROB_HIDDEN_POSITIONS_KEY
   OLD_LOGPROB_HIDDEN_POSITION_MASK_KEY
   OLD_LOGPROB_OWNER_RANK_KEY
   OLD_LOGPROB_AUX_LAYER_IDS_KEY
   OLD_LOGPROB_HIDDEN_CAPTURE_IMPL_KEY
   OLD_LOGPROB_HIDDEN_LAYOUT_KEY
   ```

4. `actor_rollout_wg.compute_log_prob(batch_td)` 远程执行 actor 前向。
5. `verl_speco/integration/oldlogprob_runtime.py` 根据 `forward_hook` 或 `output_hidden_states` 捕获指定层，并把结果作为 tensor、Ray object ref 或分块 ref 返回。
6. driver 的 `_speco_collect_oldlogprob_features()` 把 actor 输出还原为逐样本字典：

   ```python
   sample = {
       "input_ids": Tensor[1, P + R],
       "prompts": Tensor[1, P],
       "responses": Tensor[1, R],
       "hidden_positions": Tensor[1, Hrows],
       "hidden_states": Tensor[1, Hrows, Hdim],  # 或 *_ref / *_ref_chunks
       "hidden_states_layout": "dflash_aux" | "eagle3_aux_plus_last",
       "hidden_position_start": int,
       "hidden_position_end": int,
       "global_step": int,
       "replica_rank": int,
   }
   ```

7. `OldLogProbCollectionAdapter.prepare_payload()` 根据显式 `owners` 把样本分到 drafter owner buckets。
8. scheduler 执行 collection transaction，Ray RPC 参数本质上是每个 owner 对应的 `list[dict]`。
9. `SpecoWorker._commit_rollout_features(collection_id, samples)` 解析 hidden tensor/ref，调用 `_store_rollout_sample()`。
10. `_store_rollout_sample()` 调 `DrafterBaseTrainer.collect_online_data()`，把 CPU 数据写入当前 step 或跨 step buffer。
11. `update_actor_with_speco()` 调 `_speco_on_before_actor_update()`；scheduler 根据已收集数据产生 training plan，然后执行 actor update 和 drafter training。

因此，现有 worker、buffer 和训练后半段并不关心 hidden state 是由 actor 还是 vLLM 产生。需要替换的主要是第 2～6 步的数据生产方式。

## 3. 当前 standalone producer 的详细流程

### 3.1 哪些部分可以复用

`verl_speco/standalone_tq_producer.py` 当前是一个三段式异步流水线：

```text
read_inputs
  -> request_queue
  -> N 个 request_worker
  -> publish_queue
  -> publish_results
  -> TQ
```

其中只有最后的 TQ publish 和最前面的文件 reader 是 standalone 专属。中间部分已经包含 co-train 需要的核心能力。

#### `VllmFeatureClientPool`

文件：`verl_speco/producer/vllm_feature_client.py`

职责：

- 解析多个 `VllmEndpoint`；
- 维护全局并发 semaphore 和每 endpoint semaphore；
- 优先选择当前 inflight 较少的 endpoint；
- 通过 OpenAI completions 接口发送 token IDs；
- 对连接错误、read error、超时等执行指数退避重试；
- 从响应的 `kv_transfer_params.hidden_states_path` 取得 safetensors 路径；
- 等待文件完成，读取 hidden state、token IDs 和相关字段；
- 读取完成后删除临时 safetensors 与 lock 文件。

请求不是让 vLLM 再生成一段文本，而是一次 prefill 请求：

```python
await client_pool.request_prefill(
    prompt_token_ids=request.vllm_prompt_token_ids,
    sample_id=request.sample_id,
)
```

返回的 `RawVllmFeature` 仍是 vLLM 原始坐标系下的数据，例如：

```python
RawVllmFeature(
    token_ids=Tensor[Tprefill],
    hidden_states=Tensor[Tprefill, L, D],
    hidden_position_start=...,
    hidden_position_end=...,
    ...,
)
```

其中 `L` 是导出的 target layer 数量，`D` 是 target hidden size。

#### `prepare_generated_prefill_request`

文件：`verl_speco/producer/input_reader.py`

standalone 遇到仅有 prompt 的数据时，先生成 response，再调用该函数把 prompt 和生成结果组成训练请求。核心规则是：

```python
full_ids = prompt_ids + response_ids
vllm_prompt_token_ids = full_ids[:-1]
```

去掉最后一个 token 的原因是：位置 `i` 的 target hidden state 用于预测后续 token，最后一个 token 后面没有本样本内的监督 token。co-train 已经拥有 rollout response，所以只需要直接执行这一步，不需要再次生成 response。

当前 `TokenizedRequest` 包含：

```python
TokenizedRequest(
    sequence_no: int,
    sample_id: str,
    input_ids: list[int],             # prompt + response
    loss_mask: list[float],           # prompt 为 0，有效 response 为 1
    position_ids: list[int],
    feature_positions: list[int],     # 选中的 target hidden 绝对位置
    draft_position_ids: list[int],
    source_metadata: dict,
    vllm_prompt_token_ids: list[int], # 发给 vLLM 的 full_ids[:-1]
)
```

#### `feature_from_vllm_payload`

文件：`verl_speco/trainer/target_feature_replay.py`

该函数把 `RawVllmFeature + TokenizedRequest + FeatureContract` 转成算法训练侧统一使用的 `DraftFeatureSample`。它负责：

- 检查 vLLM 返回的 token IDs 是否与请求一致；
- 检查 hidden rows、层数、hidden size 和 layout；
- 按 `feature_positions` 选择训练窗口；
- 对齐 `input_ids`、`loss_mask`、positions 与 hidden state；
- hidden state 不完整或位置对不上时拒绝该样本，不把错误样本交给训练。

典型结果：

```python
DraftFeatureSample(
    input_ids=Tensor[T],
    loss_mask=Tensor[T],
    hidden_states=Tensor[Hrows, L * D],
    target_logprobs=None,
    position_ids=Tensor[T],
    feature_positions=Tensor[Hrows],
    draft_position_ids=Tensor[Hrows],
    metadata={...},
)
```

这里的协议和算法处理应继续由已有 `DraftFeatureSample`、backend 和 contract 决定，不能在新 co-train 组件里再次硬编码 DSpark。

### 3.2 哪些部分不能直接搬入 co-train

以下 standalone 逻辑不能原样调用：

- 从 JSONL 循环读 epoch；co-train 的输入来自当前 rollout `DataProto`。
- prompt-only 时调用生成接口；co-train 的 response 已经生成。
- `sequence_no/run_id/tag/EOS/max_pending_samples`；这些用于 TQ 流式生产消费，不属于单个 RL step。
- `publish_results()` 和 TQ clear；co-train 已有 scheduler collection transaction 和 worker buffer。
- standalone owner/consumer 生命周期；co-train 由 Ray trainer 和 worker group 管理。

正确的复用方式是抽取“给定 tokenized rollout sample，异步取得并归一化 hidden state”的核心，而不是在 co-train 内启动一个 `standalone_tq_producer` 进程。

## 4. 建议的新数据流

### 4.1 完整顺序

```text
1. rollout vLLM 生成 response
2. driver 得到 DataProto(prompts, responses, masks)
3. scheduler 判断本 step 是否需要采集 VLLM_PREFILL
4. driver 按采样计划选择样本、去 padding、构造 TokenizedRequest
5. CotrainVllmFeatureProducer.submit_batch() 提交并发 prefill
6. hidden-state vLLM endpoint 执行 prompt+response[:-1] prefill
7. client 读取 safetensors，执行 token/shape/position 对齐
8. 得到 DraftFeatureSample；失败或不完整样本在此处过滤
9. 将 DraftFeatureSample 转为现有 SpecoWorker collection sample
10. VllmPrefillCollectionAdapter 按 replica owner 分 buckets
11. scheduler stage -> Ray commit RPC -> finalize
12. SpecoWorker._store_rollout_sample() -> collect_online_data() -> buffer
13. scheduler 产生 training plan
14. actor update 与 drafter training 按现有顺序执行
```

步骤 5 提交后不应立刻阻塞等待。driver 可以继续 reward、reference log-prob、advantage、actor old-logprob 等工作；在 drafter collection 必须完成的边界再 `await/result()`。这样 vLLM prefill 与 RL 侧计算重叠。

### 4.2 vLLM 请求的具体 token 对齐

给定一个 rollout 样本：

```python
prompt_ids   = prompts[i][prompt_mask]      # [P]
response_ids = responses[i][response_mask] # [R]
full_ids     = cat(prompt_ids, response_ids) # [P + R]
prefill_ids  = full_ids[:-1]                 # [P + R - 1]
```

构造：

```python
loss_mask = zeros(P + R)
loss_mask[P:P + R] = 1
```

然后再应用现有 collection plan 的窗口限制。必须保证：

```text
返回 token_ids == prefill_ids
hidden rows 能覆盖 feature_positions
feature_positions 非空
选中区域对应的 loss_mask 中存在有效训练 token
```

只含 prompt、有效 response 长度为 0、hidden rows 为 0、token 不一致或窗口为空的样本，都在 producer 转换阶段丢弃，不进入 scheduler payload。这样 producer 的“发布成功数”和 consumer 的“可接收数”天然一致，不会把无效条目带入 collection transaction。

### 4.3 scheduler 到 worker 的样本格式

建议保留 worker 当前已经支持的 collection sample 外形，不大改训练后半段：

```python
worker_sample = {
    "input_ids": Tensor[1, T],
    "prompts": Tensor[1, P],
    "responses": Tensor[1, R],
    "hidden_positions": Tensor[1, Hrows],
    "hidden_states": Tensor[1, Hrows, HiddenWidth],
    "hidden_states_layout": str,
    "hidden_position_start": int,
    "hidden_position_end": int,
    "global_step": int,
    "replica_rank": int,
}
```

`HiddenWidth` 取决于现有 backend/layout。例如多个层已经按最后一维拼接时为 `L * D`。该转换必须调用 `DraftFeatureSample` 已有字段和 metadata，不在 adapter 中按算法猜测。

`replica_rank` 不是 vLLM 返回的数据，而是 driver 根据现有 drafter owner 路由计划为样本分配的控制字段。scheduler 只用它决定该样本发给哪个 `SpecoWorker` owner。

## 5. 代码修改方案

### 5.1 新增 co-train producer 核心

新增：`verl_speco/producer/cotrain_vllm_feature_producer.py`

建议接口：

```python
@dataclass
class CotrainFeatureRequest:
    batch_index: int
    owner_rank: int
    request: TokenizedRequest
    prompt_ids: torch.Tensor
    response_ids: torch.Tensor


@dataclass
class CotrainFeatureResult:
    batch_index: int
    owner_rank: int
    sample: DraftFeatureSample


class CotrainVllmFeatureProducer:
    def __init__(self, config, *, contract: FeatureContract): ...

    def submit_batch(
        self,
        requests: list[CotrainFeatureRequest],
    ) -> Future[list[CotrainFeatureResult]]: ...

    async def _produce_one(
        self,
        request: CotrainFeatureRequest,
    ) -> CotrainFeatureResult | None: ...

    def close(self) -> None: ...
```

内部直接复用：

```python
raw = await self.client_pool.request_prefill(...)
sample = feature_from_vllm_payload(raw, request.request, self.contract)
```

组件应持有一个长期存在的 `VllmFeatureClientPool`，不能每 step 重建 HTTP client、semaphore 和线程池。由于 PPO driver 主流程通常是同步代码，第一版可让组件内部持有一个后台 asyncio event loop thread，`submit_batch()` 返回 `concurrent.futures.Future`。训练结束时统一 `close()`，取消未完成任务并关闭 HTTP client。

### 5.2 增加从 rollout tensor 构造请求的函数

修改：`verl_speco/producer/input_reader.py`

新增纯函数，复用现有长度截断、feature window、position 和 loss-mask 规则：

```python
def build_rollout_prefill_request(
    *,
    sample_id: str,
    sequence_no: int,
    prompt_ids: Sequence[int],
    response_ids: Sequence[int],
    producer_cfg,
    source_metadata: dict,
) -> TokenizedRequest:
    ...
```

它不接收文本、不调用 tokenizer、不生成 response，只做：

1. 拼接有效 prompt/response token；
2. 按已有 `max_feature_length` 等规则截取 response；
3. 建 loss mask、positions；
4. 设置 `vllm_prompt_token_ids=full_ids[:-1]`。

必须把 standalone 与 co-train 的公共构造逻辑下沉到同一个私有 helper，避免两个路径以后出现 off-by-one 或截断规则差异。

### 5.3 扩展 scheduler 的 collection source

修改：

- `verl_speco/trainer/scheduler/schedule_types.py`
- `verl_speco/trainer/scheduler/collection_adapter.py`
- `verl_speco/trainer/scheduler/drafter_scheduler.py`

新增：

```python
class DrafterCollectionSource(str, Enum):
    SGLANG = "sglang"
    OLD_LOGPROB = "oldlogprob"
    VLLM_PREFILL = "vllm_prefill"
```

新增 `VllmPrefillCollectionAdapter`。它只负责：

- 校验每个 sample 有 `replica_rank`；
- 使用 `_build_payload()` 按 owner 分桶；
- 设置 `CollectionPayload.source=VLLM_PREFILL`。

它不负责请求 vLLM、不解码 hidden state、不实现算法逻辑。

同时更新 collection source 的稳定排序值、adapter registry 和 metrics source label。

### 5.4 在 `SpecoRayPPOTrainer` 接入异步生产

修改：`verl_speco/trainer/speco_ray_trainer.py`

新增或调整以下职责：

```python
def _speco_vllm_prefill_collection_requested(self) -> bool: ...
def _speco_vllm_prefill_collection_enabled(self) -> bool: ...
def _speco_get_cotrain_vllm_producer(self) -> CotrainVllmFeatureProducer: ...
def _speco_build_vllm_prefill_requests(self, batch, collection_plan): ...
def _speco_submit_vllm_prefill_collection(self, batch): ...
def _speco_finish_vllm_prefill_collection(self) -> int: ...
def _speco_close_vllm_prefill_producer(self) -> None: ...
```

接入点建议如下：

1. rollout 返回 `gen_batch_output` 并合并成训练 batch 后，调用 `_speco_submit_vllm_prefill_collection(batch)`。
2. 提交函数先调用 scheduler 的 `plan_collection(VLLM_PREFILL)`；未命中 interval 时不发 HTTP 请求。
3. 继续执行 reward、ref、old-logprob 和 advantage。
4. 在 `_speco_on_before_actor_update()` 生成 training plan 之前调用 `_speco_finish_vllm_prefill_collection()`：
   - 等待 Future；
   - 过滤失败样本；
   - 转成 worker sample；
   - adapter 分桶；
   - `_speco_execute_collection()`。
5. 原 `compute_old_log_prob_with_speco()` 在该模式下走普通 `original_compute_old_log_prob()`，不再注入 hidden capture keys。
6. fit 的 `finally` 中关闭 producer。

这里“等待点必须在 training plan 之前”是必要条件。否则 scheduler 看到的 buffer version 仍是旧值，本 step 可能错误判断没有可训练样本。

### 5.5 worker 和训练侧尽量不改

`verl_speco/workers/speco_worker.py` 的以下路径可以直接复用：

```text
collect_rollout_features / collection transaction RPC
  -> _commit_rollout_features
  -> _store_rollout_sample
  -> DrafterBaseTrainer.collect_online_data
```

第一版只在必要时增加一个从 `DraftFeatureSample` 转现有 sample dict 的小 helper；不要新增另一套 buffer，也不要让 worker 连接 standalone TQ。

如果 `DraftFeatureSample.to_training_item()` 与 `collect_online_data()` 的 metadata 表达存在差异，应在一个公共转换函数中补齐，而不是在 driver、adapter、worker 分别写一套字段映射。

### 5.6 禁用 old-logprob hidden capture，但保留 PPO old-logprob

修改配置判定和 hook 分支：

```yaml
collect_hidden_states_from_sgl: false
collect_hidden_states_from_old_logprob: false
collect_hidden_states_from_vllm: true
```

当 `collect_hidden_states_from_vllm=true` 时：

- 不设置 `OLD_LOGPROB_HIDDEN_*` keys；
- 不安装/启用 `oldlogprob_runtime` hidden hooks；
- 不调用 `_speco_collect_oldlogprob_features()`；
- 仍执行标准 `actor_rollout_wg.compute_log_prob()`，得到 PPO 的 log-prob 和 entropy。

三种来源第一版必须互斥：

```python
sum([
    collect_hidden_states_from_sgl,
    collect_hidden_states_from_old_logprob,
    collect_hidden_states_from_vllm,
]) <= 1
```

## 6. 配置建议

修改：`verl_speco/config/actor/actor.yaml` 或本仓实际承载 drafter training 默认值的配置文件，并在示例脚本暴露关键参数。

建议结构：

```yaml
actor_rollout_ref:
  rollout:
    drafter:
      training:
        mode: online
        collect_hidden_states_from_sgl: false
        collect_hidden_states_from_old_logprob: false
        collect_hidden_states_from_vllm: true

        vllm_feature_source:
          endpoints:
            - http://127.0.0.1:8000/v1
            - http://127.0.0.1:8001/v1
          model: /path/to/target-model
          max_inflight_requests: 128
          per_endpoint_concurrency: 64
          request_timeout_seconds: 600
          max_retries: 3
          retry_base_delay_seconds: 1.0
          max_sequence_length: 8192
```

`target_layer_ids`、`max_feature_length`、hidden layout、算法类型等应继续读取现有 drafter 配置，不在 `vllm_feature_source` 重复定义。`FeatureContract` 也从同一份运行配置构建，从而保证 vLLM 导出层和 trainer 预期一致。

### vLLM 服务要求

当前 `VllmFeatureClientPool` 使用 OpenAI HTTP endpoint 和 `kv_transfer_params.hidden_states_path`。因此第一版要求：

- co-train 可访问一个或多个已启动的 hidden-state vLLM 服务；
- 服务加载的 target model/tokenizer 与 rollout/actor 使用的版本一致；
- `extract_hidden_states` 的 layer IDs 与 drafter contract 一致；
- vLLM hidden 文件目录对 driver 可见；
- prefix caching 对 hidden-state 导出必须关闭或已验证能返回所有所需 rows。

verl 内部 rollout vLLM worker 不一定天然暴露当前 client 所需的 OpenAI 地址和共享 hidden 文件路径。第一版建议使用独立启动的 hidden-state vLLM endpoints。后续若 rollout 服务能够暴露同等接口，再把 endpoint discovery 接入 worker group，producer 核心无需改变。

## 7. 是否在 co-train 中使用 TQ

### 7.1 第一版建议：不使用 TQ

第一版直接把 CPU hidden tensor 放入现有 scheduler payload，必要时沿用 Ray object ref/chunk ref 机制。理由：

- co-train 已经有 scheduler collection transaction、owner 路由和 worker buffer；
- 当前 standalone TQ 的 run ID、pending、EOS、clear 语义针对跨进程无限流，不适合直接套在单个 RL step 上；
- 少改 worker 和生命周期，能先验证 vLLM hidden 与 actor hidden 的数值/训练等价性。

此时的控制面和数据面是：

```text
控制面：driver -> scheduler -> Ray RPC(collection_id, owner bucket)
数据面：CPU tensor 随 Ray 参数，或先 ray.put 后传 ObjectRef
```

### 7.2 第二阶段可选：TQ 承载大 tensor

若 Ray object store 压力明显，再让 producer 将 `DraftFeatureSample` 编码后写 TQ，而 scheduler sample 只传：

```python
{
    "feature_key": str,
    "replica_rank": int,
    "global_step": int,
}
```

worker commit 时按 key `get + decode`，commit 成功后 clear，rollback 时保留或清理。这需要定义 co-train 专属的 step-scoped key 和事务清理规则，不能复用 standalone EOS。该阶段会增加失败恢复复杂度，不建议和第一版一起提交。

## 8. 错误处理与一致性

### 8.1 单样本错误

以下错误在 `_produce_one()` 内记录 sample ID、batch index、endpoint 和原因，然后丢弃该样本：

- response 为空；
- token IDs 不一致；
- hidden rows 为 0 或覆盖不了选择位置；
- layer/hidden size/layout 不符合 contract；
- 截断后没有有效训练 token。

只有成功转换成 `DraftFeatureSample` 的样本才计入 `CollectionPayload.collected_samples`。

### 8.2 请求级错误

连接/读取错误先使用现有 client pool 重试。超过最大重试后，第一版建议默认让当前 collection 失败并终止本 step，而不是静默用不完整 batch 训练；可以后续增加 `failure_policy=fail_step|skip_sample|skip_collection`。

### 8.3 多 rank 一致性

vLLM 请求和样本过滤都发生在 driver；driver 形成最终成功样本列表后才按 owner 分桶并发 RPC。因此每个 drafter owner 收到的数量是 scheduler 已知的，不让各训练 rank 自行请求 vLLM、各自过滤。这避免某 rank 接受、另一个 rank 拒绝后进入不同 collective 顺序。

## 9. 指标与日志

建议增加：

```text
drafter/vllm_prefill/candidate_samples
drafter/vllm_prefill/submitted_samples
drafter/vllm_prefill/succeeded_samples
drafter/vllm_prefill/dropped_samples
drafter/vllm_prefill/request_elapsed_sec
drafter/vllm_prefill/wait_elapsed_sec
drafter/vllm_prefill/overlap_elapsed_sec
drafter/vllm_prefill/payload_mib
drafter/vllm_prefill/retry_count
drafter/vllm_prefill/per_endpoint_inflight
```

每次 collection 至少记录：`global_step`、`collection_id`、候选数、提交数、成功数、按 owner 分桶数量、hidden rows、payload bytes 和等待时间。单样本拒绝日志记录 sample ID 和失败检查项，但不要打印完整 token 或 hidden tensor。

## 10. 测试计划

### 10.1 单元测试

1. padded `prompts/responses` 能还原正确有效 token。
2. `prefill_ids == prompt_ids + response_ids[:-1]` 的边界测试，包括 response 长度 0/1。
3. co-train request builder 与 standalone builder 对同一 token 输入产生一致的 loss mask、feature positions 和截断结果。
4. 多 endpoint 并发、重试和 endpoint 选择沿用现有 client pool 测试。
5. token 不一致、hidden rows=0、缺层、空窗口均被 producer 拒绝，且不进入 payload。
6. `VllmPrefillCollectionAdapter` 能按 replica owner 正确分桶。
7. vLLM source 开启时 old-logprob batch 不包含任何 hidden capture key。
8. 标准 old-logprob 结果仍正确返回给 PPO。
9. producer Future 在训练计划生成前完成 collection；关闭时无残留线程和 HTTP client。

### 10.2 集成测试

用小模型和两个 hidden-state endpoints 运行数个 co-train steps，对比：

- old-logprob capture 与 vLLM prefill 的 token IDs、positions、hidden shape；
- 相同权重和样本下 drafter loss/metrics 是否接近；
- actor PPO metrics 是否不变；
- collect interval 未命中时没有 vLLM hidden 请求；
- endpoint 临时失败时重试后能继续；
- 多 drafter owner 下每个 owner 收到预期样本数。

## 11. 建议实施顺序与文件清单

### 阶段 A：抽取公共 producer 能力

- 修改 `verl_speco/producer/input_reader.py`：增加 rollout-token request builder，共享截断/对齐 helper。
- 新增 `verl_speco/producer/cotrain_vllm_feature_producer.py`：长期 client pool、异步 batch submit、转换与过滤、关闭逻辑。
- 不改 standalone TQ publish 行为。

### 阶段 B：接入 scheduler 和 driver

- 修改 `verl_speco/trainer/scheduler/schedule_types.py`：增加 `VLLM_PREFILL`。
- 修改 `verl_speco/trainer/scheduler/collection_adapter.py`：增加 owner 分桶 adapter。
- 修改 `verl_speco/trainer/scheduler/drafter_scheduler.py`：注册 adapter。
- 修改 `verl_speco/trainer/speco_ray_trainer.py`：提交 future、等待、构造 payload、执行 collection、关闭 producer。

### 阶段 C：配置、示例和验证

- 修改默认 drafter training 配置：增加 source 开关与 vLLM client 参数。
- 修改 co-train example：关闭 old-logprob hidden capture，填写 hidden-state endpoints。
- 增加 request builder、adapter、driver hook 和端到端测试。
- 用同一批固定 token 对照 actor-captured hidden 与 vLLM hidden，再进行正式性能测试。

### 阶段 D：可选 TQ 数据面

- 仅在 Ray object store 成为瓶颈后实施。
- 新增 co-train TQ key/ref adapter、worker decode 和 collection finalize/rollback 清理。
- 不改变独立训练现有 TQ 协议。

## 12. 最终推荐

推荐第一版采用：

```text
独立 hidden-state vLLM endpoints
  + 复用 VllmFeatureClientPool
  + 复用 TokenizedRequest / FeatureContract / feature_from_vllm_payload
  + 新增 VLLM_PREFILL scheduler source
  + 复用现有 SpecoWorker collection/buffer/train
  + 暂不在 co-train 中引入 TQ
```

这样修改范围集中在“hidden-state 来源”和“异步接入点”，不会重写已经稳定的 drafter worker 与训练逻辑，也不会影响 PPO 必需的 actor old-logprob 计算。等这一版验证 hidden 数值、loss 和吞吐后，再决定是否把 Ray 中的大 tensor 数据面替换为 TQ。
