# 独立 DSpark 训练 TQ Consumer 实现说明

## 1. 文档范围和当前结论

本文只说明当前仓库中已经实现的独立训练 Consumer。这里的 Consumer 是由 `torchrun` 启动的 DSpark 草稿模型训练任务：它持续从 TransferQueue（下文简称 TQ）发现样本，各训练 rank 分别取得自己负责的 Tensor，复用原有 DSpark 训练逻辑完成一次 optimizer step，然后由 rank 0 删除这一整个 global batch 对应的 TQ 记录。

当前已完成的能力是：

1. `feature_store.type=tq` 可以作为独立训练的数据源，不要求磁盘 `path`。
2. 每个训练 rank 都连接同一个 Ray 集群、同一个 TQ Controller 和同一个 partition。
3. 只有 rank 0 调用 `kv_list` 发现 ready key，并把 key/tag 分配给各 rank。
4. key 和 tag 通过 `torch.distributed.broadcast_object_list` 传输；hidden states 等 Tensor 不经过该广播。
5. 每个 rank 根据分配到的 key，直接调用 TQ `kv_batch_get` 获取自己的 Tensor。
6. TQ Tensor 被解码成原训练代码已经认识的 `DraftFeatureSample`，然后复用 `DrafterBaseTrainer` 的 batch 构造、DSpark loss、反向传播和 optimizer step。
7. 只有当所有 rank 都成功完成该 step 后，rank 0 才调用 `kv_clear` 删除整个 global batch。
8. Producer 发布 EOS 后，如果剩余样本不足一个 global batch，当前第一版会丢弃并清理这部分尾样本，然后正常结束训练迭代。

本文不会把尚未实现的 Producer 写成现有能力。Producer 后续需要复用本文第 6 节所述的公共协议，调用 `encode_sample()` 生成 fields，再使用 bridge 写入相同 TQ。

## 2. 本次涉及的文件

### 2.1 本次新增的 Consumer 核心文件

| 文件 | 实现的组件 | 作用 |
|---|---|---|
| `verl_speco/trainer/tq_feature_store.py` | `TQFeatureStore`、`ReadyEntry`、`EosMetadata` | 将公共 TQ bridge 包装成 Consumer 数据访问层，负责连接、发现、批量读取、解码、删除和读取 EOS |
| `verl_speco/trainer/tq_sample_source.py` | `TQFeatureDataLoader`、`TQLocalBatch`、`build_assignments()` | 实现多 rank 流式取数：rank 0 发现样本并分配 key，各 rank 自己从 TQ 取 Tensor |
| `tools/run_dspark_tq_consumer.sh` | Consumer 测试启动工具 | 给出一套完整的 DSpark、offline、TQ 配置和 `torchrun` 启动方式 |
| `tests/unit/test_tq_consumer.py` | Consumer 单元测试 | 覆盖 store 构建、ready 过滤排序、协议解码、rank 分配、EOS、清理和非 rank 0 行为 |

### 2.2 本次修改的既有文件

| 文件 | 修改内容 | 为什么要改 |
|---|---|---|
| `verl_speco/trainer/feature_store.py` | factory 新增 `type=tq` 分支 | 让既有独立训练入口能够像选择磁盘 feature store 一样选择流式 TQ 数据源 |
| `verl_speco/trainer/draft_training_loop.py` | 接入 TQ store/loader、跨 rank 连接检查、训练成功后清理 | 将流式取数接入原训练循环，同时保留原 DSpark trainer、loss、optimizer、metric 和 checkpoint 逻辑 |
| `verl_speco/draft_train_launcher.py` | 增加 TQ 启动参数的 fail-fast 检查 | 在启动多个 torchrun 子进程前检查 `enable`、Ray address 和 `run_id`，避免各 rank 启动后才失败 |
| `verl_speco/config/speco_base.yaml` | 标注 `feature_store.type=tq` 为无路径流式数据源 | 保留统一 Hydra 配置入口；TQ 的公共配置仍位于 sibling `training.transfer_queue` |
| `tools/tq_connection_smoke.py` | 将原连接 smoke 扩展为真实 Consumer 路径测试 | 验证 owner、真实 TQ、Consumer 读取、EOS、清理和仅关闭本地 client |
| `tests/unit/test_draft_train_launcher.py` | 增加 TQ 参数检查测试 | 验证必要配置缺失时 launcher 直接拒绝启动 |
| `tests/unit/test_draft_training_loop.py` | 增加连接和 clear 时序测试 | 验证只由 rank 0 清理、clear 失败会报告、连接失败会传播 |

### 2.3 直接复用的公共基础

以下文件不是这次 Consumer 才创造的概念，但 Consumer 直接使用它们：

| 文件 | 被复用的能力 |
|---|---|
| `verl_speco/integration/transferqueue_bridge.py` | 屏蔽 TQ 0.1.7 API 细节，提供连接、`kv_list`、`kv_batch_get`、`kv_clear` 和本地关闭接口 |
| `verl_speco/transport/drafter_sample_protocol.py` | 定义 key、tag、fields、metadata 格式，以及 `encode_sample()` / `decode_sample()` |
| `verl_speco/trainer/feature_store.py` | 复用 `DraftFeatureSample`，使 TQ 数据进入训练侧后与磁盘 feature sample 类型一致 |
| `verl_speco/trainer/base_trainer.py` 及既有 backend | 复用 `DrafterBaseTrainer.prepare_training_batch_from_samples()` 和 `training_step_from_batch()` 等训练实现 |

## 3. 运行时角色

### 3.1 TQ Owner

TQ Owner 是单独的普通 Python 进程。它连接指定 Ray 集群，并以带配置的 `tq.init(config)` 创建任务级 named Controller 和 storage actors。Owner 持有全局 TQ 生命周期；Consumer 结束时不能关闭它。

Owner 不是训练 rank，也不执行 DSpark 模型。它的主要作用是让 Producer 和 Consumer 能通过同一个 Ray actor registry 找到同一个 TQ Controller。

### 3.2 Producer

Producer 是后续需要实现的独立推理进程。它应并行调用 vLLM hidden-state 接口，构造一条条 `DraftFeatureSample` 和 `SampleMetadata`，再写入 TQ。

Producer 与 Consumer 不通过 Ray RPC 互相调用，也不通过 HTTP 直接传 Tensor。二者只需满足：

- 连接同一个 Ray address；
- 使用同一个 Ray namespace；
- 使用同一个 TQ partition；
- 使用同一个 `run_id` 和协议版本。

### 3.3 Consumer launcher

`python -m verl_speco.draft_train_launcher` 是父进程。它检查命令行 override，构造 `python -m torch.distributed.run ...` 命令，然后启动训练子进程。

launcher 自己不连接 TQ、不取样本、也不持有 GPU 模型。

### 3.4 Consumer training rank

`torchrun --nproc_per_node=N` 会启动 N 个训练 OS 进程。每个进程有独立的：

- global rank；
- local rank；
- GPU；
- `DrafterBaseTrainer`；
- `TQFeatureStore` 和本地 TQ client；
- DSpark 模型分片及 optimizer 状态。

这些 rank 共同执行一个分布式草稿模型训练任务。rank 0 额外负责发现和删除 TQ key；但所有 rank 都会取得各自的训练 Tensor，并参加模型 collective、梯度同步和 optimizer step。

### 3.5 Ray 和 torch.distributed 的职责不同

本方案仍然使用 Ray，但只因为 TQ 0.1.7 通过 Ray named actor 找 Controller。Consumer 不创建用于训练的 Ray actor，训练本身仍由 `torchrun` 和 `torch.distributed` 执行。

两种通信分别是：

- Ray/TQ：Owner、Producer、每个 Consumer rank 连接共享 TQ；大 Tensor 通过 TQ backend 传输。
- `torch.distributed`：训练 rank 之间广播小型 key/tag 命令、同步成功状态、训练模型 collective。

## 4. 共同配置以及“连接同一个 TQ”的实现

关键配置位于：

```yaml
actor_rollout_ref:
  rollout:
    drafter:
      training:
        mode: offline
        feature_store:
          type: tq
          path: null
        transfer_queue:
          enable: true
          ray:
            address: 127.0.0.1:6379
            namespace: speco-drafter
          partition_id: speco_drafter_features
          run_id: dspark-standalone-run
          schema_version: 1
          poll_interval_seconds: 0.5
          drop_last: true
```

这些字段的含义如下：

| 字段 | 使用者 | 含义 |
|---|---|---|
| `feature_store.type=tq` | Consumer | 选择流式 TQ source，而不是磁盘 shard/replay source |
| `feature_store.path=null` | Consumer | TQ 不从本地路径读文件，因此无需 path |
| `transfer_queue.enable` | Owner、Producer、Consumer | 开启 bridge 的 TQ 路径 |
| `ray.address` | 三端 | 连接同一个 Ray 集群 |
| `ray.namespace` | 三端 | 在同一 actor namespace 查找 named Controller |
| `partition_id` | 三端 | 对同一个 TQ KV 分区执行 put/list/get/clear |
| `run_id` | Producer、Consumer | 在共享 partition 中区分本次训练数据；Consumer 只接收匹配的样本 |
| `schema_version` | Producer、Consumer | 共同使用的数据协议版本 |
| `poll_interval_seconds` | Consumer rank 0 | ready 数量不足时的轮询间隔 |
| `drop_last` | Consumer | 第一版必须为 true；EOS 后不足 global batch 的尾样本被清理 |

三端并不是通过共享 Python 对象得到这些配置。每个进程都各自读取相同取值，然后执行：

```python
configure_transfer_queue(config)
connect_ray_cluster(ray_address, ray_namespace)
connect_transfer_queue_client()
```

`connect_ray_cluster()` 内部调用 `ray.init(address=..., namespace=...)`。`connect_transfer_queue_client()` 再调用无参数 `tq.init()`；TQ 由此在当前 Ray namespace 查找 Owner 创建的 named Controller。之后所有 KV 操作都显式携带相同的 `partition_id`。

因此，“连接同一个 TQ”实际由三层身份共同决定：同一 Ray 集群、同一 namespace 下的同一 named Controller、同一 `partition_id`。

## 5. 一条样本在 TQ 中的实际格式

### 5.1 一条 key 对应一个 sample

本协议没有把一个训练 batch 存成一个 TQ key。一条 key 对应一条独立训练样本。假设：

```text
run_id = dspark-run-001
sequence_no = 17
sample_id = prompt-000017
```

则 key 为：

```text
drafter:v1:dspark-run-001:000000000017:prompt-000017
```

`sequence_no` 是本次 run 内的样本顺序号，不是 batch 编号，也不是 optimizer step。Consumer 用它稳定排序，之后每次从有序 ready 列表前部取一个 global batch。

### 5.2 tag：用于轻量发现和过滤

该 key 的 tag 是普通小字典：

```python
{
    "record_type": "sample",
    "status": "ready",
    "schema_version": 1,
    "run_id": "dspark-run-001",
    "sequence_no": 17,
    "sample_id": "prompt-000017",
    "algorithm": "DSPARK",
}
```

tag 存在 TQ 的 KV 元信息中。`kv_list(partition_id=...)` 返回 `key -> tag`，不需要先加载 hidden states。rank 0 正是依靠 tag 筛选当前 run、当前 schema、DSPARK 且状态为 ready 的记录。

### 5.3 fields：真正的 Tensor payload

同一 key 的 fields 是一个 Tensor 字典：

```python
{
    "input_ids":       Tensor[int64,   shape=[L]],
    "loss_mask":      Tensor[float32, shape=[L]],
    "position_ids":   Tensor[int64,   shape=[L]],
    "hidden_states":  Tensor[dtype,   shape=[L, D]],
    "metadata_json":  Tensor[uint8,   shape=[M]],
    # 以下是可选字段：
    "last_hidden_states": Tensor[..., ...],
    "target":             Tensor[..., ...],
    "target_logprobs":    Tensor[..., ...],
}
```

这里 `L` 是 feature window 的 token 数，`D` 是目标模型 hidden size，`M` 是 metadata JSON 序列化后的 UTF-8 字节数。

`hidden_states` 等大 Tensor 只存于 fields，通过 TQ `kv_batch_get` 传输；不会放进 tag，也不会通过训练 rank 的 object broadcast。

### 5.4 metadata_json：内容丰富但仍随 fields 读取

TQ fields 只能承载 Tensor，因此结构化 metadata 被编码为 `uint8` Tensor。解码后的字典格式是：

```python
{
    "schema_version": 1,
    "run_id": "dspark-run-001",
    "sample_id": "prompt-000017",
    "sequence_no": 17,
    "algorithm": "DSPARK",
    "target_model_id": "/models/Qwen3-8B",
    "target_model_revision": "main",
    "tokenizer_fingerprint": "...",
    "target_layer_ids": [35],
    "hidden_states_layout": "token_major",
    "hidden_dtype": "bfloat16",
    "hidden_shape": [L, D],
    "feature_length": L,
    "full_sequence_length": 256,
    "feature_start": 64,
    "feature_end": 64 + L,
    "use_logits": False,
}
```

字段分工是：

- tag：只放发现、过滤、排序所需的小字段；`kv_list` 可直接得到。
- fields：放训练 Tensor 和完整 metadata；只有被某个 rank 选中后才 `kv_batch_get`。
- key：把 tag 和 fields 重新关联起来，也是清理记录时传给 `kv_clear` 的标识。

### 5.5 控制记录

控制记录与 sample 放在同一 partition，但通过 tag 的 `record_type=control` 区分。

Owner readiness key：

```text
control:v1:<run_id>:owner-ready
```

EOS key：

```text
control:v1:<run_id>:eos
```

EOS tag 包含 `status=eos` 和 `total_samples`。EOS 表示 Producer 不会再为本次 run 增加新样本；它不是一条训练样本。

## 6. 公共协议如何把 Producer 输出还原为训练对象

Producer 应调用：

```python
fields = encode_sample(sample, metadata)
key = make_sample_key(metadata)
tag = make_ready_tag(metadata)
put_sample(key, fields, tag=tag)
```

`encode_sample()` 会将所有 Tensor detach、转到 CPU、整理为 contiguous，并统一 `input_ids/position_ids` 为 int64、`loss_mask` 为 float32。随后校验 token 长度、hidden shape 和 metadata 一致，再把 metadata JSON 编成 uint8 Tensor。

Consumer 的逆过程位于 `TQFeatureStore.get_many()`：

```python
records = get_samples([entry.key for entry in entries])
sample = decode_sample(
    key=key,
    tag=entry.tag,
    fields=fields,
    expected_config=self.expected_config,
)
```

`get_samples()` 最终调用一次 TQ `kv_batch_get(keys=[...], partition_id=...)`。bridge 将 TQ 返回的 batched TensorDict 或 mapping 拆成与请求 key 顺序一致的普通 fields 字典。

`decode_sample()` 随后：

1. 检查必需 fields 是否存在。
2. 将 `metadata_json` 从 uint8 Tensor 还原为字典和 `SampleMetadata`。
3. 根据 metadata 重新计算 key，并与实际 key 比较。
4. 比较 tag 与 metadata 的公共身份字段。
5. 检查 Consumer 的 expected config。
6. 将 Tensor detach 到 CPU，统一基础 dtype/shape。
7. 检查所有主 Tensor 第一维等于 `feature_length`，hidden shape/dtype 与 metadata 一致。
8. 构造 `DraftFeatureSample`。

输出不再是 TQ 专用对象，而是既有训练代码使用的：

```python
DraftFeatureSample(
    input_ids=...,
    loss_mask=...,
    position_ids=...,
    hidden_states=...,
    metadata=...,
    ...,
)
```

这是能够复用原训练逻辑的关键边界：TQ 只负责上游存储和传输，`decode_sample()` 后的数据类型与磁盘 feature store 读取结果一致。

## 7. Consumer 从启动到结束的完整执行流程

### 阶段 1：launcher 检查配置并启动 torchrun

执行者是 launcher 父进程。入口是 `verl_speco.draft_train_launcher.main()`。

当 override 中出现 `feature_store.type=tq`，`validate_tq_launch_config()` 会要求：

- `training.transfer_queue.enable=true`；
- `training.transfer_queue.ray.address` 非空；
- `training.transfer_queue.run_id` 非空。

检查成功后构造：

```text
python -m torch.distributed.run
  --nnodes=...
  --nproc_per_node=...
  -m verl_speco.draft_train
  <全部 Hydra overrides>
```

配置参数是普通子进程命令行参数。此阶段没有 TQ Tensor 传输。

### 阶段 2：每个 rank 初始化训练运行时

每个 torchrun 子进程进入 `run_standalone_draft_training()`，调用 `_init_distributed()` 得到 `rank/local_rank/world_size`，绑定本 rank GPU，然后构造原有 `DrafterBaseTrainer` 和 DSpark backend。

`speculative_algorithm=DSPARK` 决定 backend 和 DSpark 模型训练实现；`feature_store.type=tq` 只改变数据来源，不替换 trainer。

### 阶段 3：factory 创建 TQFeatureStore

训练循环调用：

```python
store = build_feature_store_from_config(
    feature_store_cfg,
    read_only=True,
    transfer_queue_cfg=training_cfg.get("transfer_queue"),
)
```

factory 在 `type=tq` 时不读取 `feature_store.path`，而是把 sibling `training.transfer_queue` 交给 `TQFeatureStore.from_config()`。

TQ store 被限定为 `read_only=True`，意思是它是训练 Consumer source。这里的“read only”不表示永不修改 TQ；成功消费后仍可通过明确的 `clear_many()` 删除记录，但不会把它当作通用 feature writer。

### 阶段 4：所有 rank 分别连接同一个 TQ

训练循环调用 `_connect_tq_store_across_ranks()`。每个 rank 都独立执行 `store.connect()`：

```text
configure_transfer_queue
→ ray.init(address, namespace)
→ tq.init() 连接 named Controller
→ 本 rank 设置 _connected=True
```

之后 `_all_ranks_true()` 使用 `dist.all_reduce(MIN)` 汇总连接结果。只要一个 rank 连接失败，所有 rank 都停止，不允许部分 rank 进入后续 broadcast 或 FSDP collective。

这里没有“rank 0 建一个 client 给其他 rank 共用”。TQ client 是进程本地对象，N 个 rank 有 N 个 client，但它们指向同一 Controller/partition。

### 阶段 5：创建 TQFeatureDataLoader

每个 rank 构造自己的 loader，参数包括相同的 `batch_size_per_gpu`、`world_size`、轮询间隔和 drop-last，以及不同的 `rank`。

假设：

```text
world_size = 2
batch_size_per_gpu = 2
global_batch_size = 4
```

那么只有 ready 数量至少为 4，rank 0 才发布一个 batch 命令。

### 阶段 6：rank 0 发现 ready key

rank 0 首先检查 `owner_ready()`。Owner 尚未发布 readiness marker 时，rank 0 sleep 后继续轮询，不会让其他 rank 开始取数。

Owner ready 后，rank 0 调用 `list_ready()`，其底层是：

```text
tq.kv_list(partition_id)
→ key -> tag
→ 按 record_type/status/run/schema/algorithm 过滤
→ 按 (sequence_no, key) 排序
```

此阶段没有读取 fields，因此 hidden states 尚未传到训练进程。

### 阶段 7：rank 0 切分 global batch

若排序后的前四条是 `k0、k1、k2、k3`，`build_assignments()` 产生：

```python
assignments = [
    [ReadyEntry(k0, tag0), ReadyEntry(k1, tag1)],  # rank 0
    [ReadyEntry(k2, tag2), ReadyEntry(k3, tag3)],  # rank 1
]
```

每条样本只出现在一个 rank 的 assignment 中，因此各 rank 不会取得同一训练样本。这里采用连续、不重叠的切片。

rank 0 随后构造普通 Python 命令字典：

```python
{
    "kind": "batch",
    "global_keys": [k0, k1, k2, k3],
    "assignments": [
        [{"key": k0, "tag": tag0}, {"key": k1, "tag": tag1}],
        [{"key": k2, "tag": tag2}, {"key": k3, "tag": tag3}],
    ],
}
```

`global_keys` 只用于 rank 0 在训练完成后一次清理整个 batch；`assignments` 用于每个 rank 知道自己应该 get 哪些 key。

### 阶段 8：小型命令通过 torch.distributed 广播

各 rank 同时进入：

```python
dist.broadcast_object_list(payload, src=0)
```

rank 0 的 payload 中是上述字典，其他 rank 的初始值是 `None`。PyTorch 会序列化这个普通 Python 对象并广播给所有 rank。

这条边界只传输字符串、整数和小字典 tag。`hidden_states`、`input_ids` 等 fields 不在命令中，所以不会经 rank 0 中转，也不会随 broadcast 复制完整 global batch Tensor。

### 阶段 9：每个 rank 直接从 TQ 取本地 payload

每个 rank 从 `assignments[self.rank]` 还原自己的 `ReadyEntry`：

```python
local_entries = [_entry_from_wire(item) for item in assignments[self.rank]]
samples = self.store.get_many(local_entries)
```

在上述例子中：

- rank 0 调用 `kv_batch_get(keys=[k0, k1], partition_id=...)`；
- rank 1 调用 `kv_batch_get(keys=[k2, k3], partition_id=...)`。

大 Tensor 的数据面因此是 TQ storage 到目标训练 rank，不经过训练 rank 0 的 Python 内存。每个 rank 得到两个 CPU `DraftFeatureSample`。

loader yield：

```python
TQLocalBatch(
    local_keys=[本 rank 的 key],
    local_samples=[本 rank 的 DraftFeatureSample],
    global_keys=[完整 global batch key] if rank == 0 else None,
)
```

非 rank 0 不保存 `global_keys`，避免多个 rank 都尝试 clear。

### 阶段 10：复用已有训练 batch 构造

训练循环识别 `TQLocalBatch` 后，只取：

```python
samples = tq_local_batch.local_samples
```

然后调用原有接口：

```python
batch = trainer.prepare_training_batch_from_samples(
    materialized_samples,
    step=optimizer_step,
)
```

TQ 路径禁止同时开启 `target_feature_pipeline`，因为样本已经包含目标模型 hidden states，不需要训练侧再访问 vLLM materialize 一次。

此时 TQ 专用的 key/tag 已不参与 DSpark 数学计算；训练接口看到的是普通 `DraftFeatureSample`，并按原逻辑整理 input ids、hidden states、mask、position ids 和 DSpark 训练所需输入。

### 阶段 11：所有 rank 同步 batch 是否可训练

每个 rank 判断 `batch is not None`，再通过 `_all_ranks_true()` 做 `all_reduce(MIN)`。

只有所有 rank 都成功构造 batch，才能进入训练。如果任一 rank 解码或 batch 构造失败，TQ 路径直接报错，而且这些 key 不会被删除。

### 阶段 12：执行原有 DSpark training step

每个 rank 调用：

```python
ok = await trainer.training_step_from_batch(batch, optimizer_step)
```

该调用复用既有模型 forward、DSpark loss（包括配置开启时的 L1 loss）、backward、梯度同步和 optimizer step。TQ 新代码没有重新实现 loss 或 optimizer。

之后再次以 `_all_ranks_true(ok)` 同步。只有所有 rank 都返回成功，才认为这一个 global batch 已经安全消费。

### 阶段 13：训练成功后由 rank 0 删除 global batch

训练循环调用 `_clear_tq_batch_across_ranks()`：

1. rank 0 使用 `tq_local_batch.global_keys` 调用 `loader.clear_completed_batch()`。
2. loader 调用 `store.clear_many(global_keys)`。
3. bridge 最终调用 `tq.kv_clear(keys=[k0,k1,k2,k3], partition_id=...)`。
4. 所有 rank 通过 `all_reduce(MAX)` 同步 clear 是否失败。

删除发生在 optimizer step 全 rank 成功之后。不是“某个 rank get 完就删除”，因为 get 完只代表 Tensor 已读取，不能代表训练 step 已成功。

clear 成功后才增加 `successful_steps`，然后复用原有 metrics 和 checkpoint 调度。

### 阶段 14：EOS 和尾 batch

当 ready 样本少于一个 global batch时，rank 0 查询 EOS：

- 没有 EOS：说明 Producer 以后仍可能写入更多样本，sleep 后继续轮询。
- 已有 EOS 且 ready 为空：广播 `{"kind": "stop"}`，所有 rank 结束迭代。
- 已有 EOS 且存在不足一个 global batch 的尾样本：rank 0 先 clear 这些尾 key，再广播 stop。

第一版强制 `drop_last=true`，因此不会构造各 rank batch size 不一致的最后一步。

### 阶段 15：checkpoint 和退出清理

正常 step 完成后仍按原 `save_interval_steps` 保存 checkpoint；循环结束后按 `save_final_checkpoint` 决定是否保存最终 checkpoint。

`finally` 中每个 rank 调用 `store.close()`。对 `TQFeatureStore` 而言，这只是：

```text
关闭本进程 TQ client
→ 如果本进程自行 ray.init，则 ray.shutdown()
```

它不会调用全局 `tq.close()`，不会杀死 Owner 创建的 Controller，也不会影响仍在运行的 Producer 或其他 rank。

## 8. 控制面和数据面的完整边界

| 数据 | 从哪里到哪里 | 传输机制 | 是否经过 rank 0 |
|---|---|---|---|
| 启动配置 | launcher 到 torchrun 子进程 | 命令行 Hydra overrides | 每个 rank 都收到 |
| ready key/tag | TQ Controller 到 rank 0 | `tq.kv_list` | 是，只有 rank 0 list |
| batch assignment | rank 0 到全部 rank | `dist.broadcast_object_list` | 由 rank 0 发出 |
| hidden states 等 fields | TQ storage 到被分配的 rank | `tq.kv_batch_get` | rank 1 的 Tensor 不经过 rank 0 |
| batch 准备/训练成功状态 | 全部 rank 之间 | Tensor `all_reduce` | collective，无单点 payload relay |
| clear 请求 | rank 0 到 TQ | `tq.kv_clear(global_keys)` | 只有 rank 0 发起 |
| 梯度和模型 collective | 训练 rank 之间 | 既有 PyTorch distributed/FSDP 路径 | 与 TQ 无关 |

## 9. 当前“最简单校验”具体简单在哪里

`TQFeatureStore` 构造的 expected config 只固定：

```python
ExpectedFeatureConfig(
    run_id=<当前训练 run_id>,
    schema_version=<当前 schema>,
)
```

TQ 会保留 Producer 写入的 `SampleMetadata.algorithm`，但不使用它选择训练 backend，
也不额外与启动配置比较。与原有离线 feature-store 训练一致，实际 trainer/backend 只由
`rollout.drafter.speculative_algorithm` 和既有 backend factory 决定。

因此当前不会拿 Consumer 配置额外比较：

- target model ID/revision；
- tokenizer fingerprint；
- target layer IDs；
- hidden layout；
- hidden dtype 的外部预期值。

但这不等于完全不校验。`decode_sample()` 仍然强制检查：

- 必需 fields 存在；
- key、tag、metadata 三者身份一致；
- schema/run/algorithm 符合 Consumer；
- Tensor 类型正确；
- input/mask/position/hidden 长度一致；
- hidden 实际 shape/dtype 与该样本 metadata 一致；
- feature window 合法。

这满足“第一版少做外部模型身份检查”，同时避免把结构损坏或错 run 的数据送入训练。

## 10. 失败、删除和重复消费语义

当前实现遵循以下规则：

1. 连接失败：所有 rank 同步停止。
2. rank 0 list/EOS 失败：rank 0 广播 error 命令，其他 rank 不会永久等待 batch broadcast。
3. 某 rank get/decode 失败：`_next_batch_across_ranks()` 将失败同步给全部 rank，不进入模型训练 collective。
4. 某 rank 无法构造 batch：报错，global keys 保留在 TQ。
5. 某 rank training step 失败：报错，global keys 保留在 TQ。
6. 全 rank training step 成功：rank 0 clear 整个 global batch。
7. clear 失败：错误传播到全部 rank，训练停止；不会把该 step 继续当成已正常完成。
8. 达到 `max_steps`：循环停止；尚未选择的 ready 样本保留在 TQ。

第一版尚未实现完整的崩溃恢复协议。尤其是“optimizer step 已成功，但进程在 clear 前崩溃”时，key 仍存在；重新启动 Consumer 可能再次读取它。要实现严格 exactly-once，需要把 checkpoint step、已消费 sequence 或事务状态纳入协议。该能力应作为后续增强，而不是当前已实现能力。

## 11. 如何启动和检查

推荐启动顺序：

1. 启动 Ray head。
2. 启动 TQ Owner，并保持该进程存活。
3. 启动 Producer，使用相同 Ray address、namespace、partition 和 run ID。
4. 启动 `tools/run_dspark_tq_consumer.sh`。
5. Producer 完成全部样本后发布 EOS。
6. Consumer 消费完成并退出后，再停止 Owner/Ray。

示例：

```bash
MODEL_PATH=/models/Qwen3-8B \
DRAFTER_PATH=/models/dspark-drafter \
DRAFT_CKPTS_DIR=/checkpoints/dspark-tq \
TRAIN_DEVICES=0,1,2,3 \
TRAIN_GPUS=4 \
RAY_ADDRESS=127.0.0.1:6379 \
SPECO_TQ_RUN_ID=dspark-run-001 \
bash tools/run_dspark_tq_consumer.sh
```

脚本中的 namespace 固定为 `speco-drafter`，默认 partition 来自公共配置 `speco_drafter_features`。Owner 和 Producer 必须使用相同值。

## 12. 已完成的测试

### 12.1 Consumer/factory/协议/launcher 单元测试

已执行：

```text
python -m pytest \
  tests/unit/test_tq_consumer.py \
  tests/unit/test_draft_train_launcher.py \
  tests/unit/test_transferqueue_bridge.py \
  tests/unit/test_drafter_sample_protocol.py \
  tests/unit/test_draft_feature_store.py \
  -q
```

结果：`44 passed`。

### 12.2 真实 TQ 0.1.7 跨进程 smoke

`tools/tq_connection_smoke.py` 使用真实 Ray + TQ Owner 和另一个 Consumer 进程验证了：

- Owner 发布 owner-ready；
- 两条 sample 写入 TQ；
- Consumer 经 `TQFeatureStore` 和 `TQFeatureDataLoader` 读到两条样本；
- hidden shape 正确；
- Consumer clear 已完成 batch；
- EOS 后迭代停止；
- Consumer 只关闭本地 client，Owner 仍能继续观察完成标记并正常关闭。

实际 smoke 输出包含：

```text
CLIENT_OK samples=2 shape=(3,4)
CLIENT_CLOSED_LOCAL_ONLY
OWNER_OBSERVED_SAMPLES_CLEARED
OWNER_CLOSED
```

### 12.3 当前环境未覆盖的部分

完整 `tests/unit/test_draft_training_loop.py` 在当前 Windows 环境无法完整收集，因为上游 `verl/ray` 依赖不齐；新增训练循环测试代码已通过 Python 编译检查，连接/clear helper 也通过针对性单元逻辑验证。真实多 GPU DSpark 训练仍需要在目标 Linux GPU 环境执行集成测试。

## 13. 当前限制和后续建议

当前第一版有意不实现以下复杂能力：

1. Producer 本身尚未在本次 Consumer 改动中实现。
2. TQ 公共协议和 Consumer 已不再写死 `DSPARK`；当前测试 Producer、启动脚本和已验证的
   feature 语义仍是 DSPARK。其他算法若能复用当前公共 dense fields，只需由对应 Producer
   生成正确的 `DraftFeatureSample`；若字段结构不同，则在协议模块增加对应 codec，不需要改
   TQ 的 key/tag 发现、rank 分配和 clear 流程。
3. 只支持 `drop_last=true`。
4. 不支持 TQ 与 `target_feature_pipeline.enabled=true` 同时开启。
5. 不提供严格的 crash exactly-once 或 checkpoint/queue 联合恢复。
6. rank 0 仍通过 `kv_list` 轮询整个 partition；数据量很大时可考虑 cursor/ready queue 优化。
7. 当前外部 expected config 校验较简化，后续可把 model revision、tokenizer fingerprint、layer/layout/dtype 预期接入 Hydra 配置。
8. 尚需在真实多机、多 GPU、Mooncake backend 环境验证吞吐、背压、Owner 生命周期和网络故障行为。

建议下一阶段优先完成 Producer，并严格复用 `drafter_sample_protocol.py`，不要在 Producer 另造一套 key/tag/fields 格式。完成 Producer 后，首先跑 world size 1 的端到端训练，再跑多 rank 验证每条 key 只分配给一个 rank、训练成功后只由 rank 0 clear。

## 14. 最终路径摘要

```text
Producer（待实现）
  vLLM 并行 prefill
  → DraftFeatureSample + SampleMetadata
  → encode_sample 得到 Tensor fields
  → TQ kv_put(key, fields, tag)

Consumer rank 0
  kv_list 只取 key/tag
  → 过滤并按 sequence_no 排序
  → 切出 global batch
  → broadcast 每个 rank 的 key/tag assignment

每个 Consumer rank
  取 assignments[rank]
  → kv_batch_get 本 rank keys
  → decode_sample 得到 DraftFeatureSample
  → 原 prepare_training_batch_from_samples
  → 原 DSpark training_step_from_batch

全部 rank
  同步确认 optimizer step 成功
  → rank 0 kv_clear(global_keys)
  → 原 metrics/checkpoint
  → 下一批

Producer 发布 EOS
  → rank 0 确认没有完整 global batch
  → 清理不足一批的尾样本
  → broadcast stop
  → 各 rank 关闭本地 TQ client 并退出
```
