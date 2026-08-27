# Standalone TQ 公共基础层实现说明

Last updated: 08/21/2026

## 1. 文档范围和已验证结论

本文解释当前仓库已经实现并测试通过的 TQ 公共基础层：

```text
verl_speco/transport/drafter_sample_protocol.py
verl_speco/integration/transferqueue_bridge.py
verl_speco/config/speco_base.yaml
verl_speco/tq_owner.py
tests/unit/test_drafter_sample_protocol.py
tests/unit/test_transferqueue_bridge.py
pyproject.toml
```

当前已经实现：

1. Producer 和 Consumer 共用的样本 key、tag、fields 和 metadata 协议；
2. 普通进程连接 Ray 集群；
3. TQ Owner 创建 named `TransferQueueController`；
4. 独立 Client 发现并连接同一个 Controller；
5. 单样本 put、元数据 list、批量 get 和批量 clear；
6. Owner 与 Client 不同的关闭边界；
7. 独立 Owner 入口和共享 Hydra 配置；
8. mock 单元测试和真实双进程 TQ 0.1.7 smoke test。

当前还没有实现：

1. Producer 读取 JSONL、并发访问多个 vLLM endpoint 的完整 pipeline；
2. `feature_store.type=tq` 工厂分支；
3. `TQFeatureStore` 和 `TQFeatureDataLoader`；
4. rank 0 选择 global keys、各 rank 读取 local keys；
5. TQ batch 接入 DSpark optimizer step；
6. optimizer step 成功后的 rank 0 clear。

因此当前代码已经证明“两个独立进程能通过 Ray 连接同一个 TQ，并按共享协议批量传输多个样本”，但尚未接到正式 Producer 和 DSpark Consumer 主循环。

## 2. 运行时角色和术语

### 2.1 Ray head

Ray head 是 Ray 集群的控制节点。TQ 0.1.7 使用 Ray 管理 named Controller 和 storage actors。

Ray 不传输 standalone hidden-state payload。当前代码没有对这些样本调用：

```python
ray.put(hidden_states)
```

### 2.2 TQ Owner

TQ Owner 是普通 Python OS 进程，入口为：

```text
python -m verl_speco.tq_owner
```

它不是 Ray actor。它先调用 `ray.init(address=...)` 加入 Ray，再调用带完整配置的 `tq.init(config)` 创建 TQ Controller 和 storage。

Owner 是唯一允许调用全局 `tq.close()` 的进程。

### 2.3 Named TransferQueueController

TQ 0.1.7 内部创建：

```python
TransferQueueController.options(
    name="TransferQueueController"
).remote(...)
```

`name` 将 Controller actor 注册到 Ray actor registry。其他进程加入同一 Ray address 和 namespace 后，通过：

```python
ray.get_actor("TransferQueueController")
```

取得 actor handle，再读取 TQ backend 配置。

Controller 保存控制信息和数据位置；使用 MooncakeStore 时，大 tensor 本身保存在 MooncakeStore。

### 2.4 TQ Client

Owner、Producer、每个 Consumer rank 都在各自 OS 进程内拥有独立 TQ Client。

普通 Client 也传入相同的 native 配置：

```python
tq.init(native_config)
```

发现 named Controller并初始化本进程的 storage manager。Client 不是 Ray actor，Producer 和 torchrun rank 也不需要改成 Ray actor。

### 2.5 Partition、key、tag 和 fields

当前固定 partition：

```text
speco_drafter_features
```

TQ 中一条记录逻辑上是：

```text
partition_id
└── key
    ├── tag：轻量 dict，由 kv_list 发现
    └── fields：Tensor payload，由 kv_batch_get 读取
```

## 3. 共享配置如何工作

共享配置定义在 `verl_speco/config/speco_base.yaml`：

```yaml
transfer_queue:
  enable: false
  package_version: "0.1.7"
  ray:
    address: null
    namespace: speco-drafter
  partition_id: speco_drafter_features
  run_id: null
  schema_version: 1
  connect_timeout_seconds: 120
  poll_interval_seconds: 0.5
  drop_last: true
  controller:
    polling_mode: true
  backend:
    storage_backend: SimpleStorage
    SimpleStorage:
      total_storage_size: 100000
      num_data_storage_units: 8
    MooncakeStore:
      auto_init: false
      metadata_server: localhost:50050
      master_server_address: localhost:50051
      local_hostname: ""
      protocol: tcp
      global_segment_size: 4294967296
      local_buffer_size: 1073741824
      device_name: ""
```

### 3.1 Ray 连接字段

```yaml
ray:
  address: 10.0.0.1:6379
  namespace: speco-drafter
```

它们决定当前进程连接哪个 Ray 集群，以及在哪个 namespace 查找 `TransferQueueController`。Owner、Producer 和所有 Consumer ranks 必须使用相同值。

### 3.2 SPECO 协议字段

```yaml
partition_id: speco_drafter_features
run_id: dspark-20260819-a
schema_version: 1
```

这些字段用于路由和校验，不属于 TQ 原生配置。`run_id` 用于隔离不同 pipeline 的记录。

### 3.3 TQ 原生字段

```yaml
controller: ...
backend: ...
```

只有这些字段应传给 `tq.init(full_config)`。bridge 的 `_native_tq_config()` 会删除：

```text
enable
package_version
ray
partition_id
run_id
schema_version
connect_timeout_seconds
poll_interval_seconds
drop_last
```

对象变化为：

```text
完整 SPECO transfer_queue dict
→ _native_tq_config()
→ controller/backend等TQ字段
→ OmegaConf DictConfig
→ tq.init(same native config)
```

## 4. Bridge 的进程内状态

`verl_speco/integration/transferqueue_bridge.py` 在每个 OS 进程内分别维护：

```python
_state = {
    "enabled": False,
    "configured": False,
    "initialized": False,
    "config": None,
    "owner": False,
    "ray_initialized_here": False,
    "ray_address": None,
    "ray_namespace": None,
}
```

该 dict 不跨进程共享。Owner、Producer、每个 rank 分别拥有自己的 `_state`。

| 字段 | 含义 |
|---|---|
| `enabled` | 当前进程配置是否开启 TQ |
| `configured` | 是否调用过 `configure_transfer_queue()` |
| `initialized` | 当前进程是否执行过 `tq.init()` |
| `config` | 当前进程保存的普通 dict 配置 |
| `owner` | 当前进程是否创建了全局 Controller/Storage |
| `ray_initialized_here` | bridge 是否负责调用了本进程的 `ray.init()` |
| `ray_address/namespace` | 本进程的 Ray 连接信息 |

`_state_lock` 只保护同一进程内多个线程同时初始化，不是分布式锁。

## 5. Owner 的完整启动数据流

Owner 入口是 `verl_speco/tq_owner.py`，直接加载 Hydra 主配置
`verl_speco/config/speco_base.yaml`。`run_owner()` 会复制其中的
`transfer_queue` 子配置，并仅在 Owner 进程的副本中强制设置
`enable=true`；普通训练任务看到的共享默认值仍是 `false`。

### 阶段 1：读取配置

执行者：Owner OS 进程。

入口：

```python
run_owner(config)
```

取得：

```python
training_cfg = config.actor_rollout_ref.rollout.drafter.training
tq_cfg = training_cfg.transfer_queue
```

然后调用：

```python
configure_transfer_queue(training_cfg)
```

该函数只把 OmegaConf 转成普通 dict并更新当前进程 `_state`，不会连接 Ray，也不会创建 TQ。

### 阶段 2：连接 Ray

Owner 调用：

```python
connect_ray_cluster(ray_address, namespace)
```

内部执行：

```python
if not ray.is_initialized():
    ray.init(address=ray_address, namespace=namespace)
```

边界类型是 Ray control-plane connection。此时还没有传输训练 tensor。

### 阶段 3：创建 Controller 和 Storage

Owner 调用：

```python
start_transfer_queue_owner(tq_cfg)
```

执行顺序：

1. `_extract_tq_config()` 得到普通 dict；
2. 检查 `enable=true`；
3. 检查 `TransferQueue` 包可用；
4. 防止本进程重复初始化；
5. `_native_tq_config()` 删除 SPECO 字段；
6. `_as_tq_config()` 转 OmegaConf；
7. `tq.init(native_config)` 创建 Controller、Storage 和 Owner Client；
8. 设置 `_state.owner=True`、`initialized=True`。

Ray 中形成：

```text
Ray cluster / namespace
├── named actor: TransferQueueController
└── storage backend
    ├── SimpleStorage actors
    └── 或 MooncakeStore connection/process
```

### 阶段 4：发布 owner-ready

调用：

```python
publish_owner_ready(run_id, schema_version)
```

生成：

```python
key = "control:v1:<run_id>:owner-ready"
fields = {"marker": torch.tensor([1], dtype=torch.uint8)}
tag = {
    "record_type": "control",
    "status": "owner_ready",
    "schema_version": 1,
    "run_id": run_id,
}
```

这是一条控制记录，不进入训练 batch。

### 阶段 5：常驻和关闭

Owner 安装 `SIGINT/SIGTERM` handler，并等待：

```python
stop_event.wait()
```

收到信号后调用：

```python
close_transfer_queue_owner()
```

执行：

```text
验证owner身份
→ tq.close()
→ 清理Controller/Storage
→ ray.shutdown()
```

Owner 必须在 Producer 和 Consumer 退出后才能关闭。

## 6. 普通 Client 如何连接同一个 TQ

Producer 和每个 Consumer rank 后续使用相同顺序：

```python
configure_transfer_queue(training_cfg)
connect_ray_cluster(ray_address, namespace)
connect_transfer_queue_client()
```

`connect_transfer_queue_client()` 最终调用：

```python
tq.init(same_native_config)
```

TQ 0.1.7 内部通过：

```python
ray.get_actor("TransferQueueController")
```

找到 Owner 创建的 Controller，读取 backend 配置，然后创建当前进程的 TQ Client。

对象和边界变化：

```text
actor名称字符串
→ Ray actor registry
→ Controller actor handle
→ Controller.get_config.remote()
→ TQ DictConfig
→ 当前进程TransferQueueClient
→ 同一个SimpleStorage/MooncakeStore
```

## 7. 一条具体样本的初始对象

真实 smoke test使用：

```python
sample = DraftFeatureSample(
    algorithm="DSPARK",
    input_ids=torch.tensor([1, 2, 3]),          # int64[3], CPU
    loss_mask=torch.tensor([0.0, 1.0, 1.0]),   # float32[3], CPU
    position_ids=torch.tensor([0, 1, 2]),      # int64[3], CPU
    hidden_states=torch.arange(
        12, dtype=torch.float32
    ).reshape(3, 4),                            # float32[3,4], CPU
)
```

同时构造：

```python
meta = SampleMetadata(
    schema_version=1,
    run_id="codex-batch-smoke",
    sample_id="smoke-0000",
    sequence_no=0,
    algorithm="DSPARK",
    target_model_id="smoke-target",
    target_model_revision="smoke-revision",
    tokenizer_fingerprint="smoke-tokenizer",
    target_layer_ids=[0],
    hidden_states_layout="dflash_aux",
    hidden_dtype="float32",
    hidden_shape=[3, 4],
    feature_length=3,
    full_sequence_length=3,
    feature_start=0,
    feature_end=3,
    use_logits=False,
)
```

`DraftFeatureSample` 和 `SampleMetadata` 都是进程内 Python 对象，不直接经过 TQ。

## 8. Key 的生成和两个同名函数

共享协议调用：

```python
make_sample_key(meta)
```

输出：

```text
drafter:v1:codex-batch-smoke:000000000000:smoke-0000
```

字段顺序：

```text
drafter / schema version / run_id / 12位sequence_no / sample_id
```

`sequence_no` 是输入文件 record 序号，不是训练 batch 编号。

bridge 为兼容 PR #48 还保留另一个：

```python
transferqueue_bridge.make_sample_key(
    global_step,
    replica_rank,
    request_id,
)
```

它生成：

```text
speco:<global_step>:<replica_rank>:<request_id>
```

standalone Producer 必须从 `verl_speco.transport.drafter_sample_protocol` import `make_sample_key`，不能使用 bridge 中的 PR #48 旧函数。

## 9. Tag 如何生成

```python
tag = make_ready_tag(meta)
```

输出：

```python
{
    "record_type": "sample",
    "status": "ready",
    "schema_version": 1,
    "run_id": "codex-batch-smoke",
    "sequence_no": 0,
    "sample_id": "smoke-0000",
    "algorithm": "DSPARK",
}
```

tag 只包含发现、过滤和排序需要的标量/字符串。`list_samples()` 只返回 key/tag，不读取 hidden states。

## 10. `encode_sample()` 如何生成 fields

调用：

```python
fields = encode_sample(sample, meta)
```

### 10.1 校验

执行：

```text
SampleMetadata.validate()
DraftFeatureSample.validate(strict=True)
```

随后检查：

1. hidden states 是一个 dense tensor；
2. ids/mask/position 长度等于 `feature_length`；
3. hidden 第一维等于 `feature_length`；
4. hidden shape 等于 metadata；
5. hidden dtype 等于 metadata；
6. feature window 长度正确。

### 10.2 Tensor 规范化

```text
input_ids      → CPU contiguous int64[L]
loss_mask      → CPU contiguous float32[L]
position_ids   → CPU contiguous int64[L]
hidden_states  → CPU contiguous，保持模型dtype
```

没有 `position_ids` 时生成 `torch.arange(L, dtype=int64)`。

### 10.3 Metadata JSON 编码

```text
SampleMetadata dataclass
→ dict
→ JSON UTF-8 bytes
→ torch.uint8[M]
```

实现等价于：

```python
raw = json.dumps(metadata).encode("utf-8")
metadata_json = torch.tensor(list(raw), dtype=torch.uint8)
```

### 10.4 最终 fields

```python
fields = {
    "input_ids": int64[3],
    "loss_mask": float32[3],
    "position_ids": int64[3],
    "hidden_states": float32[3,4],
    "metadata_json": uint8[M],
}
```

如果存在，协议也保留 `last_hidden_states`、`target` 和 `target_logprobs`。

## 11. Bridge 如何写入 TQ

调用：

```python
put_sample(key, fields, tag=tag)
```

bridge 执行：

1. 检查 TQ 已启用；
2. 丢弃 fields 中非 tensor 值；
3. 确保本进程已经使用相同 native 配置执行 `tq.init(config)`；
4. 取得配置中的 partition；
5. 调用：

```python
tq.kv_put(
    key=key,
    partition_id="speco_drafter_features",
    fields=fields,
    tag=tag,
)
```

使用 MooncakeStore 时，大 tensor 路径是：

```text
Producer CPU tensor
→ Producer TQ Client
→ MooncakeStore
```

不是 Ray `ObjectRef`。

## 12. Consumer 如何发现 key

调用：

```python
records = list_samples()
```

内部调用：

```python
tq.kv_list(partition_id="speco_drafter_features")
```

标准化返回类型：

```python
dict[str, dict[str, Any]]
```

示例：

```python
{
    "drafter:v1:...:smoke-0000": {
        "record_type": "sample",
        "status": "ready",
        "run_id": "codex-batch-smoke",
        "sequence_no": 0,
        ...
    }
}
```

bridge 兼容 `key → tag` 和 `partition → key → tag` 两种 wrapper。这一步不读取 fields。

## 13. Consumer 如何批量取样本

输入：

```python
keys = [key0, key1]
```

调用：

```python
records = get_samples(keys)
```

bridge 只调用一次：

```python
result = tq.kv_batch_get(
    keys=keys,
    partition_id="speco_drafter_features",
)
```

TQ 0.1.7 返回带 batch 维的 TensorDict。bridge 检查 `result.batch_size`，再执行：

```python
rows = [result[index] for index in range(len(keys))]
```

每行转成普通 dict，最终返回：

```python
[
    (key0, fields0),
    (key1, fields1),
]
```

返回顺序与输入 keys 一致。重复 key 会提前报错。

## 14. `decode_sample()` 如何恢复训练对象

调用：

```python
sample = decode_sample(
    key,
    tag,
    fields,
    expected_config,
)
```

### 14.1 Metadata 解码

```text
metadata_json uint8[M]
→ bytes
→ UTF-8
→ json.loads
→ dict
→ SampleMetadata.from_dict
```

### 14.2 身份一致性

代码根据 metadata 重新生成 key，要求输入 key 完全相等；然后逐项校验 tag 的：

```text
record_type/status/schema_version/run_id/sequence_no/sample_id/algorithm
```

所以 key、tag 和 payload metadata 不能来自不同样本。

### 14.3 Consumer 合同

Consumer 提供：

```python
ExpectedFeatureConfig(
    run_id="codex-batch-smoke",
    schema_version=1,
    algorithm="DSPARK",
    target_model_id="smoke-target",
    target_model_revision=None,
    tokenizer_fingerprint=None,
    target_layer_ids=None,
    hidden_states_layout="dflash_aux",
    hidden_dtype="float32",
)
```

值为 `None` 的字段不检查；其他字段必须完全一致。正式 Consumer 应填写 target checkpoint、tokenizer、layers、layout 和 dtype，避免使用错误 target 特征。

### 14.4 输出

完成 tensor 类型、长度、shape、dtype 校验后，构造：

```python
DraftFeatureSample.from_dict(payload, strict=True)
```

输出可以交给现有 `trainer.prepare_training_batch_from_samples()`。当前 smoke test验证到这里，正式 `TQFeatureDataLoader` 尚未实现。

## 15. EOS 控制记录

调用：

```python
key, fields, tag = make_eos_record(run_id, total_samples)
```

输出：

```python
key = "control:v1:<run_id>:eos"
fields = {"marker": torch.tensor([1], dtype=torch.uint8)}
tag = {
    "record_type": "control",
    "status": "eos",
    "schema_version": 1,
    "run_id": run_id,
    "total_samples": total_samples,
}
```

EOS 表示不会再发布新样本。Consumer 应先 drain ready samples 再退出。协议函数已实现，正式 Producer/Consumer 尚未调用。

## 16. Clear 和数据生命周期

bridge 提供：

```python
clear_samples(keys)
```

内部调用：

```python
tq.kv_clear(keys=keys, partition_id="speco_drafter_features")
```

基础层只执行删除，不决定删除时机。正式 Consumer 必须遵守：

```text
rank 0选择global keys
→ 各rank读取local keys
→ 所有rank完成同一optimizer step
→ 汇总global success
→ rank 0 clear global keys
```

不能在 `get_samples()` 后立即 clear，因为 get 成功不代表训练 step 成功。

## 17. Client close 和 Owner close

### 17.1 Client close

Producer/rank 调用：

```python
close_transfer_queue_client()
```

执行：

```text
tq.get_client()
→ 当前进程client.close()
→ 如果bridge负责ray.init，则ray.shutdown()
```

它不调用全局 `tq.close()`，不会 kill Controller。调用后本进程不能继续使用 TQ。

### 17.2 Owner close

Owner 调用：

```python
close_transfer_queue_owner()
```

执行：

```text
tq.close()
→ Controller/Storage全局清理
→ ray.shutdown()
```

Owner 若误用 Client close，bridge 会抛 `RuntimeError`。

## 18. PR #48 兼容边界

bridge 继续保留：

```python
init_transfer_queue(config)
get_sample(key)
close_transfer_queue()
```

PR #48 的 `SpecoTaskRunner` 已是 Ray actor，所以不调用 standalone 的 `connect_ray_cluster()`。旧 worker 继续使用单 key `get_sample()`。

standalone 后续使用新增的：

```python
list_samples()
get_samples(keys)
clear_samples(keys)
```

因此没有修改 PR #48 现有调用点的函数签名。

## 19. 依赖和命令入口

`pyproject.toml` 新增：

```toml
[project.optional-dependencies]
transfer-queue = ["TransferQueue==0.1.7"]
```

安装：

```bash
pip install -e ".[transfer-queue]"
```

TQ 0.1.7 会安装 Ray，并要求 `numpy<2.0.0`。若现有包要求 NumPy 2，需要隔离环境或重新解决依赖。

Owner 命令：

```text
verl-speco-tq-owner
```

## 20. 单元测试

协议测试 `tests/unit/test_drafter_sample_protocol.py` 覆盖：

1. encode/decode round trip；
2. key 格式；
3. tag 身份冲突；
4. Consumer contract 冲突；
5. hidden shape 冲突；
6. EOS 格式。

bridge 测试 `tests/unit/test_transferqueue_bridge.py` 覆盖：

1. Ray address/namespace 参数；
2. Owner 只向 TQ 传原生配置；
3. Client 使用相同 native 配置调用 `tq.init(config)`；
4. put/list/get-many/clear；
5. batch 返回顺序；
6. Client close 不调用全局 close；
7. Owner 不能误用 Client close。

运行：

```bash
python -m pytest \
  tests/unit/test_drafter_sample_protocol.py \
  tests/unit/test_transferqueue_bridge.py \
  -q
```

## 21. 真实双进程 smoke test

早期用于该验证的临时双进程工具已经移除；正式入口统一由
`verl_speco.standalone_tq_training_launcher` 管理 Owner、Producer 和 Consumer 生命周期。

Owner 路径：

```text
连接Ray
→ tq.init(full config)
→ 写sample 0和sample 1
→ 等待client-done
→ clear done marker
→ 全局关闭
```

Client 路径：

```text
连接同一个Ray
→ tq.init(same native config)
→ kv_list发现两个key
→ 一次kv_batch_get([k0,k1])
→ 拆成两个fields dict
→ 分别decode_sample
→ clear两个sample keys
→ 写client-done
→ 只关闭本地client
```

已验证输出：

```text
OWNER_READY keys=[k0, k1]
CLIENT_OK samples=2 shape=(3, 4)
CLIENT_CLOSED_LOCAL_ONLY
OWNER_OBSERVED_SAMPLES_CLEARED
OWNER_CLOSED
```

这证明：

1. 两个普通进程能连接同一个 TQ；
2. named Controller 发现有效；
3. 0.1.7 的 `kv_list/kv_batch_get/kv_clear` 参数有效；
4. TensorDict batch 能按 key 顺序拆开；
5. 共享协议能恢复 `DraftFeatureSample`；
6. Client close 不会杀掉 Owner；
7. Owner 能最终统一关闭。

## 22. 当前完整路径总结

```text
Owner
→ ray.init(address, namespace)
→ tq.init(native config)
→ named TransferQueueController

普通Client
→ ray.init(same address, same namespace)
→ tq.init(same native config)
→ 找到同一个Controller

DraftFeatureSample + SampleMetadata
→ make_sample_key
→ make_ready_tag
→ encode_sample
→ fields + metadata_json tensor
→ bridge.put_sample
→ tq.kv_put
→ SimpleStorage/MooncakeStore

Consumer/测试Client
→ bridge.list_samples
→ key + tag
→ bridge.get_samples(keys)
→ tq.kv_batch_get
→ TensorDict batch
→ 每个key对应一个fields dict
→ decode_sample
→ DraftFeatureSample

正式训练成功后（待实现）
→ bridge.clear_samples(global_keys)

Client退出
→ close_transfer_queue_client

所有业务进程退出
→ Owner close_transfer_queue_owner
→ tq.close
→ ray.shutdown
```

## 23. 下一阶段接入约束

后续代码不能重新定义协议或直接访问 TQ 私有对象。

Producer 应复用：

```text
SampleMetadata
make_sample_key
make_ready_tag
encode_sample
bridge.put_sample
make_eos_record
```

Consumer 应复用：

```text
bridge.list_samples
bridge.get_samples
decode_sample
bridge.clear_samples
```

下一阶段需要新增：

```text
verl_speco/trainer/tq_feature_store.py
verl_speco/trainer/tq_sample_source.py
feature_store.py 的 type=tq 分支
draft_training_loop.py 的流式训练分支
Producer入口、输入读取和并发vLLM文件
```

这些文件应建立在本文已经实现和真实验证过的连接、协议、KV 和关闭接口之上。
