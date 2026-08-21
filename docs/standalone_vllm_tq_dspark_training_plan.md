# 独立 vLLM Producer + TQ + DSpark Consumer 第一版方案

Last updated: 08/21/2026

## 1. 第一版要实现什么

只实现下面这条主链路：

```text
verl prompt-only 数据或包含 prompt + response 的输入文件
→ Producer 并发请求 vLLM prefill
→ Producer 将每条训练样本写入 TQ
→ Consumer 从同一个 TQ 取样本
→ 独立 torchrun/FSDP DSpark 训练
→ 一个 optimizer step 成功后删除该 step 的 TQ 样本
```

第一版允许 **TQ 基础设施内部使用 Ray**，因为实测 `TransferQueue==0.1.7` 通过 Ray named actor 发现 `TransferQueueController`。Producer 和 Consumer 仍是普通 OS 进程，不改成 Ray actor；二者不使用 Ray RPC、Ray `ObjectRef` 或 Ray object store 传输训练样本。hidden-state payload 仍通过 TQ 的 `kv_put/kv_batch_get` 和配置的 MooncakeStore backend 传输。第一版不使用 DataProto/WorkerGroup，不生成长期 hidden-state feature store，也暂不实现复杂重试、自动重启和严格 checkpoint 数据恢复。

需要运行的组件：

| 组件 | 数量 | 作用 |
|---|---:|---|
| vLLM server | 一个或多个 | 加载 target model，执行并行 prefill |
| Ray head | 1 个集群 | 保存 TQ named Controller/Storage actors；不承载 Producer/Consumer 业务 RPC |
| TQ owner | 1 个普通进程 | 连接 Ray，创建并保持 TQ controller/storage，最后统一关闭 TQ |
| Producer | 1 个进程 | 读文件、并发请求 vLLM、写 TQ |
| Consumer | 1 个 torchrun 任务 | 多个 rank 从 TQ 取数并训练 DSpark |

Producer 和 Consumer 不互相调用，也不通过 HTTP 或 Ray 传样本。二者先连接同一个 Ray 集群，再通过携带相同 native 配置的 `tq.init(config)` 找到同一个 TQ，最后使用 TQ KV API 读写样本。已有 Controller 时 TransferQueue 0.1.7 会忽略后续配置并只连接；若 Client 意外先初始化，同一配置可避免默认 backend 抢先生效。

## 2. 共同的数据约定

这部分由两位开发者共同完成并先合入。建议文件：

```text
verl_speco/transport/drafter_sample_protocol.py
tests/unit/test_drafter_sample_protocol.py
```

### 2.1 一个 key 对应一条样本

第一版固定：

```text
一个输入文件 record
→ 一个 sequence_no
→ 一个 sample_id
→ 一个 TQ sample_key
→ 一个单样本 payload
```

`sequence_no` 是输入文件中的样本序号，不是 batch 编号。多个 sample keys 到 Consumer 后才组成训练 batch。

例如：

```python
run_id = "dspark-20260818-a"
sequence_no = 17
sample_id = "train-000017"

partition_id = "speco_drafter_features"  # 第一版沿用 PR 固定值
sample_key = (
    "drafter:v1:dspark-20260818-a:"
    "000000000017:train-000017"
)
```

### 2.2 Partition、key、tag 和 payload 的关系

TQ 中逻辑上是：

```text
TQ 实例
└── partition_id
    └── sample_key
        ├── tag
        └── fields/payload
```

- TQ 实例：Producer 和 Consumer 共同连接的 controller/storage；
- `partition_id`：第一版沿用 PR 固定的 `speco_drafter_features`；每次启动新的 TQ owner，保证该 TQ 实例初始为空；
- `sample_key`：该分区中一条训练样本的地址；
- `tag`：轻量索引，Consumer 用 `kv_list()` 看见；
- `fields/payload`：真正的 Tensor 数据，Consumer 用 `kv_batch_get()` 读取。

Producer 写入：

```python
tq.kv_put(
    partition_id=partition_id,
    key=sample_key,
    fields=fields,
    tag=tag,
)
```

Consumer 先发现 key：

```python
all_records = tq.kv_list()
tags_by_key = all_records[partition_id]
```

这一步只拿 key 和 tag，不搬运 hidden states。

Consumer 再取数据：

```python
result = tq.kv_batch_get(
    partition_id=partition_id,
    keys=selected_keys,
)
```

`kv_batch_get` 中的 batch 表示“一次读取多个独立 sample keys”，不是这些样本在 Producer 写入时就属于同一个对象。

### 2.3 Payload 字段

一个 sample key 对应的 `fields`：

```python
fields = {
    "input_ids": input_ids,            # CPU int64[L]
    "loss_mask": loss_mask,            # CPU float32[L]
    "position_ids": position_ids,      # CPU int64[L]
    "hidden_states": hidden_states,    # CPU bf16[L,D]
    "metadata_json": metadata_bytes,   # CPU uint8[M]
}
```

| field | 含义 | Consumer 中的用途 |
|---|---|---|
| `input_ids` | 经过 feature window 选择后的 token IDs | DSpark token 输入 |
| `loss_mask` | 每个 token 是否参与 loss | 排除 prompt/padding/无效位置 |
| `position_ids` | 每个 row 对应的序列位置 | 位置编码和对齐校验 |
| `hidden_states` | target 指定层 hidden 沿最后一维拼接 | DSpark target/context 特征 |
| `metadata_json` | 模型、layers、layout、shape、样本身份 | Consumer 校验并恢复 metadata |

符号：

- `L`：这条训练 feature 保留的 token row 数；
- `H`：target model hidden size；
- `C`：DSpark context layer 数；
- L1 关闭：`D=C*H`，layout=`dflash_aux`；
- L1 开启：`D=C*H+H`，layout=`dflash_aux_plus_last`。

示例：`H=4096,C=5,L=1536`，开启 L1：

```python
input_ids.shape == [1536]
loss_mask.shape == [1536]
position_ids.shape == [1536]
hidden_states.shape == [1536, 24576]
```

`metadata_json` 使用 JSON UTF-8 编码为 Tensor，因为参考 PR 的 bridge 只把 Tensor 放入 TQ fields：

```python
raw = json.dumps(metadata, sort_keys=True).encode("utf-8")
metadata_bytes = torch.tensor(list(raw), dtype=torch.uint8)
```

### 2.4 Tag 字段

```python
tag = {
    "record_type": "sample",
    "status": "ready",
    "schema_version": 1,
    "run_id": "dspark-20260818-a",
    "sequence_no": 17,
    "sample_id": "train-000017",
    "algorithm": "DSPARK",
}
```

tag 只存用于发现和筛选的 flat scalar/string。Consumer 只选择：

```text
record_type=sample
status=ready
schema_version=1
run_id=当前 run
algorithm=DSPARK
```

### 2.5 Metadata 字段

`metadata_json` 解码后至少包含：

```python
metadata = {
    "schema_version": 1,
    "run_id": "dspark-20260818-a",
    "sample_id": "train-000017",
    "sequence_no": 17,
    "algorithm": "DSPARK",
    "target_model_id": "/models/Qwen3-8B",
    "target_model_revision": "revision-or-checksum",
    "tokenizer_fingerprint": "sha256:...",
    "target_layer_ids": [2, 8, 14, 20, 26, -1],
    "hidden_states_layout": "dflash_aux_plus_last",
    "hidden_dtype": "bfloat16",
    "hidden_shape": [1536, 24576],
    "feature_length": 1536,
    "full_sequence_length": 1800,
    "feature_start": 264,
    "feature_end": 1800,
    "use_logits": False,
}
```

其中：

- `target_model_id/revision`：hidden states 来自哪个 target checkpoint；
- `tokenizer_fingerprint`：Producer 使用的 tokenizer/template 版本；
- `target_layer_ids`：vLLM 返回和参与拼接的层；
- `hidden_states_layout`：Consumer 应如何拆 hidden 最后一维；
- `feature_length`：payload 中四个主要 Tensor 的第一维；
- `full_sequence_length`：完整 prompt+response 的 token 数；
- `[feature_start,feature_end)`：feature 在完整序列中的范围。

### 2.6 共享协议接口

Producer 和 Consumer 都 import 同一个模块，不各自手写字段名：

```python
@dataclass(frozen=True)
class SampleMetadata:
    schema_version: int
    run_id: str
    sample_id: str
    sequence_no: int
    algorithm: str
    target_model_id: str
    target_model_revision: str
    tokenizer_fingerprint: str
    target_layer_ids: list[int]
    hidden_states_layout: str
    hidden_dtype: str
    hidden_shape: list[int]
    feature_length: int
    full_sequence_length: int
    feature_start: int
    feature_end: int
    use_logits: bool

def make_sample_key(meta: SampleMetadata) -> str: ...
def make_ready_tag(meta: SampleMetadata) -> dict: ...
def encode_sample(sample, meta: SampleMetadata) -> dict[str, Tensor]: ...
def decode_sample(key, tag, fields, expected_config) -> DraftFeatureSample: ...
def make_eos_record(run_id: str, total_samples: int): ...
```

Producer 使用 `SampleMetadata/make_sample_key/make_ready_tag/encode_sample`；Consumer 使用 `decode_sample`。

`SampleMetadata` Python 对象本身不经过 TQ：

```text
Producer SampleMetadata
→ JSON
→ uint8 Tensor
→ TQ metadata_json
→ uint8 Tensor
→ JSON
→ Consumer metadata dict
```

`decode_sample()` 负责：

1. 解码 `metadata_json`；
2. 校验 key、tag、metadata 中的 sample 身份一致；
3. 校验模型、tokenizer、layers 和 layout 与 Consumer 配置一致；
4. 校验 Tensor 必需字段、dtype 和 shape；
5. 返回现有 `DraftFeatureSample`。

### 2.7 EOS

Producer 完成全部输入后写一个控制 record：

```python
eos_key = f"control:v1:{run_id}:eos"
eos_fields = {"marker": torch.tensor([1], dtype=torch.uint8)}
eos_tag = {
    "record_type": "control",
    "status": "eos",
    "schema_version": 1,
    "run_id": run_id,
    "total_samples": total_samples,
}
```

EOS 不进入训练 batch。Consumer 看到 EOS 后继续处理剩余 ready samples；`EOS 已出现且 ready 为空` 时结束。

## 3. TQ 怎么启动，Producer 和 Consumer 怎么连接同一个 TQ

### 3.1 已验证的 TQ 0.1.7 连接机制

`TransferQueue==0.1.7` 没有提供“把 Controller 地址直接传给第二个进程”的高层连接接口。`tq.init(config)` 会先尝试以下已有 Controller 连接逻辑；存在时忽略传入配置，不存在时才用配置创建服务：

```python
_TQ_CONTROLLER = ray.get_actor("TransferQueueController")
conf = ray.get(_TQ_CONTROLLER.get_config.remote())
_maybe_create_tq_client(conf)
```

因此所有进程必须先加入同一个 Ray 集群。Ray 在本方案中只承担 TQ 控制面：保存 named `TransferQueueController`、返回 TQ 配置、管理 TQ 创建的 actor。业务进程不通过 Ray 发送 sample key 或 hidden-state tensor。

实际连接链路是：

```text
TQ owner：ray.init(address) → tq.init(full_tq_config) → 创建 named Controller
Producer：ray.init(address) → tq.init(same config) → ray.get_actor() → 创建本地 TQ client
Consumer rank 0..N：ray.init(address) → tq.init(same config) → ray.get_actor() → 创建各自 TQ client
```

### 3.2 直接移植并扩展 PR #48 的 bridge

参考文件：

```text
C:/Users/xxyyrr/Desktop/上班/verl-SpeCo/
  verl_speco/integration/transferqueue_bridge.py
```

第一版不新增 `standalone_tq.py`，直接把该文件移植到目标项目并扩展。保留 PR 已有的 import 检查、进程内幂等状态、`kv_put`、`kv_batch_get`、TensorDict 解包和 owner-only shutdown；新增 Ray 显式连接、批量 list/get/clear 和本地 client 关闭接口。

目标文件必须实现以下函数，而不只是提供一个笼统的 transport class：

```python
def configure_transfer_queue(config: Mapping[str, Any]) -> bool: ...
def connect_ray_cluster(ray_address: str, namespace: str | None = None) -> None: ...
def start_transfer_queue_owner(tq_config: Mapping[str, Any]) -> None: ...
def connect_transfer_queue_client() -> None: ...
def make_sample_key(run_id: str, sequence_no: int, sample_id: str) -> str: ...
def put_sample(key: str, fields: dict[str, Tensor], tag: dict[str, Any]) -> None: ...
def list_samples() -> dict[str, dict[str, Any]]: ...
def get_samples(keys: list[str]) -> list[tuple[str, dict[str, Tensor]]]: ...
def clear_samples(keys: list[str]) -> None: ...
def close_transfer_queue_client() -> None: ...
def close_transfer_queue_owner() -> None: ...
```

逐个函数的责任如下。

#### `configure_transfer_queue(config)`

- 从 Hydra/OmegaConf 中读取 Ray address、可选 namespace、固定 partition、run ID、schema version 和 TQ native backend 配置；
- 转成普通 Python dict，保存在进程内 `_state`；
- 校验 `TransferQueue==0.1.7` 可 import；
- 不连接 Ray，不创建 TQ，不产生跨进程副作用；
- 返回该进程是否启用了 TQ。

#### `connect_ray_cluster(ray_address, namespace)`

- 若 `ray.is_initialized()` 为 false，调用 `ray.init(address=ray_address, namespace=namespace)`；
- 若已经初始化，校验现有 Ray context 指向期望集群，不能悄悄连到另一个本地 Ray；
- Owner、Producer 和所有 torchrun ranks 都调用它；
- 此函数只建立当前 OS 进程到 Ray control plane 的连接。

#### `start_transfer_queue_owner(tq_config)`

- 仅由 `tq_owner.py` 调用；
- 前置条件是 `connect_ray_cluster()` 已成功；
- 调用一次 `tq.init(OmegaConf.create(tq_config))`；
- 将 `_state.owner=True`、`_state.initialized=True`；
- TQ 0.1.7 会创建名为 `TransferQueueController` 的 Ray actor，并创建所选 storage backend；
- 重复调用必须报错，不能启动第二套同名 Controller。

#### `connect_transfer_queue_client()`

- 由 Producer 和每个 Consumer rank 调用；
- 前置条件是当前进程已经连接 Ray；
- 调用 `tq.init(same native config)`，通过 `ray.get_actor("TransferQueueController")` 发现 owner；已有 Controller 时配置会被忽略，意外抢先时则以相同配置创建；
- 只创建当前进程的 TQ client，不创建新的 Controller；
- 成功后设置 `_state.initialized=True`；重复调用直接返回。

#### `put_sample/list_samples/get_samples/clear_samples`

- 全部固定使用 `_SPECO_TQ_PARTITION = "speco_drafter_features"`；
- `put_sample()` 调用单样本 `tq.kv_put()`；
- `list_samples()` 调用 `tq.kv_list(partition_id=...)`，只返回 key/tag 元数据；
- `get_samples()` 一次调用 `tq.kv_batch_get(keys=...)`，再按输入 key 顺序解包为普通 dict；
- `clear_samples()` 调用 `tq.kv_clear(keys=..., partition_id=...)`；
- 这些函数不调用 Ray RPC，不把 tensor 放入 Ray object store。

#### `close_transfer_queue_client()` 与 `close_transfer_queue_owner()`

TQ 0.1.7 的公共 `tq.close()` 会 kill 共享 Controller，所以两者不能写成同一个实现：

- `close_transfer_queue_client()`：通过 0.1.7 已公开的 `tq.get_client()` 取得当前进程 client并调用它的 `close()`，然后 `ray.shutdown()`；绝不能调用会 kill Controller 的全局 `tq.close()`。这个 client close 只用于进程退出阶段，关闭后本进程不能再次调用 TQ；
- `close_transfer_queue_owner()`：仅当 `_state.owner=True` 时调用 `tq.close()`，清理 Controller/Storage，最后 `ray.shutdown()`；
- Producer 或任一训练 rank 提前退出都不能关闭全局 TQ。

### 3.3 共享配置

Owner、Producer 和 Consumer 必须使用相同的 Ray address/namespace，并通过同一个 named Controller 获得 backend 配置。第一版 partition 固定，不按任务动态创建：

```yaml
transfer_queue:
  enable: true
  package_version: "0.1.7"
  ray:
    address: "ray-head-node:6379"
    namespace: "speco-drafter"
  partition_id: "speco_drafter_features"
  run_id: "dspark-20260819-a"
  schema_version: 1
  backend:
    storage_backend: MooncakeStore
    MooncakeStore:
      auto_init: false
      metadata_server: "node0:50050"
      master_server_address: "node0:50051"
      local_hostname: ""
      protocol: tcp
      global_segment_size: 4294967296
      local_buffer_size: 1073741824
      device_name: ""
```

`partition_id/run_id/schema_version` 用于过滤和校验样本；它们不能帮助进程发现 TQ。真正让三个任务连接到同一 TQ 的是“连接同一个 Ray 集群和 namespace，然后找到同名 Controller”。

依赖也必须锁定并单独验证。实机安装 `TransferQueue==0.1.7` 会安装 Ray，并要求 `numpy<2.0.0`；当前测试环境中的 `twinkle-kit` 要求 `numpy>=2.0.0`，两者冲突。开发时应使用专门的 TQ/训练环境或重新确认整套依赖约束，不能直接把 0.1.7 安装进已有生产环境后忽略 resolver warning。

### 3.4 `tq_owner.py` 要实现的入口和函数

新增：

```text
verl_speco/tq_owner.py
tools/run_dspark_tq_owner.sh
```

`tq_owner.py` 建议明确实现：

```python
def install_signal_handlers(stop_event: threading.Event) -> None: ...
def publish_owner_ready(run_id: str, schema_version: int) -> None: ...
def wait_until_stopped(stop_event: threading.Event) -> None: ...
def run_owner(config: DictConfig) -> int: ...
def main() -> None: ...
```

`run_owner()` 的执行顺序必须是：

```text
configure_transfer_queue(config)
→ connect_ray_cluster(ray.address, ray.namespace)
→ start_transfer_queue_owner(full TQ native config)
→ put owner_ready 控制 record
→ 安装 SIGINT/SIGTERM handler
→ 保持 owner 进程存活
→ 收到停止信号
→ close_transfer_queue_owner()
```

Owner 必须常驻。TQ 0.1.7 创建 Controller 时没有设置 `lifetime="detached"`，不能在初始化后立即退出。

### 3.5 启动和关闭顺序

第一版由外部脚本管理全生命周期：

```text
1. ray start --head，记录 Ray address
2. 启动 Mooncake metadata/master（若 auto_init=false）
3. 启动 TQ owner；owner 连接 Ray并调用 tq.init(full config)
4. 等待 owner_ready
5. 启动一个或多个 vLLM servers
6. 启动 Consumer；每个 torchrun rank 连接 Ray，然后 tq.init(same native config)
7. 启动 Producer；连接 Ray，然后 tq.init(same native config)
8. Producer 写 EOS，关闭本地 client并退出
9. Consumer drain、保存 final checkpoint，所有 ranks 关闭本地 client并退出
10. 给 TQ owner 发送 SIGTERM；仅 owner 执行 tq.close()
11. 等 owner 退出后执行 ray stop
12. 停止 Mooncake 服务
```

外部脚本要用 `trap` 保证异常退出也按“Producer/Consumer → owner → Ray → Mooncake”的顺序清理。不能在 Consumer rank 的 `finally` 中调用全局 `tq.close()`。

## 4. Producer 要实现什么

### 4.1 Producer 完整顺序

```text
读取共享配置
→ 连接 TQ并校验 owner_ready
→ 初始化 tokenizer
→ 初始化多个 vLLM endpoint clients
→ 流式读取输入文件
→ 为每条输入分配 sequence_no/sample_id
→ 缺少 response 时由 target vLLM 生成；构造 input_ids/loss_mask
→ 并发请求 vLLM prefill
→ 读取 vLLM hidden-state 临时结果
→ 转换成 DSpark DraftFeatureSample
→ 构造 SampleMetadata
→ encode_sample 得到 fields/tag/key
→ TQ kv_put 一条 sample
→ 删除该请求临时文件
→ 所有输入完成后写 EOS
→ close_transfer_queue_client()并退出
```

### 4.2 并发模型

Producer 是一个进程，内部并发请求多个 endpoint：

```text
InputReader
→ bounded asyncio input_queue
→ N 个 RequestWorker
→ bounded publish_queue
→ TQ Publisher
```

- `vllm_endpoints` 是列表；
- 每个 endpoint 有独立 semaphore；
- 总并发由 `max_inflight_requests` 限制；
- input/publish queue 必须有上限；
- TQ `kv_put` 如果是同步 API，用 `asyncio.to_thread()` 调用；
- TQ ready 数达到 `max_pending_samples` 时暂停继续请求，形成简单背压。

`sequence_no` 在 InputReader 中分配，不按 vLLM 完成顺序分配。因此并发乱序不会改变 sample key。

### 4.3 vLLM 结果转换

复用现有 `TargetFeatureReplayer` 的：

- OpenAI-compatible vLLM 请求；
- `prompt_token_ids` 校验；
- `kv_transfer_params.hidden_states_path`；
- safetensors 加载；
- `[seq,layers,hidden]` 校验；
- feature positions 选择；
- aux layers flatten；
- DSpark L1 时拼 final hidden。

不要复制 `_feature_from_vllm_payload()`；把纯转换逻辑提取成公共函数，让旧 file backend 和新 Producer 共同调用。

临时文件顺序：

```text
加载
→ 校验/转换
→ TQ put 成功
→ 删除
```

第一版仍允许每个并发请求产生短期临时文件，但不生成全量 hidden-state 数据集。

### 4.4 Producer 文件分工

| 文件 | 实现内容 |
|---|---|
| `verl_speco/standalone_tq_producer.py` | CLI/Hydra 入口，连接 TQ，启动 asyncio pipeline，写 EOS和汇总指标 |
| `verl_speco/producer/input_reader.py` | 流式读输入、分配 `sequence_no/sample_id`、tokenize、构造 loss mask |
| `verl_speco/producer/vllm_feature_client.py` | endpoint pool、并发控制、HTTP 请求、临时结果读取和删除 |
| `verl_speco/trainer/target_feature_replay.py` | 提取可复用的 vLLM payload → `DraftFeatureSample` 纯转换函数 |
| `examples/run_dspark_tq_producer.sh` | Producer 配置和启动命令 |
| `tests/unit/test_drafter_sample_protocol.py` | 协议编码、字段和 shape 测试（两人共同） |
| `tests/integration/test_tq_producer_smoke.py` | fake/短 vLLM → TQ sample + EOS |

Producer 开发者同时负责移植/扩展 `integration/transferqueue_bridge.py` 和实现 `tq_owner.py`，因为这部分与 TQ 写入和连接直接相关。

### 4.5 Producer 各文件的函数级实现规格

#### `verl_speco/standalone_tq_producer.py`

需要实现：

```python
@dataclass
class ProducerStats:
    input_count: int
    published_count: int
    failed_count: int
    pending_bytes: int

async def publish_one(result: PreparedFeature, transport) -> str: ...
async def run_producer(config: DictConfig) -> ProducerStats: ...
def validate_producer_config(config: DictConfig) -> None: ...
def main() -> None: ...
```

`main()` 只负责 Hydra/日志/退出码。`run_producer()` 是可测试的业务入口，执行：连接 Ray → 连接 TQ client → 创建 InputReader 和 vLLM client pool → 启动有界 asyncio pipeline → 等待所有请求及 `kv_put` 完成 → 发布 EOS → 关闭本地 client。它不能启动 Ray head、不能创建 TQ owner、不能调用全局 `tq.close()`。

`publish_one()` 接收已经完成转换的一条 `PreparedFeature`，调用共享协议的 `encode_sample()` 得到 `(key, fields, tag)`，再通过 bridge 的 `put_sample()` 发布。只有 `put_sample()` 成功返回，`published_count` 才增加，vLLM 临时文件才允许删除。

#### `verl_speco/producer/input_reader.py`

需要实现：

```python
@dataclass(frozen=True)
class InputRecord:
    sequence_no: int
    sample_id: str
    prompt: str
    response: str | None
    source_metadata: dict[str, Any]

def iter_input_records(path: str) -> Iterator[InputRecord]: ...
def tokenize_record(record: InputRecord, tokenizer, config) -> TokenizedRequest: ...
def build_loss_mask(input_ids: Tensor, prompt_length: int) -> Tensor: ...
```

`iter_input_records()` 流式读取 JSONL/Parquet，不把全文件载入内存，并按文件顺序分配稳定的 `sequence_no`。已有 response 时 `tokenize_record()` 直接拼接；prompt-only verl 数据通过 chat template 编码后由 target vLLM 生成 response，并设置 `include_output_tokens=true` 同步提取输出 hidden states。

#### `verl_speco/producer/vllm_feature_client.py`

需要实现：

```python
@dataclass(frozen=True)
class VllmEndpoint:
    base_url: str
    max_concurrency: int

class VllmFeatureClientPool:
    async def start(self) -> None: ...
    async def prefill(self, request: TokenizedRequest) -> RawVllmFeature: ...
    async def close(self) -> None: ...

async def request_prefill(endpoint, request) -> VllmResponse: ...
def choose_endpoint(endpoints, state) -> VllmEndpoint: ...
def load_hidden_state_result(response) -> RawVllmFeature: ...
def delete_temporary_result(raw: RawVllmFeature) -> None: ...
```

`prefill()` 必须允许多个 coroutine 同时运行；全局 semaphore 限制总并发，每个 endpoint 另有独立 semaphore。`request_prefill()` 只负责 HTTP 请求和响应校验；`load_hidden_state_result()` 负责读取 vLLM 返回的临时 safetensors/path。临时结果的删除不放在 `load_hidden_state_result()`，而由 `publish_one()` 成功后触发。

#### `verl_speco/trainer/target_feature_replay.py`

把当前类内部的纯转换部分抽成：

```python
def feature_from_vllm_payload(
    payload: RawVllmFeature,
    request: TokenizedRequest,
    feature_config: FeatureContract,
) -> DraftFeatureSample: ...
```

它不进行 HTTP、不访问 TQ、不删除文件，只完成 shape/layout/layer 校验、feature-position 选择、aux layer flatten 和可选 L1 final hidden 拼接。现有 file replay backend 和新 Producer 都调用这一函数，避免两套转换规则。

#### `examples/run_dspark_tq_producer.sh`

负责提供同一套：

```text
RAY_ADDRESS / Ray namespace
run_id / schema_version / 固定 partition
Mooncake/TQ backend 配置
输入文件和 tokenizer/model 配置
vLLM endpoint 列表
max_inflight_requests / per_endpoint_concurrency
```

脚本只启动 Producer，不启动 TQ owner 或 Consumer，便于两位开发者独立调试。

## 5. Consumer 要实现什么

### 5.1 不新写另一套训练器

继续使用现有入口：

```text
draft_train_launcher.py
→ draft_train.py
→ trainer/draft_training_loop.py
→ DrafterBaseTrainer
→ DSparkTrainerBackend
```

训练模式继续使用现有 standalone `training.mode=offline`，用 `feature_store.type=tq` 选择 TQ 数据源。不复制 FSDP、loss、optimizer 或 checkpoint 代码。

当前 `feature_store.type` 的作用是告诉 `build_feature_store_from_config()` 应创建哪一种训练数据来源：

| 当前 type | 对象 | 数据来源 |
|---|---|---|
| `torch_shard` | `TorchShardFeatureStore` | 本地 `.pt` shard，当前 separate-training 示例使用它 |
| `token_replay` | `TokenReplayFeatureStore` | 保存 token replay 的 shard |
| `vllm_safetensors` / `safetensors` | `VllmSafetensorsFeatureStore` | 已生成的 safetensors feature |
| `jsonl_token_replay` / `jsonl` | `JsonlTokenReplayFeatureStore` | JSONL token/text replay |

第一版新增：

```yaml
actor_rollout_ref:
  rollout:
    drafter:
      training:
        mode: offline
        feature_store:
          type: tq
          path: null
          shuffle: false
          repeat: false
```

这里 `offline` 的含义是“独立于 RL 的 standalone drafter training”，不等于数据必须来自磁盘。

不能只给现有工厂增加一个 `type=tq` 然后继续使用通用 `DraftFeatureDataLoader`。当前 loader 会：

```python
keys = list(store.iter_keys(...))
```

它假设数据集 keys 是一个静态快照；空 store 会直接结束，并且各 rank 独立枚举时可能在 Producer 持续写入的过程中看到不同快照。TQ 是流式、会新增并删除 keys 的数据源，所以 `type=tq` 必须选择专用的 `TQFeatureDataLoader`，由 rank 0 统一发现和分配 keys。

#### 5.1.1 当前磁盘 feature store 为什么每个 rank 都会取 keys

`draft_train_launcher.py` 通过 `torchrun` 启动多个训练进程。每个进程都是一个 rank，并且每个 rank 都会独立进入 `run_standalone_draft_training()`、创建 `DraftFeatureDataLoader`、执行它的 `__iter__()`。当前 loader 的核心逻辑是：

```python
keys = list(
    self.store.iter_keys(
        shuffle=self.shuffle,
        seed=self.seed + epoch,
    )
)
rank_keys = keys[rank::world_size]

for key in rank_keys:
    batch.append(self.store.read(key))
```

因此当前并不是 rank 0 枚举 keys 后再发送给其他 rank，而是所有 rank 都访问相同的静态 feature-store 路径：

```text
rank 0：iter_keys() 得到完整静态列表 → keys[0::world_size] → read 自己的 samples
rank 1：iter_keys() 得到完整静态列表 → keys[1::world_size] → read 自己的 samples
...
```

例如 store 中固定存在：

```python
keys = ["k0", "k1", "k2", "k3", "k4", "k5", "k6", "k7"]
```

当 `world_size=2` 时，两个 rank 都先得到上述完整列表，然后分别计算：

```python
# rank 0
rank_keys = keys[0::2]  # ["k0", "k2", "k4", "k6"]

# rank 1
rank_keys = keys[1::2]  # ["k1", "k3", "k5", "k7"]
```

这种实现能够成立，是因为 `.pt`、safetensors 或 replay 文件在训练期间是静态数据集。只要所有 rank 使用同一路径和相同 shuffle seed，`iter_keys()` 就会产生相同顺序的 key 快照，各 rank 可以无通信地算出互不重叠的子集。

#### 5.1.2 TQ 为什么必须改成 rank 0 发现 keys

TQ 中的 keys 会在训练期间动态变化：Producer 持续 `kv_put`，训练成功后 Consumer 又执行 `kv_clear`。如果所有 rank 仍然各自调用 `kv_list`，不同调用时刻可能看到不同快照：

```python
# rank 0 较早调用
rank0_keys = ["k0", "k1", "k2", "k3"]

# Producer 随后写入 k4、k5，rank 1 较晚调用
rank1_keys = ["k0", "k1", "k2", "k3", "k4", "k5"]
```

各 rank 再独立切片后，可能得到不同数量的 local samples，进而无法保证它们以相同顺序进入 forward、backward 和梯度 collective。为此，TQ 专用 loader 必须把控制面和数据面分开：

```text
控制面：rank 0 执行 kv_list，固定本 step 的 global_keys，并向各 rank 分发 local_keys
数据面：每个 rank 使用自己的 local_keys 直接执行 kv_batch_get，从 TQ/Mooncake 读取 hidden-state payload
```

rank 0 只发送较小的 key/tag 元数据，不读取并转发其他 rank 的 hidden-state tensor。所有 rank 仍然都会连接同一个 TQ，也都会调用批量读取接口；只有动态 key 的发现和本 step 的 global batch 决策集中在 rank 0。

因此 `feature_store.type=tq` 不是只替换底层 `read()`：它同时改变了 key 的发现、分配、等待和删除语义，需要专用 `TQFeatureDataLoader`，并要求训练循环在所有 rank 成功完成 optimizer step 后，由 rank 0 对本 step 的 `global_keys` 执行 `kv_clear`。

### 5.2 Consumer 完整顺序

```text
torchrun 启动多个 ranks
→ 每个 rank 初始化 torch.distributed
→ 每个 rank 连接同一个 TQ
→ rank 0 校验 owner_ready，并 broadcast 结果
→ 初始化现有 DSpark trainer
→ rank 0 kv_list 查找 ready sample keys
→ rank 0 选一个 global batch并分给各 rank
→ 每个 rank kv_batch_get 自己的 local keys
→ decode_sample 得到 list[DraftFeatureSample]
→ prepare_training_batch_from_samples()
→ training_step_from_batch()
→ 所有 rank 汇总 success
→ 成功后 rank 0 kv_clear 这个 global batch 的 keys
→ 继续下一批
→ 看到 EOS 且 ready 为空
→ 保存 final checkpoint
→ 所有 ranks close_transfer_queue_client()并退出
```

### 5.3 多 rank 如何分 key

例如：

```text
world_size=2
batch_size_per_gpu=2
global batch size=4
```

rank 0 选出：

```python
global_keys = ["k10", "k11", "k12", "k13"]
assignments = [
    ["k10", "k11"],  # rank 0
    ["k12", "k13"],  # rank 1
]
```

通过 `dist.scatter_object_list` 或 broadcast 发送短字符串列表。每个 rank 直接从 TQ 读取自己的 payload，不通过 rank 0 转发 hidden states。

### 5.4 从 TQ 到训练 batch

每个 rank：

```python
records = tq_transport.get_samples(local_keys)

samples = [
    decode_sample(
        key=key,
        tag=tags_by_key[key],
        fields=fields,
        expected_config=expected_contract,
    )
    for key, fields in records
]

batch = trainer.prepare_training_batch_from_samples(
    samples,
    step=optimizer_step,
)

ok = await trainer.training_step_from_batch(
    batch,
    optimizer_step,
)
```

`records` 是多个独立单样本 payload；`samples` 是变长 `DraftFeatureSample` 列表。Consumer transport 层不要直接 `torch.stack()`，现有 trainer/backend 负责对齐和组 batch。

### 5.5 删除与结束

第一版采用简单逻辑：

```text
所有 rank get/decode/train 都成功
→ all_reduce(global_success)=True
→ rank 0 kv_clear(global_batch_keys)
```

任何 rank 失败都不 clear。程序报错退出，第一版不自动恢复。

EOS 后不足一个 global batch 的尾部，第一版使用 `drop_last=True`：记录数量，rank 0 clear 这些尾部 keys，然后正常结束。

checkpoint 仍使用现有 `save_interval_steps` 和 final checkpoint；第一版不保证崩溃后已 clear 数据能够严格重放。

### 5.6 Consumer 文件分工

| 文件 | 实现内容 |
|---|---|
| `verl_speco/trainer/feature_store.py` | 工厂增加 `type=tq`，返回 `TQFeatureStore`；TQ 类型不要求 `path` |
| `verl_speco/trainer/tq_feature_store.py` | 实现固定 partition 上的 list/get-many/clear/EOS，不伪装静态 `iter_keys()` |
| `verl_speco/trainer/tq_sample_source.py` | 实现专用 `TQFeatureDataLoader`：rank 0 动态 list/filter/sort、key 分配、各 rank get/decode、EOS/tail |
| `verl_speco/trainer/draft_training_loop.py` | 保持 `mode=offline`；`type=tq` 时跳过 `feature_store.path` 必填检查并选择专用 loader；调用现有 prepare/train，global success 后 clear，final checkpoint |
| `verl_speco/draft_train_launcher.py` | `feature_store.type=tq` 时不要求 feature-store path，透传 torchrun/TQ 配置 |
| `verl_speco/config/speco_base.yaml` | 新增 TQ connection、run、poll、batch 等默认配置 |
| `tools/run_dspark_tq_consumer.sh` | Consumer GPU、batch、checkpoint 和共享 TQ 配置 |
| `tests/unit/test_tq_sample_source.py` | key 过滤/排序/分 rank、EOS、decode 调用测试 |
| `tests/integration/test_dspark_tq_consumer_smoke.py` | 两 rank 取不同样本，完成训练并 clear |

### 5.7 Consumer 各文件的函数级实现规格

#### `verl_speco/trainer/feature_store.py`

修改现有工厂：

```python
def build_feature_store_from_config(feature_store_cfg, read_only=False):
    store_type = str(feature_store_cfg.get("type", "torch_shard")).lower()
    if store_type == "tq":
        return TQFeatureStore.from_config(feature_store_cfg)
    ...
```

要求：

- 保留 `torch_shard/token_replay/vllm_safetensors/jsonl_token_replay` 的当前行为；
- `type=tq` 时不读取 `path`；
- 工厂只创建对象，不在 import 阶段连接 Ray/TQ；
- `TQFeatureStore` 是流式数据源适配器，不强行实现有误导性的静态 `iter_keys()` 和逐条 `read()`。

#### `verl_speco/trainer/tq_feature_store.py`

需要实现：

```python
@dataclass(frozen=True)
class ReadyEntry:
    key: str
    tag: dict[str, Any]

class TQFeatureStore:
    @classmethod
    def from_config(cls, cfg) -> "TQFeatureStore": ...
    def connect(self) -> None: ...
    def list_ready(self, run_id: str) -> list[ReadyEntry]: ...
    def get_many(self, entries: list[ReadyEntry]) -> list[DraftFeatureSample]: ...
    def clear_many(self, keys: list[str]) -> None: ...
    def read_eos(self, run_id: str) -> EosMetadata | None: ...
    def close_local(self) -> None: ...
```

`connect()` 调用共享 bridge 的 `connect_ray_cluster()` 和 `connect_transfer_queue_client()`。`list_ready()` 调用 `list_samples()` 后只保留 `tag.status=ready`、匹配 `run_id/schema_version` 的数据 key，并按 `(sequence_no,key)` 排序。`get_many()` 一次批量读取 fields，然后逐条调用共享 `decode_sample()`；返回顺序必须和 entries 相同。`clear_many()` 只允许 rank 0 在全局 step 成功后调用。`close_local()` 只关闭本 rank client，不能关闭 Controller。

#### `verl_speco/trainer/tq_sample_source.py`

需要实现：

```python
@dataclass
class TQLocalBatch:
    local_keys: list[str]
    local_samples: list[DraftFeatureSample]
    global_keys: list[str] | None

class TQFeatureDataLoader:
    def __iter__(self) -> Iterator[TQLocalBatch]: ...
    def _select_global_batch(self) -> tuple[list[str], dict[str, ReadyEntry]]: ...
    def _broadcast_assignments(self, assignments) -> list[ReadyEntry]: ...
    def _handle_eos_and_tail(self) -> bool: ...
    def clear_completed_batch(self, global_keys: list[str] | None) -> None: ...
```

执行责任必须明确：

- 所有 rank 创建 loader 并调用 `store.connect()`；
- 只有 rank 0 执行 `_select_global_batch()` 和 `kv_list`；
- rank 0 生成 `list[list[ReadyEntry]]` assignments，使用 `torch.distributed.scatter_object_list` 或 broadcast 分发小型 key/tag；
- 每个 rank 对自己的 local entries 调用 `store.get_many()`，hidden states 从 TQ/Mooncake 直接进入该 rank；
- loader yield `TQLocalBatch`，不能只 yield samples，因为训练后清理还需要 `global_keys`；
- 暂时为空时轮询等待，不能像静态 loader 一样结束；只有 EOS 已出现并且 ready 数据 drain 完才停止；
- `drop_last=True` 时由 rank 0 记录并清理不足 global batch 的尾部 key。

#### `verl_speco/trainer/draft_training_loop.py`

需要新增或调整：

```python
def build_training_source(config, rank, world_size): ...
def all_ranks_succeeded(local_ok: bool, device) -> bool: ...
async def run_tq_training_loop(trainer, loader, config) -> dict[str, Any]: ...
```

`build_training_source()` 根据 `feature_store.type` 分支：静态类型继续创建 `DraftFeatureDataLoader`；`tq` 创建 `TQFeatureDataLoader`，跳过 `feature_store.path` 必填检查，并禁止 `shuffle/repeat`。`run_tq_training_loop()` 对每个 `TQLocalBatch` 调用现有 `prepare_training_batch_from_samples()` 和 `training_step_from_batch()`；所有 rank 通过 collective 得到 global success 后，才让 rank 0 调用 `clear_completed_batch(global_keys)`。异常路径不 clear，finally 只执行 `store.close_local()`。

#### `verl_speco/draft_train_launcher.py`

保留当前 `torch.distributed.run` 启动方式，只增加配置校验和环境透传：

```python
def validate_tq_launch_config(overrides, launch_config) -> None: ...
def build_child_env(config) -> dict[str, str]: ...
```

它不启动 Ray head、不调用 `tq.init()`。职责是确认 `type=tq` 时提供了 Ray address/namespace，且不要求 feature-store path；随后把同一 Ray connection 配置传给每个 torchrun 子进程。每个子 rank 自己建立 Ray/TQ client，不能在 launcher 父进程建一个 client后期待 fork 继承。

#### `verl_speco/config/speco_base.yaml`

增加默认字段：

```yaml
feature_store:
  type: torch_shard
  path: null
  shuffle: true
  repeat: true
  tq:
    ray_address: null
    ray_namespace: speco-drafter
    partition_id: speco_drafter_features
    run_id: null
    schema_version: 1
    poll_interval_seconds: 0.5
    connect_timeout_seconds: 120
    drop_last: true
```

当 `type=tq` 时运行期覆盖为 `shuffle=false/repeat=false`。backend 的完整 owner 配置只需要 TQ owner 使用；Producer/Consumer client 从 named Controller 获取它，不各自重新创建 backend。

#### Consumer 测试必须覆盖的函数边界

- `test_feature_store_factory_builds_tq_without_path()`；
- `test_rank0_filters_and_sorts_ready_entries()`；
- `test_nonzero_rank_never_calls_kv_list()`；
- `test_assignments_are_disjoint_and_global_batch_complete()`；
- `test_each_rank_gets_only_local_keys()`；
- `test_decode_preserves_hidden_states_layout()`；
- `test_clear_only_after_all_ranks_success()`；
- `test_failure_does_not_clear()`；
- `test_eos_drains_ready_then_stops()`；
- `test_client_close_does_not_kill_owner()`。

## 6. 两个人怎么分工

### 共同先完成

1. `drafter_sample_protocol.py`；
2. Ray/TQ connection 配置字段；
3. 一个小型 golden sample；
4. 启动一个测试 Ray head 和 TQ owner，独立进程 A put、独立进程 B list/get/clear 的 smoke test；
5. 验证 client 退出不会 kill owner，只有 owner shutdown 才销毁 Controller。

### 开发者 A：Producer/TQ

负责：

```text
integration/transferqueue_bridge.py
tq_owner.py
standalone_tq_producer.py
producer/input_reader.py
producer/vllm_feature_client.py
target_feature_replay.py 的公共转换函数
owner/producer 启动脚本
Producer/TQ 测试
```

开发者 A 的可交付接口不是“提供一个 TQ 类”，而是：

```text
bridge：connect_ray_cluster/start_owner/connect_client/put/list/get_many/clear/client_close/owner_close
owner：run_owner/main/signal handler/owner_ready
producer：run_producer/publish_one/统计与 EOS
input reader：iter_input_records/tokenize_record/build_loss_mask
vLLM client：endpoint pool/request_prefill/load/delete
feature conversion：feature_from_vllm_payload
```

开发者 B 可以先针对这些接口写 fake transport，不需要等待真实 vLLM 和 Mooncake 联通。

### 开发者 B：Consumer/训练

负责：

```text
feature_store.py 的 type=tq 工厂分支
tq_feature_store.py
tq_sample_source.py / TQFeatureDataLoader
draft_training_loop.py 的 offline + type=tq 分支
draft_train_launcher.py 配置适配
speco_base.yaml Consumer 配置
Consumer 启动脚本
Consumer/DSpark 测试
```

开发者 B 的可交付接口是：

```text
feature-store factory：type=tq 分支
TQFeatureStore：connect/list_ready/get_many/clear_many/read_eos/close_local
TQFeatureDataLoader：select global batch/distribute local keys/get/yield/EOS-tail
training loop：build source/train/global success/clear/final checkpoint
launcher：TQ 配置校验和 torchrun 子进程环境透传
```

### 联调入口

建议再提供：

```text
examples/run_dspark_tq_pipeline_local.sh
```

只用于单机联调，顺序启动：

```text
ray start --head
→ Mooncake metadata/master
→ TQ owner（ray.init + tq.init(full config)）
→ owner_ready
→ vLLM health check
→ Consumer
→ Producer
→ 等 Producer/Consumer 退出
→ SIGTERM TQ owner（owner 执行 tq.close）
→ ray stop
→ 停止 Mooncake
```

最小联调：Producer 发布 8 条样本，2 个 Consumer ranks、每 rank batch size 2，完成 2 个 optimizer steps，8 个 sample keys 被清理，EOS 后保存 final checkpoint。

## 7. 第一版验收标准

1. TQ owner、Producer 和所有 Consumer ranks 日志显示相同 Ray address/namespace、TQ Controller、partition 和 run ID。
2. 在同一 Ray 集群中，独立 owner 创建 TQ 后，独立进程 A put，独立进程 B 能 list/get/clear。
3. Producer 对多个 vLLM endpoints 并发请求，不串行访问。
4. 一个输入 record 只生成一个 sample key 和一个 payload。
5. `kv_list` 只拿 tag；`kv_batch_get` 才拿 hidden states。
6. Consumer 各 rank 读取不同 local keys，不通过 rank 0 搬运 Tensor。
7. `decode_sample` 能拒绝模型、layer、layout、dtype 或 shape 不匹配的数据。
8. 所有 rank 训练成功后才 clear 当前 global batch。
9. Producer 先完成时，Consumer 能 drain 后再退出。
10. 不产生长期 hidden-state feature store。
11. Producer 或任一 Consumer rank 退出不会销毁 TQ Controller；只有 owner 调用全局 `tq.close()`。
12. Ray object store 中不承载 hidden-state payload，训练 tensor 通过 TQ/Mooncake 路径读取。

## 8. 后续建议：第一版跑通后再做

以下内容不进入第一版开发：

- Producer HTTP/TQ 复杂重试和 endpoint 熔断；
- Producer 发布 journal，避免重启后重复生成已 clear 样本；
- 自动重启；第一版失败后人工停止整条 pipeline并使用新 `run_id` 重跑；
- Consumer 从最新 checkpoint 自动恢复；
- checkpoint 成功后再 clear 的严格提交窗口；
- TQ owner/storage 整体丢失后的数据重建；
- 多个独立 Consumer 竞争同一 partition；
- lease、ack、超时回收和 exactly-once；
- 动态扩缩容；
- vLLM server 直接写 TQ。

第一版先保证：同一个 TQ 能连通、Producer 能并发生产、Consumer 能正确取数训练、每步成功后能及时清理。
