# TQ `DraftFeatureSample` 通用传输协议重构方案

> Last updated: 08/27/2026

## 1. 目标与结论

当前 standalone TQ 流程已经在 Consumer 侧恢复为 `DraftFeatureSample`，然后调用既有的
`DrafterTrainer.prepare_training_batch_from_samples()`。但是传输层仍然通过一套较重的
`SampleMetadata` 重新描述 hidden-state 布局、shape 和训练字段，导致：

- `DraftFeatureSample.metadata` 不能完整往返；
- TQ codec 了解过多 DSpark/hidden-state 布局细节；
- 新算法即使已经能构造 `DraftFeatureSample`，仍可能需要修改 TQ 协议；
- Producer 和 Consumer 分别实现 ready tag 过滤，容易产生统计口径不一致。

本次重构采用以下边界：

1. 保留 `SampleMetadata`，但将它缩减为 **TQ 控制信封**；
2. 完整训练数据只由 `DraftFeatureSample` 表达；
3. TQ codec 对 `DraftFeatureSample` 做通用、无损、算法无关的编码和解码；
4. Consumer 解码后直接把 `DraftFeatureSample` 交给现有训练流程；
5. 算法差异只保留在 Producer 的样本构造和 Trainer backend 中；
6. Producer 和 Consumer 复用同一个 ready-tag 解析函数。
7. Producer 写入资格与 Consumer 读取资格使用同一个共享判定，禁止两端分别实现近似校验；
8. hidden-state token/position/row 对齐失败的样本在 Producer 侧直接丢弃，不得部分截取后写入 TQ。

TQ 不能直接存放 Python dataclass 实例。TQ 0.1.7 的数据面接收 tensor fields，因此仍然
需要 `encode_sample()` / `decode_sample()`；这里要删除的是自定义的训练数据定义，而不是
必要的传输编码。

## 2. 重构后的职责划分

### 2.1 `SampleMetadata`：只负责队列控制

建议保留以下字段：

```python
@dataclass(frozen=True)
class SampleMetadata:
    protocol_schema_version: int
    run_id: str
    sample_id: str
    sequence_no: int
```

字段含义：

| 字段 | 用途 |
| --- | --- |
| `protocol_schema_version` | TQ key/tag/fields 编码格式的版本，不是模型算法版本 |
| `run_id` | 隔离不同 standalone 训练任务 |
| `sample_id` | 保留输入样本身份，便于定位错误 |
| `sequence_no` | 为并发完成的样本建立确定顺序，并生成唯一 key |

从 `SampleMetadata` 删除以下字段：

```text
algorithm
target_model_id
target_model_revision
tokenizer_fingerprint
target_layer_ids
hidden_states_layout
hidden_dtype
hidden_shape
feature_length
full_sequence_length
feature_start
feature_end
use_logits
```

这些字段如果训练需要，应保存在 `DraftFeatureSample.algorithm` 或
`DraftFeatureSample.metadata` 中。TQ 控制层不再验证其算法语义。

### 2.2 TQ tag：控制面索引

tag 保留可在不加载 tensor payload 的情况下完成发现、排序和 run 隔离所需的信息：

```python
tag = {
    "record_type": "sample",
    "status": "ready",
    "protocol_schema_version": 2,
    "run_id": "dspark-a1b2c3",
    "sequence_no": 6500,
    "sample_id": "train-006500",
}
```

tag 不再携带 `algorithm`、layer IDs、hidden shape 等训练信息。EOS 仍使用独立 control tag：

```python
tag = {
    "record_type": "control",
    "status": "eos",
    "protocol_schema_version": 2,
    "run_id": "dspark-a1b2c3",
    "total_samples": 15000,
}
```

### 2.3 TQ fields：完整 `DraftFeatureSample`

fields 使用原生 tensor 字段加一个 JSON manifest：

```python
fields = {
    "sample__input_ids": Tensor,
    "sample__loss_mask": Tensor,
    "sample__hidden_states": Tensor,
    "sample__position_ids": Tensor,          # 可选
    "sample__last_hidden_states": Tensor,    # 可选
    "sample__target": Tensor,                # 可选
    "sample__target_logprobs": Tensor,       # 可选
    "sample__metadata_tensor__000000": Tensor,
    "sample__manifest_json": UInt8Tensor,
}
```

`sample__manifest_json` 的逻辑内容示例：

```json
{
  "draft_feature_schema_version": 1,
  "algorithm": "DSPARK",
  "present_fields": [
    "input_ids",
    "loss_mask",
    "hidden_states",
    "position_ids"
  ],
  "hidden_states_kind": "tensor",
  "metadata": {
    "hidden_states_layout": "dflash_aux_plus_last",
    "target_layer_ids": [1, 12, 23, 34, 45],
    "feature_start": 31,
    "feature_end": 543,
    "hidden_positions": {
      "__tq_tensor_ref__": "sample__metadata_tensor__000000"
    }
  }
}
```

manifest 和 tensor fields 合起来必须能够完整恢复：

```python
DraftFeatureSample.from_dict(payload, strict=True)
```

## 3. 通用 metadata codec

`DraftFeatureSample.metadata` 不能简单 `json.dumps()`，因为当前代码会在其中保存
`hidden_positions` 等 tensor。新 codec 采用递归 tree 编码。

直接写入 JSON 的类型：

```text
None、bool、int、float、str
dict[str, value]
list[value]
tuple[value]（manifest 记录 tuple 类型，解码后恢复 tuple）
```

tensor 的处理方式：

```text
metadata中的Tensor
→ 转为CPU contiguous Tensor
→ 单独写入fields
→ manifest原位置写tensor field引用
```

例如：

```python
metadata = {
    "feature_start": 31,
    "hidden_positions": torch.tensor([31, 32, 33]),
}
```

编码为：

```python
fields["sample__metadata_tensor__000000"] = tensor([31, 32, 33])

manifest["metadata"] = {
    "feature_start": 31,
    "hidden_positions": {
        "__tq_tensor_ref__": "sample__metadata_tensor__000000"
    },
}
```

不支持的对象不能静默执行 `str(value)`，否则协议不是无损的。第一版应 fail closed，错误中打印
metadata 路径和实际类型。后续如果确实存在 NumPy scalar/array，可显式增加稳定编码规则。

## 4. `hidden_states` 两种表示

`DraftFeatureSample.hidden_states` 支持：

```python
torch.Tensor | list[torch.Tensor]
```

单 tensor：

```python
fields["sample__hidden_states"] = hidden
manifest["hidden_states_kind"] = "tensor"
```

tensor list：

```python
fields["sample__hidden_states__000000"] = hidden_0
fields["sample__hidden_states__000001"] = hidden_1
manifest["hidden_states_kind"] = "list"
manifest["hidden_states_fields"] = [
    "sample__hidden_states__000000",
    "sample__hidden_states__000001",
]
```

这样协议不会再因某个算法使用 tensor list 而报错。

## 5. 需要修改的文件和函数

### 5.1 `verl_speco/transport/drafter_sample_protocol.py`

这是主要重构文件。

修改内容：

1. 将 `SampleMetadata` 缩减为控制信封；
2. 将 `PROTOCOL_SCHEMA_VERSION` 从 1 升到 2；
3. 修改 `make_sample_key()`，继续使用 protocol version、run、sequence 和 sample ID；
4. 修改 `make_ready_tag()`，只生成控制面字段；
5. 重写 `encode_sample(sample, meta)`：
   - 调用 `DraftFeatureSample.to_dict()`；
   - 编码所有 dataclass tensor 字段；
   - 编码 hidden-state tensor list；
   - 递归编码 metadata；
   - 生成 manifest；
6. 重写 `decode_sample(key, tag, fields, expected_config)`：
   - 解析并校验控制信封；
   - 解析 manifest；
   - 恢复所有 tensor 和 metadata；
   - 调用 `DraftFeatureSample.from_dict(..., strict=True)`；
7. 将 `_validate_primary_tensors()` 中与具体 hidden layout/shape 的约束删除；
8. 新增并导出统一函数：

```python
parse_ready_tag(tag) -> SampleMetadata | None
is_ready_sample_tag(tag, *, run_id, protocol_schema_version) -> bool
```

Producer backpressure 和 Consumer discovery 必须复用这两个函数，禁止再分别复制过滤条件。

此外增加统一的发布资格函数：

```python
validate_publishable_sample(sample) -> None
```

`encode_sample()` 和 Consumer 的 `decode_sample()` 都调用同一组 sample 结构校验。Producer 只有
通过该校验后才能生成 ready tag；这样不存在“Producer 写入成功，但 Consumer 按另一套规则过滤”的
中间状态。协议错误必须在 `put_sample()` 前暴露。

建议新增内部函数：

```python
_encode_metadata_tree(value, fields, path) -> JSONValue
_decode_metadata_tree(value, fields, path) -> Any
_encode_hidden_states(value, fields, manifest) -> None
_decode_hidden_states(fields, manifest) -> Tensor | list[Tensor]
_json_to_uint8_tensor(value) -> Tensor
_uint8_tensor_to_json(value) -> Any
```

### 5.2 `verl_speco/standalone_tq_producer.py`

修改内容：

1. 保留 `PreparedFeature.metadata: SampleMetadata`，但它现在只是 TQ 信封；
2. 简化 `_sample_metadata()`，只读取：

```text
run_id
request.sample_id
request.sequence_no
protocol_schema_version
```

3. 删除 `_sample_metadata()` 对 feature shape、layout、target model 和 logits 的复制；
4. `publish_one()` 仍保持：

```python
fields = encode_sample(result.sample, result.metadata)
tag = make_ready_tag(result.metadata)
transport.put_sample(key, fields, tag=tag)
```

5. `_wait_for_pending_capacity()` 使用协议模块的
   `is_ready_sample_tag()`，与 Consumer 使用完全相同的过滤规则；
6. 保持“只有 TQ put 成功后才删除 vLLM 临时文件”的生命周期不变。

Producer 还必须对 hidden-state 对齐失败做样本级丢弃：

```text
token_ids 与请求 token IDs 不一致
feature positions 超出 hidden-state rows
hidden-state rows 不能覆盖完整训练窗口
hidden-state layer 数不足
→ 记录 sample_id/sequence_no/原因
→ 删除本次 vLLM 临时文件和 lock
→ dropped_count += 1
→ 不进入 publish_queue
→ 不写 ready tag/fields
→ request worker 继续处理下一条样本
```

不能沿用当前“只丢弃越界 positions、使用剩余 positions 继续训练”的行为。TQ Producer 应启用严格
对齐模式：只要一个目标位置无法与 hidden-state row 对应，整条样本就无效。普通网络错误、TQ put
错误和协议编程错误仍然 fail fast，不能被误当作脏样本吞掉。

Producer 的算法相关职责仍然保留在：

```python
feature_from_vllm_payload(raw, request, feature_contract)
```

也就是说，Producer 必须先构造正确且完整的 `DraftFeatureSample`，TQ codec 不负责推断算法布局。

### 5.3 `verl_speco/trainer/tq_feature_store.py`

修改内容：

1. `list_ready()` 使用统一 `parse_ready_tag()`；
2. 删除本文件中重复的 tag 字段校验；
3. `get_many()` 继续调用 `decode_sample()`，返回类型仍为
   `list[DraftFeatureSample]`；
4. `ExpectedFeatureConfig` 只检查：

```text
run_id
protocol_schema_version
```

5. 不再在 TQ store 中检查 algorithm、target model、layer IDs、dtype 和 layout；
6. EOS 解析使用同一个 protocol version 字段命名。

### 5.4 `verl_speco/trainer/tq_sample_source.py`

主流程不需要改变：

```text
rank 0 list_ready
→ 按sequence_no排序
→ 为各rank分配key
→ 各rank get_many
→ 得到DraftFeatureSample
```

只需确保诊断日志中的 ready 统计也调用统一 tag parser，避免日志口径和正式读取口径不同。

### 5.5 `verl_speco/trainer/feature_store.py`

第一版不修改 `DraftFeatureSample` 公共字段，避免影响现有离线 store、PR #48 和非 TQ 路径。

可以新增一个小的公共字段列表，供 store 和 TQ codec 复用，例如：

```python
DRAFT_FEATURE_OPTIONAL_TENSOR_FIELDS = (
    "last_hidden_states",
    "target",
    "target_logprobs",
    "position_ids",
)
```

不要让 TQ codec 再维护一份不同的 optional-field 列表。

### 5.6 `verl_speco/trainer/target_feature_replay.py`

不修改训练转换逻辑。`feature_from_vllm_payload()` 继续负责把不同算法的 vLLM 输出转换为
`DraftFeatureSample`。

当前它支持：

```text
EAGLE3、DFLASH、DSPARK
```

以后增加新算法时，只需要在这里或对应算法 converter 中实现：

```text
RawVllmFeature + TokenizedRequest → DraftFeatureSample
```

如果新的 `DraftFeatureSample` 字段都能被通用 codec 表达，则无需再次修改 TQ 传输层。

### 5.7 `verl_speco/trainer/base_trainer.py`

不需要修改。Consumer 解码结果继续走：

```python
trainer.prepare_training_batch_from_samples(samples, step=optimizer_step)
```

其中每个元素已经是完整 `DraftFeatureSample`，随后调用现有：

```python
sample.to_training_item()
```

算法 backend 仍由启动配置的 `speculative_algorithm` 选择。

## 6. 重构后的端到端流程

### Producer

1. 读取 prompt/response；
2. 请求 vLLM generate/prefill；
3. `feature_from_vllm_payload()` 按算法生成完整 `DraftFeatureSample`；
4. 生成简化 `SampleMetadata` 控制信封；
5. `encode_sample()` 无损编码 sample；
6. 将 tensor fields 和 ready tag 写入 TQ；
7. TQ put 成功后删除 vLLM safetensors 临时文件；
8. 全部样本完成后写 EOS。

### Consumer

1. rank 0 使用统一 tag parser 发现当前 run 的 ready keys；
2. 按 `sequence_no` 排序并切出一个 global batch；
3. 广播各 rank 的 key/tag 分配；
4. 各 rank 从 TQ 获取自己的 tensor fields；
5. `decode_sample()` 无损恢复 `DraftFeatureSample`；
6. 调用 `prepare_training_batch_from_samples()`；
7. 由已选择的算法 backend 完成训练；
8. 所有 rank 训练成功后，rank 0 清理这一 global batch 的 keys。

## 7. 测试修改

### 7.1 `tests/unit/test_drafter_sample_protocol.py`

替换以 DSpark shape 为中心的测试，增加完整 round-trip：

1. 单 tensor hidden states；
2. hidden-state tensor list；
3. 所有 optional tensor 字段；
4. metadata 嵌套 dict/list/tuple；
5. metadata 中包含 tensor；
6. EAGLE3/DFLASH/DSPARK/DOMINO 的 algorithm 字符串均能往返；
7. 不支持的 metadata 对象明确报错并包含字段路径；
8. key/tag run、sequence、sample ID 不匹配时 fail closed；
9. protocol version 不匹配时 fail closed。

核心断言不是只比较 shape，而是逐字段比较原始 sample 与恢复 sample。

### 7.2 `tests/unit/test_tq_producer.py`

增加：

1. Producer 写入 fields 后能完整恢复其原始 `DraftFeatureSample.metadata`；
2. pending capacity 使用统一 ready parser；
3. 其他 run、错误 protocol version 和非 sample control tag 不计入 pending；
4. TQ put 失败时仍不删除 vLLM 临时文件。
5. token IDs 不匹配时删除临时文件、增加 dropped 计数且不调用 TQ put；
6. 任一 feature position 越界时整条样本丢弃，不生成部分长度 sample；
7. 丢弃无效样本后 worker 能继续发布后续有效样本并最终写 EOS；
8. EOS 的 `total_samples` 使用实际成功发布数，不包含 dropped 样本。

### 7.3 `tests/unit/test_tq_consumer.py`

增加：

1. TQ store 恢复完整 `DraftFeatureSample`；
2. metadata tensor 完整恢复；
3. hidden-state list 完整恢复；
4. ready discovery 与 Producer pending 统计对同一组 tags 给出相同结果；
5. 多算法 sample 都能进入 `prepare_training_batch_from_samples()` 的现有入口。

### 7.4 回归测试

必须继续运行：

```bash
pytest -q tests/unit/test_drafter_sample_protocol.py
pytest -q tests/unit/test_tq_producer.py
pytest -q tests/unit/test_tq_consumer.py
pytest -q tests/unit/test_target_feature_replay.py
pytest -q tests/unit/test_draft_feature_store.py
```

然后运行一组 Producer/TQ/Consumer smoke test，至少确认：

```text
put → list → get → decode → train one step → clear → EOS
```

## 8. 版本与兼容策略

建议将协议版本提升到 2，不实现 v1/v2 混读。

理由：

- standalone TQ 是在线临时队列，不是长期离线数据集；
- Producer 和 Consumer 本来就应作为同一版本部署；
- 同一 run 中混用两种 fields 格式会增加错误恢复复杂度；
- fail closed 比错误地恢复训练数据更安全。

启动时 Producer、Owner 和 Consumer 必须使用同一个 protocol version。旧 run 的 TQ 数据不能被新
Consumer 接续；重启完整 pipeline 时使用新的 `run_id`。

## 9. 实现顺序

建议按以下顺序实施，每一步都能单独测试：

1. 简化 `SampleMetadata`，确定 v2 tag/key/EOS 格式；
2. 实现 metadata tree codec；
3. 实现完整 `DraftFeatureSample` encode/decode；
4. 完成 protocol round-trip 单测；
5. 修改 Producer 的控制信封构造和统一 ready 统计；
6. 为 TQ Producer 增加严格 hidden-state 对齐和样本级丢弃；
7. 修改 Consumer store 的统一 tag 解析；
8. 修改 Producer/Consumer 单测；
9. 运行单进程 TQ round-trip smoke；
10. 运行多 rank Consumer 一步训练；
11. 分别用 EAGLE3、DFLASH、DSPARK 的合成 sample 验证通用传输；
12. 最后再为尚未支持的算法增加 vLLM-output-to-sample converter。

## 10. 完成标准

满足以下条件才算重构完成：

- TQ codec 中不存在 `if algorithm == "DSPARK"` 一类分支；
- `decode_sample(encode_sample(sample))` 能无损恢复所有公共字段和 metadata；
- Producer 和 Consumer 使用同一个 ready tag parser；
- Consumer 解码后直接得到 `DraftFeatureSample`；
- `base_trainer.py` 的后续训练入口无需为 TQ 添加算法分支；
- 新算法只要能构造现有 `DraftFeatureSample`，传输层无需修改；
- ready backpressure 与 Consumer discovery 对同一组 tags 的计数完全一致；
- TQ put 成功前不删除 vLLM 临时文件，训练成功前不清理 TQ sample。
- Producer 与 Consumer 对 sample fields 调用同一组结构校验；
- hidden-state 对齐不完整的样本不会以缩短后的 feature window 进入 TQ；
- dropped 样本有明确计数和限频日志，且不会阻止后续有效样本和 EOS。
