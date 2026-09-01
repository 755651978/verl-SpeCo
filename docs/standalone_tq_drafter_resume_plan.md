# Standalone TQ 独立训练断点续训方案

## 1. 第一版目标

当前 DSpark 独立训练已经能够从 checkpoint 恢复模型权重、optimizer、LR scheduler、`optimizer_steps_total` 和 `training_steps`，但没有恢复数据进度。重启后 producer 会从文件开头重新生产，导致 checkpoint 之前已经训练的数据再次进入训练。

第一版采用最直接的方案：

```text
consumer 记录已经成功训练的 sequence_no
→ checkpoint 保存这些 sequence_no
→ 重启时 producer 加载这个集合
→ 读文件时跳过已消费编号
→ 其他样本仍按原逻辑并行请求 vLLM
```

不修改 producer 的并发请求和完成顺序，不恢复旧 TQ，也不引入顺序发布。

## 2. 当前流程已经具备的条件

### 2.1 输入已经有编号

`standalone_tq_producer.py::read_inputs()` 当前执行：

```python
record = replace(source_record, sequence_no=stats.input_count)
```

编号进入现有 TQ tag：

```python
tag = {
    "record_type": "sample",
    "status": "ready",
    "schema_version": 2,
    "run_id": "...",
    "sample_id": "...",
    "sequence_no": 1234,
}
```

所以不需要新增样本身份协议，直接使用现有 `sequence_no`。前提是续训使用相同输入文件且行顺序不变。

### 2.2 consumer rank 0 已经知道 batch 编号

`TQFeatureDataLoader.__iter__()` 中 rank 0 执行：

```python
ready = self.store.list_ready()
selected = ready[:global_batch_size]
```

`selected` 中每个 `ReadyEntry` 都有 `entry.tag["sequence_no"]`。因此 rank 0 已经知道本 global batch 实际使用了哪些输入编号。

### 2.3 消费成功边界已经明确

训练循环当前顺序为：

```text
training_step_from_batch()
→ 所有 rank 确认成功
→ rank 0 clear_completed_batch(global_keys)
→ successful_steps += 1
→ 按间隔保存 checkpoint
```

规定：

```text
训练成功 + TQ clear 成功 = 本 batch 的 sequence_no 已消费
```

训练或 clear 失败时不能更新已消费集合。

## 3. checkpoint 增加什么

当前 checkpoint：

```text
draft_step_60000/
├── config.json
├── model.safetensors 或模型 shards
├── metadata.json
└── optimizer/
```

新增：

```text
draft_step_60000/consumed_sequence_nos.pt
```

内容为排序、去重的 CPU int64 tensor：

```python
tensor([0, 1, 2, 4, 5, 8, ...], dtype=torch.int64)
```

允许存在间隔：alignment 失败的数据、尚未训练的 TQ 数据和仍在推理的数据都不在集合中。

空间开销约为：

```text
100 万个编号：7.6 MiB
1000 万个编号：76 MiB
```

相比模型和 optimizer checkpoint 很小。

`metadata.json` 增加：

```python
"standalone_data_progress": {
    "version": 1,
    "consumed_sequence_file": "consumed_sequence_nos.pt",
    "consumed_sequence_count": int,
    "input_fingerprint": {...},
}
```

## 4. 为什么乱序推理不影响这个方案

假设 vLLM 完成顺序为：

```text
5, 1, 8, 2, 4, 0, 3
```

consumer 实际训练：

```text
batch 1: [1, 5]
batch 2: [0, 2]
```

checkpoint 保存：

```python
consumed_sequence_nos = tensor([0, 1, 2, 5])
```

重启后 producer 只跳过 0、1、2、5，其余编号重新请求。因此不要求已消费数据连续，也不需要 producer 按顺序写 TQ。

## 5. consumer 修改

### 5.1 `TQLocalBatch` 携带 global 编号

文件：`verl_speco/trainer/tq_sample_source.py`

改为：

```python
@dataclass(frozen=True)
class TQLocalBatch:
    local_keys: list[str]
    local_samples: list[DraftFeatureSample]
    global_keys: list[str] | None
    global_sequence_nos: list[int] | None
```

rank 0 构造 command 时增加：

```python
"global_sequence_nos": [
    int(entry.tag["sequence_no"])
    for entry in selected
]
```

只有 rank 0 需要保存 `global_sequence_nos`，其他 rank 保持 `None`。

### 5.2 clear 成功后更新集合

文件：`verl_speco/trainer/draft_training_loop.py`

启动时：

```python
consumed_sequence_nos = load_consumed_sequence_nos(drafter_cfg.model_path)
```

在 `_clear_tq_batch_across_ranks()` 成功返回以后：

```python
if rank == 0:
    consumed_sequence_nos.update(
        tq_local_batch.global_sequence_nos or []
    )
```

运行时使用 `set[int]` 便于去重；保存前转换为不可变 snapshot：

```python
consumed_snapshot = torch.tensor(
    sorted(consumed_sequence_nos),
    dtype=torch.int64,
)
```

## 6. checkpoint 修改

不修改 `verl_speco/trainer/base_trainer.py`。该类同时被 co-train 使用，把独立训练的数据进度塞进它的公共 checkpoint 接口，会扩大影响范围。

独立训练仍先调用现有的：

```python
trainer.save_checkpoint(step=step, wait=wait)
```

然后只在 `draft_training_loop.py` 的 `_save_standalone_checkpoint()` 中追加独立训练 sidecar：

```text
draft_step_N/
├── 原有模型、optimizer、scheduler 和 metadata
├── consumed_sequence_nos.pt
└── standalone_resume.json
```

其中 `standalone_resume.json` 记录 consumed 数量、输入 fingerprint 和 sidecar 版本。只有原 checkpoint 已成功保存后，才发布这两个文件。

使用临时文件原子写入：

```python
temporary = checkpoint_path / "consumed_sequence_nos.pt.incomplete"
final = checkpoint_path / "consumed_sequence_nos.pt"
torch.save(consumed_snapshot, temporary)
os.replace(temporary, final)
```

异步保存时，`_save_standalone_checkpoint()` 先复制不可变 snapshot，再给现有 checkpoint future 注册 callback。callback 只在模型 checkpoint future 成功后原子写 sidecar，不能让后台 callback 直接读取仍在变化的 Python set。

如果进程恰好在模型 checkpoint 完成、sidecar 尚未完成时崩溃，这个目录不能用于“精确数据续训”，应回退到上一个同时具备完整模型 checkpoint 和完整 sidecar 的目录。

加载时校验：

- 文件存在；
- 一维 `torch.int64`；
- 所有值非负；
- 已排序、无重复；
- tensor数量与 metadata一致。

旧 checkpoint没有该文件时，默认提示它只能恢复训练状态、不能精确恢复数据；严格模式下拒绝续训。

## 7. producer 修改

### 7.1 新配置

```yaml
speco:
  standalone_tq_producer:
    consumed_sequence_path: null
```

launcher 在续训时传入：

```text
<draft_step_N>/consumed_sequence_nos.pt
```

### 7.2 启动时加载

文件：`verl_speco/standalone_tq_producer.py`。

```python
consumed_sequence_nos = load_consumed_sequence_nos(
    producer_cfg.get("consumed_sequence_path")
)
```

### 7.3 扫描时跳过

需要拆开两个计数：

```python
source_sequence_no  # 所有扫描过的输入，跳过也递增
queued_count        # 本次真正送入 input_queue 的数量
```

第一版保持 `iter_input_records()` 接口不变，在它产出记录后、tokenizer 和 vLLM 之前，根据 epoch 内扫描顺序算出全局 `sequence_no`：

```python
sequence_no = source_sequence_no
source_sequence_no += 1

if sequence_no in consumed_sequence_nos:
    continue

record = replace(source_record, sequence_no=sequence_no)
await input_queue.put(request)
queued_count += 1
```

因此恢复时仍需从输入文件开头顺序读取并解析一次，以重建稳定编号，但已消费行不会进入 tokenizer、input queue 或 vLLM。集合查询是平均 O(1)，后续仍使用原来的多个 `request_worker()`，不会降低 producer 并发。

这里不是每个训练 step 都重新读取 checkpoint 文件。`consumed_sequence_nos.pt` 只在 producer 启动时加载一次；之后每扫描到一个源记录，只做一次内存集合查询。对百万级样本，主要额外成本是一次顺序读文件和 JSON/Parquet 行解析，通常远小于 tokenizer 和 vLLM 推理。只有实际测量发现启动扫描成为瓶颈后，才考虑给 input reader 增加解析前跳过、文件 offset 索引或 bitmap，第一版不做。

## 8. 多 epoch 编号

编号必须跨 epoch 连续：

```text
数据集 10000 行
epoch 0：0～9999
epoch 1：10000～19999
epoch 2：20000～29999
```

加入 skip 后不能继续用一个 `stats.input_count` 同时表示扫描位置和排队数量，否则跳过数据后编号会改变。必须使用独立的 `source_sequence_no`。

## 9. 本次需要生产多少数据

独立训练的 `MAX_STEPS` 改为目标总 optimizer step，与首次训练配置保持一致：

```text
MAX_STEPS = 训练完成时的总步数
```

例如 checkpoint为 60000，目标总步数 930000：

```bash
DRAFTER_PATH=/path/to/draft_step_60000
MAX_STEPS=930000
LR_WARMUP_STEPS=<仍使用首次训练时的原值>
```

checkpoint 已恢复 optimizer 和 scheduler 状态，所以 warmup 不会从头开始，当前 learning rate 和 scheduler 计数从 checkpoint 继续。用户不需要手算剩余步数，也不需要改其他训练超参；正常情况下只新增/替换 `DRAFTER_PATH`。

训练循环不再用本次进程的 `successful_steps < max_steps` 判断结束，而使用恢复后的总步数：

```python
while max_steps <= 0 or trainer.optimizer_steps_total < max_steps:
    ...
```

launcher计算 producer 配额时使用：

```python
remaining_steps = max(max_steps - resumed_optimizer_step, 0)
max_samples = remaining_steps * batch_size_per_gpu * world_size
```

但 producer应使用 `queued_count` 判断本次配额。跳过的已消费编号不计入 `queued_count`。

该行为要求传入的是完整训练 checkpoint，里面具有 optimizer 和 scheduler 状态。只有模型权重的目录只能作为初始化权重，不能保证 learning rate、warmup 和 optimizer 状态精确续接。

## 10. 输入文件校验

编号只有在输入文件内容和顺序不变时才稳定。checkpoint至少保存：

```python
"input_fingerprint": {
    "path": str,
    "size_bytes": int,
    "mtime_ns": int,
}
```

推荐增加 SHA-256。续训时 fingerprint不一致则默认报错，避免旧 `sequence_no` 对应到新数据。

## 11. 崩溃语义

- vLLM已完成但未训练：不在 checkpoint集合，重启后重新推理。
- 已写 TQ但未训练：不在集合，重启后重新推理。
- optimizer成功但新 checkpoint未完成：恢复上一个模型和集合，重新训练上个 checkpoint之后的数据。
- checkpoint完整：模型状态和 consumed集合对应同一步，已消费数据会被跳过。
- checkpoint写到一半：恢复时忽略不完整目录，回退上一个 `complete=true` checkpoint。

## 12. 文件修改清单

### 新增 `verl_speco/trainer/standalone_resume.py`

```python
load_consumed_sequence_nos(path) -> set[int]
save_consumed_sequence_nos(path, values) -> dict
build_input_fingerprint(path) -> dict
validate_input_fingerprint(saved, current) -> None
```

### 修改 `verl_speco/trainer/tq_sample_source.py`

- `TQLocalBatch.global_sequence_nos`；
- rank 0从 selected tags提取编号。

### 修改 `verl_speco/trainer/draft_training_loop.py`

- 启动时加载集合；
- clear成功后更新集合；
- checkpoint完成后写排序 int64 snapshot和 standalone sidecar；
- 使用 `optimizer_steps_total < max_steps` 作为独立训练终止条件。

`base_trainer.py` 及 co-train 入口不修改。standalone sidecar 的校验和加载全部收敛在 `standalone_resume.py` 与独立训练入口中。

### 修改 `verl_speco/standalone_tq_producer.py`

- 加载 consumed集合；
- 拆分 source sequence和 queued count；
- tokenizer和vLLM前排除已消费编号；
- 其他并发逻辑保持不变。

### 修改 `verl_speco/standalone_tq_training_launcher.py`

- 从 resume checkpoint获取 consumed文件；
- 把路径传给 producer；
- 校验输入 fingerprint。
- 从 checkpoint读取已恢复 optimizer step，并仅生产剩余总步数需要的样本。

### 修改配置和 example

- 增加 `consumed_sequence_path`；
- 说明只需设置 `DRAFTER_PATH=draft_step_N`；`MAX_STEPS`、warmup等保持首次训练配置。

## 13. 测试

1. consumer训练乱序编号 batch，clear成功后全部加入集合。
2. 训练失败或 clear失败时不更新集合。
3. checkpoint文件排序、去重，metadata数量一致。
4. 输入 `0～9`，集合 `{0,2,5}`，producer只请求 `1,3,4,6,7,8,9`。
5. 跳过的数据不占本次 `max_samples`配额。
6. 多 epoch编号连续且重启后稳定。
7. vLLM worker数量和请求并发与修改前一致。
8. 端到端保存、重启后，producer不再请求 checkpoint集合中的编号。

## 14. 最终流程

首次训练：

```text
producer并行请求 vLLM
→ TQ
→ consumer训练 batch
→ clear成功
→ rank 0记录 sequence_no
→ checkpoint保存模型、optimizer和 consumed_sequence_nos.pt
```

续训：

```text
加载 draft_step_N
→ 恢复模型、optimizer、LR和 step
→ 加载 consumed_sequence_nos.pt
→ 校验输入文件
→ 创建新 Ray/TQ run
→ producer跳过已消费编号
→ 其余样本继续并行请求 vLLM
```

该方案不改变 producer并发或 TQ消费顺序，改动集中在“consumer记录已成功数据”和“producer排除已消费数据”，适合作为第一版实现。
