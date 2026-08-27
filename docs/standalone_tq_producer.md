# Standalone vLLM → TransferQueue Producer

Last updated: 08/21/2026

本文解释 standalone Producer：它可以直接读取 verl 的 prompt-only Parquet（包括
DAPO-Math-17k 的 chat-message `prompt`），也兼容已有 `prompt`/`response` 的 JSONL
或 Parquet。缺少 response 时由 target vLLM 生成，并在同一请求中提取 prompt 与
output hidden states，之后把样本写到已存在的 TransferQueue（TQ）。

这条路径面向第一版 DSpark standalone 训练：Producer、TQ owner 和 Consumer
是三个独立 OS 进程；Ray 只用于让它们找到同一个 TQ Controller，hidden states
不通过 Ray object store 传输。

## 为什么需要这个 Producer

此前仓库已经有两块基础能力：

- `drafter_sample_protocol.py`：规定一条 TQ sample 的 key、tag、Tensor 字段和
  EOS record；
- `transferqueue_bridge.py` 与 `tq_owner.py`：负责连接 Ray/TQ、写读清理样本和
  owner 生命周期。

缺少的是把预先生成的文本变成 DSpark 训练特征并发布到 TQ 的独立进程。新增的
Producer 补上这一段，不引入第二套协议或 feature store。

## 数据流

```text
verl prompt Parquet 或 prompt/response JSONL/Parquet
  │
  │  按文件顺序分配 sequence_no 和 sample_id
  ▼
Tokenizer
  │  input_ids / loss_mask / feature window
  ▼
多个 vLLM endpoint（有界并发）
  │  OpenAI completions 请求 → 临时 safetensors 文件
  ▼
公共 hidden-state 转换函数
  │  DSpark DraftFeatureSample + SampleMetadata
  ▼
TransferQueue kv_put（一条输入记录对应一条 sample）
  │
  ├─ put 成功：删除该请求的临时文件
  └─ 全部成功：写一个 EOS control record
```

Producer 在开始请求前会等到对应 `run_id` 的 `owner_ready` 控制记录。它不会创建
Ray head、TQ Controller 或 storage backend；这些由 `verl-speco-tq-owner` 管理。

## 输入文件

输入可以是 JSONL 或 Parquet。`prompt` 可以是字符串，也可以是 verl 常用的
`[{"role": ..., "content": ...}]` chat-message 列表。`response` 是可选字符串：
存在时直接 replay；不存在时由 target vLLM 生成。Parquet 通过
`data.train_files` 直接传入，不需要转换。

```json
{"sample_id":"train-000017","prompt":"Question: 1 + 1 = ","response":"2"}
{"prompt":"Translate hello: ","response":"你好"}
```

- `sequence_no` 按非空行的文件顺序从 0 分配；并发完成顺序不会影响它。
- `sample_id` 可选；省略时生成 `train-000000`、`train-000001` 等稳定值。
- verl 数据的 `extra_info.index` 存在时会优先作为稳定 `sample_id`。
- chat-message prompt 通过 target tokenizer 的 `apply_chat_template()` 编码，并加上
  generation prompt；不能把 `reward_model.ground_truth` 当作模型 response。
- Producer tokenize `prompt` 和 `prompt + response`。后者必须以 prompt 的 token IDs
  为前缀；否则会报错，而不会猜测 response 的 loss-mask 边界。
- `loss_mask` 中 prompt token 为 0，response token 为 1。
- feature window 从 response 前一个 token 开始，长度由
  `max_feature_length` 限制；传给 vLLM 的 token IDs 截止于该 window 末端。
- 其他 JSON 字段目前只作为 Producer 进程内来源元数据；第一版协议不会把它们写入
  TQ，所以 Consumer 不能读取这些字段。

## vLLM 与 hidden states

对已有 response，Producer 使用 OpenAI-compatible completions API 做 prefill。
对 prompt-only 数据，Producer 在一次请求中生成 response 并要求保存输出 hidden：

```text
prompt=<chat-template token IDs>
max_tokens=<内部有界长度>
extra_body={
  "return_token_ids": true,
  "kv_transfer_params": {"include_output_tokens": true}
}
```

响应必须同时满足：

1. 若返回 `choices[0].prompt_token_ids`，它必须等于请求的 token IDs；
2. `kv_transfer_params.hidden_states_path` 必须存在；
3. 该文件必须含 `token_ids` 和形状为 `[seq, layers, hidden]` 的 `hidden_states`。

vLLM 0.23 已内置满足这个合同的 `ExampleHiddenStatesConnector`。不需要 SpeCo
Mooncake connector。在线服务必须关闭 chunked prefill，并显式配置一个 Producer
可见的临时目录。例如：

```bash
export MODEL_PATH=/path/to/target-model
export HIDDEN_STATES_DIR=/dev/shm/speco-hidden-states
mkdir -p "${HIDDEN_STATES_DIR}"

vllm serve "${MODEL_PATH}" \
  --host 0.0.0.0 \
  --port 8000 \
  --speculative-config \
  '{"method":"extract_hidden_states","num_speculative_tokens":1,"draft_model_config":{"hf_config":{"eagle_aux_hidden_state_layer_ids":[1,9,17,25,33,36]}}}' \
  --kv-transfer-config \
  "{\"kv_connector\":\"ExampleHiddenStatesConnector\",\"kv_role\":\"kv_producer\",\"kv_connector_extra_config\":{\"shared_storage_path\":\"${HIDDEN_STATES_DIR}\",\"use_synchronization_lock\":true}}" \
  --no-enable-chunked-prefill
```

上面的 layer IDs 只是 Qwen3-4B 示例。实际值必须按 target 模型和训练配置确定；
DSpark L1 开启时，vLLM 列表是 auxiliary layer IDs 加 final layer，而 Producer 的
`TARGET_LAYER_IDS` 只填写 auxiliary 部分。

官方 connector 使用持久存在的 `.lock` 文件和 `flock` 协调异步落盘。Producer
读取前等待文件锁释放；TQ `put_sample` 成功后同时删除 safetensors 和 `.lock`。

`feature_from_vllm_payload()` 是从旧 replay 路径提取出的公共纯函数。它校验 token
对齐、选择 feature rows、拼接 auxiliary layers；DSpark L1 开启时额外拼接 final
hidden state。旧 replay 路径仍通过薄封装调用此函数，避免两套转换规则。

## TQ 写入和失败语义

每个输入 record 只写一个协议 key：

```text
drafter:v1:<run_id>:<12位sequence_no>:<sample_id>
```

写入顺序是严格的：

```text
加载临时 safetensors
→ 校验并转换
→ TQ kv_put
→ 删除临时文件
```

因此：

- `kv_put` 失败时临时文件保留，且 Producer 不写 EOS；
- 任一请求、转换或写入失败会停止整条 Producer，不做自动重试或 endpoint 熔断；
- 只有所有 sample 都发布完成，才写 `control:v1:<run_id>:eos`；
- 进程退出时只调用 `close_transfer_queue_client()`，不会调用全局 `tq.close()`，
  不会销毁共享 Controller。只有 owner 可以关闭 TQ。

`max_pending_samples` 是简单背压：当前 run 的 ready sample 数达到该阈值时，新的
vLLM 请求会暂停，等待 Consumer 清理已成功训练的 key。

## 配置与启动

默认 Producer 配置位于
`speco.standalone_tq_producer`，TQ 连接配置仍位于
`actor_rollout_ref.rollout.drafter.training.transfer_queue`。

必须设置的 Producer 字段：

| 字段 | 含义 |
| --- | --- |
| `input_path` | 上述 JSONL 或 Parquet 文件 |
| `tokenizer_path` / `tokenizer_fingerprint` | 用于 tokenization 和 Consumer 合同校验 |
| `target_model_id` / `target_model_revision` | target checkpoint 身份 |
| `target_layer_ids` | auxiliary target layer IDs；DSpark L1 时 wire metadata 会额外写 `-1` 表示 final layer |
| `vllm_endpoints` / `vllm_model` | 一个或多个 OpenAI-compatible vLLM endpoint 与模型名 |

必须与 owner/Consumer 一致的 TQ 字段：

| 字段 | 固定要求 |
| --- | --- |
| `package_version` | `0.1.7` |
| `partition_id` | `speco_drafter_features` |
| `schema_version` | `1` |
| `run_id`、Ray address、Ray namespace | 三个进程必须相同 |

单独调试时可以通过安装后的命令入口运行；正式训练由统一launcher启动Producer：

```bash
verl-speco-tq-producer \
  actor_rollout_ref.rollout.drafter.training.transfer_queue.enable=true \
  actor_rollout_ref.rollout.drafter.training.transfer_queue.ray.address=<ray-address> \
  actor_rollout_ref.rollout.drafter.training.transfer_queue.run_id=<run-id> \
  speco.standalone_tq_producer.input_path=<samples.jsonl> \
  speco.standalone_tq_producer.tokenizer_path=<tokenizer> \
  speco.standalone_tq_producer.tokenizer_fingerprint=<fingerprint> \
  speco.standalone_tq_producer.target_model_id=<target> \
  speco.standalone_tq_producer.target_model_revision=<revision> \
  speco.standalone_tq_producer.target_layer_ids='[2,8,14,20,26]' \
  speco.standalone_tq_producer.vllm_endpoints='[http://node0:8000/v1]' \
  speco.standalone_tq_producer.vllm_model=<target>
```

完整生命周期顺序仍是：Ray/TQ backend → TQ owner → Consumer → Producer → Consumer
drain → owner shutdown。Producer 完成不代表训练完成，EOS 只表示不会再有新样本。
正式独立训练入口
`examples/run_qwen3-8b_drafter_separate_training.sh` 会通过
`verl_speco.standalone_tq_training_launcher` 自动管理这套生命周期；上面的 Producer
脚本仅用于单独调试 Producer。

## 测试覆盖与未验证项

新增测试覆盖：JSONL/真实 Parquet 解析、DAPO chat prompt、target response generation、
token 边界、多个 endpoint 的并发限制、ready 队列背压、
成功时 sample 后 EOS 与临时文件删除、失败时无 EOS 且保留临时文件，以及旧 EAGLE3
转换路径仍可复用公共函数。

这些测试使用 fake vLLM/TQ。真实 Ray + TransferQueue + vLLM 的多进程
联调没有在当前环境执行；运行前仍需确认 vLLM 版本能返回上述
`hidden_states_path` 以及 TQ 0.1.7 依赖环境可用。
