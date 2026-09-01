# Standalone TQ 独立训练参数说明

> Last updated: 08/27/2026

本文说明下面两个脚本暴露的参数：

- `tools/run_qwen3-8b_drafter_hidden_state_vllm.sh`：按可见设备与 TP 自动启动一个或多个 target vLLM 服务。
- `examples/run_qwen3-8b_drafter_separate_training.sh`：启动 Producer、TQ 和 DSpark Consumer 训练。

参数可以通过环境变量设置，例如：

```bash
MAX_STEPS=1000 \
BATCH_SIZE_PER_GPU=2 \
DSPARK_CE_LOSS_ALPHA=0.1 \
DSPARK_L1_LOSS_ALPHA=0.9 \
bash examples/run_qwen3-8b_drafter_separate_training.sh
```

## 1. vLLM 服务参数

以下参数由 `run_qwen3-8b_drafter_hidden_state_vllm.sh` 使用。

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `MODEL_PATH` | `/path/to/Qwen3-8B` | 所有 target vLLM 服务加载的模型路径。 |
| `DEVICE_ENV` | `ASCEND_RT_VISIBLE_DEVICES` | 控制设备可见性的环境变量。GPU 环境可设为 `CUDA_VISIBLE_DEVICES`。 |
| `VLLM_DEVICES` | `0,1,2,3,4,5` | 分配给vLLM的完整设备列表，脚本按连续的 `VLLM_TP` 张设备切成多个实例。 |
| `VLLM_TP` | `1` | 每个vLLM实例的 tensor parallel 大小；设备总数必须能被该值整除。 |
| `VLLM_HOST` | `127.0.0.1` | 所有vLLM服务监听的主机地址。 |
| `VLLM_BASE_PORT` | `8000` | 第一个实例的端口；后续实例依次使用 `8001`、`8002` 等。 |
| `VLLM_GPU_MEMORY_UTILIZATION` | `0.8` | 单个 vLLM 服务允许使用的设备显存比例。 |
| `VLLM_MAX_NUM_SEQS` | `256` | 单个 vLLM 服务最多同时调度的 sequence 数。 |
| `VLLM_HIDDEN_STATE_LAYER_IDS` | `[1,9,17,25,33,36]` | vLLM 导出的 hidden-state 层；前面的辅助层必须与训练侧 `DSPARK_TARGET_LAYER_IDS` 相同，最后一层用于构造 L1 loss 所需的 target 概率分布。 |
| `HIDDEN_STATES_DIR` | `/tmp/speco-vllm-hidden-states` | vLLM connector 临时写 hidden-state 文件的根目录，每个实例使用独立的 `service-N` 子目录。 |

如果每个服务使用两张卡：

```bash
MODEL_PATH=/nas/disk1/Qwen3-4B \
VLLM_DEVICES=0,1,2,3,4,5 \
VLLM_TP=2 \
bash tools/run_qwen3-8b_drafter_hidden_state_vllm.sh
```

## 2. 基础训练参数

以下参数由 `run_qwen3-8b_drafter_separate_training.sh` 使用。

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `MODEL_PATH` | `/path/to/Qwen3-8B` | Target model 路径，同时用于 tokenizer、模型配置和 target embedding/LM head。 |
| `TRAIN_FILE` | `/path/to/train_file.parquet` | Producer 读取的 JSONL 或 Parquet 数据文件。 |
| `DRAFTER_PATH` | 空 | 可选的已有 drafter/checkpoint 路径；为空时根据 target 配置从头初始化 DSpark。 |
| `DRAFT_CKPTS_DIR` | `/path/to/dspark_draft_checkpoints` | 保存 drafter checkpoint 的目录。 |
| `TRAIN_DEVICES` | `2,3` | Consumer 训练使用的设备。不能与任何 vLLM 实例占用的设备重叠。 |
| `TRAIN_GPUS` | `2` | 本节点启动的训练 rank 数，通常等于 `TRAIN_DEVICES` 中的设备数量。 |
| `DEVICE_ENV` | `ASCEND_RT_VISIBLE_DEVICES` | 训练侧设备可见性环境变量；GPU 环境可改为 `CUDA_VISIBLE_DEVICES`。 |
| `SPECO_VLLM_ENDPOINTS` | `[http://127.0.0.1:8000/v1,http://127.0.0.1:8001/v1]` | Producer 并行访问的 vLLM endpoint 列表。 |
| `VLLM_READY_TIMEOUT_SECONDS` | `120` | 启动训练前等待所有 vLLM endpoint 就绪的最长时间。 |
| `PYTHON_BIN` | `python3` | 启动 Python 模块所用的解释器。 |
| `PROJECT_NAME` | `verl_dspark_drafter` | 实验项目名。 |
| `EXP_NAME` | `qwen3_8b_dspark_separate_training` | 本次实验名称。 |

## 3. Producer和请求并发参数

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `VLLM_REQUEST_TIMEOUT` | `120` | 单次 generate 或 prefill HTTP 请求的超时时间，单位为秒。 |
| `VLLM_MAX_INFLIGHT_REQUESTS` | `16` | Producer 整体最多同时存在的 vLLM 请求数。 |
| `VLLM_PER_ENDPOINT_CONCURRENCY` | `4` | 每个 vLLM endpoint 独立的并发请求上限。 |
| `PRODUCER_INPUT_QUEUE_SIZE` | `32` | 已读取、等待 vLLM worker 处理的请求队列容量。 |
| `PRODUCER_PUBLISH_QUEUE_SIZE` | `16` | 已完成推理、等待写入 TQ 的样本队列容量。 |
| `PRODUCER_MAX_PENDING_SAMPLES` | `1024` | TQ 中尚未被 Consumer 训练并删除的样本数量上限，用于限制积压。 |
| `PRODUCER_PENDING_POLL_INTERVAL` | `0.5` | TQ 积压达到上限后，Producer 重新检查容量的间隔，单位为秒。 |
| `PRODUCER_MAX_SEQUENCE_LENGTH` | `8192` | prompt 和 response 处理前允许的最大总 token 长度。 |
| `PRODUCER_MAX_FEATURE_LENGTH` | `512` | 每个样本最终保留用于训练的最大 token 窗口。 |
| `PRODUCER_GENERATION_MAX_TOKENS` | `512` | 输入没有 response 时，vLLM 最多生成的 completion token 数。 |

两个 endpoint 下的有效客户端并发近似为：

```text
min(VLLM_MAX_INFLIGHT_REQUESTS,
    endpoint数量 × VLLM_PER_ENDPOINT_CONCURRENCY)
```

`VLLM_MAX_NUM_SEQS` 是 vLLM 服务端调度上限；`VLLM_MAX_INFLIGHT_REQUESTS` 和 `VLLM_PER_ENDPOINT_CONCURRENCY` 是 Producer 客户端请求上限。

## 4. 训练过程参数

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `MAX_STEPS` | `10` | 本次运行最多完成的训练 step 数。Producer 会根据它和全局 batch size 计算需要生成多少样本。 |
| `BATCH_SIZE_PER_GPU` | `2` | 每个训练 rank 每个 step 使用的样本数。全局 batch size 为该值乘训练 rank 总数。 |
| `SAVE_INTERVAL_STEPS` | `5` | 每隔多少 optimizer step 保存一次 checkpoint；设为0表示不做周期保存。 |
| `SAVE_FINAL_CHECKPOINT` | `true` | 训练结束时是否保存最终 checkpoint。 |
| `LEARNING_RATE` | `1e-6` | Drafter optimizer 的基础学习率。 |
| `LR_WARMUP_STEPS` | `0` | 学习率 warmup 的 step 数。 |
| `LR_SCHEDULER_TYPE` | `constant` | 学习率调度类型，支持 `constant`、`cosine`、`linear`、`global_cosine`。 |
| `LR_DECAY_STEPS` | `100` | 需要衰减的 scheduler 使用的衰减 step 数。 |
| `MIN_LR_RATIO` | `0.1` | 学习率衰减后的最小值与基础学习率的比例。 |
| `PARAM_OFFLOAD` | `true` | FSDP 是否把模型参数 offload 到 CPU。 |
| `OPTIMIZER_OFFLOAD` | `true` | FSDP 是否把 optimizer state offload 到 CPU。 |

Producer 需要发布的样本数为：

```text
MAX_STEPS × BATCH_SIZE_PER_GPU × TRAIN_GPUS × 节点数
```

如果文件样本不足，Producer 会重新从文件开头读取并再次请求 vLLM，达到所需样本数后才发布 EOS。

## 5. DSpark模型和采样参数

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `DSPARK_BLOCK_SIZE` | `7` | 每个 anchor 并行预测的 token 数。 |
| `DSPARK_NUM_ANCHORS` | `32` | 每个样本选取的 anchor 数；越大训练计算量和显存占用越高。 |
| `DSPARK_MAX_WINDOW` | `512` | DSpark 从输入样本中取出的最大训练窗口长度。 |
| `DSPARK_NUM_TARGET_LAYERS` | `5` | 输入 DSpark 的 target 辅助 hidden-state 层数量。 |
| `DSPARK_NUM_HIDDEN_LAYERS` | `5` | DSpark drafter 自身 transformer 层数。 |
| `DSPARK_TARGET_LAYER_IDS` | `[1,9,17,25,33]` | Target model 中采集的辅助 hidden-state 层编号。必须与 vLLM 服务侧配置的辅助层一致。 |
| `DSPARK_MARKOV_RANK` | `256` | Markov head 的低秩维度。 |
| `DSPARK_MARKOV_HEAD_TYPE` | `vanilla` | Markov head 类型。当前独立训练路径建议使用 `vanilla`。 |

## 6. DSpark损失参数

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `DSPARK_LOSS_MODE` | `full_vocab` | CE 计算方式，可使用 `full_vocab`、`restricted_ce` 或 `sampled_ce`。 |
| `DSPARK_SAMPLED_CE_NEGATIVES` | `0` | `sampled_ce` 模式下采样的负类数量。 |
| `DSPARK_LOSS_DECAY_GAMMA` | `7` | block 内不同预测位置的指数衰减系数。 |
| `DSPARK_CE_LOSS_ALPHA` | `0.1` | Token CE loss 在总 loss 中的权重。 |
| `DSPARK_L1_LOSS_ALPHA` | `0.45` | Draft 与 target token 概率分布之间的 L1 loss 权重。Target 概率由 final hidden state 和 LM head 计算；设为0可关闭 L1 loss。 |
| `DSPARK_L1_CHUNK_SIZE` | `0` | L1 loss 分块计算大小；0表示不主动分块。显存不足时可设置正整数。 |
| `DSPARK_CONFIDENCE_LOSS_ALPHA` | `0.0` | Confidence loss 权重。当前 standalone 协议没有 acceptance target，必须保持0。 |

当前 DSpark 总损失为：

```text
loss = DSPARK_CE_LOSS_ALPHA × ce_loss
     + DSPARK_L1_LOSS_ALPHA × l1_loss
```

只训练 CE 的配置：

```bash
DSPARK_CE_LOSS_ALPHA=1.0
DSPARK_L1_LOSS_ALPHA=0.0
```

## 7. 调试参数

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `DSPARK_DEBUG_LOG` | `false` | 是否输出 DSpark forward 的详细调试日志。 |
| `DSPARK_DEBUG_LOG_FIRST_N` | `2` | 开启调试日志后，前多少次 forward 必定打印。 |
| `DSPARK_DEBUG_LOG_INTERVAL` | `100` | 前几次之后，每隔多少次 forward 打印一次调试信息。 |

## 8. Layer ID一致性

默认配置为：

```text
训练侧 DSPARK_TARGET_LAYER_IDS       = [1,9,17,25,33]
服务侧 VLLM_HIDDEN_STATE_LAYER_IDS = [1,9,17,25,33,36]
```

服务侧前五项是辅助层，必须与训练侧完全相同。最后的 `36` 是 Qwen3-4B 对应的 final hidden-state layer，供 DSpark L1 loss 使用。更换 target model 或辅助层配置时，两边需要一起修改。
