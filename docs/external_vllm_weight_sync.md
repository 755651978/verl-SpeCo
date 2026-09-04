# co-train 外部 vLLM 权重热更新

Last updated: 2026-09-03

## 范围

依据 `bench_weight_transfer.py` 和本机 py311 中实际安装的 vLLM 0.23.0 源码实现。控制消息通过 HTTP，权重 Tensor 由 actor rank 0 通过 NCCL 直接发送，不经过 driver、Producer Actor 或 TQ。

当前支持 CUDA 上的 FSDP/FSDP2/VeOmni actor 导出的完整 HF 权重，包括引擎导出的 merged 权重。不支持未合并 LoRA adapter；Megatron 等其他引擎的单 rank 完整导出语义未验证，暂时明确拒绝。当前 vLLM NCCL API 使用 CUDA，不能直接用于 NPU/HCCL。

## 启动配置

外部 hidden-state vLLM 保留原来的 hidden-state connector 等启动参数，另外增加：

```bash
export VLLM_SERVER_DEV_MODE=1
# 在原 vllm serve 命令上添加：
# --weight-transfer-config '{"backend":"nccl"}'
```

不必使用 `--load-format dummy`；如果使用，必须确认首次权重同步成功后才发送推理请求。Dev API 只应暴露在可信网络中。

co-train 配置示例（放在原 drafter.training 下）：

```yaml
collect_hidden_states_from_old_logprob: false
collect_hidden_states_from_sgl: false
collect_hidden_states_from_vllm: true
vllm_feature_source:
  endpoints:
    - http://10.0.0.10:8000/v1
    - http://10.0.0.11:8000/v1
  weight_hot_update:
    enabled: true
    master_address: null
    timeout_seconds: 600
    bucket_size_mb: 256
    packed: true
    packed_num_buffers: 2
```

- `endpoints`：复用 Producer 的列表，不再单独填一份。程序去掉末尾 `/v1` 后访问 `/get_world_size` 等管理接口。
- `master_address`：默认使用 actor rank 0 所在机器的地址；手动填写时必须是 vLLM workers 能访问到的 actor rank 0 地址，不是 driver 地址。端口在该 actor 进程上自动选择。
- `timeout_seconds`：HTTP 等待超时，不是所有底层 NCCL 故障的统一退出期限。
- `bucket_size_mb`：发送端分批暂存权重的目标大小，避免自己额外保存一整份完整模型。单个 Tensor 超过这个值时独占一批，packed buffer 同步扩容；不代表总显存上限，也不改变训练引擎自身导出过程的显存需求。
- `packed` / `packed_num_buffers`：使用 vLLM 原生打包发送及缓冲数量，两端接收相同配置。

actor rank 0 和 Producer Actor 都必须能访问所有 endpoint。跨机器不要使用 `127.0.0.1`。外部 vLLM 应由这个训练任务独占更新，不要让其他任务同时写权重或发送请求。每个 endpoint 建立单独 NCCL 组：一个发送 rank，加该 endpoint 的 TP × PP × DP workers。

## 执行顺序

1. 初始化 actor worker 后，driver 调用所有 actor ranks 的初始化 RPC。仅 rank 0 查询 vLLM worker 数量、在后台发 HTTP 初始化请求，同时在本地调用 `NCCLWeightTransferEngine.trainer_init()` 完成握手。
2. `fit()` 恢复 actor checkpoint 后，在原来的 rollout 权重同步边界先同步外部 vLLM。因此首次发送的是当前训练模型（含恢复的 checkpoint），不是另加载的模型文件。
3. 当前轮 Producer 使用这份权重 prefill 当前 rollout 数据，仍按原 scheduler 分桶、提交给 drafter。
4. actor 更新、drafter 训练后，在下一次 rollout 权重同步边界再次同步外部 vLLM，再执行原 rollout target/drafter 权重发布。没有延后一轮的逻辑。先发送外部权重，是为了尽量在同卡 rollout 恢复显存前释放 actor 导出的暂存 Tensor。
5. 每次发送的 HTTP 顺序为 `pause(mode=wait, clear_cache=true)`、`start_weight_update`、一批或多批 `update_weights`、`finish_weight_update`、`resume`。后台 HTTP 等待接收时，主线程调用 `trainer_send_weights()`。所有 endpoint 都 finish 成功后才开始 resume。
6. 所有 actor ranks 消费 `get_per_tensor_param()` 返回的 iterator，完成其内部的分片汇集操作；只有 rank 0 通过 NCCL 发送。需要参数 offload 的引擎在完成后恢复 CPU offload。
7. 退出时关闭 sender 的 HTTP Session，调用本地 communicator 的 `destroy()`。不关闭用户自己启动的 vLLM 服务。

`pause` 清理旧权重的 KV/prefix cache，不等于永久禁用 prefix caching。原 hidden-state 服务为保证完整 hidden rows 所需的禁用 prefix cache 等配置仍须保留。

## 改了哪些代码

- `verl_speco/integration/external_vllm_weight_sync.py`：替换原占位接口，实现 HTTP/NCCL sender、分批发送、driver 和 actor 侧生命周期。
- `verl_speco/integration/rollout_publish.py`：在已有 `DraftWeightPublishMixin` 上新增初始化、更新、关闭三个 `ONE_TO_ALL` RPC，复用已有 actor worker，不新加载目标模型。
- `verl_speco/trainer/speco_ray_trainer.py`：连接初始化、checkpoint 恢复后的首次发送、每轮权重同步边界和最终关闭。未修改上游 verl 源码。
- `verl_speco/config/speco_base.yaml`：补充上述配置；默认 `enabled: false`，原有模式不发起权重传输。
- `tests/unit/test_external_vllm_weight_sync.py`：模拟控制面、sender、分批、失败处理和 driver hook 的测试。
- `tests/special_sanity/check_device_api_usage.py`：声明这一个集成文件使用的是 CUDA 专属 vLLM API。

## 日志与失败处理

### Producer 的 final norm 同步

EAGLE3 和开启 L1 的 DSpark 使用辅助层加最终 hidden；vLLM connector 的最后一块是 final norm 前的 residual，Producer 在公共转换函数内补目标模型 final norm，辅助层不变。DFlash 和关闭 L1 的 DSpark 不需要这一步。

Producer 初始化时复用独立训练的 norm loader，只加载 final norm 的 checkpoint 权重。开启 `weight_hot_update.enabled` 后，不能一直使用初始化时的权重：Producer 报告 norm 对应的 HF 参数名；actor rank 0 在同一次完整权重导出中复制这几个小 tensor 到 CPU。发送 vLLM 成功后，现有 Ray RPC 把这些 tensor 返回 driver，再更新 Producer 的 norm。只有两边都成功才记录 `last_synced_step`，之后才进入下一轮取数。不是重新读磁盘，也没有新增一个大模型副本或额外 FSDP 汇集。

未开启热更新时仍使用 checkpoint 的 norm，因此 endpoint 必须服务同一份固定目标模型。此模式不能在外部自行更新 endpoint 权重而不更新 Producer。

握手成功：`[external vLLM weights] connected endpoint=... workers=...`

一轮所有 endpoint 更新成功：`[external vLLM weights] updated step=... tensors=... endpoints=...`

失败会使当前训练同步抛错，不标记成功，也不自动恢复可能只更新了部分权重的服务。后台 HTTP 异常会通过 future.result() 传回；真正 NCCL 故障仍可能等待底层超时。第一版不做失败重试或自动重连，失败后应检查日志并重启服务/任务。

## 验证状态

已完成 CPU stand-in/mock 测试与相关回归测试（104 passed，8 skipped）。跳过项为本机旧 Transformers 缺少 Qwen3 类的数值测试；包括 Producer norm 更新及失败传播测试，但这不是多卡端到端证明。

未启动真实 vLLM 服务。用户将自行在服务器测试实际 HTTP/NCCL 更新、训练与推理结果。
