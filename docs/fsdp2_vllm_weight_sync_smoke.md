# 四卡独立测试：FSDP2 → vLLM TP=2 权重热更新

Last updated: 2026-09-03

## 这个测试做什么

不启动 co-train、Ray、Producer 或 TQ，只测试实际 FSDP2 模型的权重导出、HTTP/NCCL 传输和 vLLM 加载。

测试会真实执行一次或几次 SGD 更新，但不保存模型、optimizer 或 checkpoint，不修改原模型目录。测试结束后内存中的更新权重随进程释放。

| 物理 GPU | 进程 | 用途 |
|---|---|---|
| 0、1 | 一个 vLLM 服务的两个 TP workers | 接收并加载权重，执行推理 |
| 2、3 | torchrun 的两个 FSDP2 ranks | 加载分片模型、训练、导出参数 |

使用有空闲显存的四张 NVIDIA GPU。当前代码不是 NPU/HCCL 测试。先选普通 dense Qwen3-0.6B、Qwen3-4B 或 Llama 类本地模型；不要先用 MoE、量化、LoRA 或多模态模型。该脚本的模型需具有 `model.model.layers`。

## 0. 准备代码和环境

把当前分支的代码同步到服务器。教程假设仓库在 `/model/xyr/verl-SpeCo-ls`，模型在 `/nas/disk1/Qwen3-4B`；按实际位置修改。

激活服务器现有的、能运行 vLLM 0.23.0 的 CUDA Python 环境。两个终端使用同一个环境。

```bash
cd /model/xyr/verl-SpeCo-ls
nvidia-smi
python -c 'import torch; from importlib.metadata import version; print("torch:", torch.__version__, "CUDA:", torch.version.cuda, "vllm:", version("vllm")); assert torch.cuda.is_available(), "CUDA unavailable"'
```

不要为此随意升级 PyTorch。已核对的 vLLM 0.23.0 包要求 `torch==2.11.0`，实际 CUDA wheel 还需要与服务器驱动兼容。报驱动过旧、CUDA unavailable 或动态库错误时先修环境，尚未进入本测试逻辑。

测试脚本会自动优先导入当前仓库，不必重新 `pip install verl-speco`；运行时会打印 `SOURCE=.../verl_speco/integration/external_vllm_weight_sync.py`，确认没有使用旧安装包。

## 1. 终端 A：启动 vLLM，使用卡 0、1

确认这两个 GPU 没有被其他训练占用，并确认端口 8000 空闲。

```bash
cd /model/xyr/verl-SpeCo-ls
MODEL_PATH=/nas/disk1/Qwen3-4B

CUDA_VISIBLE_DEVICES=0,1 VLLM_SERVER_DEV_MODE=1 \
vllm serve "$MODEL_PATH" \
  --host 127.0.0.1 \
  --port 8000 \
  --served-model-name weight-sync-smoke \
  --tensor-parallel-size 2 \
  --dtype bfloat16 \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.5 \
  --enforce-eager \
  --generation-config vllm \
  --no-enable-prefix-caching \
  --load-format dummy \
  --weight-transfer-config '{"backend":"nccl"}'
```

保持这个终端运行，等服务就绪。

- `VLLM_SERVER_DEV_MODE=1`：启用权重管理接口。只在可信环境使用，这里仅监听本机回环地址。
- `TP=2`：vLLM 内部把目标模型按 Tensor Parallel 分到两张卡。
- `--load-format dummy`：不先加载 checkpoint 权重，首次有效推理依赖发送端同步成功。不要在首次同步前用输出判断模型质量。
- `--generation-config vllm`：避免模型目录的采样默认值影响概率对比。
- 不需要 hidden-state connector，也不需要启动草稿模型，这一步隔离验证目标模型权重传输。

本机四卡测试使用 `127.0.0.1`。如果两端不在同一台服务器，需要另外配置监听地址、网络和 actor rank 0 的可达地址，本教程不覆盖跨机部署。

## 2. 终端 B：先检查管理接口

```bash
curl --noproxy '*' -fsS http://127.0.0.1:8000/get_world_size
```

预期：

```json
{"world_size":2}
```

返回 404 通常表示没有启用 dev API，或者访问了错误的 vLLM 版本/端口。连接拒绝说明服务尚未起来。不是 2 则先核对 vLLM 的 TP/PP/DP 配置。

这个接口只是读取服务配置，不会执行推理或修改权重。

## 3. 终端 B：用卡 2、3 跑 FSDP2 测试

```bash
cd /model/xyr/verl-SpeCo-ls
MODEL_PATH=/nas/disk1/Qwen3-4B

CUDA_VISIBLE_DEVICES=2,3 \
python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=2 \
  tools/fsdp2_vllm_weight_sync_smoke.py \
  --model "$MODEL_PATH" \
  --endpoint http://127.0.0.1:8000/v1 \
  --served-model-name weight-sync-smoke \
  --bucket-size-mb 256 \
  --train-steps 1 \
  --lr 0.01 \
  --logprob-atol 0.1
```

不需要 `ray start`。

参数含义：

- `--model`：必须与 vLLM 是同一架构、同一份模型；脚本只从本地目录加载，不下载。
- `--endpoint`：带 `/v1` 的推理地址。生产代码会自动去掉 `/v1` 来访问管理接口。
- `--served-model-name`：与终端 A 一致，是 HTTP 请求里的模型名，不是文件路径。
- `--bucket-size-mb`：沿用实际发送代码的分批目标大小；超大单 Tensor 会独占一批并扩展 packed buffer。
- `--train-steps`：首次对比后真实训练几步，默认 1。
- `--lr`：本测试的临时 SGD 学习率，不是正式训练的推荐学习率。
- `--logprob-atol`：不同推理实现下，同一 token 的 logprob 允许的绝对误差。
- `--no-packed`：可选，用逐 Tensor 广播替代默认 packed。不是绕过 NCCL。
- `--timeout`：HTTP 超时秒数，默认 600；NCCL 的底层故障可能仍需要等待其自身超时。

模型首先在每个进程的 CPU 内存中加载完整 checkpoint，再按 transformer layer 进行 FSDP2 分片并放到设备。因此 CPU 内存也要足够，不适合直接从超大模型开始。

## 4. 应看到哪些日志

```text
SOURCE=.../verl_speco/integration/external_vllm_weight_sync.py
FSDP_READY rank=0 global_shape=(...) local_shape=(...)
FSDP_READY rank=1 global_shape=(...) local_shape=(...)
CONNECT_OK
INITIAL_SYNC_OK
COMPARE prompt=0 hf_top1=... vllm_top1=... max_logprob_error=...
...
INITIAL_COMPARE_OK
TRAIN_STEP_OK step=1 loss=...
UPDATED_SYNC_OK
COMPARE prompt=0 hf_top1=... vllm_top1=... max_logprob_error=...
...
CHECKED_TOKEN_REFERENCE_CHANGE=...
UPDATED_COMPARE_OK
PASS: initial and trained FSDP2 weights match external vLLM; no checkpoint written
```

`FSDP_READY` 会同时打印全局参数形状与本 rank 的分片形状，确认不是两个完整的普通模型。

脚本用三个固定 prompt 的 token IDs，分别对比 FSDP2 模型和 vLLM 的下一 token top-5 logprob。不会用“生成的文字是否一样”作为唯一依据，也不会只凭 HTTP 200 判成功。浮点差异可能让非常接近的 top-1 token 换位，脚本同时检查其概率差距。

第二轮还检查被比较 token 的参考概率是否相对训练前充分变化。如果变化不足以排除旧权重，打印 `INCONCLUSIVE` 并退出失败，而不声称热更新已验证。遇到这种情况：在终端 A 停止并重新启动测试服务，再把终端 B 改为 `--train-steps 3 --lr 0.05` 重试。不建议直接放宽误差掩盖问题。

## 5. 结束与排查

发送端正常完成后会自行关闭 NCCL sender、HTTP Session 和 FSDP2 进程组，不保存文件。vLLM 是你单独启动的，因此到终端 A 按 Ctrl+C 关闭。

发送失败后不要继续使用这个测试 vLLM：部分权重可能已更新，服务可能保持暂停。先停止发送端，再停止并重新启动终端 A 的测试服务，然后重试。不要使用 `pkill -f ray` 或其他会误杀无关任务的命令。

常见定位：

- 没到 `CONNECT_OK`：检查 dev API、两端网络可达性和 NCCL 握手。传输 rendezvous 默认使用发送 rank 0 的自动检测 IP，可用 `--master-address <该机器可达IP>` 指定。
- 卡在 `INITIAL_SYNC`：结合发送端和 vLLM 日志，检查 `update_weights` 错误、NCCL 错误及显存。发送端两张卡都必须参与 FSDP2 导出。
- 首次 `COMPARE` 超阈值：核对模型、dtype、是否量化、参数名、两端包版本；先不要提高阈值。
- 初次正确、训练后不正确：重点检查后续权重更新和旧缓存。发送代码在 pause 时清缓存，本教程还禁用了 prefix caching。
- OOM：先换小模型，或减少 bucket 大小。普通模型加载、FSDP2 导出、vLLM TP 权重和 packed 接收缓冲都有显存需求，`bucket_size_mb` 不是总显存上限。

## 6. 与正式 co-train 的关系

这份脚本复用的是现有 `initialize_worker_weight_sync()`、`update_worker_weights()`、`close_worker_weight_sync()` 和 `ExternalVllmWeightSender`，不是另写一套 NCCL。

只有测试引擎适配部分不同：脚本用原生 FSDP2 的 `state_dict()` / `DTensor.full_tensor()`；正式 co-train 使用实际 verl engine 的 `get_per_tensor_param()`。因此测试通过能验证真实分片汇集和传输，但不能证明 MoE 参数转换、LoRA、offload、Ray 调度或整个 co-train 都已经通过。

两个 FSDP ranks 都参与汇集；只有 FSDP rank 0 发给两个 vLLM TP workers。FSDP 训练通信组为 2 个进程，外部权重传输组为 3 个进程（sender + 两个 TP workers），不是 4。

本机未运行此 GPU/vLLM 测试，脚本和步骤供服务器手动验证。
