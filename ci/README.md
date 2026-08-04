# CI layers

The repository uses six workflow layers:

- `cpu_unit_tests.yml`: required PR checks without installing this repository or using accelerator runtimes.
- `gpu_vllm_unit_tests.yml`: scheduled/manual vLLM example-script runs on GPU.
- `gpu_sglang_unit_tests.yml`: scheduled/manual SGLang example-script runs on GPU.
- `gpu_drafter_training_smoke.yml`: scheduled/manual standalone drafter training smoke tests on GPU.
- `npu_vllm_unit_tests.yml`: PR, push, scheduled, and manual vLLM example-script runs on NPU.
- `npu_sglang_unit_tests.yml`: PR, push, scheduled, and manual SGLang example-script runs on NPU.

The hardware workflows require self-hosted runner labels:

- GPU: `self-hosted`, `linux`, `x64`, `gpu`
- NPU: `self-hosted`, `linux-aarch64-a2-8`

The NPU vLLM workflow targets an 8-card NPU runner by default and sets
`SPECO_ACCELERATOR_COUNT=8` unless the GitHub environment variable or manual
workflow input overrides it.

Pull requests, including pull requests from forks, run this smoke matrix:

- vLLM + DSpark
- SGLang + EAGLE3
- SGLang + DFlash

Push, scheduled, and manual NPU runs use the broader matrix:

- vLLM + EAGLE3
- vLLM + DFlash
- vLLM + DSpark
- SGLang + EAGLE3
- SGLang + DFlash

Like verl's CI, the hardware workflows assume the runner image has the runtime
stack and a small default model cache. Most workflows default to:

- `/home/runner/models`
- `/home/runner/models/hf_data`

The default paths are intentionally tiny-runner friendly and can be overridden
with GitHub environment variables or manual workflow inputs.

The NPU vLLM workflow follows verl's Ascend CI layout. Its image provides the
accelerator runtime plus pre-cached model and dataset assets under
`/root/.cache/models`; the workflow links that cache to `~/models`. It checks
out the pull-request revision and runs `pip install --no-deps -e .`, then verifies that
`verl_speco` imports from the current GitHub workspace rather than the image's
preinstalled copy.

The NPU vLLM workflow does not download models or datasets. Before launching the
example script it checks that the selected target model directory, selected
draft model directory, training parquet, and validation parquet exist. Override
the defaults with GitHub environment variables or manual inputs when the image
uses different paths.

The CPU layer uses `PYTHONPATH=$PWD` and checks out the upstream verl commit
from `REQUIRED_VERL.txt`. It runs:

```bash
python -m compileall verl_speco
bash -n examples/*.sh
python -m pytest tests/compat tests/config tests/examples tests/integration -q
```

The vLLM and SGLang hardware workflows call `ci/run_example_test.sh`, which
selects one of the repository-owned scripts in `examples/` and passes CI
variables as Hydra overrides.

The GPU drafter training workflow directly runs the standalone tests under
`tests/special_standalone/` for EAGLE-1, EAGLE-2, Domino, and P-EAGLE. Its
default target model is `/home/runner/models/Qwen/Qwen3-4B`, and it runs three
optimizer steps per matrix entry. Configure these defaults with
`SPECO_TRAINING_SMOKE_TARGET_MODEL`, `SPECO_TRAINING_SMOKE_STEPS`, and
`SPECO_TRAINING_SMOKE_LR`.

Configure CI variables in the `speco-gpu-ci` and `speco-npu-ci` GitHub
environments, or pass them as manual workflow inputs where available:

- `SPECO_MODEL_ROOT`
- `SPECO_DATA_ROOT`
- `SPECO_TARGET_MODEL`
- `SPECO_TRAINING_SMOKE_TARGET_MODEL`
- `SPECO_TRAINING_SMOKE_STEPS`
- `SPECO_TRAINING_SMOKE_LR`
- `SPECO_EAGLE3_DRAFT_MODEL`
- `SPECO_DFLASH_DRAFT_MODEL`
- `SPECO_DSPARK_DRAFT_MODEL`
- `SPECO_TRAIN_FILE`
- `SPECO_TEST_FILE`
- `SPECO_CKPT_DIR`
- `SPECO_ACCELERATOR_COUNT`
- `SPECO_TENSOR_PARALLEL_SIZE`
- `SPECO_SEQUENCE_PARALLEL_SIZE`
- `SPECO_ENABLE_TRAINING`
- `SPECO_SPEC_STEPS`
- `SPECO_SPEC_TOPK`
- `SPECO_SPEC_VERIFY_TOKENS`
- `SPECO_DFLASH_NUM_ANCHORS`
- `SPECO_DFLASH_MAX_WINDOW`
- `SPECO_DSPARK_BLOCK_SIZE`
- `SPECO_DSPARK_SPEC_STEPS`
- `SPECO_DSPARK_SPEC_VERIFY_TOKENS`
- `SPECO_DSPARK_NUM_ANCHORS`
- `SPECO_DSPARK_MAX_WINDOW`
- `SPECO_TOTAL_TRAINING_STEPS`
- `SPECO_TRAIN_MAX_SAMPLES`
- `SPECO_VAL_MAX_SAMPLES`
- `SPECO_DATALOADER_NUM_WORKERS`
- `SPECO_EXTRA_HYDRA_ARGS`

For NPU runs, `ci/run_example_test.sh` generates
`ASCEND_RT_VISIBLE_DEVICES=0,...,N-1` from `SPECO_ACCELERATOR_COUNT` when the
caller has not already set `ASCEND_RT_VISIBLE_DEVICES`. If the caller provides
`ASCEND_RT_VISIBLE_DEVICES`, the script preserves it and checks that
`SPECO_ACCELERATOR_COUNT` does not exceed the visible device count.

PR smoke jobs force lightweight settings through environment variables:

- `SPECO_TOTAL_TRAINING_STEPS=1`
- `SPECO_TRAIN_MAX_SAMPLES=1`
- `SPECO_VAL_MAX_SAMPLES=1`
- `SPECO_DATALOADER_NUM_WORKERS=0`

The runner image is responsible for providing the matching verl, vLLM/SGLang,
PyTorch accelerator runtime, and model files. Hardware workflows deliberately
fail closed when required models or datasets are absent.

Fork pull requests execute repository code on the NPU self-hosted runner.
Keep the `speco-npu-ci` environment free of privileged secrets, require review
before first-time contributors can run workflows, and use isolated or
ephemeral runners where possible.

## Testing CI locally

Run the CPU layer checks from the repository root:

```bash
python -m compileall verl_speco
bash -n examples/*.sh
bash -n ci/run_example_test.sh
python -m pytest tests/compat tests/config tests/examples tests/integration -q
```

On Windows, prefer Git for Windows bash when WSL is not installed:

```powershell
D:\git\bin\bash.exe -n examples/*.sh
D:\git\bin\bash.exe -n ci/run_example_test.sh
python -m pytest tests/compat tests/config tests/examples tests/integration -q
```

You can inspect the selected script and Hydra overrides without launching a
model by setting `SPECO_DRY_RUN=true`:

```bash
SPECO_DRY_RUN=true \
SPECO_TARGET_MODEL=/models/target \
SPECO_DSPARK_DRAFT_MODEL=/models/dspark \
SPECO_TRAIN_FILE=/data/train.parquet \
SPECO_TEST_FILE=/data/test.parquet \
SPECO_CKPT_DIR=/tmp/speco \
bash ci/run_example_test.sh npu vllm dspark
```

To test the hardware workflows on GitHub, open Actions and choose one of:

- `gpu_drafter_training_smoke`
- `gpu_vllm_unit_tests`
- `gpu_sglang_unit_tests`
- `npu_vllm_unit_tests`
- `npu_sglang_unit_tests`

Run the selected workflow without inputs after preparing the default paths
above. Fill the manual inputs only when you want to override the defaults for
one run.
