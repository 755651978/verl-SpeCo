# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Reconstruct standalone draft-training features from compact token samples."""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, cast

import torch
from torch import nn

from verl_speco.integration.oldlogprob_layer_ids import (
    resolve_oldlogprob_aux_layer_ids,
)
from verl_speco.trainer.feature_store import DraftFeatureSample, DraftReplaySample

logger = logging.getLogger(__name__)


def _config_value(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _parse_dtype(value: Any) -> torch.dtype:
    normalized = str(value or "bfloat16").strip().lower()
    dtypes = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if normalized not in dtypes:
        raise ValueError(
            f"Unsupported target feature replay dtype {value!r}; "
            "expected bfloat16, float16, or float32"
        )
    return dtypes[normalized]


def _tensor_from_module_output(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return cast(torch.Tensor, value)
    if isinstance(value, (tuple, list)) and value and torch.is_tensor(value[0]):
        return cast(torch.Tensor, value[0])
    last_hidden_state = getattr(value, "last_hidden_state", None)
    if torch.is_tensor(last_hidden_state):
        return cast(torch.Tensor, last_hidden_state)
    raise TypeError(
        f"Target feature replay expected tensor-like module output, got {type(value)!r}"
    )


def _get_module_by_path(root: Any, path: str) -> Any:
    current = root
    for part in path.split("."):
        if not part:
            continue
        current = getattr(current, part, None)
        if current is None:
            return None
    return current


def _find_layers_and_final_norm(model: nn.Module) -> tuple[list[nn.Module], nn.Module]:
    roots: list[nn.Module] = [model]
    base_model = getattr(model, "base_model", None)
    if isinstance(base_model, nn.Module) and base_model is not model:
        roots.append(base_model)

    candidates = (
        ("model.layers", "model.norm"),
        ("base_model.model.layers", "base_model.model.norm"),
        ("model.decoder.layers", "model.decoder.final_layer_norm"),
        ("transformer.h", "transformer.ln_f"),
        ("gpt_neox.layers", "gpt_neox.final_layer_norm"),
    )
    for root in roots:
        for layers_path, norm_path in candidates:
            layers = _get_module_by_path(root, layers_path)
            norm = _get_module_by_path(root, norm_path)
            if (
                isinstance(layers, (nn.ModuleList, list, tuple))
                and len(layers) > 0
                and isinstance(norm, nn.Module)
            ):
                return list(layers), norm

        for name, child in root.named_modules():
            if not isinstance(child, nn.ModuleList) or len(child) <= 0:
                continue
            if not (name.endswith("layers") or name.endswith("h")):
                continue
            parent_path = name.rsplit(".", 1)[0] if "." in name else ""
            for norm_name in ("norm", "final_layer_norm", "ln_f"):
                norm_path = f"{parent_path}.{norm_name}" if parent_path else norm_name
                norm = _get_module_by_path(root, norm_path)
                if isinstance(norm, nn.Module):
                    return list(child), norm

    raise RuntimeError(
        "Target feature replay could not find transformer layers and final norm"
    )


def _hidden_capture_target(layer_id: int, num_layers: int) -> tuple[str, int | None]:
    hidden_state_index = (
        int(layer_id) + 1 if int(layer_id) >= 0 else num_layers + 1 + int(layer_id)
    )
    if hidden_state_index <= 0 or hidden_state_index > num_layers:
        raise IndexError(
            f"Target replay layer id {layer_id} resolved to hidden-state index "
            f"{hidden_state_index}, but the model has {num_layers} layers"
        )
    if hidden_state_index == num_layers:
        return "final", None
    return "layer", hidden_state_index - 1


def _load_json_config(path: Any) -> dict[str, Any] | None:
    if not path:
        return None
    config_path = os.path.join(os.fspath(path), "config.json")
    try:
        with open(config_path, encoding="utf-8") as config_file:
            value = json.load(config_file)
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


class BoundedReplayCache:
    """Per-rank disk cache with a hard least-recently-used size budget."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        max_size_gb: float,
        rank: int,
        world_size: int,
    ):
        global_max_bytes = max(int(float(max_size_gb) * 1024**3), 0)
        self.max_bytes = global_max_bytes // max(int(world_size), 1)
        self.path = Path(path) / f"rank{int(rank):05d}"
        self.path.mkdir(parents=True, exist_ok=True)
        self._entries: dict[Path, tuple[int, float]] = {}
        self._total_bytes = 0
        self._scan()

    @property
    def enabled(self) -> bool:
        return self.max_bytes > 0

    def _scan(self) -> None:
        self._entries = {}
        self._total_bytes = 0
        for path in self.path.glob("*.pt"):
            try:
                stat = path.stat()
            except OSError:
                continue
            size = int(stat.st_size)
            self._entries[path] = (size, float(stat.st_mtime))
            self._total_bytes += size

    def get(self, key: str) -> DraftFeatureSample | None:
        if not self.enabled:
            return None
        path = self.path / f"{key}.pt"
        if not path.exists():
            return None
        try:
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                payload = torch.load(path, map_location="cpu")
            sample = DraftFeatureSample.from_dict(payload, strict=True)
            now = time.time()
            os.utime(path, (now, now))
            size = int(path.stat().st_size)
            self._entries[path] = (size, now)
            return sample
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Discard invalid target replay cache entry %s: %s", path, exc
            )
            self._forget(path)
            try:
                path.unlink()
            except OSError:
                pass
            return None

    def put(self, key: str, sample: DraftFeatureSample) -> bool:
        if not self.enabled:
            return False
        path = self.path / f"{key}.pt"
        if path.exists():
            return True
        with tempfile.NamedTemporaryFile(
            prefix=path.name,
            suffix=".tmp",
            dir=self.path,
            delete=False,
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)
        try:
            torch.save(sample.to_dict(), tmp_path)
            size = int(tmp_path.stat().st_size)
            if size > self.max_bytes:
                return False
            self._evict_until_fits(size)
            os.replace(tmp_path, path)
            now = time.time()
            self._entries[path] = (size, now)
            self._total_bytes += size
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to write target replay cache entry %s: %s", path, exc
            )
            return False
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

    def _evict_until_fits(self, incoming_size: int) -> None:
        entries = sorted(self._entries.items(), key=lambda item: item[1][1])
        for path, _ in entries:
            if self._total_bytes + incoming_size <= self.max_bytes:
                break
            try:
                path.unlink()
            except OSError:
                continue
            self._forget(path)

    def _forget(self, path: Path) -> None:
        previous = self._entries.pop(path, None)
        if previous is not None:
            self._total_bytes = max(self._total_bytes - int(previous[0]), 0)

    def metrics(self) -> dict[str, float]:
        return {
            "replay/cache_size_gb": self._total_bytes / float(1024**3),
            "replay/cache_budget_gb_per_rank": self.max_bytes / float(1024**3),
        }


class TargetFeatureReplayer:
    """Materialize target hidden states only for standalone token replay."""

    def __init__(
        self,
        config: Any,
        *,
        rank: int,
        world_size: int,
        device: torch.device,
    ):
        self.config = config
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.device = torch.device(device)
        self.draft_config = config.actor_rollout_ref
        self.drafter_cfg = self.draft_config.rollout.drafter
        self.training_cfg = self.drafter_cfg.training
        self.replay_cfg = self.training_cfg.get("target_feature_replay", {}) or {}
        configured_model_path = _config_value(self.replay_cfg, "model_path", None)
        model_path = configured_model_path or self.draft_config.model.path
        if not model_path:
            raise ValueError(
                "Token replay requires target_feature_replay.model_path or "
                "actor_rollout_ref.model.path"
            )
        self.model_path = os.fspath(model_path)
        self.target_revision = str(
            _config_value(self.replay_cfg, "target_revision", None) or self.model_path
        )
        self.dtype = _parse_dtype(_config_value(self.replay_cfg, "dtype", "bfloat16"))
        self.trust_remote_code = bool(
            _config_value(self.replay_cfg, "trust_remote_code", False)
        )
        self.strict_target_model_path = bool(
            _config_value(self.replay_cfg, "strict_target_model_path", False)
        )
        self.algorithm = str(self.drafter_cfg.speculative_algorithm).upper()
        if self.algorithm not in {"EAGLE3", "DFLASH", "DSPARK"}:
            raise ValueError(
                f"Token replay does not support drafter algorithm {self.algorithm!r}"
            )
        self.use_logits = bool(self.training_cfg.get("use_logits", False))
        self.logits_topk = int(self.training_cfg.get("logits_topk", 128) or 128)
        self.logits_chunk_rows = max(
            int(_config_value(self.replay_cfg, "logits_chunk_rows", 32) or 32), 1
        )

        from transformers import AutoConfig

        self.target_config = AutoConfig.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
        )
        self.target_num_hidden_layers = int(
            getattr(
                getattr(self.target_config, "text_config", self.target_config),
                "num_hidden_layers",
            )
        )
        model_configs = [
            value
            for value in (
                _load_json_config(self.drafter_cfg.get("model_path", None)),
                _load_json_config(self.drafter_cfg.get("checkpoint_path", None)),
            )
            if value is not None
        ]
        layer_ids = resolve_oldlogprob_aux_layer_ids(
            self.drafter_cfg,
            target_num_hidden_layers=self.target_num_hidden_layers,
            model_configs=model_configs,
        )
        if not layer_ids:
            raise RuntimeError(
                "Token replay could not resolve target auxiliary layer ids"
            )
        self.target_layer_ids = [int(layer_id) for layer_id in layer_ids]
        dspark_l1_enabled = (
            self.algorithm == "DSPARK"
            and float(self.training_cfg.get("dspark_l1_loss_alpha", 0.9) or 0.0) > 0
        )
        self.hidden_layout = (
            "dflash_aux_plus_last"
            if dspark_l1_enabled
            else "dflash_aux"
            if self.algorithm in {"DFLASH", "DSPARK"}
            else "eagle3_aux_plus_last"
        )
        config_json = json.dumps(
            self.target_config.to_dict(), sort_keys=True, default=str
        ).encode()
        self.target_config_fingerprint = hashlib.sha256(config_json).hexdigest()

        self.cache: BoundedReplayCache | None = None
        cache_cfg = _config_value(self.replay_cfg, "cache", {}) or {}
        if bool(_config_value(cache_cfg, "enabled", False)):
            cache_path = _config_value(cache_cfg, "path", None)
            if not cache_path:
                feature_path = os.fspath(self.training_cfg.feature_store.path)
                cache_path = f"{feature_path}.hidden_cache"
            self.cache = BoundedReplayCache(
                cache_path,
                max_size_gb=float(_config_value(cache_cfg, "max_size_gb", 0.0) or 0.0),
                rank=self.rank,
                world_size=self.world_size,
            )

        self.model: nn.Module | None = None
        self.layers: list[nn.Module] = []
        self.final_norm: nn.Module | None = None
        self.backbone: nn.Module | None = None
        self.output_embedding: nn.Module | None = None
        self.cache_hits = 0
        self.cache_misses = 0
        self.materialized_samples = 0
        self.target_forward_seconds = 0.0

    def materialize(
        self, samples: Iterable[DraftReplaySample | DraftFeatureSample]
    ) -> list[DraftFeatureSample]:
        materialized: list[DraftFeatureSample] = []
        for sample in samples:
            if isinstance(sample, DraftFeatureSample):
                materialized.append(sample)
                continue
            if not isinstance(sample, DraftReplaySample):
                raise TypeError(
                    f"Target feature replay expected DraftReplaySample, got {type(sample)!r}"
                )
            self._validate_target_path(sample)
            key = self._cache_key(sample)
            cached = self.cache.get(key) if self.cache is not None else None
            if cached is not None:
                self.cache_hits += 1
                materialized.append(cached)
                continue
            self.cache_misses += 1
            replayed = self._materialize_one(sample)
            if self.cache is not None:
                self.cache.put(key, replayed)
            materialized.append(replayed)
        self.materialized_samples += len(materialized)
        return materialized

    def _validate_target_path(self, sample: DraftReplaySample) -> None:
        if sample.algorithm.upper() != self.algorithm:
            raise ValueError(
                "Token replay algorithm mismatch: "
                f"sample={sample.algorithm!r} training={self.algorithm!r}"
            )
        collected_layer_ids = sample.metadata.get("target_layer_ids")
        if collected_layer_ids is not None:
            normalized_layer_ids = (
                [int(collected_layer_ids)]
                if isinstance(collected_layer_ids, int)
                else [int(value) for value in collected_layer_ids]
            )
            if normalized_layer_ids != self.target_layer_ids:
                raise ValueError(
                    "Token replay target layer mismatch: "
                    f"collected={normalized_layer_ids} "
                    f"replay={self.target_layer_ids}"
                )
        collected_layout = sample.metadata.get("hidden_states_layout")
        if collected_layout and str(collected_layout) != self.hidden_layout:
            raise ValueError(
                "Token replay hidden layout mismatch: "
                f"collected={collected_layout!r} replay={self.hidden_layout!r}"
            )
        if not self.strict_target_model_path:
            return
        collected_path = sample.metadata.get("target_model_path")
        if collected_path and os.path.normpath(
            os.fspath(collected_path)
        ) != os.path.normpath(self.model_path):
            raise ValueError(
                "Token replay target model path mismatch: "
                f"collected={collected_path!r} replay={self.model_path!r}"
            )

    def _cache_key(self, sample: DraftReplaySample) -> str:
        digest = hashlib.sha256()
        contract = {
            "target_revision": self.target_revision,
            "target_config": self.target_config_fingerprint,
            "algorithm": self.algorithm,
            "target_layer_ids": self.target_layer_ids,
            "hidden_layout": self.hidden_layout,
            "dtype": str(self.dtype),
            "use_logits": self.use_logits,
            "logits_topk": self.logits_topk,
        }
        digest.update(json.dumps(contract, sort_keys=True).encode())
        for tensor in (
            sample.input_ids,
            sample.attention_mask,
            sample.position_ids,
            sample.feature_positions,
            sample.draft_position_ids,
            sample.loss_mask,
        ):
            contiguous = tensor.detach().cpu().contiguous()
            digest.update(str(contiguous.dtype).encode())
            digest.update(str(tuple(contiguous.shape)).encode())
            digest.update(contiguous.numpy().tobytes())
        return digest.hexdigest()

    def _ensure_model(self) -> None:
        if self.model is not None:
            return
        from transformers import AutoModelForCausalLM

        logger.warning(
            "Loading frozen target model for standalone token replay: path=%s dtype=%s device=%s",
            self.model_path,
            self.dtype,
            self.device,
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
            trust_remote_code=self.trust_remote_code,
            low_cpu_mem_usage=True,
        )
        model.eval()
        model.requires_grad_(False)
        model.to(self.device)
        self.layers, self.final_norm = _find_layers_and_final_norm(model)
        base_model_prefix = str(getattr(model, "base_model_prefix", "") or "")
        backbone = (
            getattr(model, base_model_prefix, None) if base_model_prefix else None
        )
        self.backbone = backbone if isinstance(backbone, nn.Module) else model
        output_embedding = model.get_output_embeddings()
        self.output_embedding = (
            output_embedding if isinstance(output_embedding, nn.Module) else None
        )
        self.model = model

    def _materialize_one(self, sample: DraftReplaySample) -> DraftFeatureSample:
        self._ensure_model()
        assert self.backbone is not None
        assert self.final_norm is not None

        feature_positions = sample.feature_positions.detach().cpu().long()
        feature_end = int(feature_positions[-1].item()) + 1
        input_ids = sample.input_ids[:feature_end].to(
            self.device, dtype=torch.long, non_blocking=True
        )
        attention_mask = sample.attention_mask[:feature_end].to(
            self.device, dtype=torch.long, non_blocking=True
        )
        position_ids = sample.position_ids[:feature_end].to(
            self.device, dtype=torch.long, non_blocking=True
        )
        captures: dict[str, torch.Tensor] = {}
        handles = []

        def capture(key: str):
            def hook(_module, _inputs, output):
                captures[key] = _tensor_from_module_output(output)

            return hook

        aux_keys: list[str] = []
        modules: dict[str, nn.Module] = {}
        for layer_id in self.target_layer_ids:
            kind, layer_index = _hidden_capture_target(
                layer_id, self.target_num_hidden_layers
            )
            if kind == "final":
                key = "final"
                module = self.final_norm
            else:
                assert layer_index is not None
                key = f"layer:{layer_index}"
                module = self.layers[layer_index]
            aux_keys.append(key)
            modules[key] = module
        include_final = self.hidden_layout in {
            "eagle3_aux_plus_last",
            "dflash_aux_plus_last",
        }
        need_final = include_final or (self.algorithm == "EAGLE3" and self.use_logits)
        if need_final:
            modules["final"] = self.final_norm
        for key, module in modules.items():
            handles.append(module.register_forward_hook(capture(key)))

        started = time.perf_counter()
        try:
            forward_kwargs = {
                "input_ids": input_ids.unsqueeze(0),
                "attention_mask": attention_mask.unsqueeze(0),
                "position_ids": position_ids.unsqueeze(0),
                "use_cache": False,
                "return_dict": True,
            }
            forward_kwargs = _supported_forward_kwargs(
                self.backbone.forward, forward_kwargs
            )
            with torch.inference_mode():
                self.backbone(**forward_kwargs)
        finally:
            for handle in handles:
                handle.remove()
        self.target_forward_seconds += time.perf_counter() - started

        required_keys = list(aux_keys)
        if need_final:
            required_keys.append("final")
        missing = [key for key in required_keys if key not in captures]
        if missing:
            raise RuntimeError(
                f"Target feature replay missed hidden-state captures: {missing}"
            )

        device_positions = feature_positions.to(self.device)
        hidden_parts = [
            captures[key].squeeze(0).index_select(0, device_positions)
            for key in aux_keys
        ]
        selected_final = (
            captures["final"].squeeze(0).index_select(0, device_positions)
            if need_final
            else None
        )
        if include_final:
            assert selected_final is not None
            hidden_parts.append(selected_final)
        hidden_states = torch.cat(hidden_parts, dim=-1).to(
            device="cpu", dtype=self.dtype
        )

        target_logprobs = None
        if self.algorithm == "EAGLE3" and self.use_logits:
            assert selected_final is not None
            target_logprobs = self._build_sparse_target_logprobs(selected_final[:-1])

        selected_input_ids = sample.input_ids.index_select(0, feature_positions).long()
        selected_loss_mask = sample.loss_mask.index_select(0, feature_positions).float()
        metadata = dict(sample.metadata)
        feature_start = int(feature_positions[0].item())
        feature_end = int(feature_positions[-1].item()) + 1
        metadata.update(
            {
                "source": "token_replay",
                "target_model_path": self.model_path,
                "target_revision": self.target_revision,
                "target_config_fingerprint": self.target_config_fingerprint,
                "target_layer_ids": list(self.target_layer_ids),
                "hidden_states_layout": self.hidden_layout,
                "feature_start": feature_start,
                "feature_end": feature_end,
                "hidden_position_start": feature_start,
                "hidden_position_end": feature_end,
                "hidden_positions": feature_positions,
                "sequence_length": int(selected_input_ids.numel()),
                "full_sequence_length": int(sample.input_ids.numel()),
                "use_logits": self.use_logits,
            }
        )
        if target_logprobs is not None:
            metadata["target_logprobs_position_start"] = feature_start + 1
            metadata["target_logprobs_position_end"] = feature_end

        return DraftFeatureSample(
            algorithm=self.algorithm,
            input_ids=selected_input_ids,
            loss_mask=selected_loss_mask,
            hidden_states=hidden_states,
            target_logprobs=target_logprobs,
            position_ids=sample.draft_position_ids.long(),
            metadata=metadata,
        )

    def _build_sparse_target_logprobs(
        self, final_hidden_states: torch.Tensor
    ) -> torch.Tensor:
        if self.output_embedding is None:
            raise RuntimeError(
                "EAGLE3 token replay with use_logits=true requires target output embeddings"
            )
        rows: list[torch.Tensor] = []
        topk = max(self.logits_topk, 1)
        with torch.inference_mode():
            for start in range(
                0, int(final_hidden_states.size(0)), self.logits_chunk_rows
            ):
                hidden = final_hidden_states[start : start + self.logits_chunk_rows]
                logits = self.output_embedding(hidden).float()
                local_topk = min(topk, int(logits.size(-1)))
                values, ids = logits.topk(local_topk, dim=-1)
                values = values - torch.logsumexp(logits, dim=-1, keepdim=True)
                rows.append(
                    torch.stack((values, ids.to(dtype=values.dtype)), dim=-1).cpu()
                )
        if not rows:
            return torch.empty(0, topk, 2, dtype=torch.float32)
        return torch.cat(rows, dim=0).contiguous()

    def metrics(self) -> dict[str, float]:
        metrics = {
            "replay/cache_hits_total": float(self.cache_hits),
            "replay/cache_misses_total": float(self.cache_misses),
            "replay/materialized_samples_total": float(self.materialized_samples),
            "replay/target_forward_time_total": float(self.target_forward_seconds),
        }
        total = self.cache_hits + self.cache_misses
        if total > 0:
            metrics["replay/cache_hit_ratio"] = self.cache_hits / float(total)
        if self.cache is not None:
            metrics.update(self.cache.metrics())
        return metrics

    def close(self) -> None:
        if self.model is None:
            return
        try:
            self.model.to("cpu")
        except Exception:  # noqa: BLE001
            pass
        self.model = None
        self.layers = []
        self.final_norm = None
        self.backbone = None
        self.output_embedding = None


def _supported_forward_kwargs(forward: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(forward)
    except (TypeError, ValueError):
        return kwargs
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in signature.parameters}
