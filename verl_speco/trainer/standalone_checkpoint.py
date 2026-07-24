"""Standalone drafter checkpoint runtime config helpers."""

from __future__ import annotations

from copy import deepcopy
import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def _ensure_dict_child(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if isinstance(value, dict):
        return value
    value = {}
    config[key] = value
    return value


def _source_drafter_model_path(trainer: Any) -> str | None:
    model_path = getattr(getattr(getattr(trainer, "config", None), "rollout", None), "drafter", None)
    model_path = getattr(model_path, "model_path", None)
    if not model_path:
        return None
    return os.fspath(model_path)


def _target_model_path(trainer: Any) -> str | None:
    model_path = getattr(getattr(trainer, "config", None), "model", None)
    model_path = getattr(model_path, "path", None)
    if not model_path:
        return None
    return os.fspath(model_path)


def _load_source_drafter_config(trainer: Any) -> dict[str, Any] | None:
    model_path = _source_drafter_model_path(trainer)
    if not model_path:
        return None
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        return None
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to load source drafter config %s: %s", config_path, exc)
        return None
    return loaded if isinstance(loaded, dict) else None


def _load_target_runtime_config(trainer: Any) -> dict[str, Any] | None:
    model_path = _target_model_path(trainer)
    if not model_path:
        return None
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        return None
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to load target model config %s: %s", config_path, exc)
        return None
    return loaded if isinstance(loaded, dict) else None


def _target_runtime_model_type(target_config: dict[str, Any] | None) -> str | None:
    if not isinstance(target_config, dict) or target_config.get("model_type") is None:
        return None
    return str(target_config["model_type"])


def _fill_if_missing(dst: dict[str, Any], src: dict[str, Any], keys: tuple[str, ...]) -> None:
    for key in keys:
        if key in src and key not in dst:
            dst[key] = deepcopy(src[key])


def _copy_if_present(dst: dict[str, Any], src: dict[str, Any] | None, keys: tuple[str, ...]) -> None:
    if src is None:
        return
    for key in keys:
        if key in src:
            dst[key] = deepcopy(src[key])


def _drop_keys(config: dict[str, Any], keys: tuple[str, ...]) -> None:
    for key in keys:
        config.pop(key, None)


def _target_rope_parameters(target_config: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(target_config, dict):
        return None
    rope_parameters = target_config.get("rope_parameters")
    if isinstance(rope_parameters, dict):
        return deepcopy(rope_parameters)
    rope_theta = target_config.get("rope_theta")
    if rope_theta is not None:
        return {"rope_theta": rope_theta, "rope_type": "default"}
    return None


def _dspark_runtime_architecture(model_type: Any) -> str | None:
    normalized = str(model_type or "").lower()
    if normalized.startswith("qwen3"):
        return "Qwen3DSparkModel"
    if normalized.startswith("deepseek"):
        return "DeepSeekDSparkModel"
    return None


def _normalize_block_runtime_model_type(
    runtime_config: dict[str, Any],
    target_model_type: str | None,
    backend_type: str,
) -> None:
    training_model_types = {"dflash", "dspark", "qwen3_dspark"}
    model_type = str(runtime_config.get("model_type") or "")
    if model_type in training_model_types and target_model_type is not None:
        runtime_config["model_type"] = target_model_type
        runtime_config.setdefault("draft_model_type", backend_type)
        runtime_config.setdefault("speculative_algorithm", backend_type.upper())


def _normalize_dspark_runtime_architecture(
    runtime_config: dict[str, Any],
    target_model_type: str | None,
) -> None:
    model_type = runtime_config.get("model_type") or target_model_type
    architecture = _dspark_runtime_architecture(model_type)
    if architecture is None:
        architectures = runtime_config.get("architectures") or []
        if architectures == ["DSparkDraftModel"]:
            logger.warning(
                "Standalone DSpark runtime config keeps generic architecture for unsupported model_type=%r",
                model_type,
            )
        return

    architectures = runtime_config.get("architectures") or []
    if architectures != [architecture]:
        runtime_config["architectures"] = [architecture]


def rewrite_standalone_runtime_config(
    trainer: Any,
    checkpoint_path: str,
    completed_future: Any = None,
) -> str | None:
    """Rewrite standalone DFlash/DSpark checkpoints with runtime-facing config.

    Returns the source drafter model path so callers can perform additional
    checkpoint post-processing such as appending lm_head.weight.
    """

    backend_type = getattr(getattr(trainer, "backend", None), "model_type", None)
    if backend_type not in {"dflash", "dspark"}:
        return None

    if completed_future is not None:
        try:
            completed_future.result()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skip standalone runtime config rewrite because checkpoint save failed: %s", exc)
            return None

    config_path = os.path.join(checkpoint_path, "config.json")
    if not os.path.exists(config_path):
        logger.warning("Cannot rewrite standalone runtime config: missing %s", config_path)
        return None

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            training_config = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Cannot rewrite standalone runtime config %s: %s", config_path, exc)
        return None
    if not isinstance(training_config, dict):
        logger.warning("Cannot rewrite standalone runtime config %s: expected object", config_path)
        return None

    training_config_path = os.path.join(checkpoint_path, "speco_training_config.json")
    try:
        with open(training_config_path, "w", encoding="utf-8") as f:
            json.dump(training_config, f, indent=2, sort_keys=True)
    except OSError as exc:
        logger.warning("Failed to write standalone training config copy %s: %s", training_config_path, exc)

    source_model_path = _source_drafter_model_path(trainer)
    source_runtime_config = _load_source_drafter_config(trainer)
    runtime_config = source_runtime_config
    target_runtime_config = _load_target_runtime_config(trainer)
    target_model_type = _target_runtime_model_type(target_runtime_config)
    if runtime_config is None:
        runtime_config = deepcopy(training_config)
        if target_model_type is not None:
            runtime_config["model_type"] = target_model_type
            runtime_config.setdefault("draft_model_type", backend_type)
            runtime_config.setdefault("speculative_algorithm", backend_type.upper())
        else:
            logger.warning(
                "Source drafter config is unavailable and target model_type could not be inferred; "
                "standalone checkpoint keeps SpeCo training config as runtime config"
            )

    _normalize_block_runtime_model_type(runtime_config, target_model_type, backend_type)
    if backend_type == "dspark":
        _normalize_dspark_runtime_architecture(runtime_config, target_model_type)

    runtime_config["speco_training_model_type"] = backend_type
    target_runtime_keys = ("head_dim", "rope_theta", "max_position_embeddings")
    common_alias_keys = ("head_dim", "rope_theta", "target_layer_ids", "mask_token_id", "num_context_layers")
    _fill_if_missing(runtime_config, training_config, common_alias_keys)
    _copy_if_present(runtime_config, target_runtime_config, target_runtime_keys)

    dflash_config = _ensure_dict_child(runtime_config, "dflash_config")
    _fill_if_missing(dflash_config, training_config, common_alias_keys)
    _copy_if_present(dflash_config, target_runtime_config, ("head_dim", "rope_theta"))

    if backend_type == "dspark":
        dspark_config = _ensure_dict_child(runtime_config, "dspark_config")
        _fill_if_missing(
            dspark_config,
            training_config,
            (
                "block_size",
                "head_dim",
                "rope_theta",
                "num_anchors",
                "markov_rank",
                "markov_head_type",
                "confidence_head_alpha",
                "confidence_head_with_markov",
                "ce_loss_alpha",
                "l1_loss_alpha",
                "loss_decay_gamma",
                "target_layer_ids",
                "num_context_layers",
                "num_target_layers",
                "target_num_hidden_layers",
                "mask_token_id",
            ),
        )
        _copy_if_present(dspark_config, target_runtime_config, ("head_dim", "rope_theta"))
    else:
        dspark_config = {}

    source_has_rope_theta = isinstance(source_runtime_config, dict) and "rope_theta" in source_runtime_config
    if backend_type == "dspark" and str(target_model_type or "").lower().startswith("qwen3") and not source_has_rope_theta:
        _drop_keys(runtime_config, ("rope_theta",))
        _drop_keys(dflash_config, ("rope_theta",))
        _drop_keys(dspark_config, ("rope_theta",))
        if "rope_parameters" not in runtime_config:
            rope_parameters = _target_rope_parameters(target_runtime_config)
            if rope_parameters is not None:
                runtime_config["rope_parameters"] = rope_parameters

    target_layer_ids = (
        runtime_config.get("target_layer_ids")
        or dflash_config.get("target_layer_ids")
        or dspark_config.get("target_layer_ids")
    )
    if target_layer_ids is not None and "eagle_aux_hidden_state_layer_ids" not in runtime_config:
        try:
            runtime_config["eagle_aux_hidden_state_layer_ids"] = [int(layer_id) + 1 for layer_id in target_layer_ids]
        except (TypeError, ValueError):
            logger.warning("Invalid target_layer_ids in standalone exported config: %r", target_layer_ids)

    try:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(runtime_config, f, indent=2, sort_keys=True)
            f.write("\n")
    except OSError as exc:
        logger.warning("Failed to write standalone runtime config %s: %s", config_path, exc)
        return None

    return source_model_path
