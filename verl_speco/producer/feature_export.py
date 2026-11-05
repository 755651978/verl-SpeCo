# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
"""Bounded background export of already computed, CPU-bound features.

This component never calls a model. The caller owns device synchronization and
supplies a completion callback plus the objects whose lifetime it protects.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import logging
import os
import threading
import time
import uuid

logger = logging.getLogger(__name__)


class FeatureExportQueue:
    def __init__(self, *, max_bytes=256 << 20, max_pending=16, workers=2):
        self.max_bytes = max_bytes
        self.max_pending = max_pending
        self._bytes = 0
        self._pending = 0
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="speco-feature-export"
        )

    def reserve(self, size):
        with self._lock:
            if (
                size < 0
                or self._bytes + size > self.max_bytes
                or self._pending >= self.max_pending
            ):
                return False
            self._bytes += size
            self._pending += 1
            return True

    def release(self, size):
        with self._lock:
            self._bytes -= size
            self._pending -= 1

    def submit_reserved(
        self, directory, payload, *, size, wait=lambda: None, owners=()
    ):
        """Transfer a reservation and all buffer ownership to the worker.

        Caller releases the reservation only if submission raises. The pending
        marker exists before control returns to the engine/frontend.
        """
        marker = Path(directory) / f"{uuid.uuid4().hex}.pending"
        marker.touch()
        try:
            self._executor.submit(self._write, marker, payload, size, wait, owners)
        except BaseException:
            marker.unlink(missing_ok=True)
            raise

    def _write(self, marker, payload, size, wait, owners):
        temporary = marker.with_suffix(".tmp")
        try:
            import torch

            wait()
            if not (marker.parent / "live").exists():
                return
            torch.save(payload, temporary)
            if (marker.parent / "live").exists():
                os.replace(temporary, marker.with_suffix(".pt"))
        except Exception:
            logger.exception("Feature export failed")
            try:
                if (marker.parent / "live").exists():
                    (marker.parent / "error").touch()
            except OSError:
                logger.warning("Could not persist feature export error")
        finally:
            # Keep device snapshots and pinned destinations alive through wait
            # and serialization, including cancellation and error paths.
            try:
                del owners
                temporary.unlink(missing_ok=True)
                marker.unlink(missing_ok=True)
                if not (marker.parent / "live").exists():
                    try:
                        marker.parent.rmdir()
                    except OSError:
                        pass
            finally:
                self.release(size)

    def close(self):
        self._executor.shutdown(wait=True)


def wait_for_exports(directory, *, timeout=120.0):
    deadline = time.monotonic() + timeout
    while any(Path(directory).glob("*.pending")):
        if (Path(directory) / "error").exists() or (
            Path(directory) / "skipped"
        ).exists():
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("Timed out waiting for captured feature export")
        time.sleep(0.005)


def rollout_feature_fields(prompt, response, hidden, positions):
    """Shared SGLang/vLLM sample fields; no cloning or model execution.

    Inputs are unbatched tensors already owned by the caller. CPU conversion is
    a no-op for exporter results and preserves the existing SGLang contract.
    """
    import torch

    if positions is not None:
        if positions.ndim != 1 or positions.numel() != hidden.shape[0]:
            raise ValueError("Hidden positions must match feature rows")
    return {
        "input_ids": torch.cat([prompt, response], dim=0).unsqueeze(0),
        "prompts": prompt.unsqueeze(0),
        "responses": response.unsqueeze(0),
        "hidden_states": hidden.unsqueeze(0).cpu(),
        "hidden_positions": positions.unsqueeze(0).cpu()
        if positions is not None
        else None,
    }
