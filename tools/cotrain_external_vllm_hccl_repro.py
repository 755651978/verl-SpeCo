#!/usr/bin/env python3
"""Run the full co-train Worker initialization path and stop at external HCCL.

This is a dedicated diagnostic entrypoint.  The implementation stays in
``cotrain_external_vllm_connect_smoke`` so the reproducer and its tests cannot
silently diverge.
"""

from __future__ import annotations

import os


os.environ.setdefault("SPECO_CONNECT_SMOKE_PHASE", "before_init_model")

from tools.cotrain_external_vllm_connect_smoke import main  # noqa: E402


if __name__ == "__main__":
    main()
