# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the restore-time compiled-kernel rebuild hook."""

import sys
from unittest import mock

sys.modules.setdefault("dynamo.vllm.args", mock.MagicMock())
sys.modules.setdefault("dynamo.vllm.handlers", mock.MagicMock())
sys.modules.setdefault("dynamo.vllm.worker_factory", mock.MagicMock())

from dynamo.vllm.snapshot import _rebuild_compiled_kernel_state


def test_rebuild_compiled_kernel_state_clears_dynamo_and_inductor(monkeypatch):
    dynamo = mock.MagicMock()
    inductor_utils = mock.MagicMock()
    torch = mock.MagicMock()
    torch._dynamo = dynamo
    torch._inductor.utils = inductor_utils
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch._dynamo", dynamo)
    monkeypatch.setitem(sys.modules, "torch._inductor", torch._inductor)
    monkeypatch.setitem(sys.modules, "torch._inductor.utils", inductor_utils)

    _rebuild_compiled_kernel_state()

    dynamo.reset.assert_called_once_with()
    inductor_utils.clear_caches.assert_called_once_with()
