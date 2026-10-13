"""Runtime selection must be explicit, fixed per process and fail without fallback."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.ai import inference, runtime
from app.ai.contracts.errors import ProcessingError


@pytest.fixture(autouse=True)
def reset_runtime(monkeypatch):
    monkeypatch.setenv("NUTRITION_AI_RUNTIME", "cpu")
    monkeypatch.delenv("SPACES_ZERO_GPU", raising=False)
    runtime.get_runtime.cache_clear()
    runtime.get_device.cache_clear()
    monkeypatch.setattr(inference, "_zero_executor", None)
    yield
    runtime.get_runtime.cache_clear()
    runtime.get_device.cache_clear()


def test_cpu_is_explicit_even_when_cuda_is_available(monkeypatch):
    cuda = Mock(return_value=True)
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=cuda))
    )
    assert runtime.get_device() == "cpu"
    cuda.assert_not_called()


def test_runtime_changes_require_a_process_restart(monkeypatch):
    assert runtime.get_runtime() == "cpu"
    monkeypatch.setenv("NUTRITION_AI_RUNTIME", "cuda")
    assert runtime.get_runtime() == "cpu"


@pytest.mark.parametrize("value", ["", "auto", "gpu"])
def test_invalid_runtime_fails_configuration(monkeypatch, value):
    monkeypatch.setenv("NUTRITION_AI_RUNTIME", value)
    with pytest.raises(ProcessingError) as caught:
        runtime.get_runtime()
    assert caught.value.code == "runtime_configuration_error"


@pytest.mark.parametrize("available", [True, False])
def test_cuda_requires_available_hardware(monkeypatch, available):
    monkeypatch.setenv("NUTRITION_AI_RUNTIME", "cuda")
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: available)),
    )
    if available:
        assert runtime.get_device() == "cuda"
    else:
        with pytest.raises(ProcessingError) as caught:
            runtime.get_device()
        assert caught.value.code == "runtime_configuration_error"


def test_zerogpu_cannot_be_started_as_regular_rest(monkeypatch):
    monkeypatch.setenv("NUTRITION_AI_RUNTIME", "zerogpu")
    with pytest.raises(ProcessingError, match="space_app.py"):
        runtime.initialize_runtime()


def test_zero_requires_hosted_hardware(monkeypatch):
    monkeypatch.setenv("NUTRITION_AI_RUNTIME", "zerogpu")
    with pytest.raises(ProcessingError, match="hardware"):
        runtime.initialize_runtime(allow_zerogpu=True)


def test_persistent_cuda_is_rejected_on_zero_hardware(monkeypatch):
    monkeypatch.setenv("NUTRITION_AI_RUNTIME", "cuda")
    monkeypatch.setenv("SPACES_ZERO_GPU", "true")
    with pytest.raises(ProcessingError, match="persistent cuda"):
        runtime.initialize_runtime()


@pytest.mark.parametrize("version,cuda", [("2.4.1", "12.1"), ("2.8.0+cpu", None)])
def test_zero_rejects_unsupported_torch(monkeypatch, version, cuda):
    monkeypatch.setenv("NUTRITION_AI_RUNTIME", "zerogpu")
    monkeypatch.setenv("SPACES_ZERO_GPU", "true")
    monkeypatch.setenv("SPACES_ZERO_DEVICE_API_URL", "https://scheduler.invalid")
    monkeypatch.setenv("SPACE_ID", "test/space")
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setitem(sys.modules, "spaces", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "spaces.config",
        SimpleNamespace(Config=SimpleNamespace(zero_gpu=True)),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(__version__=version, version=SimpleNamespace(cuda=cuda)),
    )
    with pytest.raises(ProcessingError) as caught:
        runtime.initialize_runtime(allow_zerogpu=True)
    assert caught.value.code == "runtime_configuration_error"


def test_zero_dispatch_never_executes_local_fallback(monkeypatch):
    monkeypatch.setenv("NUTRITION_AI_RUNTIME", "zerogpu")
    local = Mock()
    executor = Mock(
        side_effect=ProcessingError(
            "Quota exhausted", code="zerogpu_quota_exceeded", status_code=503
        )
    )
    monkeypatch.setattr(inference, "run_local_inference", local)
    inference.install_zero_executor(executor)
    with pytest.raises(ProcessingError) as caught:
        inference.run_inference("translate", {"text": "أرز"})
    assert caught.value.code == "zerogpu_quota_exceeded"
    local.assert_not_called()


def test_zero_without_executor_fails(monkeypatch):
    monkeypatch.setenv("NUTRITION_AI_RUNTIME", "zerogpu")
    with pytest.raises(ProcessingError) as caught:
        inference.run_inference("translate", {"text": "أرز"})
    assert caught.value.code == "runtime_configuration_error"


def test_zero_uses_startup_emulation_without_querying_gpu_availability(monkeypatch):
    monkeypatch.setenv("NUTRITION_AI_RUNTIME", "zerogpu")
    monkeypatch.setenv("SPACES_ZERO_GPU", "true")
    monkeypatch.setenv("SPACES_ZERO_DEVICE_API_URL", "https://scheduler.invalid")
    monkeypatch.setenv("SPACE_ID", "test/space")
    monkeypatch.setenv("HF_TOKEN", "test-token")
    available = Mock(return_value=False)
    monkeypatch.setitem(sys.modules, "spaces", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "spaces.config",
        SimpleNamespace(Config=SimpleNamespace(zero_gpu=True)),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            __version__="2.8.0",
            version=SimpleNamespace(cuda="12.8"),
            cuda=SimpleNamespace(is_available=available),
        ),
    )
    assert runtime.get_device() == "cuda"
    available.assert_not_called()
