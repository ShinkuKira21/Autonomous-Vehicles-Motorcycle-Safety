import inspect
import importlib
import pytest


def test_module_exists():
    try:
        template_module = importlib.import_module("modules.models.template")
    except ImportError:
        pytest.fail("my_ml_module does not exist")


def test_function_exists():
    template_module = importlib.import_module("modules.models.template")
    assert inspect.isfunction(getattr(template_module, "build", None))
