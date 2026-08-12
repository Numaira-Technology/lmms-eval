"""Registration and configuration checks for the retained evaluation tasks."""

import importlib
import os

import pytest
import yaml

from lmms_eval.tasks import TaskManager


@pytest.fixture(scope="module")
def task_manager():
    return TaskManager("WARNING")


RETAINED_TASKS = {
    "physical_conflict",
    "vsibench",
    "vstat",
}

PYTHON_TASKS = {"physical_conflict", "vstat"}
YAML_SUBTASKS = RETAINED_TASKS - PYTHON_TASKS

TASK_UTILS = {
    "physical_conflict": "lmms_eval.tasks.physical_conflict.utils",
    "vsibench": "lmms_eval.tasks.vsibench.utils",
    "vstat": "lmms_eval.tasks.vstat.utils",
}


class _FunctionTag:
    """Placeholder for ``!function`` tags while safely loading task YAML."""

    def __init__(self, value):
        self.value = value


def _function_constructor(loader, node):
    return _FunctionTag(loader.construct_scalar(node))


def _load_task_yaml(yaml_path: str) -> dict:
    loader = type("SafeLoaderCopy", (yaml.SafeLoader,), {})
    loader.add_constructor("!function", _function_constructor)
    with open(yaml_path) as handle:
        return yaml.load(handle, Loader=loader)


def test_registry_contains_only_retained_tasks(task_manager):
    assert set(task_manager.all_tasks) == RETAINED_TASKS
    assert set(task_manager.all_subtasks) == YAML_SUBTASKS
    assert all(task_manager.task_index[name]["type"] == "python_task" for name in PYTHON_TASKS)
    assert task_manager.all_groups == []
    assert task_manager.all_tags == []


@pytest.mark.parametrize("task_name", sorted(RETAINED_TASKS))
def test_task_yaml_is_valid(task_manager, task_name):
    entry = task_manager.task_index[task_name]
    yaml_path = entry["yaml_path"]

    assert os.path.isfile(yaml_path)
    config = _load_task_yaml(yaml_path)
    assert isinstance(config, dict)
    assert config["task"] == task_name
    assert "dataset_path" in config or "include" in config
    assert any(key in config or "include" in config for key in ("doc_to_messages", "doc_to_text", "doc_to_visual"))


@pytest.mark.parametrize("task_name,module_path", sorted(TASK_UTILS.items()))
def test_task_utils_are_importable(task_name, module_path):
    module = importlib.import_module(module_path)
    assert callable(getattr(module, f"{task_name}_doc_to_text"))
    assert callable(getattr(module, f"{task_name}_process_results"))


def test_no_duplicate_task_names(task_manager):
    assert len(task_manager.all_subtasks) == len(set(task_manager.all_subtasks))
