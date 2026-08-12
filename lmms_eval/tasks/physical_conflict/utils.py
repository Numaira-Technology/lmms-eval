"""Physical Conflict benchmark task helpers.

The task follows the same ``ConfigurableTask`` flow as VSTAT: load the
annotation file, flatten its examples in ``process_docs``, resolve each video
from the document, construct the prompt and target, and score model output.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict
from loguru import logger as eval_logger

from lmms_eval import utils as lmms_utils
from lmms_eval.api.task import ConfigurableTask
from lmms_eval.tasks._task_utils.mcq_extract import extract_mcq_answer

_DEFAULT_QA_FILENAME = "clean_samples_1000_human_reviewed_with_qa.jsonl"
_DEFAULT_SPLIT_FILENAME = "splits/test.jsonl"
_DEFAULT_CACHE_DIR = "physical_conflict"
_CHOICE_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_NUMBER_PATTERN = re.compile(r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)")

NUMERIC_QUESTION_TYPES = {
    "first_conflict_start_time",
    "first_conflict_duration",
    "total_non_overlapping_conflict_duration",
}

_downloaded_data_root: Path | None = None
_configured_video_root: Path | None = None

_SETUP_HINT = (
    "Set PHYSICAL_CONFLICT_QA_PATH to "
    "clean_samples_1000_human_reviewed_with_qa.jsonl and "
    "PHYSICAL_CONFLICT_VIDEO_ROOT to the prepared video directory. "
    "Set PHYSICAL_CONFLICT_SPLIT_PATH when the test split is not located at "
    "splits/test.jsonl relative to the QA file."
)


def _resolve_path(path: str | os.PathLike[str]) -> Path:
    expanded = Path(path).expanduser()
    return expanded if expanded.is_absolute() else Path.cwd() / expanded


def _hf_home() -> Path:
    return Path(os.path.expanduser(os.path.expandvars(os.getenv("HF_HOME", "~/.cache/huggingface/"))))


def _cache_root_from_config(dataset_kwargs: dict[str, Any] | None) -> Path:
    kwargs = dataset_kwargs or {}
    cache_dir = str(kwargs.get("cache_dir") or _DEFAULT_CACHE_DIR)
    return Path(lmms_utils.resolve_cache_dir(cache_dir, base_dir=str(_hf_home())))


def _configured_path(env_name: str, dataset_kwargs: dict[str, Any], *config_keys: str) -> Path | None:
    override = os.environ.get(env_name)
    if override:
        return _resolve_path(override)
    for key in config_keys:
        value = dataset_kwargs.get(key)
        if value:
            return _resolve_path(value)
    return None


def _qa_path_from_config(dataset_path: str | None, dataset_kwargs: dict[str, Any] | None) -> Path:
    kwargs = dataset_kwargs or {}
    configured = _configured_path("PHYSICAL_CONFLICT_QA_PATH", kwargs, "qa_file", "qa_path")
    if configured is not None:
        return configured

    data_files = kwargs.get("data_files")
    if isinstance(data_files, dict):
        data_file = data_files.get("test") or data_files.get("qa") or next(iter(data_files.values()), None)
    else:
        data_file = data_files
    if data_file:
        return _resolve_path(data_file)

    if dataset_path:
        candidate = _resolve_path(dataset_path)
        if candidate.is_file():
            return candidate
        if candidate.is_dir():
            nested = candidate / str(kwargs.get("qa_filename") or _DEFAULT_QA_FILENAME)
            if nested.exists():
                return nested

    cached = _cache_root_from_config(kwargs) / str(kwargs.get("qa_filename") or _DEFAULT_QA_FILENAME)
    if cached.exists():
        return cached
    raise FileNotFoundError(f"Missing Physical Conflict QA file: {cached}\n{_SETUP_HINT}")


def _split_path_from_config(qa_path: Path, dataset_kwargs: dict[str, Any] | None) -> Path | None:
    kwargs = dataset_kwargs or {}
    configured = _configured_path("PHYSICAL_CONFLICT_SPLIT_PATH", kwargs, "split_file", "split_path")
    if configured is not None:
        return configured
    candidate = qa_path.parent / str(kwargs.get("split_filename") or _DEFAULT_SPLIT_FILENAME)
    return candidate if candidate.exists() else None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing Physical Conflict annotation file: {path}\n{_SETUP_HINT}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} must contain a JSON object.")
            records.append(value)
    return records


def _load_split_ids(split_path: Path | None) -> set[str] | None:
    if split_path is None:
        return None
    sample_ids = {str(record.get("id") or record.get("sample_id") or "") for record in _read_jsonl(split_path)}
    sample_ids.discard("")
    if not sample_ids:
        raise ValueError(f"Physical Conflict split contains no sample IDs: {split_path}")
    return sample_ids


def _normalize_sample_for_arrow(sample: dict[str, Any]) -> dict[str, Any]:
    """Keep only evaluation fields and normalize heterogeneous QA answers."""

    normalized_qas: list[dict[str, Any]] = []
    for qa in sample.get("qa_pairs") or []:
        normalized_qas.append(
            {
                "qa_id": str(qa.get("qa_id") or ""),
                "question_type": str(qa.get("question_type") or ""),
                "question": str(qa.get("question") or ""),
                "answer_type": str(qa.get("answer_type") or "single_choice"),
                "options": _normalize_options(qa.get("options")),
                "answer_json": json.dumps(qa.get("answer"), separators=(",", ":")),
                "target_value": qa.get("target_value"),
                "tolerance": qa.get("tolerance"),
                "unit": qa.get("unit"),
            }
        )
    return {
        "id": str(sample.get("id") or sample.get("sample_id") or ""),
        "video_path": str(sample.get("video_path") or ""),
        "qa_pairs": normalized_qas,
    }


class PhysicalConflictTask(ConfigurableTask):
    """ConfigurableTask that expands the Physical Conflict QA JSONL."""

    def __init__(self, *args, config: dict[str, Any] | None = None, **kwargs) -> None:
        if config is not None:
            config = dict(config)
            config.pop("class", None)
        super().__init__(*args, config=config, **kwargs)

    def download(self, dataset_kwargs: dict[str, Any] | None = None) -> None:
        global _configured_video_root, _downloaded_data_root

        kwargs = dict(dataset_kwargs or {})
        qa_path = _qa_path_from_config(self.config.dataset_path, kwargs)
        split_path = _split_path_from_config(qa_path, kwargs)
        split_ids = _load_split_ids(split_path)
        _downloaded_data_root = qa_path.parent
        _configured_video_root = _configured_path("PHYSICAL_CONFLICT_VIDEO_ROOT", kwargs, "video_root")

        samples = _read_jsonl(qa_path)
        if split_ids is not None:
            samples = [sample for sample in samples if str(sample.get("id", "")) in split_ids]
        normalized_samples = [_normalize_sample_for_arrow(sample) for sample in samples]

        split = self.config.test_split
        self.dataset = DatasetDict({split: Dataset.from_list(normalized_samples)})
        if self.config.process_docs is not None:
            self.dataset[split] = self.config.process_docs(self.dataset[split])
        self.dataset_no_image = self.dataset.copy()
        eval_logger.info(f"Loaded Physical Conflict annotations from {qa_path} " f"({len(samples)} videos, {len(self.dataset[split])} questions).")


def _normalize_options(raw_options: Any) -> list[dict[str, str]]:
    options: list[dict[str, str]] = []
    for index, option in enumerate(raw_options or []):
        if isinstance(option, dict):
            option_id = str(option.get("id") or _CHOICE_LETTERS[index]).upper()
            option_text = str(option.get("text") or option.get("value") or "").strip()
        else:
            option_id = _CHOICE_LETTERS[index]
            option_text = str(option).strip()
        if not option_text:
            raise ValueError("Physical Conflict options must contain non-empty text.")
        options.append({"id": option_id, "text": option_text})
    if not options:
        raise ValueError("Physical Conflict questions must provide answer options.")
    return options


def physical_conflict_process_docs(dataset: Dataset) -> Dataset:
    flat_docs: list[dict[str, Any]] = []
    for sample in dataset:
        sample_id = str(sample.get("id") or sample.get("sample_id") or "")
        video_path = str(sample.get("video_path") or "")
        if not sample_id or not video_path:
            raise ValueError("Physical Conflict samples require id and video_path.")

        for qa in sample.get("qa_pairs") or []:
            options = _normalize_options(qa.get("options"))
            option_ids = [option["id"] for option in options]
            answer_type = str(qa.get("answer_type") or "single_choice").lower()
            is_multiple_choice = answer_type == "multiple_choice"
            raw_answer = json.loads(qa["answer_json"]) if "answer_json" in qa else qa.get("answer")
            if is_multiple_choice:
                if not isinstance(raw_answer, list) or not raw_answer:
                    raise ValueError("Multiple-choice Physical Conflict answers must be non-empty lists.")
                answer_ids = sorted({str(value).upper() for value in raw_answer}, key=option_ids.index)
                answer_text = json.dumps(answer_ids, separators=(",", ":"))
            else:
                answer_ids = [str(raw_answer).upper()]
                answer_text = answer_ids[0]
            if any(answer not in option_ids for answer in answer_ids):
                raise ValueError(f"Physical Conflict answer is not present in its options: {answer_ids}")

            question_type = str(qa.get("question_type") or "")
            is_numeric = question_type in NUMERIC_QUESTION_TYPES
            if is_numeric:
                target_value = qa.get("target_value")
                if target_value is None:
                    raise ValueError(f"Numeric Physical Conflict question {question_type!r} requires target_value.")
                answer_text = str(target_value)
            flat_docs.append(
                {
                    "qa_id": str(qa.get("qa_id") or f"{sample_id}__{question_type}"),
                    "sample_id": sample_id,
                    "video_path": video_path,
                    "question_type": question_type,
                    "question": str(qa.get("question") or "").strip(),
                    "answer_type": answer_type,
                    "is_numeric": is_numeric,
                    "is_multiple_choice": is_multiple_choice,
                    "options": options,
                    "answer_ids": answer_ids,
                    "answer_text": answer_text,
                    "target_value": qa.get("target_value"),
                    "tolerance": qa.get("tolerance"),
                    "unit": qa.get("unit"),
                }
            )
    return Dataset.from_list(flat_docs)


def _candidate_video_roots() -> list[Path]:
    roots: list[Path] = []
    for env_name in ("PHYSICAL_CONFLICT_VIDEO_ROOT", "PHYSICAL_CONFLICT_DATA_ROOT"):
        value = os.environ.get(env_name)
        if value:
            roots.append(_resolve_path(value))
    if _configured_video_root is not None:
        roots.append(_configured_video_root)
    if _downloaded_data_root is not None:
        roots.append(_downloaded_data_root)
    roots.extend((_cache_root_from_config(None), Path.cwd() / "data" / _DEFAULT_CACHE_DIR))

    deduped: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            deduped.append(root)
            seen.add(key)
    return deduped


def _resolve_video_path(video_path: str) -> Path:
    raw_path = Path(video_path).expanduser()
    if raw_path.is_absolute():
        return raw_path
    for root in _candidate_video_roots():
        candidate = root / raw_path
        if candidate.exists():
            return candidate
    roots = _candidate_video_roots()
    return (roots[0] if roots else Path.cwd()) / raw_path


def physical_conflict_doc_to_visual(doc: dict[str, Any]) -> list[str]:
    path = _resolve_video_path(str(doc["video_path"]))
    if not path.exists():
        raise FileNotFoundError(f"Missing Physical Conflict video file: {path}\n{_SETUP_HINT}")
    return [str(path)]


def physical_conflict_doc_to_text(doc: dict[str, Any], lmms_eval_specific_kwargs: dict[str, Any] | None = None) -> str:
    kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = kwargs.get("pre_prompt", "")
    option_lines = "\n".join(f"{option['id']}. {option['text']}" for option in doc["options"])
    body = f"Watch the full video carefully before answering.\n\nQuestion: {doc['question']}\n\nOptions:\n{option_lines}"
    if doc["is_numeric"]:
        post_prompt = kwargs.get("numeric_post_prompt", "")
    elif doc["is_multiple_choice"]:
        post_prompt = kwargs.get("multiple_choice_post_prompt", 'Return only a JSON array of option IDs, for example ["A","C"].')
    else:
        post_prompt = kwargs.get("mcq_post_prompt", "")
    return f"{pre_prompt}{body}\n\n{post_prompt}".strip()


def physical_conflict_doc_to_target(doc: dict[str, Any]) -> str:
    return str(doc["answer_text"])


def _normalize_multiple_choice_prediction(doc: dict[str, Any], prediction: str) -> list[str] | None:
    option_ids = [option["id"] for option in doc["options"]]
    try:
        parsed = json.loads(prediction)
    except json.JSONDecodeError:
        parsed = re.findall(r"(?<![A-Za-z])[A-Z](?![A-Za-z])", prediction.upper())
    if not isinstance(parsed, list) or not parsed:
        return None
    values = [str(value).upper() for value in parsed]
    if len(values) != len(set(values)) or any(value not in option_ids for value in values):
        return None
    return sorted(values, key=option_ids.index)


def _extract_last_number(text: str) -> float | None:
    matches = _NUMBER_PATTERN.findall(str(text).replace(",", ""))
    return float(matches[-1]) if matches else None


def physical_conflict_normalize_prediction(doc: dict[str, Any], raw_prediction: str) -> str | float | list[str] | None:
    prediction = str(raw_prediction).strip()
    if doc["is_numeric"]:
        return _extract_last_number(prediction)
    if doc["is_multiple_choice"]:
        return _normalize_multiple_choice_prediction(doc, prediction)

    option_ids = [option["id"] for option in doc["options"]]
    option_text = {option["text"].strip().casefold(): option["id"] for option in doc["options"]}
    if prediction.casefold() in option_text:
        return option_text[prediction.casefold()]
    return extract_mcq_answer(prediction, choices=option_ids) or None


def physical_conflict_process_results(doc: dict[str, Any], results: list[str]) -> dict[str, float]:
    prediction = physical_conflict_normalize_prediction(doc, results[0] if results else "")
    if doc["is_numeric"]:
        absolute_error = abs(float(prediction) - float(doc["target_value"])) if isinstance(prediction, float) else float("inf")
        within_tolerance = float(absolute_error <= float(doc.get("tolerance") or 0.5))
        return {
            "Overall_Exact_Accuracy": within_tolerance,
            "Numeric_Accuracy_at_0_5s": within_tolerance,
            "Numeric_Option_MAE": absolute_error,
        }

    predicted_set = {prediction} if isinstance(prediction, str) else set(prediction or [])
    target_set = set(doc["answer_ids"])
    exact = float(prediction is not None and predicted_set == target_set)
    metrics: dict[str, float] = {"Overall_Exact_Accuracy": exact}

    if doc["is_multiple_choice"]:
        intersection = predicted_set & target_set
        metrics.update(
            {
                "MultipleChoice_Exact_Set_Accuracy": exact,
                "MultipleChoice_Macro_Precision": len(intersection) / len(predicted_set) if predicted_set else 0.0,
                "MultipleChoice_Macro_Recall": len(intersection) / len(target_set) if target_set else float(not predicted_set),
            }
        )
        return metrics

    metrics["SingleChoice_Accuracy"] = exact
    if doc["question_type"] == "conflict_presence":
        metrics["Binary_Accuracy"] = exact
    return metrics


def physical_conflict_aggregate_mean(results: list[float]) -> float:
    return sum(float(result) for result in results) / len(results) if results else 0.0
