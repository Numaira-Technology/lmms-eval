"""Snapshot tests for VSI-Bench prompt construction."""

import json
from pathlib import Path

import pytest

SNAPSHOT_DIR = Path(__file__).parent / "snapshots"


def _import_vsibench():
    from lmms_eval.tasks.vsibench.utils import vsibench_doc_to_text

    return vsibench_doc_to_text


CASES = {
    "vsibench__mca": {
        "task": "vsibench",
        "variant": "multiple-choice",
        "get_fn": _import_vsibench,
        "default_kwargs": {
            "pre_prompt": "",
            "mca_post_prompt": "Answer with the option's letter from the given choices directly.",
            "na_post_prompt": "Please answer the question using a single word or phrase.",
        },
        "gen_kwargs": {
            "max_new_tokens": 16,
            "temperature": 0,
            "top_p": 1.0,
            "num_beams": 1,
            "do_sample": False,
        },
        "fixture": {
            "question": "Which direction is the sofa relative to the dining table?",
            "question_type": "object_rel_direction_easy",
            "options": ["To the left", "To the right", "In front", "Behind"],
            "ground_truth": "A",
            "dataset": "scannet",
            "scene_name": "scene0001_00",
        },
    },
    "vsibench__na": {
        "task": "vsibench",
        "variant": "numerical-answer",
        "get_fn": _import_vsibench,
        "default_kwargs": {
            "pre_prompt": "",
            "mca_post_prompt": "Answer with the option's letter from the given choices directly.",
            "na_post_prompt": "Please answer the question using a single word or phrase.",
        },
        "gen_kwargs": {
            "max_new_tokens": 16,
            "temperature": 0,
            "top_p": 1.0,
            "num_beams": 1,
            "do_sample": False,
        },
        "fixture": {
            "question": "How many chairs are visible in this room?",
            "question_type": "object_counting",
            "options": [],
            "ground_truth": "4",
            "dataset": "scannet",
            "scene_name": "scene0001_00",
        },
    },
}


@pytest.mark.parametrize("case_name", sorted(CASES))
def test_prompt_stable(case_name, update_snapshots):
    case = CASES[case_name]
    prompt = case["get_fn"]()(case["fixture"], case["default_kwargs"])
    snapshot_path = SNAPSHOT_DIR / f"{case_name}.json"

    if update_snapshots:
        snapshot = {
            "task": case["task"],
            "variant": case["variant"],
            "doc_id": "unknown",
            "prompt_text": prompt,
            "gen_kwargs": case["gen_kwargs"],
        }
        snapshot_path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n")
        pytest.skip("snapshot updated")

    assert snapshot_path.exists(), f"No snapshot found for '{case_name}'"
    expected = json.loads(snapshot_path.read_text())
    assert prompt == expected["prompt_text"]
    assert case["gen_kwargs"] == expected["gen_kwargs"]


@pytest.mark.parametrize("case_name", sorted(CASES))
def test_gen_kwargs_complete(case_name):
    assert "max_new_tokens" in CASES[case_name]["gen_kwargs"]
