import json
import os
from pathlib import Path

import pytest
from datasets import Dataset

from lmms_eval.tasks import TaskManager
from lmms_eval.tasks.physical_conflict import utils


def _presence_qa(sample_id: str, *, answer: str = "B") -> dict:
    return {
        "qa_id": f"{sample_id}__conflict_presence",
        "question_type": "conflict_presence",
        "question": "Does this video contain a physical conflict?",
        "answer_type": "single_choice",
        "options": [{"id": "A", "text": "No"}, {"id": "B", "text": "Yes"}],
        "answer": answer,
    }


def _quarter_qa(sample_id: str) -> dict:
    return {
        "qa_id": f"{sample_id}__conflict_quarter_coverage",
        "question_type": "conflict_quarter_coverage",
        "question": "Which quarters of the video contain a physical conflict?",
        "answer_type": "multiple_choice",
        "options": [
            {"id": "A", "text": "Q1"},
            {"id": "B", "text": "Q2"},
            {"id": "C", "text": "Q3"},
            {"id": "D", "text": "Q4"},
        ],
        "answer": ["A", "C"],
    }


def _numeric_qa(sample_id: str) -> dict:
    return {
        "qa_id": f"{sample_id}__first_conflict_duration",
        "question_type": "first_conflict_duration",
        "question": "How long does the first physical conflict last?",
        "answer_type": "single_choice",
        "options": [
            {"id": "A", "text": "2.00"},
            {"id": "B", "text": "5.00"},
            {"id": "C", "text": "8.00"},
            {"id": "D", "text": "11.00"},
        ],
        "answer": "B",
        "target_value": 5.0,
        "unit": "seconds",
        "tolerance": 0.5,
    }


def _sample(sample_id: str = "sample_001", *, qa_pairs: list[dict] | None = None, video_path: str = "video.mp4") -> dict:
    return {
        "id": sample_id,
        "video_path": video_path,
        "qa_pairs": qa_pairs if qa_pairs is not None else [_presence_qa(sample_id)],
    }


def _docs(sample: dict) -> list[dict]:
    normalized = utils._normalize_sample_for_arrow(sample)
    return list(utils.physical_conflict_process_docs(Dataset.from_list([normalized])))


def test_physical_conflict_task_is_registered():
    assert "physical_conflict" in TaskManager("ERROR").all_tasks


def test_process_docs_flattens_questions_like_vstat():
    sample_id = "sample_001"
    docs = _docs(_sample(sample_id, qa_pairs=[_quarter_qa(sample_id), _presence_qa(sample_id)]))

    assert len(docs) == 2
    assert docs[0]["sample_id"] == sample_id
    assert docs[0]["video_path"] == "video.mp4"
    assert docs[0]["answer_ids"] == ["A", "C"]
    assert docs[0]["answer_text"] == '["A","C"]'
    assert docs[1]["answer_text"] == "B"


def test_prompt_and_target_use_flat_document():
    doc = _docs(_sample())[0]

    prompt = utils.physical_conflict_doc_to_text(doc)

    assert "Watch the full video carefully" in prompt
    assert "A. No" in prompt
    assert "B. Yes" in prompt
    assert utils.physical_conflict_doc_to_target(doc) == "B"


def test_single_choice_prediction_uses_vstat_mcq_extraction():
    doc = _docs(_sample())[0]

    assert utils.physical_conflict_normalize_prediction(doc, "B") == "B"
    assert utils.physical_conflict_normalize_prediction(doc, "B. Yes") == "B"
    assert utils.physical_conflict_normalize_prediction(doc, "The answer is B") == "B"
    assert utils.physical_conflict_normalize_prediction(doc, "Yes") == "B"


def test_multiple_choice_prediction_and_metrics():
    sample_id = "sample_001"
    doc = _docs(_sample(sample_id, qa_pairs=[_quarter_qa(sample_id)]))[0]

    assert utils.physical_conflict_normalize_prediction(doc, '["C","A"]') == ["A", "C"]
    assert utils.physical_conflict_normalize_prediction(doc, "A and C") == ["A", "C"]
    assert utils.physical_conflict_normalize_prediction(doc, '["A","A"]') is None

    result = utils.physical_conflict_process_results(doc, ['["A","B"]'])
    assert result["Overall_Exact_Accuracy"] == 0.0
    assert result["MultipleChoice_Exact_Set_Accuracy"] == 0.0
    assert result["MultipleChoice_Macro_Precision"] == 0.5
    assert result["MultipleChoice_Macro_Recall"] == 0.5


def test_numeric_metrics_use_selected_option_value():
    sample_id = "sample_001"
    doc = _docs(_sample(sample_id, qa_pairs=[_numeric_qa(sample_id)]))[0]

    correct = utils.physical_conflict_process_results(doc, ["B"])
    wrong = utils.physical_conflict_process_results(doc, ["A"])

    assert correct["Numeric_Option_MAE"] == 0.0
    assert correct["Numeric_Accuracy_at_0_5s"] == 1.0
    assert wrong["Numeric_Option_MAE"] == 3.0
    assert wrong["Numeric_Accuracy_at_0_5s"] == 0.0


def test_doc_to_visual_resolves_video_from_document(tmp_path: Path):
    video = tmp_path / "fixture.mp4"
    video.write_bytes(b"fixture")
    doc = _docs(_sample(video_path=str(video)))[0]

    assert utils.physical_conflict_doc_to_visual(doc) == [str(video)]


def test_split_loader_accepts_vstat_style_sample_ids(tmp_path: Path):
    split_path = tmp_path / "test.jsonl"
    split_path.write_text(json.dumps({"id": "sample_001"}) + "\n")

    assert utils._load_split_ids(split_path) == {"sample_001"}


@pytest.mark.skipif(not os.environ.get("GOODVISION_REPO_ROOT"), reason="GOODVISION_REPO_ROOT is not configured")
def test_current_goodvision_release_projects_to_334_test_questions():
    root = Path(os.environ["GOODVISION_REPO_ROOT"])
    main_path = root / "data/samples/v1/clean_samples_1000_human_reviewed_with_qa.jsonl"
    split_path = root / "data/samples/v1/splits/test.jsonl"
    split_ids = utils._load_split_ids(split_path)
    samples = [sample for sample in utils._read_jsonl(main_path) if sample["id"] in split_ids]
    normalized = [utils._normalize_sample_for_arrow(sample) for sample in samples]
    docs = utils.physical_conflict_process_docs(Dataset.from_list(normalized))

    assert len(samples) == 149
    assert len(docs) == 334
