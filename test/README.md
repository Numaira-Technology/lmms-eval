# Test Suite

This directory contains framework tests plus focused coverage for the task set shipped by this fork: VSI-Bench and Physical Conflict.

## Running the tests

Run the retained-task checks:

```bash
python -m pytest \
  test/eval/test_task_pipeline.py \
  test/eval/test_physical_conflict.py \
  test/eval/prompt_stability/ -q
```

## Retained task coverage

- `test/eval/test_task_pipeline.py` verifies that only the Physical Conflict and VSI-Bench task variants are registered, their YAML files parse, and their utility modules import.
- `test/eval/test_physical_conflict.py` validates Physical Conflict data integrity, prompt construction, prediction normalization, scoring, and opaque media resolution.
- `test/eval/prompt_stability/` protects the multiple-choice and numerical-answer VSI-Bench prompts with golden snapshots.

To regenerate VSI-Bench snapshots after an intentional prompt change:

```bash
python -m pytest test/eval/prompt_stability/ --update-snapshots -v
```

The remaining test modules cover shared CLI, caching, evaluator, model, protocol, token accounting, and scheduling behavior.
