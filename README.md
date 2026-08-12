# Numaira LMMS Eval

Numaira's focused multimodal evaluation fork. It retains the lmms-eval runtime and three video benchmarks:

- `physical_conflict`
- `vsibench`
- `vstat`

## Install

Python 3.10 or newer is required.

```bash
python -m pip install -e .
```

List the registered tasks:

```bash
python -m lmms_eval tasks list
```

## Run an evaluation

Choose any supported model backend and one retained task:

```bash
python -m lmms_eval \
  --model qwen2_5_vl \
  --model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct \
  --tasks vsibench \
  --batch_size 1
```

The CLI also accepts `vstat` and `physical_conflict` through `--tasks`.

## Dataset configuration

### Physical Conflict

Set the prepared annotation, split, and video locations:

```bash
export PHYSICAL_CONFLICT_QA_PATH=/path/to/clean_samples_1000_human_reviewed_with_qa.jsonl
export PHYSICAL_CONFLICT_SPLIT_PATH=/path/to/splits/test.jsonl
export PHYSICAL_CONFLICT_VIDEO_ROOT=/path/to/videos
```

The task loads the selected videos, flattens their QA pairs into evaluation documents, and uses separate MCQ, numeric, and multi-select prompt/scoring paths.

### VSTAT

VSTAT downloads `nyu-visionx/vstat` into the Hugging Face cache by default. To use prepared local data:

```bash
export VSTAT_QA_PATH=/path/to/vstat_qa_clean.json
export VSTAT_VIDEO_ROOT=/path/to/vstat
```

### VSI-Bench

VSI-Bench uses `nyu-visionx/VSI-Bench` and stores videos under `$HF_HOME/vsibench` by default.

## Validation

Run the focused task checks:

```bash
python -m pytest \
  test/eval/test_task_pipeline.py \
  test/eval/test_physical_conflict.py \
  test/eval/prompt_stability/ -q
```

Formatting is enforced with:

```bash
python -m pre_commit run --all-files
```
