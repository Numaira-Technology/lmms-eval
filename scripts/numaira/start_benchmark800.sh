#!/usr/bin/env bash

# 在 Vast.ai 服务器上准备或启动最新 Benchmark-800（800 个媒体、1,501 道 QA）
# 的共享运行会话（Numaira）。
# 本脚本不会安装依赖、拉取代码、修改 SSH key，也不会自行启动模型评测。

set -Eeuo pipefail

usage() {
  cat <<'EOF'
用法：
  bash scripts/numaira/start_benchmark800.sh --check-only [选项]
  bash scripts/numaira/start_benchmark800.sh --start [选项]
  bash scripts/numaira/start_benchmark800.sh --status [选项]
  bash scripts/numaira/start_benchmark800.sh --attach [选项]

模式：
  --check-only   检查 Conda、GPU、GoodVision 和 Benchmark-800 数据，不启动会话
  --start        检查通过后，创建名为 benchmark800 的 tmux 后台会话
  --status       查看 tmux 会话是否正在运行
  --attach       进入已经运行的 tmux 会话

选项：
  --repo PATH        GoodVision 仓库路径（默认 /workspace/goodvision）
  --media-root PATH  800 个媒体资产所在目录（默认 /workspace/data/benchmark800）
  --output-root PATH 运行结果目录（默认 /workspace/results/benchmark800）
  --conda-env NAME   Conda 环境名（默认 qwen3）
  --session NAME     tmux 会话名（默认 benchmark800）
  --skip-media-check 暂时不检查 800 个媒体资产，仅用于环境搭建阶段
  --help             显示帮助

第一次建议：
  bash scripts/numaira/start_benchmark800.sh --check-only --media-root /实际/媒体路径
  bash scripts/numaira/start_benchmark800.sh --start --media-root /实际/媒体路径
  tmux attach -t benchmark800

注意：
  脚本按 GoodVision main 的 Benchmark-800 contract 校验 800 个媒体和 1,501 道 QA。
  --start 只创建准备好的终端会话，不会自动运行耗时的模型 benchmark。
EOF
}

fail() {
  printf '[ERROR] %s\n' "$*" >&2
  exit 1
}

info() {
  printf '[INFO] %s\n' "$*"
}

MODE=""
REPO_ROOT="${GOODVISION_ROOT:-/workspace/goodvision}"
MEDIA_ROOT="${GOODVISION_MEDIA_ROOT:-/workspace/data/benchmark800}"
OUTPUT_ROOT="${GOODVISION_OUTPUT_ROOT:-/workspace/results/benchmark800}"
CONDA_ENV="${CONDA_ENV:-qwen3}"
SESSION_NAME="${GOODVISION_TMUX_SESSION:-benchmark800}"
SKIP_MEDIA_CHECK=0
RESOLVED_CONDA_EXE=""

while (($#)); do
  case "$1" in
    --check-only|--start|--status|--attach)
      [[ -z "$MODE" ]] || fail "只能选择一个模式。"
      MODE="${1#--}"
      shift
      ;;
    --repo)
      [[ $# -ge 2 ]] || fail "--repo 后需要路径。"
      REPO_ROOT="$2"
      shift 2
      ;;
    --media-root)
      [[ $# -ge 2 ]] || fail "--media-root 后需要路径。"
      MEDIA_ROOT="$2"
      shift 2
      ;;
    --output-root)
      [[ $# -ge 2 ]] || fail "--output-root 后需要路径。"
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --conda-env)
      [[ $# -ge 2 ]] || fail "--conda-env 后需要环境名。"
      CONDA_ENV="$2"
      shift 2
      ;;
    --session)
      [[ $# -ge 2 ]] || fail "--session 后需要会话名。"
      SESSION_NAME="$2"
      shift 2
      ;;
    --skip-media-check)
      SKIP_MEDIA_CHECK=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      fail "未知参数：$1（使用 --help 查看帮助）"
      ;;
  esac
done

[[ -n "$MODE" ]] || fail "请选择 --check-only、--start、--status 或 --attach。"
[[ "$SESSION_NAME" =~ ^[A-Za-z0-9_.-]+$ ]] || fail "tmux 会话名只能包含字母、数字、点、下划线和连字符。"

if [[ "$MODE" == "status" ]]; then
  command -v tmux >/dev/null 2>&1 || fail "未找到 tmux。"
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    info "会话正在运行：$SESSION_NAME"
    tmux list-sessions -F '#{session_name}  created=#{session_created_string}  windows=#{session_windows}' \
      | awk -v target="$SESSION_NAME" '$1 == target'
    exit 0
  fi
  fail "会话未运行：$SESSION_NAME"
fi

if [[ "$MODE" == "attach" ]]; then
  command -v tmux >/dev/null 2>&1 || fail "未找到 tmux。"
  tmux has-session -t "$SESSION_NAME" 2>/dev/null || fail "会话未运行：$SESSION_NAME"
  exec tmux attach-session -t "$SESSION_NAME"
fi

activate_conda_env() {
  local conda_exe=""
  if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE:-}" ]]; then
    conda_exe="$CONDA_EXE"
  elif type -P conda >/dev/null 2>&1; then
    conda_exe="$(type -P conda)"
  else
    local candidate
    for candidate in \
      /opt/conda/bin/conda \
      /root/miniforge3/bin/conda \
      /root/miniconda3/bin/conda \
      /workspace/miniforge3/bin/conda; do
      if [[ -x "$candidate" ]]; then
        conda_exe="$candidate"
        break
      fi
    done
  fi

  [[ -n "$conda_exe" ]] || fail "未找到 Conda；请先创建共享环境 '$CONDA_ENV'。"
  RESOLVED_CONDA_EXE="$conda_exe"
  if [[ "${CONDA_DEFAULT_ENV:-}" == "$CONDA_ENV" ]]; then
    info "Conda 环境已激活：$CONDA_ENV"
    return
  fi
  eval "$("$conda_exe" shell.bash hook)"
  conda activate "$CONDA_ENV" || fail "无法激活 Conda 环境 '$CONDA_ENV'。"
  info "已激活 Conda 环境：$CONDA_ENV"
}

activate_conda_env

command -v python >/dev/null 2>&1 || fail "当前环境中没有 python。"
command -v git >/dev/null 2>&1 || fail "未找到 git。"
command -v nvidia-smi >/dev/null 2>&1 || fail "未找到 nvidia-smi。"

REPO_ROOT="$(cd "$REPO_ROOT" 2>/dev/null && pwd)" || fail "无法进入 GoodVision 仓库：$REPO_ROOT"
git -C "$REPO_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1 \
  || fail "不是 Git checkout：$REPO_ROOT"

MANIFEST_PATH="$REPO_ROOT/data/manifests/goodvision_multicap_v2/benchmark_800.jsonl"
QUESTIONS_PATH="$REPO_ROOT/data/qa/goodvision_multicap_v2/benchmark_800/questions.jsonl"
PRIVATE_ANSWERS_PATH="$REPO_ROOT/data/private/goodvision_multicap_v2/benchmark_800/answers.jsonl"
GENERATOR_PATH="$REPO_ROOT/scripts/generate_qa.py"

[[ -f "$MANIFEST_PATH" ]] || fail "缺少 Benchmark-800 manifest。当前 checkout 可能不是最新 main：$MANIFEST_PATH"
[[ -f "$QUESTIONS_PATH" ]] || fail "缺少公开问题文件：$QUESTIONS_PATH"
[[ -f "$PRIVATE_ANSWERS_PATH" ]] || fail "缺少受限私有答案文件：$PRIVATE_ANSWERS_PATH"
[[ -f "$GENERATOR_PATH" ]] || fail "缺少 QA 校验入口：$GENERATOR_PATH"

if [[ "$SKIP_MEDIA_CHECK" == 0 ]]; then
  MEDIA_ROOT="$(cd "$MEDIA_ROOT" 2>/dev/null && pwd)" || fail "无法进入媒体目录：$MEDIA_ROOT"
else
  info "已跳过媒体文件检查；正式运行 benchmark 前必须取消该选项。"
fi

mkdir -p "$OUTPUT_ROOT"
OUTPUT_ROOT="$(cd "$OUTPUT_ROOT" && pwd)"

export GOODVISION_ROOT="$REPO_ROOT"
export GOODVISION_MEDIA_ROOT="$MEDIA_ROOT"
export GOODVISION_OUTPUT_ROOT="$OUTPUT_ROOT"
export GOODVISION_BENCHMARK_MANIFEST="$MANIFEST_PATH"
export GOODVISION_PUBLIC_QUESTIONS="$QUESTIONS_PATH"

info "GoodVision：$REPO_ROOT"
info "Git commit：$(git -C "$REPO_ROOT" rev-parse --short=12 HEAD)"
info "输出目录：$OUTPUT_ROOT"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

(
  cd "$REPO_ROOT"
  python scripts/generate_qa.py --check
)

python - "$MANIFEST_PATH" "$QUESTIONS_PATH" "$PRIVATE_ANSWERS_PATH" "$MEDIA_ROOT" "$SKIP_MEDIA_CHECK" <<'PY'
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

manifest_path = Path(sys.argv[1])
questions_path = Path(sys.argv[2])
private_answers_path = Path(sys.argv[3])
media_root = Path(sys.argv[4])
skip_media_check = sys.argv[5] == "1"

def read_jsonl(path: Path):
    rows = []
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise SystemExit(f"{path}:{line_number}: invalid JSON: {error}") from error
    return rows

members = read_jsonl(manifest_path)
questions = read_jsonl(questions_path)
private_answers = read_jsonl(private_answers_path)

expected_members = 800
expected_questions = 1501

if len(members) != expected_members:
    raise SystemExit(f"Expected {expected_members} members, found {len(members)}")
if len(questions) != expected_questions:
    raise SystemExit(
        f"Expected {expected_questions} public questions, found {len(questions)}"
    )
if len(private_answers) != expected_questions:
    raise SystemExit(
        f"Expected {expected_questions} private answers, found {len(private_answers)}"
    )
if len({row["video_id"] for row in members}) != expected_members:
    raise SystemExit("Benchmark membership contains duplicate video_id values")
if len({row["qa_id"] for row in questions}) != expected_questions:
    raise SystemExit("Public questions contain duplicate qa_id values")
if len({row["qa_id"] for row in private_answers}) != expected_questions:
    raise SystemExit("Private answers contain duplicate qa_id values")

member_ids = {row["video_id"] for row in members}
question_video_ids = {row["video_id"] for row in questions}
if question_video_ids != member_ids:
    raise SystemExit("Public questions and Benchmark-800 membership cover different videos")
if {row["qa_id"] for row in private_answers} != {row["qa_id"] for row in questions}:
    raise SystemExit("Public questions and private answers cover different qa_id values")

capability_counts = Counter(row["capability"] for row in questions)
expected_capability_counts = {
    "counting": 310,
    "event_recognition": 490,
    "location_tracking": 342,
    "temporal_questions": 359,
}
if dict(capability_counts) != expected_capability_counts:
    raise SystemExit(f"Capability counts differ: {dict(capability_counts)}")

print("[CHECK] Benchmark membership: 800 videos")
print("[CHECK] Public QA: 1,501 questions")
print("[CHECK] Private answers: 1,501 answers")
print(f"[CHECK] Capability counts: {dict(capability_counts)}")

if skip_media_check:
    raise SystemExit(0)

missing = []
wrong_type = []
wrong_size = []
wrong_hash = []

def hash_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()

def sequence_member_path(member, candidate: Path, image: Path) -> str:
    source = member["source_dataset"]
    if source in {"mot17", "mot20"}:
        return f"{candidate.parent.name}/{candidate.name}/{image.name}"
    if source == "vscrowd":
        return f"{candidate.name}/{image.name}"
    raise SystemExit(f"Unsupported image-sequence source: {source}")

def hash_image_sequence(member, candidate: Path) -> tuple[int, str]:
    if candidate.is_symlink() or not candidate.is_dir():
        raise ValueError("expected a regular image-sequence directory")
    images = sorted(candidate.iterdir(), key=lambda path: path.name)
    if not images:
        raise ValueError("image-sequence directory is empty")
    projection = []
    total_size = 0
    for image in images:
        if image.is_symlink() or not image.is_file():
            raise ValueError(f"unexpected non-file member: {image.name}")
        size, digest = hash_file(image)
        total_size += size
        projection.append(
            {
                "member_path": sequence_member_path(member, candidate, image),
                "bytes": size,
                "sha256": digest,
            }
        )
    tree_digest = hashlib.sha256()
    for row in projection:
        payload = (
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        tree_digest.update(payload)
    return total_size, tree_digest.hexdigest()

for member in members:
    media = member["media"]
    candidate = media_root / media["path_or_name"]
    if not candidate.exists():
        missing.append((member["video_id"], candidate))
        continue
    try:
        if media["media_type"] == "file":
            if candidate.is_symlink() or not candidate.is_file():
                raise ValueError("expected a regular file")
            observed_size, observed_hash = hash_file(candidate)
        elif media["media_type"] == "image_sequence":
            observed_size, observed_hash = hash_image_sequence(member, candidate)
        else:
            raise ValueError(f"unsupported media_type={media['media_type']!r}")
    except (OSError, ValueError) as error:
        wrong_type.append((member["video_id"], candidate, str(error)))
        continue
    if observed_size != media["bytes"]:
        wrong_size.append((member["video_id"], media["bytes"], observed_size))
        continue
    if observed_hash != media["sha256"]:
        wrong_hash.append((member["video_id"], media["sha256"], observed_hash))

if missing or wrong_type or wrong_size or wrong_hash:
    details = []
    details.extend(f"missing {video_id}: {path}" for video_id, path in missing[:5])
    details.extend(
        f"type {video_id}: {path}: {error}"
        for video_id, path, error in wrong_type[:5]
    )
    details.extend(
        f"size {video_id}: expected={expected} observed={observed}"
        for video_id, expected, observed in wrong_size[:5]
    )
    details.extend(
        f"sha256 {video_id}: expected={expected} observed={observed}"
        for video_id, expected, observed in wrong_hash[:5]
    )
    raise SystemExit(
        f"Media validation failed: missing={len(missing)}, wrong_type={len(wrong_type)}, "
        f"wrong_size={len(wrong_size)}, wrong_hash={len(wrong_hash)}\n" + "\n".join(details)
    )

print("[CHECK] Media: 490 files + 310 image sequences passed size and SHA-256 validation")
PY

python - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in the active Python environment")
x = torch.ones(1, device="cuda")
print(f"[CHECK] Python: {sys.version.split()[0]}")
print(f"[CHECK] PyTorch: {torch.__version__}; CUDA: {torch.version.cuda}")
print(f"[CHECK] CUDA tensor: {x.item()}; GPU: {torch.cuda.get_device_name(0)}")
PY

info "已校验受限私有答案；不会将其路径注入模型运行会话。"

if [[ "$MODE" == "check-only" ]]; then
  info "全部检查通过；没有启动模型任务。"
  exit 0
fi

command -v tmux >/dev/null 2>&1 || fail "未找到 tmux。"
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  fail "tmux 会话已存在：$SESSION_NAME。使用 --attach 进入，或 --status 查看。"
fi

SESSION_ENV="$OUTPUT_ROOT/session_env.sh"
{
  printf '#!/usr/bin/env bash\n'
  printf 'export GOODVISION_ROOT=%q\n' "$GOODVISION_ROOT"
  printf 'export GOODVISION_MEDIA_ROOT=%q\n' "$GOODVISION_MEDIA_ROOT"
  printf 'export GOODVISION_OUTPUT_ROOT=%q\n' "$GOODVISION_OUTPUT_ROOT"
  printf 'export GOODVISION_BENCHMARK_MANIFEST=%q\n' "$GOODVISION_BENCHMARK_MANIFEST"
  printf 'export GOODVISION_PUBLIC_QUESTIONS=%q\n' "$GOODVISION_PUBLIC_QUESTIONS"
  printf 'cd %q\n' "$GOODVISION_ROOT"
  printf 'printf %q\n' "Benchmark-800 session ready. No model evaluation has started."
  printf 'exec bash -i\n'
} >"$SESSION_ENV"
chmod 700 "$SESSION_ENV"

START_COMMAND="eval \"\$(\"$RESOLVED_CONDA_EXE\" shell.bash hook)\" && conda activate $(printf '%q' "$CONDA_ENV") && exec bash $(printf '%q' "$SESSION_ENV")"
tmux new-session -d -s "$SESSION_NAME" -c "$REPO_ROOT" "$START_COMMAND"

info "已启动 tmux 会话：$SESSION_NAME"
info "进入会话：tmux attach -t $SESSION_NAME"
info "退出但保持后台运行：按 Ctrl-b，再按 d"
info "当前没有启动模型 benchmark；runner 命令确定后在该会话中运行。"
