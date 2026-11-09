"""Prepare a fresh local native trial from this maintained adapter; never execute it."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

if __package__:
    from . import VERSION
else:
    from __init__ import VERSION

HERE = Path(__file__).resolve().parent
ALLOWED_MODELS = ("deepseek-v4-pro",)
# Raised from 25 on 2026-11-03 for the relation-graph fallback's ¥50 per question.
# Trials that pass no cap keep trial_helpers' default of 25, as Flash and Pro do.
SEALED_HARD_CAP = Decimal(50)
TEMPLATE_FILES = (
    "run_native.py", "trial_helpers.py", "budget_adapter.py", "task_supervisor.py",
    "answer_label_guard.py", "tool_classification.py", "sandbox_registry.py",
    "PROFILE.yaml", "launch.ps1", "test_native.py", "test_linux_adapter.py",
    "test_budget_adapter.py", "test_verified_pricing.py", "PRICING_VERIFICATION.json",
    "TEMPLATE_ORIGIN.json",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def valid_hard_cap(value):
    """A per-trial ledger cap may only lower the sealed one."""
    try:
        cap = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError("invalid_hard_cap") from exc
    if not 0 < cap <= SEALED_HARD_CAP:
        raise ValueError("invalid_hard_cap")
    return cap


def prepare(question_file, output, *, profile_file=None, source_dir=None, model="", hard_cap=""):
    """Copy only versioned adapter inputs, an explicit profile and the current source manifest.

    The default profile and fee/model/time policies are retained from the sealed
    adapter. A new opt-in profile must be selected explicitly with profile_file.
    This function grants no paid-run authorization and makes no service calls.
    """
    question_file, output = Path(question_file).resolve(), Path(output).resolve()
    source_dir = Path(source_dir).resolve() if source_dir else HERE.parent
    profile_file = Path(profile_file).resolve() if profile_file else HERE / "PROFILE.yaml"
    question = json.loads(question_file.read_text(encoding="utf-8-sig"))
    if not isinstance(question, dict) or set(question) != {"id", "question", "question_sha256"}:
        raise ValueError("question_only_three_keys_required")
    if any(not isinstance(question[key], str) or not question[key].strip() for key in question):
        raise ValueError("question_id_text_and_hash_required")
    if hashlib.sha256(question["question"].encode()).hexdigest() != question["question_sha256"]:
        raise ValueError("question_hash_mismatch")
    if output == source_dir or source_dir in output.parents:
        raise ValueError("trial_directory_must_be_outside_source")
    if output.exists():
        raise FileExistsError("fresh_trial_directory_required")
    for name in TEMPLATE_FILES:
        if not (HERE / name).is_file():
            raise FileNotFoundError("missing_template:" + name)
    if not profile_file.is_file() or not source_dir.is_dir():
        raise ValueError("profile_and_source_required")
    source = {
        path.relative_to(source_dir).as_posix(): sha(path)
        for path in source_dir.rglob("*")
        if path.is_file() and not any(part in {"__pycache__", ".pytest_cache", ".git"} for part in path.parts)
    }
    output.mkdir(parents=True, exist_ok=False)
    for name in TEMPLATE_FILES:
        shutil.copyfile(HERE / name, output / name)
    shutil.copyfile(profile_file, output / "PROFILE.yaml")
    # Preserve Docker/fee/network branches; bind the new checkout location only.
    launch = (output / "launch.ps1").read_text(encoding="utf-8")
    old_root = "$projectRoot='__PROJECT_ROOT__'"
    old_source = "$sourceDir=Join-Path $projectRoot 'frontier_research'"
    old_docker_param = "param([ValidateSet('test','execute','cleanup')][string]$Mode='test')"
    old_docker_exe = "$dockerExe='__DOCKER_EXE__'"
    if (launch.count(old_root) != 1 or launch.count(old_source) != 1
            or launch.count(old_docker_param) != 1 or launch.count(old_docker_exe) != 1):
        raise RuntimeError("launch_template_path_anchors_changed")
    escaped_root = source_dir.parent.as_posix().replace("'", "''")
    escaped_source = source_dir.as_posix().replace("'", "''")
    launch = launch.replace(old_root, "$projectRoot='" + escaped_root + "'")
    launch = launch.replace(old_source, "$sourceDir='" + escaped_source + "'")
    launch = launch.replace(
        old_docker_param,
        "param([ValidateSet('test','execute','cleanup')][string]$Mode='test',[string]$DockerCommand='docker')",
    )
    launch = launch.replace(old_docker_exe, "$dockerExe=$DockerCommand")
    overrides = []
    if model:
        if model not in ALLOWED_MODELS:
            raise ValueError("unsupported_model_override")
        overrides.append(f"OPENAI_MODEL={model}")
    if hard_cap:
        overrides.append(f"TRIAL_HARD_CNY={valid_hard_cap(hard_cap)}")
    if overrides:
        # Docker gives -e precedence over --env-file; offline test mode is untouched.
        old_env_file = "$dockerArgs+=@('--env-file',$envFile)"
        if launch.count(old_env_file) != 1:
            raise ValueError("unsupported_model_override")
        extra = "".join(f",'-e','{item}'" for item in overrides)
        launch = launch.replace(old_env_file, f"$dockerArgs+=@('--env-file',$envFile{extra})")
    (output / "launch.ps1").write_text(launch, encoding="utf-8", newline="\n")
    save_new(output / "QUESTION.json", question)
    save_new(output / "IMAGE_SOURCE_BASELINE.json", {"files": source})
    metadata = {
        "utc": datetime.now(timezone.utc).isoformat(), "mode": "prepared_only",
        "adapter_version": VERSION, "adapter_template": str(HERE), "model_override": model,
        "hard_cap_cny": str(hard_cap),
        "question_id": question["id"], "question_sha256": question["question_sha256"],
        "question_input_sha256": sha(question_file), "source_directory": str(source_dir),
        "source_files": len(source), "profile_input": str(profile_file),
        "profile_sha256": sha(output / "PROFILE.yaml"),
        "adapter_files": {name: sha(output / name) for name in TEMPLATE_FILES},
        "prior_run_data_copied": False, "paid_runs_started": 0,
        "paid_run_authorization": "not granted by preparation",
        "parameter_policy": "original bounded fee/model/thinking/output/retry/deadline policies retained; only explicitly supplied profile changes",
        "fingerprint_coverage": "existing check_source verifies enumerated source files; no claim to full directory tamper detection",
    }
    save_new(output / "PREPARED.json", metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question-json", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--profile-file", type=Path)
    parser.add_argument("--model", default="", choices=("", *ALLOWED_MODELS))
    parser.add_argument("--hard-cap", default="", help="Ledger cap in CNY for this trial, at most the sealed 50.")
    args = parser.parse_args()
    metadata = prepare(args.question_json, args.out, profile_file=args.profile_file, model=args.model,
                       hard_cap=args.hard_cap)
    print(json.dumps({key: metadata[key] for key in
                      ("mode", "adapter_version", "question_id", "source_files", "paid_runs_started")}))


if __name__ == "__main__":
    main()
