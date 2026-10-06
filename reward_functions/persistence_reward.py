"""Custom verl reward function implementing the project's verifier-noise model.

Ports Egashira et al.'s confusion-matrix noise model (TPR/FPR flipping,
targeted subpopulations) from the TRL-based prototype (eth-sri/llm-verifier-noise),
extended with a persistence mechanism suited to verl's distributed (Ray
multi-worker) execution model.

Wire this in via (note the `reward.` prefix -- see "CONFIG KEY GOTCHA" below):
    reward.custom_reward_function.path=reward_functions/persistence_reward.py
    reward.custom_reward_function.name=compute_score

CONFIG KEY GOTCHA (confirmed the hard way): verl's docs, and its own top-level
`custom_reward_function` key, suggest the bare `custom_reward_function.path=...`
without the `reward.` prefix. That key only exists for backward-compat migration
into the async reward-loop path. The *active* reward manager for a plain
`verl.trainer.main_ppo` + `algorithm.adv_estimator=grpo` run (reward_manager
`name: naive`) reads `config.reward.custom_reward_function` directly, which stays
`path: None` if you only set the bare key. There is no error and no crash: verl
silently falls back to its own default scoring. Check the resolved config dump at
the top of the training log for `'reward': {'custom_reward_function': {'path': ...`
-- it must show this file's path, not None.

Modes (env var MIXUP_MODE):
    clean       No noise; reward is the honest binary correctness label.
    resampled   Confusion-matrix noise, freshly redrawn every call. This is
                the symmetric/per-epoch-resampled baseline (H1; Plesner et al.).
    persistent  Same noise model, but the flip for a given task is
                DETERMINISTIC -- derived by hashing the task's identity, not
                drawn from a shared mutable cache. The same task therefore
                gets the same label every time it is scored: across rollouts,
                across epochs, across every Ray worker process, with no
                coordination needed between workers. This determinism is
                what "persistence" means experimentally (H2): a standing,
                one-directional subsidy on specific tasks, not noise that
                averages out over time.

                This is a deliberate adaptation, not a literal port: the TRL
                prototype used an in-memory cache dict, which only gives a
                consistent answer within one process. verl scores responses
                from many Ray worker processes; hashing the task's identity
                gives every worker the same answer independently.

Task identity: (data_source, extra_info["split"], extra_info["index"]) when
present -- that is what verl's GSM8K/MATH preprocessing provides, and it is
unique per task. Falling back to extra_info["question"], and only as a last
resort to ground_truth. ground_truth alone is NOT a safe identity: for GSM8K it
is just the final numeric answer, so every question answering "18" would share
one coin and the mask would track answer value instead of task.

Labels: the verifier is modelled as a BINARY classifier. The honest result from
verl's own verifier is reduced to correct/incorrect (score >= MIXUP_CORRECT_
THRESHOLD, default 1.0), which also handles verifiers that return dicts, use a
+-1 scale (math_dapo), or give partial credit (code). Every mode then emits the
same {0.0, 1.0} reward scale, so conditions stay comparable -- which matters for
the Dr.GRPO ablation, where reward scale is not normalised away.

Return value is a dict (verl's naive reward manager accepts this and forwards
every key as reward_extra_info):
    score            the label the trainer actually learns from (possibly noisy)
    acc              HONEST correctness. Do not report training accuracy from
                     `score`: with a plain float return verl stores the noisy
                     score as "acc", silently measuring the corrupted signal.
    mixup_eligible   1.0 if the task matched the targeting selector
    mixup_covered    1.0 if the task carried noise at all (coverage draw)
    mixup_flipped    1.0 if score != acc
verl's naive reward manager is meant to forward these as reward_extra_info, but at
the verl commit this project pins the trainer.rollout_data_dir dump only contains
gts/input/output/score/step/uid -- they do NOT appear there, and whether they reach
training metrics is unconfirmed. Treat the per-call log below (MIXUP_LOG_DIR) as the
authoritative record of honest correctness vs. the noisy label.

Config (env vars -- verl's reward hook takes no extra kwargs beyond the fixed
signature, so this is the config surface):
    MIXUP_MODE               clean | resampled | persistent     (default: clean)
    MIXUP_TPR                P(label=1 | honest correct)        (default: 1.0)
    MIXUP_FPR                P(label=1 | honest incorrect)      (default: 0.0)
    MIXUP_COVERAGE           fraction of eligible tasks carrying noise at all;
                              the rest are labelled honestly      (default: 1.0)
    MIXUP_MASK_SEED          salt for the persistent mask and coverage draw.
                              Independent of the training seed. Same value =>
                              same corrupted tasks in every run; vary it to
                              resample *which* tasks are corrupted    (default: 0)
    MIXUP_CORRECT_THRESHOLD  honest score at/above which a response counts as
                              correct                                 (default: 1.0)
    MIXUP_TARGET_FIELD       optional extra_info field to target on. Unset =>
                              every task is eligible (diffuse error). Set =>
                              only tasks whose extra_info[field] satisfies the
                              selector are eligible. Mirrors Egashira's
                              targeted_buckets.
    MIXUP_TARGET_OP          eq | ne | lt | lte | gt | gte         (default: eq)
    MIXUP_TARGET_VALUE       comparison value (float if parseable, else string)
    MIXUP_LOG_DIR            if set, append one JSON line per scored sample to
                              <dir>/mixup_calls.<pid>.jsonl (one file per
                              process, so concurrent workers never interleave)

Resampled mode draws from random.SystemRandom, not the global `random` module:
trainer workers commonly seed the global RNG identically, which would make every
worker emit the same "random" flips. The consequence is that resampled noise is
fresh per call but not reproducible run-to-run.
"""

import hashlib
import json
import os
import random
import time

from verl.utils.reward_score import default_compute_score


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


MIXUP_MODE = os.environ.get("MIXUP_MODE", "clean").strip().lower()
MIXUP_TPR = _env_float("MIXUP_TPR", 1.0)
MIXUP_FPR = _env_float("MIXUP_FPR", 0.0)
MIXUP_COVERAGE = _env_float("MIXUP_COVERAGE", 1.0)
MIXUP_MASK_SEED = os.environ.get("MIXUP_MASK_SEED", "0").strip()
MIXUP_CORRECT_THRESHOLD = _env_float("MIXUP_CORRECT_THRESHOLD", 1.0)
MIXUP_TARGET_FIELD = os.environ.get("MIXUP_TARGET_FIELD") or None
MIXUP_TARGET_OP = os.environ.get("MIXUP_TARGET_OP", "eq").strip().lower()
MIXUP_LOG_DIR = os.environ.get("MIXUP_LOG_DIR") or None

_OPS = {
    "eq": lambda a, b: a == b,
    "ne": lambda a, b: a != b,
    "lt": lambda a, b: a < b,
    "lte": lambda a, b: a <= b,
    "gt": lambda a, b: a > b,
    "gte": lambda a, b: a >= b,
}

_rng = random.SystemRandom()


def _parse_target_value(raw):
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return raw


MIXUP_TARGET_VALUE = _parse_target_value(os.environ.get("MIXUP_TARGET_VALUE"))


def _is_eligible(extra_info) -> bool:
    """True if this item is eligible for noise under the configured selector.
    No selector configured => every item is eligible (untargeted/diffuse)."""
    if MIXUP_TARGET_FIELD is None:
        return True
    if not extra_info or MIXUP_TARGET_FIELD not in extra_info:
        return False
    field_val = extra_info[MIXUP_TARGET_FIELD]
    op = _OPS.get(MIXUP_TARGET_OP, _OPS["eq"])
    try:
        if isinstance(MIXUP_TARGET_VALUE, float) and not isinstance(field_val, str):
            return op(float(field_val), MIXUP_TARGET_VALUE)
        return op(field_val, MIXUP_TARGET_VALUE)
    except (TypeError, ValueError):
        return False


def _task_identity(data_source, ground_truth, extra_info) -> str:
    """Stable identity for a task, independent of which rollout/epoch/worker is
    scoring it. See module docstring for why ground_truth alone is not used."""
    anchor = None
    if isinstance(extra_info, dict):
        if extra_info.get("index") is not None:
            anchor = ("index", extra_info.get("split"), extra_info.get("index"))
        elif extra_info.get("question") is not None:
            anchor = ("question", extra_info.get("question"))
    if anchor is None:
        anchor = ("ground_truth", ground_truth)
    try:
        return json.dumps({"data_source": data_source, "anchor": anchor}, sort_keys=True, default=str)
    except TypeError:
        return f"{data_source}|{anchor!r}"


def _deterministic_unit_interval(key: str, salt: str) -> float:
    """Deterministic pseudo-random float in [0, 1) derived by hashing. Same
    (mask seed, salt, key) always gives the same value: in this process, in any
    other Ray worker process, this epoch, next epoch. This is the entire
    mechanism behind persistence -- no shared cache required."""
    digest = hashlib.sha256(f"{MIXUP_MASK_SEED}|{salt}|{key}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(1 << 64)


def _honest_score(honest) -> float:
    """Reduce whatever verl's verifier returned (float, bool, or dict with a
    "score" key) to a single float."""
    if isinstance(honest, dict):
        honest = honest.get("score", 0.0)
    return float(honest)


_log_handles = {}


def _log_call(record: dict) -> None:
    """Optional per-sample JSONL log. Must never be able to break scoring."""
    if MIXUP_LOG_DIR is None:
        return
    try:
        pid = os.getpid()
        handle = _log_handles.get(pid)
        if handle is None:
            os.makedirs(MIXUP_LOG_DIR, exist_ok=True)
            handle = open(os.path.join(MIXUP_LOG_DIR, f"mixup_calls.{pid}.jsonl"), "a", buffering=1)
            _log_handles[pid] = handle
        handle.write(json.dumps(record, default=str) + "\n")
    except Exception:
        pass


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    # Honest ground truth, via verl's own built-in verifier for this dataset.
    honest = default_compute_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        **kwargs,
    )
    honest_score = _honest_score(honest)
    is_correct = honest_score >= MIXUP_CORRECT_THRESHOLD

    if MIXUP_MODE not in ("clean", "resampled", "persistent"):
        raise ValueError(f"Unknown MIXUP_MODE: {MIXUP_MODE!r} (expected clean|resampled|persistent)")

    eligible = False
    covered = False
    label = is_correct
    key = None

    if MIXUP_MODE != "clean":
        eligible = _is_eligible(extra_info)
        if eligible:
            key = _task_identity(data_source, ground_truth, extra_info)
            # Coverage: only a subset of eligible tasks carry noise at all. The
            # covered *set* is deterministic (fixed by the hash), not resampled --
            # coverage is how much of the data is affected, not an extra source
            # of randomness on top of the flip itself.
            covered = _deterministic_unit_interval(key, "coverage") < MIXUP_COVERAGE
        if covered:
            threshold = MIXUP_TPR if is_correct else MIXUP_FPR
            coin = _deterministic_unit_interval(key, "flip") if MIXUP_MODE == "persistent" else _rng.random()
            label = coin < threshold

    result = {
        "score": 1.0 if label else 0.0,
        "acc": 1.0 if is_correct else 0.0,
        "mixup_eligible": 1.0 if eligible else 0.0,
        "mixup_covered": 1.0 if covered else 0.0,
        "mixup_flipped": 1.0 if label != is_correct else 0.0,
    }

    if MIXUP_LOG_DIR is not None:
        _log_call(
            {
                "ts": round(time.time(), 3),
                "mode": MIXUP_MODE,
                "tpr": MIXUP_TPR,
                "fpr": MIXUP_FPR,
                "coverage": MIXUP_COVERAGE,
                "mask_seed": MIXUP_MASK_SEED,
                "data_source": data_source,
                "task": hashlib.sha256((key or _task_identity(data_source, ground_truth, extra_info)).encode()).hexdigest()[:16],
                "honest_score": honest_score,
                **result,
            }
        )

    return result
