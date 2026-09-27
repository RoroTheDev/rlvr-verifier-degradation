"""Custom verl reward function implementing the project's verifier-noise model.

Ports Egashira et al.'s confusion-matrix noise model (TPR/FPR flipping,
targeted subpopulations) from the TRL-based prototype (eth-sri/llm-verifier-noise),
extended with a persistence mechanism suited to verl's distributed (Ray
multi-worker) execution model.

Wire this in via, e.g.:
    custom_reward_function.path=reward_functions/persistence_reward.py
    custom_reward_function.name=compute_score

Modes (env var MIXUP_MODE):
    clean       Honest scoring only. Default -- safe if nothing is configured.
    resampled   Confusion-matrix noise, freshly redrawn every call. This is
                the symmetric/per-epoch-resampled baseline (H1; Plesner et al.).
    persistent  Same noise model, but the flip for a given task is
                DETERMINISTIC -- derived by hashing the task's identity, not
                drawn from a shared mutable cache. The same task therefore
                gets the same label every time it's scored: across rollouts,
                across epochs, across every Ray worker process, with no
                coordination needed between workers. This determinism is
                what "persistence" means experimentally (H2): a standing,
                one-directional subsidy on specific tasks, not noise that
                averages out over time.

                Note this is a deliberate adaptation, not a literal port --
                the TRL prototype used an in-memory cache dict, which only
                gives a consistent answer within one process. verl scores
                responses from many Ray worker processes; hashing the task's
                identity gives every worker the same answer independently,
                which a shared cache would need explicit coordination to do.

Config (env vars -- verl's custom_reward_function hook takes no extra kwargs
beyond the fixed signature, so this is the config surface):
    MIXUP_MODE            clean | resampled | persistent      (default: clean)
    MIXUP_TPR             true positive rate                   (default: 1.0)
    MIXUP_FPR             false positive rate                  (default: 0.0)
    MIXUP_COVERAGE        fraction of eligible tasks that carry noise at all;
                           the rest are scored honestly          (default: 1.0)
    MIXUP_TARGET_FIELD    optional field name in extra_info to target on.
                           Unset => every task is eligible (untargeted/diffuse
                           error). Set => only tasks whose extra_info[field]
                           satisfies the selector are eligible (targeted
                           error); everything else stays clean. Mirrors
                           Egashira's targeted_buckets.
    MIXUP_TARGET_OP       eq | ne | lt | lte | gt | gte         (default: eq)
    MIXUP_TARGET_VALUE    comparison value (parsed as float if possible,
                           else compared as a string)
"""

import hashlib
import json
import os
import random

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
MIXUP_TARGET_FIELD = os.environ.get("MIXUP_TARGET_FIELD") or None
MIXUP_TARGET_OP = os.environ.get("MIXUP_TARGET_OP", "eq").strip().lower()

_OPS = {
    "eq": lambda a, b: a == b,
    "ne": lambda a, b: a != b,
    "lt": lambda a, b: a < b,
    "lte": lambda a, b: a <= b,
    "gt": lambda a, b: a > b,
    "gte": lambda a, b: a >= b,
}


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


def _task_identity(data_source, ground_truth) -> str:
    """Stable identity for a task, independent of which rollout/epoch/worker
    is currently scoring it. ground_truth is the anchor: it's fixed per task
    and always present, unlike solution_str, which is different every
    rollout by construction."""
    try:
        return json.dumps({"data_source": data_source, "ground_truth": ground_truth}, sort_keys=True, default=str)
    except TypeError:
        return f"{data_source}|{ground_truth}"


def _deterministic_unit_interval(key: str, salt: str) -> float:
    """Deterministic pseudo-random float in [0, 1) derived by hashing.
    Same (key, salt) always produces the same value: in this process, in any
    other Ray worker process, this epoch, next epoch. This is the entire
    mechanism behind persistence -- no shared cache required."""
    digest = hashlib.sha256(f"{salt}|{key}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(1 << 64)


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    # Honest ground truth, via verl's own built-in verifier for this dataset.
    honest_score = default_compute_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        **kwargs,
    )

    if MIXUP_MODE == "clean":
        return honest_score

    if MIXUP_MODE not in ("resampled", "persistent"):
        raise ValueError(f"Unknown MIXUP_MODE: {MIXUP_MODE!r} (expected clean|resampled|persistent)")

    if not _is_eligible(extra_info):
        return honest_score

    key = _task_identity(data_source, ground_truth)

    # Coverage: only a subset of eligible tasks carry noise at all. The
    # covered *set* is deterministic (fixed by the hash), not resampled --
    # coverage is about how much of the data is affected, not an extra
    # source of randomness on top of the flip itself.
    if _deterministic_unit_interval(key, "coverage") >= MIXUP_COVERAGE:
        return honest_score

    is_correct = bool(honest_score)
    threshold = MIXUP_TPR if is_correct else MIXUP_FPR
    coin = _deterministic_unit_interval(key, "flip") if MIXUP_MODE == "persistent" else random.random()

    return 1.0 if coin < threshold else 0.0
