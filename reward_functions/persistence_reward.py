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
    group_resampled   Plesner et al.'s headline "group x rollout" noise: with
                probability MIXUP_GROUP_P the whole outcome matrix of a prompt's
                group is inverted -- every rollout's test outcomes flip at once, so
                a rollout's reward f in [0, 1] becomes 1 - f. The coin is drawn
                per (task, encounter): all rollouts of one prompt in one step share
                it, and it is redrawn the next time that prompt is seen. The
                encounter number must be in extra_info["encounter"] (see
                data_prep/mbpp_prep.py); a missing one raises rather than silently
                degrading to persistent noise. A format failure (negative honest
                score, e.g. MBPP's -0.25 for "no code block") has no outcome matrix
                and is never flipped.
    group_persistent  Identical to group_resampled except the coin is keyed on the
                task alone, so a flipped prompt is flipped on every encounter for
                the whole run. Same marginal rate, same structure -- the ONLY
                difference between the two is persistence, which is the paper's
                H2 contrast.

Noise applies to training data only: rows whose extra_info["split"] is in
MIXUP_CLEAN_SPLITS (default validation,val,test) are always scored honestly, so
validation accuracy is measured against the real verifier even when the training
reward is corrupted. Rows with no split are treated as training data.

Task identity: (data_source, extra_info["split"], extra_info["index"]) when
present -- that is what verl's GSM8K/MATH preprocessing provides, and it is
unique per task. Falling back to extra_info["question"], and only as a last
resort to ground_truth. ground_truth alone is NOT a safe identity: for GSM8K it
is just the final numeric answer, so every question answering "18" would share
one coin and the mask would track answer value instead of task.

Labels: in the TPR/FPR modes (resampled, persistent) the verifier is modelled as a
BINARY classifier. The honest result from verl's own verifier is reduced to
correct/incorrect (score >= MIXUP_CORRECT_THRESHOLD, default 1.0), which also
handles verifiers that return dicts, use a +-1 scale (math_dapo), or give partial
credit (code). These modes emit {0.0, 1.0}.
The group modes need the honest reward's magnitude (a fraction of tests passed, to
be inverted), so they always emit it on its own scale. clean mode does too when
MIXUP_REWARD_SCALE=continuous; keep clean and noisy runs on the same scale or the
Dr.GRPO ablation (where reward scale is not normalised away) is confounded.
MBPP data (data_source "mbpp") is scored by reward_functions/mbpp_scoring.py;
everything else goes to verl's default_compute_score.

Return value is a dict (verl's naive reward manager accepts this and forwards
every key as reward_extra_info):
    score            the label the trainer actually learns from (possibly noisy)
    acc              HONEST correctness. Do not report training accuracy from
                     `score`: with a plain float return verl stores the noisy
                     score as "acc", silently measuring the corrupted signal.
    honest_reward    the honest reward on the verifier's own scale
    mixup_eligible   1.0 if the task matched the targeting selector (and is training data)
    mixup_covered    1.0 if the task carried noise at all (coverage draw; for the group
                     modes, simply eligibility)
    mixup_flipped    1.0 if the label was inverted (TPR/FPR modes: score != acc;
                     group modes: the group's outcome matrix was inverted)
verl's naive reward manager is meant to forward these as reward_extra_info, but at
the verl commit this project pins the trainer.rollout_data_dir dump only contains
gts/input/output/score/step/uid -- they do NOT appear there, and whether they reach
training metrics is unconfirmed. Treat the per-call log below (MIXUP_LOG_DIR) as the
authoritative record of honest correctness vs. the noisy label.

Config (env vars -- verl's reward hook takes no extra kwargs beyond the fixed
signature, so this is the config surface):
    MIXUP_MODE               clean | resampled | persistent |
                              group_resampled | group_persistent   (default: clean)
    MIXUP_GROUP_P            group modes: P(a prompt's whole group is inverted)
                                                                   (default: 0.0)
    MIXUP_REWARD_SCALE       binary | continuous. continuous keeps the honest reward
                              (e.g. fraction of tests passed) in clean mode; the group
                              modes are always continuous; the TPR/FPR modes are always
                              binary and raise if continuous is requested  (default: binary)
    MIXUP_CLEAN_SPLITS       comma list of extra_info["split"] values that are never
                              corrupted                      (default: validation,val,test)
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
import sys
import time

from verl.utils.reward_score import default_compute_score

# verl loads this file by path (not as part of a package), so make the sibling
# scorer importable explicitly.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import mbpp_scoring  # noqa: E402


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
MIXUP_GROUP_P = _env_float("MIXUP_GROUP_P", 0.0)
MIXUP_REWARD_SCALE = os.environ.get("MIXUP_REWARD_SCALE", "binary").strip().lower()
MIXUP_CLEAN_SPLITS = frozenset(
    s.strip() for s in os.environ.get("MIXUP_CLEAN_SPLITS", "validation,val,test").split(",") if s.strip()
)

_LABEL_MODES = ("resampled", "persistent")  # TPR/FPR noise on a binary label
_GROUP_MODES = ("group_resampled", "group_persistent")  # whole-group inversion
_ALL_MODES = ("clean",) + _LABEL_MODES + _GROUP_MODES

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


def _honest_raw(data_source, solution_str, ground_truth, extra_info, kwargs):
    """The honest verifier's raw result: our MBPP scorer, else verl's built-in one."""
    if data_source == "mbpp":
        return mbpp_scoring.score(solution_str, ground_truth)
    return default_compute_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        **kwargs,
    )


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    if MIXUP_MODE not in _ALL_MODES:
        raise ValueError(f"Unknown MIXUP_MODE: {MIXUP_MODE!r} (expected {'|'.join(_ALL_MODES)})")
    if MIXUP_REWARD_SCALE not in ("binary", "continuous"):
        raise ValueError(f"Unknown MIXUP_REWARD_SCALE: {MIXUP_REWARD_SCALE!r} (expected binary|continuous)")
    if MIXUP_MODE in _LABEL_MODES and MIXUP_REWARD_SCALE == "continuous":
        raise ValueError("MIXUP_REWARD_SCALE=continuous is not defined for the TPR/FPR modes: they emit a binary label")

    honest_score = _honest_score(_honest_raw(data_source, solution_str, ground_truth, extra_info, kwargs))
    is_correct = honest_score >= MIXUP_CORRECT_THRESHOLD
    continuous = MIXUP_MODE in _GROUP_MODES or MIXUP_REWARD_SCALE == "continuous"

    info = extra_info if isinstance(extra_info, dict) else {}
    split = info.get("split")

    eligible = False
    covered = False
    inverted = False
    label = is_correct
    score = honest_score if continuous else (1.0 if is_correct else 0.0)
    key = None

    # Noise is for training data only: validation/test rows are scored honestly.
    if MIXUP_MODE != "clean" and split not in MIXUP_CLEAN_SPLITS:
        eligible = _is_eligible(extra_info)

    if eligible:
        key = _task_identity(data_source, ground_truth, extra_info)
        if MIXUP_MODE in _LABEL_MODES:
            # Coverage: only a subset of eligible tasks carry noise at all. The
            # covered *set* is deterministic (fixed by the hash), not resampled --
            # coverage is how much of the data is affected, not an extra source
            # of randomness on top of the flip itself.
            covered = _deterministic_unit_interval(key, "coverage") < MIXUP_COVERAGE
            if covered:
                threshold = MIXUP_TPR if is_correct else MIXUP_FPR
                coin = _deterministic_unit_interval(key, "flip") if MIXUP_MODE == "persistent" else _rng.random()
                label = coin < threshold
                score = 1.0 if label else 0.0
        else:
            # Group modes: one coin per prompt group (per encounter, if resampled).
            # Every rollout of the group sees the same coin because they share the
            # row's extra_info, hence the same task identity and encounter.
            covered = True
            coin_key = key
            if MIXUP_MODE == "group_resampled":
                encounter = info.get("encounter")
                if encounter is None:
                    raise ValueError(
                        "MIXUP_MODE=group_resampled needs extra_info['encounter'] "
                        "(build the data with data_prep/mbpp_prep.py); refusing to guess"
                    )
                coin_key = f"{key}|encounter={encounter}"
            # A negative honest score is a format failure: no outcome matrix, nothing to invert.
            inverted = honest_score >= 0.0 and _deterministic_unit_interval(coin_key, "group_flip") < MIXUP_GROUP_P
            if inverted:
                score = 1.0 - honest_score

    flipped = inverted if MIXUP_MODE in _GROUP_MODES else (label != is_correct)
    result = {
        "score": float(score),
        "acc": 1.0 if is_correct else 0.0,
        "honest_reward": honest_score,
        "mixup_eligible": 1.0 if eligible else 0.0,
        "mixup_covered": 1.0 if covered else 0.0,
        "mixup_flipped": 1.0 if flipped else 0.0,
    }

    if MIXUP_LOG_DIR is not None:
        _log_call(
            {
                "ts": round(time.time(), 3),
                "mode": MIXUP_MODE,
                "tpr": MIXUP_TPR,
                "fpr": MIXUP_FPR,
                "coverage": MIXUP_COVERAGE,
                "group_p": MIXUP_GROUP_P,
                "reward_scale": "continuous" if continuous else "binary",
                "mask_seed": MIXUP_MASK_SEED,
                "data_source": data_source,
                "split": split,
                "index": info.get("index"),
                "encounter": info.get("encounter"),
                "task": hashlib.sha256((key or _task_identity(data_source, ground_truth, extra_info)).encode()).hexdigest()[:16],
                "honest_score": honest_score,
                **result,
            }
        )

    return result
