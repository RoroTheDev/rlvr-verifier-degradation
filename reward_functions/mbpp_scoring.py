"""MBPP scorer for the verl reward function (data_source "mbpp").

Follows Plesner et al.'s MBPP reward as described in their reproducibility script:
    - the first ```python fenced block of the response is the program;
    - no code block => -0.25 (format penalty), no tests are run;
    - otherwise the reward is the fraction of the problem's unit tests that pass;
    - a "correct" response is one that passes every test (score exactly 1.0).

Differences from that script, on purpose:
    - Qwen3 "thinking" output: only text after the last </think> is searched for
      code, so a fenced block inside the reasoning is never executed as the answer.
      An opened-but-never-closed <think> (cut off at the length limit) has no
      answer and gets the format penalty. With thinking off this changes nothing.
    - Result markers carry a random per-call nonce, so a response that prints
      "PASS" lines cannot award itself credit.
    - The program runs in a throw-away working directory with a minimal
      environment, so generated code cannot read this process's secrets
      (e.g. WANDB_API_KEY). It is still NOT a security sandbox -- it runs with the
      worker's permissions, so only use it inside a disposable pod.
    - A program that hits the time limit gets 0.0 (no partial credit for tests that
      ran before the hang).

ground_truth is a JSON string: {"tests": ["assert f(1) == 2", ...], "setup": "..."}.
"""

import json
import os
import re
import secrets
import subprocess
import sys
import tempfile
import textwrap

FORMAT_PENALTY = -0.25
DEFAULT_TIMEOUT_S = float(os.environ.get("MBPP_TIMEOUT_S", "10"))
_FENCE = re.compile(r"```python(.*?)```", re.DOTALL)


def extract_code(response: str):
    """The program in the response, or None if there is no usable answer."""
    if "</think>" in response:
        text = response.rsplit("</think>", 1)[1]
    elif "<think>" in response:
        return None  # still thinking when the length limit hit
    else:
        text = response
    match = _FENCE.search(text)
    return match.group(1).strip() if match else None


def _program(code: str, setup: str, tests, nonce: str) -> str:
    parts = [setup or "", code, ""]
    for i, test in enumerate(tests):
        parts.append("try:")
        parts.append(textwrap.indent(test.strip(), "    "))
        parts.append(f"    print('__R{nonce}_{i}_PASS')")
        parts.append("except BaseException:")
        parts.append(f"    print('__R{nonce}_{i}_FAIL')")
    return "\n".join(parts) + "\n"


def _clean_env() -> dict:
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0"}
    for key in ("SYSTEMROOT", "TEMP", "TMP"):  # Windows needs these to start python
        if os.environ.get(key):
            env[key] = os.environ[key]
    return env


def run_tests(code: str, tests, setup: str = "", timeout: float = DEFAULT_TIMEOUT_S):
    """One bool per test. Everything fails if the program crashes before the tests
    or exceeds the time limit."""
    nonce = secrets.token_hex(8)
    program = _program(code, setup, tests, nonce)
    with tempfile.TemporaryDirectory() as workdir:
        path = os.path.join(workdir, "solution.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(program)
        try:
            proc = subprocess.run(
                [sys.executable, "-I", path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=workdir,
                env=_clean_env(),
            )
        except subprocess.TimeoutExpired:
            return [False] * len(tests)
    return [f"__R{nonce}_{i}_PASS" in proc.stdout for i in range(len(tests))]


def score(response: str, ground_truth, timeout: float = DEFAULT_TIMEOUT_S) -> float:
    spec = json.loads(ground_truth) if isinstance(ground_truth, str) else dict(ground_truth)
    tests = list(spec["tests"])
    code = extract_code(response)
    if code is None:
        return FORMAT_PENALTY
    results = run_tests(code, tests, spec.get("setup") or "", timeout)
    return sum(results) / len(results) if results else 0.0
