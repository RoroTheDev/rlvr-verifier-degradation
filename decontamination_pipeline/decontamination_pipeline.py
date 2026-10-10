"""Remove GSM8K training examples that overlap with MATH-500 test examples."""

from __future__ import annotations

import argparse
from json import dumps
from re import sub
from difflib import SequenceMatcher
from pathlib import Path

from datasets import load_dataset
from datasketch import MinHash
from tqdm import tqdm

MATH500_DATASET = "HuggingFaceH4/MATH-500"
GSM8K_DATASET = "openai/gsm8k"
DEFAULT_QWEN_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def normalize(text: str) -> str:
    """Normalize superficial formatting differences."""
    text = text.lower()
    text = text.replace(r"\left(", "(").replace(r"\right)", ")")
    text = text.replace(r"\:", "").replace(r"\,", "")
    text = sub(r"\s+", " ", text)
    text = sub(r"\s*([+\-*/=(){}\[\]<>])\s*", r"\1", text)
    return text.strip()


def _shingles(text: str, ngram_size: int = 5) -> set[str]:
    tokens = text.split()
    if len(tokens) < ngram_size:
        return set(tokens)
    return {
        " ".join(tokens[i : i + ngram_size])
        for i in range(len(tokens) - ngram_size + 1)
    }


def minhash(text: str, num_perm: int = 128, ngram_size: int = 5) -> MinHash:
    signature = MinHash(num_perm=num_perm)
    for shingle in _shingles(text, ngram_size):
        signature.update(shingle.encode("utf-8"))
    return signature


class QwenJudge:
    """Decontamination Semantic Judge using model Qwen analyzing Math-500 and GSM8K."""

    def __init__(self, model_name: str) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Qwen mode requires torch and transformers. "
                "Install requirements-qwen.txt first."
            ) from exc

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()

    def matches(self, candidate: str, reference: str) -> bool:
        messages = [
            {
                "role": "system",
                "content": (
                    "Return only YES or NO. Return YES if the two texts describe "
                    "the same underlying math problem, even if worded differently."
                ),
            },
            {"role": "user", "content": f"REFERENCE:\n{reference}\n\nCANDIDATE:\n{candidate}"},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=3,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        answer = self.tokenizer.decode(
            output[0, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True
        ).strip().upper()
        return answer.startswith("YES")


def run(
    threshold: float = 0.80,
    use_qwen: bool = False,
    qwen_limit: int | None = None,
    output_dir: str | Path = "results",
    *,
    math500: object | None = None,
    gsm8k: object | None = None,
) -> None:
    """Run decontamination, optionally using in-memory datasets for testing."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    if qwen_limit is not None and qwen_limit < 0:
        raise ValueError("qwen_limit must be non-negative")

    if math500 is None:
        print("Loading MATH-500 test split...")
        math500 = load_dataset(MATH500_DATASET, split="test")
        if len(math500) != 500:
            raise ValueError("Expected exactly 500 MATH-500 test examples")
    if gsm8k is None:
        print("Loading GSM8K train split...")
        gsm8k = load_dataset(GSM8K_DATASET, "main", split="train")
        if len(gsm8k) != 7473:
            raise ValueError("Expected exactly 7473 GSM8K training examples")

    exact: set[str] = set()
    references: list[str] = []
    reference_shingles: list[set[str]] = []
    shingle_to_references: dict[str, set[int]] = {}

    for i, row in enumerate(math500):
        problem = row["problem"]
        normalized = normalize(problem)
        exact.add(normalized)
        references.append(problem)
        shingles = _shingles(normalized)
        reference_shingles.append(shingles)
        for shingle in shingles:
            shingle_to_references.setdefault(shingle, set()).add(i)

    judge = QwenJudge(DEFAULT_QWEN_MODEL) if use_qwen else None
    clean: list[dict] = []
    removed_exact = 0
    removed_fuzzy = 0
    removed_qwen = 0
    qwen_calls = 0

    for row in tqdm(gsm8k, desc="Checking GSM8K"):
        question = row["question"]
        normalized = normalize(question)

        if normalized in exact:
            removed_exact += 1
            continue

        candidate_shingles = _shingles(normalized)
        if threshold == 0:
            candidate_ids = range(len(references))
        elif candidate_shingles:
            candidate_ids = set().union(
                *(shingle_to_references.get(shingle, set()) for shingle in candidate_shingles)
            )
        else:
            candidate_ids = set()

        fuzzy_match = False
        for candidate_id in candidate_ids:
            reference = reference_shingles[candidate_id]
            intersection_size = len(candidate_shingles & reference)
            union_size = len(candidate_shingles) + len(reference) - intersection_size
            if union_size and intersection_size / union_size >= threshold:
                fuzzy_match = True
                break

        if fuzzy_match:
            removed_fuzzy += 1
            continue

        if judge and (qwen_limit is None or qwen_calls < qwen_limit):
            nearest = sorted(
                references,
                key=lambda reference: SequenceMatcher(
                    None, normalized, normalize(reference)
                ).ratio(),
                reverse=True,
            )[:3]
            qwen_calls += 1
            if any(judge.matches(question, reference) for reference in nearest):
                removed_qwen += 1
                continue

        clean.append(row)

    output_path_dir = Path(output_dir)
    output_path_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_path_dir / "clean_gsm8k_train.jsonl"
    with output_path.open("w", encoding="utf-8") as stream:
        for row in clean:
            stream.write(dumps(row, ensure_ascii=False) + "\n")

    test_output_path = output_path_dir / "math500_test.jsonl"
    with test_output_path.open("w", encoding="utf-8") as stream:
        for row in math500:
            stream.write(dumps(row, ensure_ascii=False) + "\n")

    report = {
        "candidate": f"{GSM8K_DATASET}:main/train",
        "reference": f"{MATH500_DATASET}:test",
        "total_math500": len(math500),
        "retained_math500": len(math500),
        "total_gsm8k": len(gsm8k),
        "removed_exact": removed_exact,
        "removed_fuzzy": removed_fuzzy,
        "removed_qwen": removed_qwen,
        "removed_total": removed_exact + removed_fuzzy + removed_qwen,
        "clean": len(clean),
        "fuzzy_method": "exact Jaccard similarity over 5-token shingles",
        "fuzzy_threshold": threshold,
        "qwen_enabled": use_qwen,
        "qwen_model": DEFAULT_QWEN_MODEL if use_qwen else None,
        "qwen_caveat": (
            "Qwen judges supplied text pairs; it does not reveal Qwen's private "
            "pretraining data."
        ),
    }
    (output_path_dir / "report.json").write_text(
        dumps(report, indent=2),
        encoding="utf-8",
    )

    semantic_method = (
        f" An optional {DEFAULT_QWEN_MODEL} judge also screened each eligible "
        "training question against its three closest reference questions, ranked "
        f"by character-sequence similarity; {qwen_calls:,} training questions "
        "were screened. This is not an exhaustive semantic comparison."
        if use_qwen else " The optional Qwen semantic judge was not used."
    )
    markdown_report = f"""# Data Split & Decontamination Report (Week 1 Replication)

## Dataset and split

Training uses GSM8K (`{GSM8K_DATASET}`, configuration `main`, split `train`).
Evaluation uses MATH-500 (`{MATH500_DATASET}`, split `test`). Filtering applies
only to the training split; the evaluation split is exported without altering
its records, fields, values, or order.

## Decontamination method (Row 13)

The decontamination script compares GSM8K questions against MATH-500 problems.
It first removes exact matches after lowercasing, whitespace normalization,
and limited mathematical-formatting normalization. It then computes exact
Jaccard similarity between sets of contiguous five-token shingles and removes
training questions with similarity at or above {threshold:.2f} to any test
problem. For texts shorter than five tokens, individual tokens are used.
An inverted shingle index retrieves all pairs with nonzero overlap; the
similarity decision itself does not use approximate MinHash/LSH retrieval.
{semantic_method.strip()}

## Filtering results

- **MATH-500 (test):** {len(math500):,}/{len(math500):,} examples retained.
- **GSM8K (train):** {len(gsm8k):,} original examples →
  {report['removed_total']:,} examples flagged by the configured criteria →
  {report['removed_total']:,} examples removed from training.
- **Removal breakdown:** {removed_exact:,} exact matches;
  {removed_fuzzy:,} additional shingle-similarity matches;
  {removed_qwen:,} additional semantic-judge matches.
- **Delivered training set:** {len(gsm8k):,} − {report['removed_total']:,} =
  **{len(clean):,} examples**.

These counts concern detected question overlap under the stated rules, not a
proof that all paraphrases or semantic equivalents have been excluded. The
procedure does not inspect model pretraining data or compare solution text.
Filtering alone cannot establish zero-shot evaluation or guarantee that
MATH-500 is out of distribution for the trained model.

## Deviations from the original paper

1. **Training-set decontamination.** This replication explicitly screens
   GSM8K training questions against MATH-500 and excludes every detected match.
   The original paper's filtering protocol has not been verified here; using
   unfiltered GSM8K in the original study must therefore not be assumed.
   The number of records excluded in this run is {report['removed_total']:,}.
   A zero-removal result leaves the training records unchanged despite the
   additional screening step.
2. **Prompt formatting.** This export introduces no training or evaluation
   system prompt, question wrapper, or chat template. Retained dataset records
   are written verbatim as JSON objects; normalization is used only for
   comparison. The original paper's prompts and the downstream modeling
   prompts have not been checked, so prompt equivalence remains unverified.
   An optional judge's prompt, if enabled, is solely a filtering instruction,
   not a modeling prompt.

## Modeling handoff

- [Clean GSM8K training data](./clean_gsm8k_train.jsonl): original `question`
  and `answer` fields, in retained source order.
- [Unmodified MATH-500 test data](./math500_test.jsonl): all original fields
  and records, in source order.
- [Machine-readable filtering counts](./report.json).

Both data artifacts are UTF-8 JSONL. Load them separately because the source
schemas differ, then place them in a `DatasetDict`:

```python
from datasets import DatasetDict, load_dataset

splits = DatasetDict({{
    "train": load_dataset("json", data_files="clean_gsm8k_train.jsonl", split="train"),
    "test": load_dataset("json", data_files="math500_test.jsonl", split="train"),
}})
```

The example assumes this directory as the working directory. Source dataset
revisions are not pinned by this pipeline; retain these exported files when
reproducing this run.
"""
    (output_path_dir / "decontamination_report.md").write_text(
        markdown_report, encoding="utf-8"
    )

    print("\nDecontamination report:")
    for key, value in report.items():
        print(f"  {key}: {value}")
    print(f"\nSaved clean dataset to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--use-qwen", action="store_true")
    parser.add_argument("--qwen-limit", type=int)
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")
    run(args.threshold, args.use_qwen, args.qwen_limit, args.output_dir)


if __name__ == "__main__":
    main()
