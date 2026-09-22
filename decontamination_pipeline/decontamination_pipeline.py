"""Remove GSM8K training examples that overlap with MATH-500 test examples."""

from __future__ import annotations

import argparse
import json
import re
from difflib import SequenceMatcher
from pathlib import Path

from datasets import load_dataset
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm

MATH500_DATASET = "HuggingFaceH4/MATH-500"
GSM8K_DATASET = "openai/gsm8k"
DEFAULT_QWEN_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def normalize(text: str) -> str:
    """Normalize superficial formatting differences."""
    text = text.lower()
    text = text.replace(r"\left(", "(").replace(r"\right)", ")")
    text = text.replace(r"\:", "").replace(r"\,", "")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*([+\-*/=(){}\[\]<>])\s*", r"\1", text)
    return text.strip()


def minhash(text: str, num_perm: int = 128, ngram_size: int = 5) -> MinHash:
    signature = MinHash(num_perm=num_perm)
    tokens = text.split()
    shingles = (
        tokens
        if len(tokens) < ngram_size
        else (" ".join(tokens[i : i + ngram_size]) for i in range(len(tokens) - ngram_size + 1))
    )
    for shingle in shingles:
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
            skip_special_tokens=True,
        ).strip().upper()
        return answer.startswith("YES")


def run(threshold: float, use_qwen: bool, qwen_limit: int | None) -> None:
    print("Loading MATH-500 test split...")
    math500 = load_dataset(MATH500_DATASET, split="test")
    print("Loading GSM8K train split...")
    gsm8k = load_dataset(GSM8K_DATASET, "main", split="train")

    exact: set[str] = set()
    references: list[str] = []
    index = MinHashLSH(threshold=threshold, num_perm=128)

    for i, row in enumerate(math500):
        problem = row["problem"]
        normalized = normalize(problem)
        exact.add(normalized)
        references.append(problem)
        index.insert(str(i), minhash(normalized))

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

        candidate_ids = index.query(minhash(normalized))
        if candidate_ids:
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

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "clean_gsm8k_train.jsonl"
    with output_path.open("w", encoding="utf-8") as stream:
        for row in clean:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "candidate": f"{GSM8K_DATASET}:main/train",
        "reference": f"{MATH500_DATASET}:test",
        "total_gsm8k": len(gsm8k),
        "removed_exact": removed_exact,
        "removed_fuzzy": removed_fuzzy,
        "removed_qwen": removed_qwen,
        "clean": len(clean),
        "qwen_enabled": use_qwen,
        "qwen_model": DEFAULT_QWEN_MODEL if use_qwen else None,
        "qwen_caveat": (
            "Qwen judges supplied text pairs; it does not reveal Qwen's private "
            "pretraining data."
        ),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nDecontamination report:")
    for key, value in report.items():
        print(f"  {key}: {value}")
    print(f"\nSaved clean dataset to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--use-qwen", action="store_true")
    parser.add_argument("--qwen-limit", type=int)
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")
    run(args.threshold, args.use_qwen, args.qwen_limit)


if __name__ == "__main__":
    main()
