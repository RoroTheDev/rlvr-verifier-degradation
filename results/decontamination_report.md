# Data Split & Decontamination Report (Week 1 Replication)

## Dataset and split

Training uses GSM8K (`openai/gsm8k`, configuration `main`, split `train`).
Evaluation uses MATH-500 (`HuggingFaceH4/MATH-500`, split `test`). Filtering applies
only to the training split; the evaluation split is exported without altering
its records, fields, values, or order.

## Decontamination method (Row 13)

The decontamination script compares GSM8K questions against MATH-500 problems.
It first removes exact matches after lowercasing, whitespace normalization,
and limited mathematical-formatting normalization. It then computes exact
Jaccard similarity between sets of contiguous five-token shingles and removes
training questions with similarity at or above 0.80 to any test
problem. For texts shorter than five tokens, individual tokens are used.
An inverted shingle index retrieves all pairs with nonzero overlap; the
similarity decision itself does not use approximate MinHash/LSH retrieval.
The optional Qwen semantic judge was not used.

## Filtering results

- **MATH-500 (test):** 500/500 examples retained.
- **GSM8K (train):** 7,473 original examples →
  0 examples flagged by the configured criteria →
  0 examples removed from training.
- **Removal breakdown:** 0 exact matches;
  0 additional shingle-similarity matches;
  0 additional semantic-judge matches.
- **Delivered training set:** 7,473 − 0 =
  **7,473 examples**.

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
   The number of records excluded in this run is 0.
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

splits = DatasetDict({
    "train": load_dataset("json", data_files="clean_gsm8k_train.jsonl", split="train"),
    "test": load_dataset("json", data_files="math500_test.jsonl", split="train"),
})
```

The example assumes this directory as the working directory. Source dataset
revisions are not pinned by this pipeline; retain these exported files when
reproducing this run.
