import json
import re
import argparse
from difflib import SequenceMatcher
from pathlib import Path

try:
    from datasets import load_dataset
    from nltk.util import ngrams
    from tqdm import tqdm
    from datasketch import MinHash, MinHashLSH
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing required Python packages. Install them with: "
        "python -m pip install -r requirements.txt"
    ) from exc

DEFAULT_QWEN_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


class QwenContaminationJudge:
    """Use an instruction-tuned Qwen model to judge semantic near-duplicates."""

    def __init__(
        self, 
        model_name: str, 
        max_new_tokens: int = 8
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ModuleNotFoundError as exc:
            raise SystemExit(
                "Qwen mode requires torch and transformers. Install them with: "
                "python -m pip install -r requirements.txt"
            ) from exc

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(model_name)
        self._model.eval()
        self._max_new_tokens = max_new_tokens

    def is_contaminated(self, question: str, reference: str) -> bool:
        messages = [
            {
                "role": "system",
                "content": (
                    "You classify dataset contamination. Return only CONTAMINATED "
                    "if the candidate is the same mathematical problem as the "
                    "reference with wording or formatting changes; otherwise return CLEAN."
                ),
            },
            {
                "role": "user",
                "content": f"REFERENCE:\n{reference}\n\nCANDIDATE:\n{question}",
            },
        ]
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(prompt, return_tensors="pt")
        with self._torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        generated = output[0, inputs["input_ids"].shape[1] :]
        answer = self._tokenizer.decode(generated, skip_special_tokens=True).upper()
        return "CONTAMINATED" in answer and "CLEAN" not in answer


def normalize_math(text: str) -> str:
    """
    Normalizes text and math notation to ensure structural variations
    do not bypass the decontamination filters.
    """
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s*([+\-*/=(){}[\]<>])\s*', r'\1', text)
    text = text.replace(r'\left(', '(').replace(r'\right)', ')')
    text = text.replace(r'\:', '').replace(r'\,', '')
    return text.strip()

def get_minhash(
    text: str, 
    num_perm: int = 128, 
    n_gram: int = 5
) -> MinHash:
    """
    Generates a MinHash signature based on word-level n-grams.
    """
    m = MinHash(num_perm=num_perm)
    tokens = text.split()
    
    if len(tokens) < n_gram:
        shingles = tokens
    else:
        shingles = [' '.join(g) for g in ngrams(tokens, n_gram)]
        
    for s in shingles:
        m.update(s.encode('utf-8'))
    return m


def decontaminate_gsm8k_with_math500(
    threshold: float = 0.80,
    qwen_model: str | None = DEFAULT_QWEN_MODEL,
    qwen_limit: int | None = None,
) -> list[dict[str, str]]:
    """
    Loads GSM8K (Train) and Math-500 (Test) from Hugging Face,
    and filters out contaminated training samples.
    """

    print("Step 1: Loading Math-500 benchmark dataset...")
    math500_test = load_dataset("HuggingFaceH4/MATH-500", split="test")
    
    print("Step 2: Loading GSM8K training dataset...")
    gsm8k_train = load_dataset("openai/gsm8k", "main", split="train")
    
    print("Step 3: Indexing Math-500 for Exact and Fuzzy matching...")
    exact_match_registry = set()
    math500_questions: list[str] = []
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    
    for idx, row in enumerate(math500_test):
        normalized_test_text = normalize_math(row['problem'])
        exact_match_registry.add(normalized_test_text)
        math500_questions.append(row["problem"])
        
        m_hash = get_minhash(normalized_test_text)
        lsh.insert(f"math500_{idx}", m_hash)
        
    print("\nStep 4: Scanning GSM8K Train for data contamination...")
    clean_gsm8k_train = []
    stats = {"exact_matches": 0, "fuzzy_matches": 0, "qwen_matches": 0, "clean": 0}
    qwen_judge = QwenContaminationJudge(qwen_model) if qwen_model else None
    
    for row in tqdm(gsm8k_train, desc="Processing GSM8K"):
        raw_train_text = row['question']
        normalized_train_text = normalize_math(raw_train_text)
        
        # Check A: Exact Match Filtering
        if normalized_train_text in exact_match_registry:
            stats["exact_matches"] += 1
            continue
            
        # Check B: Fuzzy Match Filtering (Near-Duplicates)
        train_hash = get_minhash(normalized_train_text)
        fuzzy_result = lsh.query(train_hash)
        
        if len(fuzzy_result) > 0:
            stats["fuzzy_matches"] += 1
            continue
            
        if qwen_judge and (qwen_limit is None or stats["clean"] < qwen_limit):
            candidates = sorted(
                math500_questions,
                key=lambda reference: SequenceMatcher(
                    None, normalized_train_text, normalize_math(reference)
                ).ratio(),
                reverse=True,
            )[:3]
            if any(qwen_judge.is_contaminated(raw_train_text, candidate) for candidate in candidates):
                stats["qwen_matches"] += 1
                continue

        stats["clean"] += 1
        clean_gsm8k_train.append(row)
        
    print("\nDECONTAMINATION REPORT:")
    print(f"   - Total original GSM8K samples: {len(gsm8k_train)}")
    print(f"   - Removed via Exact Match:      {stats['exact_matches']}")
    print(f"   - Removed via Fuzzy Match:      {stats['fuzzy_matches']}")
    print(f"   - Removed via Qwen semantic judge: {stats['qwen_matches']}")
    print(f"   - Verified Clean samples:       {stats['clean']}")
    print(
        "\nCAVEAT: Qwen is used as a semantic near-duplicate judge, not as evidence "
        "of Qwen's private pretraining corpus. The result is Qwen-assisted filtering "
        "against MATH-500, not proof that Qwen training data is contamination-free."
    )
    
    return clean_gsm8k_train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--qwen-model", default=DEFAULT_QWEN_MODEL)
    parser.add_argument(
        "--qwen-limit",
        type=int,
        default=None,
        help="Only run Qwen on this many otherwise-clean rows; default is all rows.",
    )
    parser.add_argument(
        "--disable-qwen",
        action="store_true",
        help="Use only exact and MinHash matching.",
    )
    args = parser.parse_args()

    clean_dataset = decontaminate_gsm8k_with_math500(
        threshold=args.threshold,
        qwen_model=None if args.disable_qwen else args.qwen_model,
        qwen_limit=args.qwen_limit,
    )

    output_dir = Path(__file__).resolve().parents[1] / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "clean_gsm8k_train.jsonl"

    with output_path.open("w", encoding="utf-8") as f:
        for item in clean_dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved clean dataset to {output_path}")
