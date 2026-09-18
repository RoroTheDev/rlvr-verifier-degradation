"""Pre-download the datasets this project uses into the Hugging Face cache.

Run once on a fresh machine so later scripts don't stall on network.
Set HF_HOME to move the cache somewhere else (e.g. scratch on a cluster).

    python download_datasets.py               # everything
    python download_datasets.py gsm8k math500 # just some
"""

import argparse
import sys

from datasets import load_dataset

DATASETS: dict[str, dict] = {
    "gsm8k": dict(path="openai/gsm8k", name="main"),
    "math500": dict(path="HuggingFaceH4/MATH-500"),
    "humanevalplus": dict(path="evalplus/humanevalplus"),
    # ~7 GB. Ships a loading script (TACO.py), hence trust_remote_code.
    "taco": dict(path="BAAI/TACO", name="ALL", trust_remote_code=True),
}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "names", nargs="*", choices=list(DATASETS), metavar="NAME",
        help="which datasets to fetch (default: all). One of: " + ", ".join(DATASETS),
    )
    args = ap.parse_args()

    failed: list[str] = []
    for name in args.names or DATASETS:
        kw = DATASETS[name]
        label = kw["path"] + (f" ({kw['name']})" if "name" in kw else "")
        print(f"[{name}] {label}")
        try:
            ds = load_dataset(**kw)
        except Exception as e:  # keep going, report at the end
            print(f"[{name}] FAILED: {e}", file=sys.stderr)
            failed.append(name)
            continue
        for split, d in ds.items():
            print(f"    {split}: {len(d):,} rows")

    if failed:
        sys.exit(f"failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
