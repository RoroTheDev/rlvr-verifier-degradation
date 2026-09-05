from datasets import load_dataset

def download_datasets():
    load_dataset("openai/gsm8k", "main")
    load_dataset("BAAI/TACO")
    load_dataset("HuggingFaceH4/MATH-500")
    load_dataset("evalplus/humanevalplus")


if __name__ == "__main__":
    download_datasets()