import os

PROMPTS_DIR = os.path.dirname(__file__)


def load_prompt(name: str) -> str:
    path = os.path.join(PROMPTS_DIR, f"{name}.txt")
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()
