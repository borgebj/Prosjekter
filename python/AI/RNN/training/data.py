
from pathlib import Path
from datasets import load_dataset


# =====================================
# Paths

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_PATH = DATA_DIR / "tinystories.txt"


# =====================================
# Text cleanup

def repair_mojibake(text: str) -> str:
    """Repair common UTF-8 punctuation misread as Windows-1252."""
    if not any(marker in text for marker in ("â€œ", "â€™", "â€”", "â€")):
        return text

    try:
        repaired = text.encode("cp1252").decode("utf-8")

        # Use the repair only if it reduces the known corruption markers.
        markers = ("â€œ", "â€", "â€™", "â€”", "â€“")
        before = sum(text.count(marker) for marker in markers)
        after = sum(repaired.count(marker) for marker in markers)

        if after < before:
            return repaired

    except (UnicodeEncodeError, UnicodeDecodeError):
        pass

    # Handle common punctuation sequences that strict CP1252 conversion
    # cannot decode because some characters are undefined in that encoding.
    replacements = {
        "â€œ": "“",
        "â€\x9d": "”",
        "â€™": "’",
        "â€”": "—",
        "â€“": "–",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


# =====================================
# Download TinyStories

def main() -> None:
    dataset = load_dataset(
        "roneneldan/TinyStories",
        split="train",
        streaming=True,
    )

    count = 0
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as file:
        for story in dataset.take(200):
            text = story["text"].strip()

            if not text:
                continue

            text = repair_mojibake(text)

            if count:
                file.write("\n<|STORY|>\n")

            file.write(text)
            count += 1

    print(f"Saved {count} stories to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
