from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from neural_network import NeuralNet

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "generated"
OUTPUT_DIR.mkdir(exist_ok=True)

EXAMPLES = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 0],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1],
    [2, 1, 1],
    [3, 1, 2],
    [4, 2, 3],
], dtype=float)

LABELS = [
    "No signal",
    "Free only",
    "Win only",
    "Offer only",
    "Free + win",
    "Free + offer",
    "Win + offer",
    "All three",
    "Strong pattern",
    "Very strong",
    "Extreme signal",
]


def make_prediction_chart(model):
    probs = model.predict(EXAMPLES).flatten()
    values = probs * 100

    fig, ax = plt.subplots(figsize=(10, 6.4), facecolor="#f8fafc")
    palette = [
        "#34d399", "#34d399", "#34d399", "#34d399",
        "#2dd4bf", "#2dd4bf", "#2dd4bf", "#38bdf8",
        "#60a5fa", "#f59e0b", "#f97316",
    ]
    ax.barh(LABELS, values, color=palette, edgecolor="#0f172a", linewidth=1.2, height=0.8)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Spam likelihood (%)")
    ax.set_title("Representative predictions from the current spam detector")
    ax.invert_yaxis()
    ax.axvline(50, color="#475569", linestyle="--", linewidth=1.0, alpha=0.8)

    for i, v in enumerate(values):
        x = min(v + 1.5, 97.5)
        ax.text(x, i, f"{v:.1f}%", va="center", ha="left", fontsize=10, color="#0f172a")

    ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "sample_predictions.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    model = NeuralNet.load(str(ROOT / "spam_detector.pt"))
    make_prediction_chart(model)
    print(f"Generated prediction preview in {OUTPUT_DIR}")
