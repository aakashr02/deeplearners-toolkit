"""
Generate demo/stream_data.csv for demo/stream_demo.ipynb.

Run from the project root:

    python demo/make_stream_data.py

The data comes from ``numcompute.benchmarking.make_stream_data``: 6,000
rows, four sensor-like features and a binary label. The labelling rule
changes halfway through (concept drift), 5% of labels are flipped, and 3%
of feature values are missing (written as empty fields).
"""

import sys
from pathlib import Path

import numpy as np

# Allow running from a clone without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from numcompute.benchmarking import make_stream_data  # noqa: E402

N_SAMPLES = 6000
OUTPUT = Path(__file__).with_name("stream_data.csv")


def _format(value):
    return "" if np.isnan(value) else f"{value:.4f}"


def main():
    X, y = make_stream_data(N_SAMPLES, n_features=4, drift_at=0.5,
                            missing_rate=0.03, label_noise=0.05, seed=2024)

    lines = ["sensor_1,sensor_2,sensor_3,sensor_4,label"]
    for row, label in zip(X, y):
        lines.append(",".join(_format(v) for v in row) + f",{label}")

    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(y)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
