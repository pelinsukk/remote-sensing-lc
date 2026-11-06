"""Shared SAT-6 course CSV reader and balanced split for both notebooks."""
from pathlib import Path

import numpy as np
import pandas as pd


def split_indices(one_hot, train=1000, validation=100, test=100, seed=42):
    if (one_hot.ndim != 2 or one_hot.shape[1] != 6
            or not np.isin(one_hot, (0, 1)).all() or not (one_hot.sum(axis=1) == 1).all()):
        raise ValueError("Expected six one-hot columns with exactly one active label per row.")
    if min(train, validation, test) < 1:
        raise ValueError("All three split sizes must be positive.")
    rng = np.random.default_rng(seed)
    parts = [[], [], []]
    for column in one_hot.T:
        indices = rng.permutation(np.flatnonzero(column))
        if len(indices) < train + validation + test:
            raise ValueError("Not enough rows in a class for the requested split.")
        for part, chunk in zip(parts, np.split(indices[:train+validation+test],
                                                [train, train+validation])):
            part.extend(chunk)
    return tuple(rng.permutation(part) for part in parts)


def channel_columns(*channels):
    """Columns for selected bands in pixel-interleaved RGB+NIR data."""
    return np.arange(3136).reshape(784, 4)[:, channels].ravel()


def load_course_data(root="data", *, train=1000, validation=100, test=100):
    root = Path(root)
    annotations = pd.read_csv(root / "sat6annotations.csv", header=None)
    all_labels = pd.read_csv(root / "y_test_sat6.csv", header=None)
    splits = split_indices(all_labels.to_numpy(), train, validation, test)
    selected = sorted(np.concatenate(splits))
    keep = set(selected)
    rows = []
    count = 0
    # Read only selected pixel rows into memory; the original CSV is ~906 MB.
    with (root / "X_test_sat6.csv").open("rb") as source:
        for i, line in enumerate(source):
            count += 1
            if i in keep:
                row = np.fromstring(line.decode("ascii").strip(), dtype=np.int64, sep=",")
                if row.size != 3136 or (row < 0).any() or (row > 255).any():
                    raise ValueError(f"Invalid pixel row {i + 1}.")
                rows.append(row.astype(np.uint8))
    if count != len(all_labels) or len(rows) != len(selected):
        raise ValueError("Image CSV and labels have different row counts; check the download.")
    pixels = pd.DataFrame(np.stack(rows), index=selected)
    return pixels, all_labels.loc[selected], annotations, splits
