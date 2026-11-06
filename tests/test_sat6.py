import tempfile
import unittest
from pathlib import Path

import numpy as np

from sat6 import channel_columns, load_course_data, split_indices


class Sat6Test(unittest.TestCase):
    def test_seeded_splits_are_balanced_disjoint_and_reproducible(self):
        labels = np.repeat(np.eye(6, dtype=int), 7, axis=0)
        groups = split_indices(labels, 3, 2, 1)
        for group, size, repeat in zip(groups, (3, 2, 1), split_indices(labels, 3, 2, 1)):
            np.testing.assert_array_equal(labels[group].sum(axis=0), [size] * 6)
            np.testing.assert_array_equal(group, repeat)
        self.assertEqual(len(np.unique(np.concatenate(groups))), 36)

    def test_interleaved_band_selection(self):
        pixels = np.arange(3136).reshape(28, 28, 4)
        selected = pixels.ravel()[channel_columns(1, 3)].reshape(28, 28, 2)
        np.testing.assert_array_equal(selected, pixels[:, :, [1, 3]])

    def test_loader_preserves_row_alignment_and_rejects_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = np.repeat(np.eye(6, dtype=int), 4, axis=0)
            np.savetxt(root / "y_test_sat6.csv", labels, fmt="%d", delimiter=",")
            (root / "sat6annotations.csv").write_text("a\nb\nc\nd\ne\nf\n")
            images = root / "X_test_sat6.csv"
            rows = [",".join([str(i)] * 3136) for i in range(len(labels))]
            images.write_text("\n".join(rows) + "\n")
            pixels, selected, _, _ = load_course_data(root, train=1, validation=1, test=1)
            np.testing.assert_array_equal(pixels.iloc[:, 0], pixels.index)
            np.testing.assert_array_equal(selected, labels[pixels.index])
            images.write_text("\n".join(rows[:-1]) + "\n")
            with self.assertRaisesRegex(ValueError, "row counts"):
                load_course_data(root, train=1, validation=1, test=1)
