"""Global constants: dataset dimensions and the class-incremental partition.

Labels are NEVER remapped: the model always has 10 outputs. Task A / Task B /
OOD are defined purely as subsets of the original 0..9 label space.
"""

from __future__ import annotations

from typing import List

INPUT_DIM: int = 40
NUM_CLASSES: int = 10

TASK_A_CLASSES: List[int] = [0, 1, 2, 3, 4]
TASK_B_CLASSES: List[int] = [5, 6, 7]
OOD_CLASSES: List[int] = [8, 9]

# Classes seen across the two training tasks (used for OOD "in-distribution").
SEEN_CLASSES: List[int] = TASK_A_CLASSES + TASK_B_CLASSES  # 0..7

# Canonical dataset provenance (used by data.download).
MNIST1D_URL: str = "https://github.com/greydanus/mnist1d/raw/master/mnist1d_data.pkl"

TASKS = {
    "A": TASK_A_CLASSES,
    "B": TASK_B_CLASSES,
    "OOD": OOD_CLASSES,
    "AB": SEEN_CLASSES,
}
