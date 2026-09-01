"""Shared pair-sampling helpers for dynamic Stage 1 datasets."""

import random

from torch.utils.data import default_collate


def sample_cslpae_pair_batch(dataset, property_name, batch_size):
    """Sample same-property pairs from a dataset split.

    The dataset must expose ``subject_indices`` and ``task_indices`` mappings
    and support ``dataset[index]`` returning one complete sample tuple.
    """
    if property_name == "subject":
        index_by_value = dataset.subject_indices
    elif property_name == "task":
        index_by_value = dataset.task_indices
    else:
        raise ValueError(f"Unsupported pair property: {property_name}")

    values = sorted(index_by_value)
    if not values:
        raise ValueError(f"No values available for property: {property_name}")

    samples_per_repeat = len(values)
    repeats = max(int(batch_size) // (2 * samples_per_repeat), 1)
    left_indices = []
    right_indices = []
    for _ in range(repeats):
        for value in values:
            candidates = index_by_value[value]
            if len(candidates) >= 2:
                left, right = random.sample(candidates, 2)
            else:
                left = right = candidates[0]
            left_indices.append(left)
            right_indices.append(right)

    return (
        default_collate([dataset[index] for index in left_indices]),
        default_collate([dataset[index] for index in right_indices]),
        repeats,
        samples_per_repeat,
    )
