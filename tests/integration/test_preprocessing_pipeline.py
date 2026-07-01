import numpy as np
import torch

from foregrounds_diffusion.preprocessing import augment_images_unique, split_data_to_tensors


def test_preprocessing_pipeline_shape_and_dtype():
    """(10, 64, 64, 2) channels-last array → normalise → split → augment."""
    rng = np.random.default_rng(42)
    # 10 samples so that 80% = 8 training patches → 8 × 8 = 64 augmented
    data = rng.standard_normal((10, 64, 64, 2)).astype(np.float32)

    train_set, val_set, test_set = split_data_to_tensors(
        data, train_size=0.8, val_size=0.1, test_size=0.1
    )

    assert train_set.shape == (8, 2, 64, 64)
    assert train_set.dtype == torch.float32
    assert val_set.shape[1:] == (2, 64, 64)
    assert test_set.shape[1:] == (2, 64, 64)


def test_augment_produces_8x_count():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((10, 64, 64, 2)).astype(np.float32)

    train_set, _, _ = split_data_to_tensors(
        data, train_size=0.8, val_size=0.1, test_size=0.1
    )
    augmented = augment_images_unique(train_set)

    assert augmented.shape == (64, 2, 64, 64)   # 8 patches × 8× augmentation
    assert augmented.dtype == torch.float32


def test_augment_no_duplicate_tensors():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((10, 64, 64, 2)).astype(np.float32)
    train_set, _, _ = split_data_to_tensors(
        data, train_size=0.8, val_size=0.1, test_size=0.1
    )
    augmented = augment_images_unique(train_set)

    # All 64 augmented images should be distinct (no exact duplicates)
    flat = augmented.reshape(64, -1).numpy()
    for i in range(64):
        for j in range(i + 1, 64):
            assert not np.allclose(flat[i], flat[j]), \
                f"Augmented images {i} and {j} are identical"
