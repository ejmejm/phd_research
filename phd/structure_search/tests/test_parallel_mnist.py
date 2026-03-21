"""Tests for ParallelMNISTStream."""

import numpy as np

from phd.structure_search.data import ParallelMNISTStream


def _make_fake_mnist(n_samples=100, input_dim=784):
    """Create small fake MNIST-like data for testing."""
    rng = np.random.default_rng(0)
    images = rng.random((n_samples, input_dim)).astype(np.float32)
    labels = rng.integers(0, 10, size=n_samples)
    return images, labels


def test_sample_batch_shapes():
    """Output shapes should be correct for K tasks."""
    images, labels = _make_fake_mnist()
    stream = ParallelMNISTStream(
        images=images, labels=labels,
        n_tasks=3, batch_size=4, seed=42,
    )
    imgs, lbls = stream.sample_batch(10)
    assert imgs.shape == (10, 4, 3 * 784), f'Got {imgs.shape}'
    assert lbls.shape == (10, 4, 3), f'Got {lbls.shape}'
    print('PASS: test_sample_batch_shapes')


def test_labels_in_range():
    """All labels should be in [0, 9]."""
    images, labels = _make_fake_mnist()
    stream = ParallelMNISTStream(
        images=images, labels=labels,
        n_tasks=5, batch_size=2, seed=42,
    )
    _, lbls = stream.sample_batch(50)
    assert lbls.min() >= 0 and lbls.max() <= 9
    print('PASS: test_labels_in_range')


def test_stationary_no_permutation():
    """With permute_period=0, labels should use identity permutation."""
    images, labels = _make_fake_mnist(n_samples=50)
    stream = ParallelMNISTStream(
        images=images, labels=labels,
        n_tasks=2, batch_size=1, seed=0, permute_period=0,
    )
    # Sample many batches — permutations should stay as identity
    for _ in range(10):
        stream.sample_batch(100)

    for k in range(2):
        np.testing.assert_array_equal(
            stream.label_permutations[k], np.arange(10),
        )
    print('PASS: test_stationary_no_permutation')


def test_permutation_fires():
    """With permute_period > 0, at least one permutation should change."""
    images, labels = _make_fake_mnist()
    stream = ParallelMNISTStream(
        images=images, labels=labels,
        n_tasks=4, batch_size=1, seed=42, permute_period=50,
    )
    # Advance past the first permutation event
    stream.sample_batch(100)

    # At least one task should have a non-identity permutation
    changed = any(
        not np.array_equal(p, np.arange(10))
        for p in stream.label_permutations
    )
    assert changed, 'No permutation fired after 100 steps with period=50'
    print('PASS: test_permutation_fires')


def test_permutation_only_affects_one_task():
    """Each permutation event should change exactly one task."""
    images, labels = _make_fake_mnist()
    stream = ParallelMNISTStream(
        images=images, labels=labels,
        n_tasks=10, batch_size=1, seed=7, permute_period=100,
    )
    # Record initial permutations
    before = [p.copy() for p in stream.label_permutations]

    # Advance past exactly one permutation event (step 100 fires in [0, 101))
    stream.sample_batch(101)

    n_changed = sum(
        not np.array_equal(before[k], stream.label_permutations[k])
        for k in range(10)
    )
    assert n_changed == 1, f'Expected 1 task changed, got {n_changed}'
    print('PASS: test_permutation_only_affects_one_task')


def test_get_test_batch_shapes():
    """get_test_batch should return correct shapes."""
    train_imgs, train_lbls = _make_fake_mnist(n_samples=100)
    test_imgs, test_lbls = _make_fake_mnist(n_samples=20)

    stream = ParallelMNISTStream(
        images=train_imgs, labels=train_lbls,
        n_tasks=3, batch_size=1, seed=0,
        test_images=test_imgs, test_labels=test_lbls,
    )
    t_imgs, t_lbls = stream.get_test_batch()
    assert t_imgs.shape == (20, 3 * 784), f'Got {t_imgs.shape}'
    assert t_lbls.shape == (20, 3), f'Got {t_lbls.shape}'
    print('PASS: test_get_test_batch_shapes')


def test_get_test_batch_applies_permutations():
    """get_test_batch should reflect current label permutations."""
    train_imgs, train_lbls = _make_fake_mnist(n_samples=100)
    test_imgs, test_lbls = _make_fake_mnist(n_samples=20)

    stream = ParallelMNISTStream(
        images=train_imgs, labels=train_lbls,
        n_tasks=2, batch_size=1, seed=0, permute_period=50,
        test_images=test_imgs, test_labels=test_lbls,
    )

    _, lbls_before = stream.get_test_batch()

    # Fire some permutations
    stream.sample_batch(200)

    _, lbls_after = stream.get_test_batch()

    # At least one task's test labels should have changed
    changed = not np.array_equal(lbls_before, lbls_after)
    assert changed, 'Test labels did not change after permutations'
    print('PASS: test_get_test_batch_applies_permutations')


def test_step_counter_advances():
    """step_counter should track total steps sampled."""
    images, labels = _make_fake_mnist()
    stream = ParallelMNISTStream(
        images=images, labels=labels,
        n_tasks=2, batch_size=1, seed=0,
    )
    stream.sample_batch(100)
    assert stream.step_counter == 100
    stream.sample_batch(50)
    assert stream.step_counter == 150
    print('PASS: test_step_counter_advances')


if __name__ == '__main__':
    test_sample_batch_shapes()
    test_labels_in_range()
    test_stationary_no_permutation()
    test_permutation_fires()
    test_permutation_only_affects_one_task()
    test_get_test_batch_shapes()
    test_get_test_batch_applies_permutations()
    test_step_counter_advances()
    print('\nAll ParallelMNISTStream tests passed!')
