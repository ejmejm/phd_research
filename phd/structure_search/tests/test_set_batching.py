"""Batch-size semantics of ``train_set.py``.

``train.batch_size`` feeds ``train_step`` a leading batch axis (see
``ParallelMNISTStream.sample_batch``, which yields ``(n_steps, B, ...)``); the
step vmaps the model over that axis and means the loss. These tests pin the
property that makes such a minibatch step *identical* to accumulating and
averaging B single-sample gradients, so the batch-size study needs no separate
accumulation path:

    one step on a batch of B samples  ==  mean of the B single-sample steps

which holds exactly for plain SGD (``update = -lr * grad``, linear in grad):

    w - lr * mean_i(g_i)  ==  mean_i(w - lr * g_i)

The same identity is checked for the contribution-utility traces, whose batch
reduction (``mean_B |x|``, ``mean_B |h|``) is likewise linear in the batch.
Both sides are produced by ``train_step`` itself, so the test tracks the real
code path rather than a re-derivation of it.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
import pytest

# The identity below is exact in real arithmetic, so the tolerances here are
# about float32 rounding only. On an Ampere+ GPU the default matmul precision is
# TF32 (~10-bit mantissa), under which one batched matmul and B separate
# single-row matmuls disagree by ~1e-4 relative -- reduced precision in the
# forward/backward, not a batching defect, but enough to swamp the signal these
# tests are looking for. Pin the highest precision so they measure the algebra.
jax.config.update('jax_default_matmul_precision', 'highest')

from phd.feature_search.jax_core.experiment_helpers import prepare_optimizer, rng_from_string
from phd.structure_search.data import ParallelMNISTStream
from phd.structure_search.experiments.train_set import (
    TrainState, _derive_sizes, _model_filter_spec, _w1_source_magnitude,
    init_model, train_step,
)


N_TASKS = 2
INPUT_DIM_PER_TASK = 32   # stand-in for MNIST's 784, kept small for speed
NUM_CLASSES = 10
LR = 0.5                  # large so the update is well above float32 noise


def _cfg(hidden=8):
    """Minimal config for the SET sizing/init/optimizer helpers."""
    input_dim = N_TASKS * INPUT_DIM_PER_TASK
    output_dim = N_TASKS * NUM_CLASSES
    budget = 4 * (input_dim + output_dim)
    return OmegaConf.create({
        'target_hidden_units': hidden,
        'model': {'activation': 'leaky_relu', 'fan_in_safety_factor': 2.0},
        'optimizer': {'name': 'sgd', 'learning_rate': LR, 'weight_decay': 0},
        'set': {'init_mode': 'uniform_p', 'connection_budget': budget},
    })


def _make_state(seed=0, hidden=8):
    cfg = _cfg(hidden)
    input_dim = N_TASKS * INPUT_DIM_PER_TASK
    output_dim = N_TASKS * NUM_CLASSES
    hidden_units, max_conns, max_fan_out, p_w1, p_w2 = _derive_sizes(
        cfg, input_dim, output_dim,
    )
    rng = jax.random.key(seed)
    model = init_model(
        cfg, input_dim, output_dim, hidden_units, max_conns, max_fan_out,
        p_w1, p_w2, key=rng_from_string(rng, 'model'),
    )
    optimizer = prepare_optimizer(
        model, cfg.optimizer.name, cfg.optimizer,
        filter_spec=_model_filter_spec(model),
    )
    # Non-zero starting traces so the EMA's decay term is exercised too (a
    # zero trace would make the identity hold for the wrong reason).
    k = rng_from_string(rng, 'util')
    return TrainState(
        model=model, optimizer=optimizer,
        step=jnp.array(0), rng=rng_from_string(rng, 'train'),
        util_w1=jax.random.uniform(k, model.network.weights.shape),
        util_w2=jax.random.uniform(k, model.network.output_weights.shape),
        util_step_w1=jnp.full(model.network.weights.shape, 7, dtype=jnp.int32),
        util_step_w2=jnp.full(model.network.output_weights.shape, 7, dtype=jnp.int32),
    )


def _fake_batch(batch_size, seed=1):
    """A (B, n_tasks*input_dim) batch with per-task labels, like the stream."""
    rng = np.random.default_rng(seed)
    images = rng.normal(size=(batch_size, N_TASKS * INPUT_DIM_PER_TASK)).astype(np.float32)
    labels = rng.integers(0, NUM_CLASSES, size=(batch_size, N_TASKS)).astype(np.int32)
    return jnp.asarray(images), jnp.asarray(labels)


def _step(state, images, labels, utility_decay=0.9):
    return train_step(
        state, (images, labels),
        num_classes=NUM_CLASSES, n_tasks=N_TASKS, utility_decay=utility_decay,
    )


@pytest.mark.parametrize('batch_size', [2, 8, 16])
def test_batched_step_equals_averaged_single_sample_steps(batch_size):
    """SGD on a batch of B == the average of the B single-sample SGD steps.

    This is the gradient-accumulation identity: if the batched step averaged
    gradients over the wrong axis, dropped samples, summed instead of meaned,
    or leaked one sample's activations into another's update, the two sides
    would disagree.
    """
    state = _make_state()
    images, labels = _fake_batch(batch_size)

    batched, batched_metrics = _step(state, images, labels)

    singles = [_step(state, images[i:i + 1], labels[i:i + 1]) for i in range(batch_size)]

    def mean_of_singles(get):
        return jnp.mean(jnp.stack([get(s) for s, _ in singles]), axis=0)

    for name, get in [
        ('weights', lambda s: s.model.network.weights),
        ('output_weights', lambda s: s.model.network.output_weights),
        ('outgoing_weights', lambda s: s.model.network.outgoing_weights),
        ('util_w1', lambda s: s.util_w1),
        ('util_w2', lambda s: s.util_w2),
    ]:
        np.testing.assert_allclose(
            np.asarray(get(batched)), np.asarray(mean_of_singles(get)),
            rtol=1e-5, atol=1e-6, err_msg=f'{name} mismatch at B={batch_size}',
        )

    # Loss and accuracy are batch means as well.
    single_metrics = jnp.mean(jnp.stack([m for _, m in singles]), axis=0)
    np.testing.assert_allclose(
        np.asarray(batched_metrics), np.asarray(single_metrics), rtol=1e-5, atol=1e-6,
    )

    # One optimizer update per step regardless of B, and the trace's per-weight
    # bias-correction counter likewise advances once (not B times).
    assert int(batched.step) == 1
    assert int(batched.util_step_w1[0].max()) == 8


def test_batched_step_changes_weights():
    """Guard for the test above: the update must be non-trivial, or the
    equality would be comparing two copies of the initial weights."""
    state = _make_state()
    images, labels = _fake_batch(8)
    new_state, _ = _step(state, images, labels)
    active = state.model.network.input_indices[0] >= 0
    delta = np.abs(np.asarray(new_state.model.network.weights[0] - state.model.network.weights[0]))
    assert delta[np.asarray(active)].max() > 1e-4
    # Inactive slots stay exactly zero (no gradient leaks into pruned weights).
    assert delta[~np.asarray(active)].max() == 0.0


@pytest.mark.parametrize('batch_size', [1, 8])
def test_w1_source_magnitude_matches_gather_then_reduce(batch_size):
    """``_w1_source_magnitude`` reduces the batch before the per-slot gather;
    it must equal the naive gather-then-reduce reference."""
    state = _make_state()
    images, _ = _fake_batch(batch_size)
    input_indices = state.model.network.input_indices[0]

    active = input_indices >= 0
    safe_idx = jnp.where(active, input_indices, 0)
    reference = jnp.mean(
        jnp.where(active, jnp.abs(images[:, safe_idx]), 0.0), axis=0,
    )

    np.testing.assert_allclose(
        np.asarray(_w1_source_magnitude(images, input_indices)),
        np.asarray(reference), rtol=1e-6, atol=1e-7,
    )


def test_w1_source_magnitude_intermediate_is_batch_free():
    """...and does so without a (B, units, conns) intermediate.

    Gathering first costs an extra (B, hidden, max_conns) buffer -- ~50 MB per
    seed at n_tasks=32 / H=1536 / B=64, on top of the one the sparse forward
    pass already needs. Compiled HLO is inspected rather than runtime memory so
    the check is cheap and deterministic.
    """
    state = _make_state(hidden=8)
    batch_size = 16
    images, _ = _fake_batch(batch_size)
    input_indices = state.model.network.input_indices[0]
    hlo = (jax.jit(_w1_source_magnitude)
           .lower(images, input_indices).compile().as_text())

    u, c = state.model.network.weights.shape[1:]
    bad = f'f32[{batch_size},{u},{c}]'
    assert bad not in hlo, f'{bad} intermediate present in the W1 trace gather'


def test_permute_events_are_per_step_not_per_sample():
    """``permute_period`` counts optimizer steps, so a given number of steps
    yields the same number of permutation events at every batch size.

    The batch-size study relies on this: to hold the non-stationarity fixed on
    a *sample* axis, the configs scale ``permute_period`` by 1/B themselves.
    """
    rng = np.random.default_rng(0)
    images = rng.random((64, INPUT_DIM_PER_TASK)).astype(np.float32)
    labels = rng.integers(0, NUM_CLASSES, size=64)

    class _CountingRng:
        """Forwards to the real generator, counting `permutation` calls --
        which `_advance_permutations` makes exactly once per event (diffing the
        label maps instead would under-count two events hitting one task)."""

        def __init__(self, rng):
            self._rng, self.n_permutations = rng, 0

        def permutation(self, *a, **kw):
            self.n_permutations += 1
            return self._rng.permutation(*a, **kw)

        def __getattr__(self, name):
            return getattr(self._rng, name)

    def n_events(batch_size, n_steps, period, chunk):
        """Permutation events over `n_steps` steps, sampled `chunk` at a time."""
        stream = ParallelMNISTStream(
            images=images, labels=labels, n_tasks=8,
            batch_size=batch_size, seed=0, permute_period=period,
        )
        stream.rng = _CountingRng(stream.rng)
        for _ in range(n_steps // chunk):
            stream.sample_batch(chunk)
        assert stream.step_counter == n_steps, 'step_counter must count steps, not samples'
        return stream.rng.n_permutations

    # Events land on multiples of `period` in [period, n_steps - 1], so over
    # n_steps there are floor((n_steps - 1) / period) of them -- a function of
    # steps and period only. Batch size does not enter, which is exactly why
    # the batch configs must scale `permute_period` by 1/B themselves to hold
    # the drift rate fixed on a sample axis.
    n_steps, period = 40, 10
    expected = (n_steps - 1) // period
    for batch_size in (1, 4, 16):
        assert n_events(batch_size, n_steps, period, chunk=1) == expected, (
            f'event count changed with batch size (B={batch_size})'
        )

    # Halving the period roughly doubles the events at fixed steps: the knob
    # behaves monotonically, so 1/B scaling does what the configs intend.
    assert n_events(4, n_steps, 5, chunk=1) == (n_steps - 1) // 5 == 7

    # Chunking is irrelevant: log_freq (how many steps are sampled per call)
    # must not change the drift schedule, since it also scales with 1/B.
    for chunk in (1, 2, 8, 40):
        assert n_events(4, n_steps, period, chunk=chunk) == expected, (
            f'event count changed with sampling chunk size ({chunk})'
        )


def test_stream_batch_shares_permutation_within_a_step():
    """All B samples in one step are drawn under a single permutation state,
    so a batch never straddles a task change."""
    rng = np.random.default_rng(0)
    # One image per class, identity-valued so a label permutation is visible.
    images = np.eye(10, INPUT_DIM_PER_TASK, dtype=np.float32)
    labels = np.arange(10)
    stream = ParallelMNISTStream(
        images=images, labels=labels, n_tasks=1, batch_size=32, seed=0,
        permute_period=1,
    )
    imgs, lbls = stream.sample_batch(4)
    assert imgs.shape == (4, 32, INPUT_DIM_PER_TASK)
    assert lbls.shape == (4, 32, 1)
    # Within each step the (image -> label) map must be a single function; a
    # mid-batch permutation would map one image row to two different labels.
    for t in range(4):
        cls = imgs[t].argmax(axis=-1)
        mapping = {}
        for c, l in zip(cls, lbls[t, :, 0]):
            mapping.setdefault(int(c), int(l))
            assert mapping[int(c)] == int(l), f'permutation changed mid-batch at step {t}'
