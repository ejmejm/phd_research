import argparse
from typing import List, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import random
import mlflow
import numpy as np
import optax
from tqdm import tqdm

from data_stream import load_dataset, ContinualClassStream
from models import build_model
from phd.jax_core.utils import count_params, stack_pytrees


UNROLL_STEPS = 4


DEFAULT_SWITCH_FREQ = {
    'mnist': 1000,
    'cifar100': 200,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Network Structure Generalization Experiment')

    # Dataset
    parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar100'],
                        required=True)

    # Model
    parser.add_argument('--model', type=str, choices=['mlp', 'cnn'], required=True)
    parser.add_argument('--activation', type=str, default='relu')

    # MLP config
    parser.add_argument('--mlp_hidden_dims', type=int, nargs='+', default=None)

    # CNN config
    parser.add_argument('--cnn_channels', type=int, nargs='+', default=None)
    parser.add_argument('--cnn_kernel_size', type=int, default=3)
    parser.add_argument('--cnn_stride', type=int, default=2)

    # Continual learning
    parser.add_argument('--switch_freq', type=int, default=None)
    parser.add_argument('--allow_resampling',
                        type=lambda x: x.lower() == 'true', default=True)

    # Training
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--total_steps', type=int, default=100_000)
    parser.add_argument('--log_interval', type=int, default=200)

    # Seed (accepts one or more values; vmapped when multiple)
    parser.add_argument('--seed', type=int, nargs='+', default=None)

    # MLflow
    parser.add_argument('--mlflow', action='store_true')
    parser.add_argument('--project', type=str,
                        default='network-structure-generalization')

    args = parser.parse_args()

    if args.switch_freq is None:
        args.switch_freq = DEFAULT_SWITCH_FREQ[args.dataset]

    return args


class TrainState(eqx.Module):
    # Static
    optimizer: optax.GradientTransformation = eqx.field(static=True)

    # Dynamic
    model: eqx.Module
    optimizer_state: optax.OptState
    step: jax.Array


class StepMetrics(eqx.Module):
    """Metrics collected from a single training step."""
    loss: jax.Array
    correct: jax.Array


def init_experiment(args: argparse.Namespace
                    ) -> Tuple[TrainState, List[ContinualClassStream], int]:
    """Initialize per-seed models/optimizers and CPU-side data streams."""
    seeds = args.seed

    # Load dataset once on CPU (shared across all streams)
    all_images, samples_per_class, num_classes, image_shape = load_dataset(args.dataset)

    streams = []
    train_states = []
    for seed in seeds:
        streams.append(ContinualClassStream(
            all_images=all_images,
            samples_per_class=samples_per_class,
            num_classes=num_classes,
            image_shape=image_shape,
            switch_freq=args.switch_freq,
            allow_resampling=args.allow_resampling,
            seed=seed,
        ))

        key = random.PRNGKey(seed)
        key, model_key = random.split(key)
        model = build_model(args, image_shape, key=model_key)

        optimizer = optax.rmsprop(args.learning_rate)
        optimizer_state = optimizer.init(eqx.filter(model, eqx.is_array))

        train_states.append(TrainState(
            optimizer=optimizer,
            model=model,
            optimizer_state=optimizer_state,
            step=jnp.array(0),
        ))

    n_params = count_params(train_states[0].model)
    print(f'Model: {args.model}, Params: {n_params}, Seeds: {seeds}')

    batched_state = stack_pytrees(train_states)
    return batched_state, streams, n_params


def train_step(train_state: TrainState, data) -> Tuple[TrainState, StepMetrics]:
    """Single training step for jax.lax.scan. Data is provided via scan xs."""
    image, label = data

    # Forward + backward
    def loss_fn(model):
        logits = model(image)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, label)
        return loss, logits

    (loss, logits), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
        train_state.model)

    # Apply gradients
    updates, new_opt_state = train_state.optimizer.update(
        grads, train_state.optimizer_state)
    new_model = eqx.apply_updates(train_state.model, updates)

    # Accuracy (pre-update prediction)
    predicted = jnp.argmax(logits)
    correct = (predicted == label).astype(jnp.int32)

    new_state = TrainState(
        optimizer=train_state.optimizer,
        model=new_model,
        optimizer_state=new_opt_state,
        step=train_state.step + 1,
    )

    return new_state, StepMetrics(loss=loss, correct=correct)


def log_metrics(metrics: StepMetrics, step: int, use_mlflow: bool = True
                ) -> Tuple[float, float]:
    """Aggregate scan-batch metrics (mean over seeds and steps) and log to MLflow."""
    mean_loss = float(metrics.loss.mean())
    mean_acc = float(metrics.correct.astype(jnp.float32).mean())
    if use_mlflow:
        mlflow.log_metrics({'loss': mean_loss, 'accuracy': mean_acc}, step=step)
    return mean_loss, mean_acc


def train(train_state: TrainState, num_steps: int, log_interval: int,
          streams: List[ContinualClassStream], use_mlflow: bool = False
          ) -> Tuple[TrainState, list, list]:
    """Outer training loop: pre-sample data on CPU, train on GPU via vmapped scan."""
    num_scans = num_steps // log_interval

    def scan_steps(state, data):
        return jax.lax.scan(train_step, state, data, unroll=UNROLL_STEPS)

    vmapped_scan = jax.jit(jax.vmap(scan_steps))

    all_losses = []
    all_accuracies = []
    pbar = tqdm(total=num_steps, desc='Training')

    for _ in range(num_scans):
        # Pre-sample one cycle of data on CPU per seed
        batch = [stream.sample_batch(log_interval) for stream in streams]
        images = jnp.array(np.stack([b[0] for b in batch]))  # (n_seeds, log_interval, *img)
        labels = jnp.array(np.stack([b[1] for b in batch]))  # (n_seeds, log_interval)

        train_state, metrics = vmapped_scan(train_state, (images, labels))

        # metrics.loss: (n_seeds, log_interval) — mean over both dims
        mean_loss, mean_acc = log_metrics(
            metrics, step=int(train_state.step[0].item()), use_mlflow=use_mlflow)
        all_losses.append(mean_loss)
        all_accuracies.append(mean_acc)

        pbar.update(log_interval)
        pbar.set_postfix({'loss': f'{mean_loss:.4f}', 'acc': f'{mean_acc:.4f}'})

    pbar.close()
    return train_state, all_losses, all_accuracies


def main():
    args = parse_args()

    # Normalize seeds to list
    if args.seed is None:
        args.seed = [np.random.randint(0, 1_000_000_000)]
    elif isinstance(args.seed, int):
        args.seed = [args.seed]

    train_state, streams, n_params = init_experiment(args)

    if args.mlflow:
        mlflow.set_experiment(args.project)
        mlflow.start_run()
        mlflow.log_params({**vars(args), 'num_params': n_params})

    train_state, all_losses, all_accuracies = train(
        train_state, args.total_steps, args.log_interval,
        streams, use_mlflow=args.mlflow)

    # Compute final metrics (already averaged over seeds)
    average_loss = float(np.mean(all_losses))
    n_tail = max(1, len(all_losses) // 10)
    asymptotic_loss = float(np.mean(all_losses[-n_tail:]))
    asymptotic_accuracy = float(np.mean(all_accuracies[-n_tail:]))

    print(f'Average loss: {average_loss:.4f}')
    print(f'Asymptotic loss: {asymptotic_loss:.4f}')
    print(f'Asymptotic accuracy: {asymptotic_accuracy:.4f}')

    if args.mlflow:
        mlflow.log_metrics({
            'average_loss': average_loss,
            'asymptotic_loss': asymptotic_loss,
            'asymptotic_accuracy': asymptotic_accuracy,
        })
        mlflow.end_run()


if __name__ == '__main__':
    main()
