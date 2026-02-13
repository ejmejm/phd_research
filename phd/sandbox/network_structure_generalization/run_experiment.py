import argparse
from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import random
import mlflow
import numpy as np
import optax
from tqdm import tqdm

from data_stream import load_dataset, ContinualClassStream
from models import build_model, count_params


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

    # Seed
    parser.add_argument('--seed', type=int, default=None)

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
    log_interval: int = eqx.field(static=True)

    # Dynamic
    model: eqx.Module
    optimizer_state: optax.OptState
    stream: ContinualClassStream
    rng: jax.Array
    step: jax.Array


class StepMetrics(eqx.Module):
    """Metrics collected from a single training step."""
    loss: jax.Array
    correct: jax.Array


def init_experiment(args: argparse.Namespace) -> Tuple[TrainState, int, int]:
    """Initialize model, optimizer, data stream, and training state."""
    seed = args.seed if args.seed is not None else np.random.randint(0, 1_000_000_000)
    key = random.PRNGKey(seed)

    # Load dataset
    all_images, samples_per_class, num_classes, image_shape = load_dataset(args.dataset)

    # Create data stream
    key, stream_key = random.split(key)
    stream = ContinualClassStream(
        all_images=all_images,
        samples_per_class=samples_per_class,
        num_classes=num_classes,
        image_shape=image_shape,
        switch_freq=args.switch_freq,
        allow_resampling=args.allow_resampling,
        key=stream_key,
    )

    # Build model
    key, model_key = random.split(key)
    model = build_model(args, image_shape, key=model_key)
    n_params = count_params(model)
    print(f'Model: {args.model}, Params: {n_params}')

    # Optimizer
    optimizer = optax.rmsprop(args.learning_rate)
    optimizer_state = optimizer.init(eqx.filter(model, eqx.is_array))

    train_state = TrainState(
        optimizer=optimizer,
        log_interval=args.log_interval,
        model=model,
        optimizer_state=optimizer_state,
        stream=stream,
        rng=key,
        step=jnp.array(0),
    )

    return train_state, seed, n_params


def train_step(train_state: TrainState, _) -> Tuple[TrainState, StepMetrics]:
    """Single training step for jax.lax.scan."""
    key, _ = random.split(train_state.rng)

    # Sample from data stream
    new_stream, (image, label) = train_state.stream.sample()

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
        log_interval=train_state.log_interval,
        model=new_model,
        optimizer_state=new_opt_state,
        stream=new_stream,
        rng=key,
        step=train_state.step + 1,
    )

    return new_state, StepMetrics(loss=loss, correct=correct)


def log_metrics(metrics: StepMetrics, step: int, use_mlflow: bool = True
                ) -> Tuple[float, float]:
    """Aggregate scan-batch metrics and optionally log to MLflow."""
    mean_loss = float(metrics.loss.mean())
    mean_acc = float(metrics.correct.mean())
    if use_mlflow:
        mlflow.log_metrics({'loss': mean_loss, 'accuracy': mean_acc}, step=step)
    return mean_loss, mean_acc


def train(train_state: TrainState, num_steps: int, use_mlflow: bool = False
          ) -> Tuple[TrainState, list, list]:
    """Outer training loop with jax.lax.scan inner loop."""
    log_interval = train_state.log_interval
    num_scans = num_steps // log_interval

    @jax.jit
    def scan_steps(state: TrainState) -> Tuple[TrainState, StepMetrics]:
        return jax.lax.scan(
            train_step,
            state,
            length=log_interval,
            unroll=UNROLL_STEPS,
        )

    all_losses = []
    all_accuracies = []
    pbar = tqdm(total=num_steps, desc='Training')

    for _ in range(num_scans):
        train_state, metrics = scan_steps(train_state)

        mean_loss, mean_acc = log_metrics(
            metrics, step=int(train_state.step.item()), use_mlflow=use_mlflow)
        all_losses.append(mean_loss)
        all_accuracies.append(mean_acc)

        pbar.update(log_interval)
        pbar.set_postfix({'loss': f'{mean_loss:.4f}', 'acc': f'{mean_acc:.4f}'})

    pbar.close()
    return train_state, all_losses, all_accuracies


def main():
    args = parse_args()
    train_state, seed, n_params = init_experiment(args)

    if args.mlflow:
        mlflow.set_experiment(args.project)
        mlflow.start_run()
        mlflow.log_params({**vars(args), 'seed': seed, 'num_params': n_params})

    train_state, all_losses, all_accuracies = train(
        train_state, args.total_steps, use_mlflow=args.mlflow)

    # Compute final metrics
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
