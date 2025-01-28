import os
import math
import pickle
from typing import Dict, Tuple, Optional, Any, Callable
import random
import tempfile

import hydra
import numpy as np
import omegaconf
import jax
import jax.numpy as jnp
from jax import random as jrandom
import optax
from flax import linen as nn
from flax.training import train_state
import tensorflow_datasets as tfds
from omegaconf import DictConfig
import wandb
from tqdm import tqdm
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def set_seed(seed: int) -> None:
  """Set random seeds for reproducibility."""
  random.seed(seed)
  np.random.seed(seed)


def get_datasets(config: DictConfig) -> Tuple[Any, Any, int, int]:
  """Initialize train and test datasets.
  
  Args:
    config: Configuration object containing dataset parameters
    
  Returns:
    Tuple of (train_dataset, test_dataset, input_dimension, num_classes)
  """
  # Set dimensions based on dataset type
  if config.data.dataset == 'mnist':
    input_dim = 28 * 28
    num_classes = 10
  else:  # cifar10
    input_dim = 32 * 32 * 3
    num_classes = 10

  if config.data.preload:
    # Create cache paths
    cache_dir = tempfile.gettempdir()
    train_cache = os.path.join(cache_dir, f'train_cache_{config.data.dataset}.pkl')
    test_cache = os.path.join(cache_dir, f'test_cache_{config.data.dataset}.pkl')
    
    # Try loading from cache first
    if os.path.exists(train_cache) and os.path.exists(test_cache):
      print('Loading dataset from cache')
      with open(train_cache, 'rb') as f:
        train_loader = pickle.load(f)
      with open(test_cache, 'rb') as f:
        test_loader = pickle.load(f)
      return train_loader, test_loader, input_dim, num_classes

  # If no cache exists or preload is disabled, load the original dataset
  class JaxTransform:
    def __call__(self, x):
      return jnp.array(x)

  if config.data.dataset == 'mnist':
    transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.1307,), (0.3081,)),
      JaxTransform()
    ])
    train_dataset = torchvision.datasets.MNIST(
      './data', train=True, download=config.data.download, transform=transform)
    test_dataset = torchvision.datasets.MNIST(
      './data', train=False, download=config.data.download, transform=transform)
  else:  # cifar10
    transform = transforms.Compose([
      transforms.ToTensor(), 
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
      JaxTransform()
    ])
    train_dataset = torchvision.datasets.CIFAR10(
      './data', train=True, download=config.data.download, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(
      './data', train=False, download=config.data.download, transform=transform)

  # If eval_samples is specified, take a subset of the test dataset
  if config.data.eval_samples is not None:
    indices = torch.randperm(len(test_dataset))[:config.data.eval_samples]
    test_dataset = torch.utils.data.Subset(test_dataset, indices)

  train_loader = DataLoader(
    train_dataset,
    batch_size=config.train.batch_size,
    shuffle=True,
    num_workers=config.data.num_workers,
  )
  test_loader = DataLoader(
    test_dataset,
    batch_size=config.train.eval_batch_size,
    shuffle=False,
    num_workers=config.data.num_workers
  )

  # Convert labels to JAX arrays
  def collate_fn(batch):
    images, labels = zip(*batch)
    return {
      'image': jnp.stack(images),
      'label': jnp.array(labels)
    }

  train_loader.collate_fn = collate_fn
  test_loader.collate_fn = collate_fn
  
  if config.data.preload:
    print('Preprocessing and caching dataset')
    # Convert to lists and cache
    train_loader = list(train_loader)
    test_loader = list(test_loader)
    
    train_loader = jax.tree.map(lambda *x: jnp.stack(x), *train_loader)
    test_loader = jax.tree.map(lambda *x: jnp.stack(x), *test_loader)
    
    with open(train_cache, 'wb') as f:
      pickle.dump(train_loader, f)
    with open(test_cache, 'wb') as f:
      pickle.dump(test_loader, f)
  
  return train_loader, test_loader, input_dim, num_classes


class SparseLinear(nn.Module):
  """Linear layer with configurable sparsity mask."""
  features: int
  sparsity: float

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    kernel = self.param('kernel',
                       nn.initializers.lecun_normal(),
                       (x.shape[-1], self.features))
    bias = self.param('bias', nn.initializers.zeros, (self.features,))
    
    # Create sparsity mask
    key = self.make_rng('params')
    mask = jrandom.uniform(key, kernel.shape) > self.sparsity
    mask = jnp.float32(mask)
    
    return jnp.dot(x, kernel * mask) + bias


class MLP(nn.Module):
  """Configurable MLP with variable width, depth and sparsity."""
  input_dim: int
  hidden_dim: int
  output_dim: int
  depth: int
  sparsity: float
  
  @nn.compact
  def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
    x = x.reshape((x.shape[0], -1))
    
    for _ in range(self.depth):
      x = SparseLinear(features=self.hidden_dim, sparsity=self.sparsity)(x)
      x = nn.relu(x)
    
    x = SparseLinear(features=self.output_dim, sparsity=self.sparsity)(x)
    return x


class TrainState(train_state.TrainState):
  """Custom train state that includes batch stats and PRNG key."""
  key: jnp.ndarray

  @classmethod
  def create(cls, *, apply_fn, params, tx, key):
    """Creates a new instance with parameters and their optimizer state."""
    return cls(
      step=0,
      apply_fn=apply_fn,
      params=params,
      tx=tx,
      opt_state=tx.init(params),
      key=key,
    )


def count_active_params(state: TrainState) -> int:
  """Count number of non-masked parameters."""
  active = 0
  # In JAX/Flax, we need to traverse the parameter tree differently
  def count_mask(params):
    if isinstance(params, dict):
      return sum(count_mask(v) for v in params.values())
    return jnp.sum(params != 0)
  
  return count_mask(state.params)


def evaluate(state: TrainState, test_ds: Any, config: DictConfig) -> Dict[str, float]:
  """Run evaluation and return metrics.
  
  Args:
    state: Training state containing model parameters and optimizer state
    test_ds: Test dataset iterator
    
  Returns:
    Dictionary of evaluation metrics averaged across batches
  """
  test_metrics = []
  for batch_idx in range(0, len(test_ds)):
    # Get next chunk of batches - handle both preloaded and DataLoader cases
    if config.data.preload:
      test_batch = jax.tree.map(lambda x: x[batch_idx], test_ds)
    else:
      test_batch = test_ds[batch_idx]
    
    metrics = eval_step(state, test_batch)
    test_metrics.append(metrics)
  
  # Stack metrics across batches and compute mean
  stacked = jax.tree.map(lambda *x: jnp.stack(x), *test_metrics)
  return jax.tree.map(lambda x: x.mean(), stacked)


def train_and_evaluate(
  state: TrainState,
  train_ds: Any,
  test_ds: Any,
  config: DictConfig,
) -> Dict[str, float]:
  """Run training loop and return summary metrics."""
  # Validate logging frequencies
  assert config.train.validation_freq % config.train.log_freq == 0, (
    'validation_freq must be a multiple of log_freq'
  )

  step = 0
  samples_seen = 0
  all_train_losses = []
  all_test_losses = []
  samples_per_epoch = len(train_ds)
  progress_bar = tqdm(total=config.train.total_samples, unit='samples')
  first_key = list(train_ds.keys())[0]

  # Process dataset in chunks of log_freq batches
  batch_idx = 0
  while True:
    if samples_seen >= config.train.total_samples:
      break
      
    # Get next chunk of batches - handle both preloaded and DataLoader cases
    if config.data.preload:
      batches = jax.tree.map(lambda x: x[batch_idx:batch_idx + config.train.log_freq], train_ds)
    else:
      batch_slice = slice(batch_idx, batch_idx + config.train.log_freq)
      batches = [train_ds[i] for i in range(*batch_slice.indices(len(train_ds)))]
      batches = jax.tree.map(lambda *x: jnp.stack(x), *batches)
        
    batch_idx += config.train.log_freq
    if batch_idx >= len(train_ds):
      batch_idx = 0
    
    running_samples = np.prod(batches[first_key].shape[:2])

    state, metrics = train_step_multiple(state, batches)
    
    train_loss = metrics['loss'].mean().item()
    train_acc = metrics['accuracy'].mean().item()
    
    all_train_losses.append(train_loss)
    step += config.train.log_freq
    samples_seen += running_samples
    current_epoch = samples_seen / samples_per_epoch
    
    if config.wandb.enabled:
      wandb.log({
        'train/loss': train_loss,
        'train/accuracy': train_acc,
        'train/step': step,
        'train/samples': samples_seen,
        'train/epoch': current_epoch
      })
    
    progress_bar.set_postfix(
      loss=f'{train_loss:.4f}',
      acc=f'{train_acc:.4f}',
      epoch=f'{current_epoch:.2f}'
    )
    
    progress_bar.update(running_samples)
    
    # Validation (will align with log_freq due to assertion)
    if step % config.train.validation_freq == 0:
      test_metrics = evaluate(state, test_ds, config)
      all_test_losses.append(test_metrics['loss'].item())
      
      if config.wandb.enabled:
        wandb.log({
          'test/loss': test_metrics['loss'],
          'test/accuracy': test_metrics['accuracy'],
          'test/step': step,
          'test/samples': samples_seen,
          'test/epoch': current_epoch
        })

  progress_bar.close()
  
  # Compute summary statistics
  early_train_loss = np.mean(all_train_losses[:len(all_train_losses)//10])
  early_test_loss = np.mean(all_test_losses[:len(all_test_losses)//10])
  late_train_loss = np.mean(all_train_losses[-len(all_train_losses)//10:])
  late_test_loss = np.mean(all_test_losses[-len(all_test_losses)//10:])
  
  final_metrics = evaluate(state, test_ds, config)
  
  return {
    'summary/early_train_loss': early_train_loss,
    'summary/early_test_loss': early_test_loss,
    'summary/final_train_loss': late_train_loss,
    'summary/final_test_loss': late_test_loss,
    'summary/final_test_accuracy': final_metrics['accuracy'],
    'summary/total_steps': step,
    'summary/total_samples': samples_seen,
    'summary/final_epoch': samples_seen / samples_per_epoch
  }


def setup_wandb(config: DictConfig) -> None:
    """Initialize Weights & Biases logging."""
    wandb_mode = 'online' if config.wandb.enabled else 'disabled'
    wandb.init(project=config.wandb.project, mode=wandb_mode)
    wandb.config.update(omegaconf.OmegaConf.to_container(
        config, resolve=True, throw_on_missing=True))
    return config


def create_train_state(
  key: jnp.ndarray,
  config: DictConfig,
  model: nn.Module,
  input_shape: Tuple
) -> TrainState:
  """Creates initial training state."""
  variables = model.init(key, jnp.ones(input_shape))
  params = variables['params']
  
  if config.train.optimizer == 'adam':
    tx = optax.adam(config.train.learning_rate)
  elif config.train.optimizer == 'rmsprop':
    tx = optax.rmsprop(config.train.learning_rate)
  else:  # sgd
    tx = optax.sgd(config.train.learning_rate)
    
  return TrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=tx,
    key=key,
  )


@jax.jit
def train_step_multiple(
  state: TrainState,
  batches: Dict[str, jnp.ndarray]
) -> Tuple[TrainState, Dict[str, jnp.ndarray]]:
  """Performs training steps on multiple batches sequentially.
  
  Args:
    state: Current training state
    batches: List of batch dictionaries containing 'image' and 'label'
    
  Returns:
    Updated state and metrics for all batches
  """
  def _step(state, batch):
    state, metrics = train_step(state, batch)
    return state, metrics

  # Run training steps sequentially
  state, metrics = jax.lax.scan(_step, state, batches)
  
  return state, metrics


@jax.jit
def train_step(state: TrainState, batch: Dict) -> Tuple[TrainState, Dict]:
  """Performs a single training step."""
  def loss_fn(params):
    logits = state.apply_fn(
      {'params': params},
      batch['image'],
      rngs={'params': state.key}
    )
    loss = optax.softmax_cross_entropy_with_integer_labels(
      logits, batch['label']).mean()
    return loss, logits

  (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
  state = state.apply_gradients(grads=grads)
  
  metrics = {
    'loss': loss,
    'accuracy': jnp.mean(jnp.argmax(logits, -1) == batch['label'])
  }
  
  return state, metrics


@jax.jit
def eval_step(state: TrainState, batch: Dict) -> Dict:
  """Performs a single evaluation step."""
  logits = state.apply_fn(
    {'params': state.params},
    batch['image'],
    rngs={'params': state.key}
  )
  
  return {
    'loss': optax.softmax_cross_entropy_with_integer_labels(
      logits, batch['label']).mean(),
    'accuracy': jnp.mean(jnp.argmax(logits, -1) == batch['label'])
  }


@hydra.main(config_path='conf', config_name='defaults')
def main(config: DictConfig) -> None:
  """Main training loop."""
  # Setup
  set_seed(config.seed)
  key = jrandom.PRNGKey(config.seed)
  
  if config.wandb.enabled:
    config = setup_wandb(config)
  
  # Data
  train_ds, test_ds, input_dim, num_classes = get_datasets(config)
  if config.data.preload:
    first_key = list(train_ds.keys())[0]
    print(f'Train size: {len(train_ds[first_key])}')
    print(f'Test size: {len(test_ds[first_key])}')
  else:
    print(f'Train size: {len(train_ds)}')
    print(f'Test size: {len(test_ds)}')
  
  # Model
  model = MLP(
    input_dim=input_dim,
    hidden_dim=config.model.width,
    output_dim=num_classes,
    depth=config.model.depth,
    sparsity=config.model.sparsity,
  )
  
  # Initialize training state
  key, init_key = jrandom.split(key)
  input_shape = (1, input_dim)
  state = create_train_state(init_key, config, model, input_shape)
  
  # Log active parameters if enabled
  if config.wandb.enabled:
    n_active = count_active_params(state)
    wandb.log({'model/active_parameters': n_active})
  
  # Training loop
  summary_metrics = train_and_evaluate(state, train_ds, test_ds, config)
  
  if config.wandb.enabled:
    wandb.log(summary_metrics)
    wandb.finish()


if __name__ == '__main__':
  main()
