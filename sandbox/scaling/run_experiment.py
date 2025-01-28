import math
from typing import Dict, Tuple, Optional
import random

import hydra
import numpy as np
import omegaconf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from omegaconf import DictConfig
import wandb
from tqdm import tqdm


def set_seed(seed: int) -> None:
  """Set random seeds for reproducibility."""
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_datasets(config: DictConfig) -> Tuple[DataLoader, DataLoader]:
  """Initialize train and test dataloaders."""
  if config.data.dataset == 'mnist':
    transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = torchvision.datasets.MNIST(
      './data', train=True, download=config.data.download, transform=transform)
    test_dataset = torchvision.datasets.MNIST(
      './data', train=False, download=config.data.download, transform=transform)
    input_dim = 28 * 28
    num_classes = 10
  else:  # cifar10
    transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = torchvision.datasets.CIFAR10(
      './data', train=True, download=config.data.download, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(
      './data', train=False, download=config.data.download, transform=transform)
    input_dim = 32 * 32 * 3
    num_classes = 10

  # If eval_samples is specified, take a subset of the test dataset
  if config.data.eval_samples is not None:
    indices = torch.randperm(len(test_dataset))[:config.data.eval_samples]
    test_dataset = torch.utils.data.Subset(test_dataset, indices)

  train_loader = DataLoader(
    train_dataset, 
    batch_size=config.train.batch_size,
    shuffle=True,
    num_workers=config.data.num_workers
  )
  test_loader = DataLoader(
    test_dataset,
    batch_size=config.train.eval_batch_size,
    shuffle=False,
    num_workers=config.data.num_workers
  )
  
  return train_loader, test_loader, input_dim, num_classes


class SparseLinear(nn.Module):
  """Linear layer with configurable sparsity mask."""
  
  def __init__(self, in_features: int, out_features: int, sparsity: float):
    super().__init__()
    self.linear = nn.Linear(in_features, out_features)
    self.mask = torch.ones_like(self.linear.weight.data)
    if sparsity > 0:
      n_zeros = int(sparsity * self.mask.numel())
      zero_idx = random.sample(range(self.mask.numel()), n_zeros)
      self.mask.view(-1)[zero_idx] = 0
    self.mask = nn.Parameter(self.mask, requires_grad=False)
  
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return nn.functional.linear(
      x, self.linear.weight * self.mask, self.linear.bias)


class MLP(nn.Module):
  """Configurable MLP with variable width, depth and sparsity."""
  
  def __init__(
    self,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    depth: int,
    sparsity: float,
    activation: str = 'relu'
  ):
    super().__init__()
    
    # Setup activation
    if activation == 'relu':
      self.activation = nn.ReLU()
    else:
      raise ValueError(f'Unknown activation: {activation}')
    
    # Build layers
    layers = []
    current_dim = input_dim
    for _ in range(depth):
      layers.append(SparseLinear(current_dim, hidden_dim, sparsity))
      layers.append(self.activation)
      current_dim = hidden_dim
    
    layers.append(SparseLinear(current_dim, output_dim, sparsity))
    self.layers = nn.Sequential(*layers)
  
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = x.view(x.size(0), -1)
    return self.layers(x)


def count_active_params(model: nn.Module) -> int:
  """Count number of non-masked parameters."""
  active = 0
  for layer in model.modules():
    if isinstance(layer, SparseLinear):
      active += (layer.mask != 0).sum().item()
  return active


def evaluate(
  model: nn.Module,
  loader: DataLoader,
  criterion: nn.Module,
  device: str
) -> Tuple[float, float]:
  """Evaluate model on given dataloader."""
  model.eval()
  total_loss = 0
  correct = 0
  total = 0
  
  with torch.no_grad():
    for data, target in loader:
      data, target = data.to(device), target.to(device)
      output = model(data)
      total_loss += criterion(output, target).item() * data.size(0)
      pred = output.argmax(dim=1)
      correct += pred.eq(target).sum().item()
      total += data.size(0)
  
  return total_loss / total, correct / total


def train_and_evaluate(
  model: nn.Module,
  train_loader: DataLoader,
  test_loader: DataLoader,
  optimizer: torch.optim.Optimizer,
  criterion: nn.Module,
  config: DictConfig,
  device: str
) -> Dict[str, float]:
  """Run training loop and return summary metrics.
  
  Args:
    model: The neural network to train
    train_loader: DataLoader for training data
    test_loader: DataLoader for test data
    optimizer: The optimizer to use
    criterion: Loss function
    config: Configuration object
    device: Device to run on ('cuda' or 'cpu')
    
  Returns:
    Dictionary containing summary metrics
  """
  step = 0
  samples_seen = 0
  running_loss = 0
  running_correct = 0
  running_total = 0
  
  # For computing early/late statistics
  all_train_losses = []
  all_test_losses = []
  
  # Calculate total samples per epoch for epoch tracking
  samples_per_epoch = len(train_loader.dataset)
  total_samples = config.train.total_samples
  
  progress_bar = tqdm(total=total_samples, unit='samples')
  while samples_seen < total_samples:
    model.train()
    for data, target in train_loader:
      if samples_seen >= total_samples:
        break
        
      data, target = data.to(device), target.to(device)
      batch_size = data.size(0)
      
      optimizer.zero_grad()
      output = model(data)
      loss = criterion(output, target)
      loss.backward()
      optimizer.step()
      
      # Update running statistics
      running_loss += loss.item()
      pred = output.argmax(dim=1)
      running_correct += pred.eq(target).sum().item()
      running_total += batch_size
      
      # Update counters
      step += 1
      samples_seen += batch_size
      current_epoch = samples_seen / samples_per_epoch
      
      # Logging
      if step % config.train.log_freq == 0:
        train_loss = running_loss / config.train.log_freq
        train_acc = running_correct / running_total
        
        metrics = {
          'train/loss': train_loss,
          'train/accuracy': train_acc,
          'train/step': step,
          'train/samples': samples_seen,
          'train/epoch': current_epoch
        }
        
        if config.wandb.enabled:
          wandb.log(metrics)
        
        progress_bar.set_postfix(
          loss=f'{train_loss:.4f}',
          acc=f'{train_acc:.4f}',
          epoch=f'{current_epoch:.2f}'
        )
        
        progress_bar.update(batch_size * config.train.log_freq)
        
        all_train_losses.append(train_loss)
        running_loss = 0
        running_correct = 0
        running_total = 0
      
      # Validation
      if step % config.train.validation_freq == 0:
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        all_test_losses.append(test_loss)
        
        if config.wandb.enabled:
          wandb.log({
            'test/loss': test_loss,
            'test/accuracy': test_acc,
            'test/step': step,
            'test/samples': samples_seen,
            'test/epoch': current_epoch
          })
  
  progress_bar.close()
  
  # Compute and log summary statistics
  early_train_loss = np.mean(all_train_losses[:len(all_train_losses)//10])
  early_test_loss = np.mean(all_test_losses[:len(all_test_losses)//10])
  late_train_loss = np.mean(all_train_losses[-len(all_train_losses)//10:])
  late_test_loss = np.mean(all_test_losses[-len(all_test_losses)//10:])
  
  final_test_loss, final_test_acc = evaluate(model, test_loader, criterion, device)
  
  return {
    'summary/early_train_loss': early_train_loss,
    'summary/early_test_loss': early_test_loss,
    'summary/final_train_loss': late_train_loss,
    'summary/final_test_loss': late_test_loss,
    'summary/final_test_accuracy': final_test_acc,
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


@hydra.main(config_path='conf', config_name='defaults')
def main(config: DictConfig) -> None:
  """Main training loop."""
  # Setup
  set_seed(config.seed)
  device = torch.device(config.device)
  
  if config.wandb.enabled:
    config = setup_wandb(config)
  
  # Data
  train_loader, test_loader, input_dim, num_classes = get_datasets(config)
  
  # Model
  model = MLP(
    input_dim=input_dim,
    hidden_dim=config.model.width,
    output_dim=num_classes,
    depth=config.model.depth,
    sparsity=config.model.sparsity,
    activation=config.model.activation
  ).to(device)
  
  # Log active parameters
  n_active = count_active_params(model)
  if config.wandb.enabled:
    wandb.log({'model/active_parameters': n_active})
  
  # Training setup
  criterion = nn.CrossEntropyLoss()
  if config.train.optimizer == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=config.train.learning_rate)
  elif config.train.optimizer == 'rmsprop':
    optimizer = optim.RMSprop(model.parameters(), lr=config.train.learning_rate)
  elif config.train.optimizer == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=config.train.learning_rate)
  
  # Train and get summary metrics
  summary_metrics = train_and_evaluate(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    optimizer=optimizer,
    criterion=criterion,
    config=config,
    device=device
  )
  
  if config.wandb.enabled:
    wandb.log(summary_metrics)
    wandb.finish()


if __name__ == '__main__':
  main()
