import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

from phd.feature_search.core.experiment_helpers import seed_from_string, StandardizationStats, standardize_targets
from phd.feature_search.core.tasks import NonlinearGEOFFTask
from phd.feature_search.core.models import LTU, MLP


SEED = 1

N_INPUTS = 20
LEARNER_HIDDEN_DIM = 10
LEARNER_N_LAYERS = 2
LEARNER_ACTIVATION = 'ltu'

TRAIN_SAMPLES = 100_000
BATCH_SIZE = 1
PRINT_FREQ = 200
LOG_FREQ = 100

OPTIMIZER_CLS = torch.optim.SGD
STEP_SIZE = 0.001


def make_task(seed):
    task = NonlinearGEOFFTask(
        n_features = N_INPUTS,
        flip_rate = 0.0,
        n_layers = 2,
        n_stationary_layers = 2,
        hidden_dim = 2,
        weight_scale = 1.0,
        activation = 'ltu',
        sparsity = 0.0,
        weight_init = 'binary',
        seed = seed_from_string(seed, 'task'),
    )

    # Init target output weights to kaiming uniform and predictor output weights to zero
    task_init_generator = torch.Generator(device=task.weights[-1].device)
    task_init_generator.manual_seed(seed_from_string(seed, 'task_init_generator'))
    torch.nn.init.kaiming_uniform_(
        task.weights[-1],
        mode = 'fan_in',
        nonlinearity = 'linear',
        generator = task_init_generator,
    )
    
    return task # Get iterator with task.get_iterator(batch_size)


def create_model(n_inputs, hidden_dim, n_layers, activation, seed):
    """Create a simple MLP model without using the complex MLP class."""
    model = MLP(
        input_dim = n_inputs,
        output_dim = 1,
        n_layers = n_layers,
        hidden_dim = hidden_dim,
        weight_init_method = 'binary',
        activation = activation,
        n_frozen_layers = 0,
        seed = seed_from_string(seed, 'model'),
        bias = False,
    )
    with torch.no_grad():
        model.layers[-1].weight.zero_()
    
    return model


def train_model(model, task_iterator, n_iterations, optimizer, print_freq):
    """Train the model and return losses throughout training."""
    model.train()
    losses = []
    recent_losses = []
    
    # Initialize standardization stats
    stats = StandardizationStats(gamma=0.99)
    
    # Create progress bar
    pbar = tqdm(range(n_iterations), desc="Training")
    
    for i in pbar:
        # Get batch from task iterator
        x, y = next(task_iterator)
        
        # Forward pass
        y_pred, _ = model(x)
        
        # Standardize targets using running statistics
        y_standardized, stats = standardize_targets(y, stats)
        
        # Compute MSE loss
        loss = nn.functional.mse_loss(y_pred, y_standardized)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Store loss
        loss_val = loss.item()
        losses.append(loss_val)
        recent_losses.append(loss_val)
        
        # Update progress bar every LOG_FREQ iterations
        if (i + 1) % print_freq == 0:
            avg_recent_loss = sum(recent_losses) / len(recent_losses)
            pbar.set_postfix({'avg_loss': f'{avg_recent_loss:.6f}'})
            recent_losses = []
    
    return losses


def plot_losses(losses, window_size):
    """Plot the training losses."""
    # Average losses over 100 steps
    n_windows = len(losses) // window_size
    
    # Calculate averages for complete windows only
    averaged_losses = []
    x_positions = []
    
    for i in range(n_windows):
        start_idx = i * window_size
        end_idx = start_idx + window_size
        window_losses = losses[start_idx:end_idx]
        avg_loss = sum(window_losses) / len(window_losses)
        averaged_losses.append(avg_loss)
        # Use the midpoint of the window for x-position
        x_positions.append(start_idx + window_size // 2)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_positions, averaged_losses)
    plt.title('Training Loss Over Time (Averaged over 100 steps)')
    plt.xlabel('Iteration')
    plt.ylabel('MSE Loss')
    # plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.show()


def main():
    task = make_task(SEED)
    task_iterator = task.get_iterator(BATCH_SIZE)
    
    # 1. Create model with N_LAYERS (N_LAYERS - 1 hidden layers) and HIDDEN_DIM hidden units per hidden layer
    #    it should have N_INPUTS inputs and 1 output
    model = create_model(N_INPUTS, LEARNER_HIDDEN_DIM, LEARNER_N_LAYERS, LEARNER_ACTIVATION, SEED)
    
    # 2. Create optimizer with OPTIMIZER_CLS and STEP_SIZE
    optimizer = OPTIMIZER_CLS(model.parameters(), lr = STEP_SIZE)
    
    # 3. Run the training loop for TRAIN_SAMPLES / BATCH_SIZE iterations
    n_iterations = TRAIN_SAMPLES // BATCH_SIZE
    losses = train_model(model, task_iterator, n_iterations, optimizer, PRINT_FREQ)
    
    # 4. Plot the losses throughout training
    plot_losses(losses, LOG_FREQ)
    
    return losses


if __name__ == '__main__':
    main()