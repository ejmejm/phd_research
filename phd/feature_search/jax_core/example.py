import jax
from jax.lib import xla_bridge
from jax import random
import time

from phd.feature_search.core.tasks import NonlinearGEOFFTask as TorchNonlinearGEOFFTask
from phd.feature_search.jax_core.tasks.geoff import NonlinearGEOFFTask

# Set JAX to use CPU
import jax

# Print device being used

def main():
    # Set random seed for reproducibility
    seed = 42
    key = random.PRNGKey(seed)
    
    # jax.config.update('jax_platform_name', 'cpu')
    print(f"JAX is using device: {xla_bridge.get_backend().platform}")
    
    print("Creating NonlinearGEOFFTask...")

    # Create a task with 8 features, medium flip rate, 2 layers
    task = NonlinearGEOFFTask(
        n_features=20,
        flip_rate=0.01,  # 1% of weights flip per step
        n_layers=2,      # 2-layer network (non-linear)
        n_stationary_layers=0,
        hidden_dim=100,
        weight_scale=1.0,
        activation='ltu',
        sparsity=0.1,    # 10% sparsity
        weight_init='binary',
        seed=seed
    )

    torch_task = TorchNonlinearGEOFFTask(
        n_features=20,
        flip_rate=0.01,  # 1% of weights flip per step
        n_layers=2,      # 2-layer network (non-linear)
        n_stationary_layers=0,
        hidden_dim=100,
        weight_scale=1.0,
        activation='ltu',
        sparsity=0.1,    # 10% sparsity
        weight_init='binary',
        seed=seed
    )
    
    batch_size = 1

    def gen_batch(task):
        return task.generate_batch(batch_size)
    
    def gen_batch_scan(task, length):
        return jax.lax.scan(lambda x, y: gen_batch(x), task, None, length=length)
    
    jitted_gen_batch = jax.jit(gen_batch)
    jitted_gen_batch_scan = jax.jit(gen_batch_scan, static_argnums=(1,))
    
    # Time comparison: jitted vs non-jitted forward pass
    print("\nComparing jitted vs non-jitted generate batch:")
    
    # Get a batch warmup
    task, _ = jitted_gen_batch(task)
    print('Traced generate batch')
    
    # Time jitted execution
    n_runs = 5000
    start_time = time.time()
    for i in range(n_runs):
        task, _ = jitted_gen_batch(task)
    jitted_time = time.time() - start_time
    
    # Generate batch scan warmup
    jitted_gen_batch_scan(task, n_runs)
    print('Traced generate batch scan')

    # JIT + scan execution
    start_time = time.time()
    # with jax.profiler.TraceAnnotation('geoff_batch_scan_session'):
    # with jax.profiler.trace("/tmp/jax-trace/gpu", create_perfetto_link=True):
    jax.block_until_ready(jitted_gen_batch_scan(task, n_runs))
    jax_scan_time = time.time() - start_time

    # Time torch execution
    iterator = torch_task.get_iterator(batch_size)
    start_time = time.time()
    for i in range(n_runs):
        next(iterator)
    non_jitted_time = time.time() - start_time
    
    # Print timing results
    print(f"Non-jitted forward pass time ({n_runs} runs): {non_jitted_time:.4f}s")
    print(f"Jitted forward pass time ({n_runs} runs): {jitted_time:.4f}s")
    print(f"Jitted scan time ({n_runs} runs): {jax_scan_time:.4f}s")
    print(f"Speedup: {non_jitted_time / jitted_time:.2f}x")
    print(f"Speedup (scan): {non_jitted_time / jax_scan_time:.2f}x")

if __name__ == "__main__":
    main() 