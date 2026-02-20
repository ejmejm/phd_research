from typing import Any, Dict, Iterator, Optional, Tuple

import equinox as eqx
import jax.numpy as jnp
from nle_prediction import OrderedNetHackDataloader

from phd.feature_search.jax_core.utils import tree_replace


class NethackPredictionTask(eqx.Module):
    # Static parameters (configuration)
    dataloader: OrderedNetHackDataloader = eqx.field(static=True)
    iterator: Iterator[Dict[str, Any]] = eqx.field(static=True)

    def __init__(
        self,
        data_dir: str = './data/nld-nao',
        format: str = 'raw',
        batch_size: int = 1,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.dataloader = OrderedNetHackDataloader(
            data_dir = data_dir,
            format = format,
            batch_size = batch_size,
        )
        self.iterator = iter(self.dataloader)
    
    def reset(self):
        return tree_replace(self, iterator = iter(self.dataloader))
    
    def generate_batch(self) -> Tuple[eqx.Module, Tuple]:
        """Generates a single batch of data.
        
        Args:
            batch_size: Size of batch to generate
            
        Returns:
            Tuple containing:
            - New task state
            - Batch data (x, y)
        """
        
        data = next(self.iterator)
        
        chars = data['tty_chars'] # (batch_size, 24, 80)
        colors = data['tty_colors'] # (batch_size, 24, 80)
        cursor = data['cursor'] # (batch_size, 2)
        
        batch_size = chars.shape[0]
        one_hot_cursor = jnp.zeros((batch_size, 24, 80), dtype=jnp.float32)
        # cursor shape: (batch_size, 2), where each element is (row, col)
        # For each batch entry, set one_hot_cursor[i, row, col] = 1
        row_idx = cursor[:, 0]
        col_idx = cursor[:, 1]
        one_hot_cursor = one_hot_cursor.at[jnp.arange(batch_size), row_idx, col_idx].set(1.0)
        
        x = jnp.stack([
            chars,
            colors + 256,
            one_hot_cursor + 512,
        ], axis=1, dtype=jnp.int32)
        
        # TODO: Set y to the next observation in the sequence
        y = x
        
        return self, (x, y)