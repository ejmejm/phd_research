"""
Same as `full_feature_search.py`, but with very detailed logging for the utilities of features.
The goal is to use this logged data to investigate my blocking hypothesis.
"""

import json
from dataclasses import dataclass
import logging
from typing import Iterator, Tuple, List, Literal, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from tqdm import tqdm

import hydra
from omegaconf import DictConfig

import sqlite3
import os

from phd.feature_search.core.experiment_helpers import (
    get_model_statistics,
    seed_from_string,
    standardize_targets,
    StandardizationStats,
)
from phd.feature_search.core.idbd import IDBD
from phd.feature_search.core.models import MLP
from phd.feature_search.core.feature_recycling import CBPTracker
from phd.feature_search.core.tasks import NonlinearGEOFFTask
from phd.research_utils.logging import *
from phd.feature_search.scripts.full_feature_search import (
    DistractorTracker,
    model_distractor_forward_pass,
    prepare_ltu_geoff_experiment,
)

# Add pickle import
import pickle
import array


logger = logging.getLogger(__name__)


@dataclass
class FeatureUtilityStats:
    """Stats for a single feature."""
    id: int
    utilities: List[float]
    feature_type: Literal['real', 'distractor']
    creation_step: int
    pruned_step: Optional[int] = None


def save_feature_utility_stats(stats: FeatureUtilityStats, conn: sqlite3.Connection, commit: bool = False):
    """Saves the feature utility stats to a SQLite database."""
    # Create table if it doesn't exist
    conn.execute('''
        CREATE TABLE IF NOT EXISTS feature_stats (
            id INTEGER PRIMARY KEY,
            creation_step INTEGER,
            pruned_step INTEGER,
            feature_type TEXT,
            utilities BLOB
        )
    ''')
    
    # Convert utilities list to raw bytes (fastest method)
    utilities_bytes = array.array('f', stats.utilities).tobytes()
    
    # Insert or replace the stats
    conn.execute('''
        INSERT OR REPLACE INTO feature_stats 
        (id, creation_step, pruned_step, feature_type, utilities)
        VALUES (?, ?, ?, ?, ?)
    ''', (
        stats.id,
        stats.creation_step,
        stats.pruned_step,
        stats.feature_type,
        utilities_bytes
    ))
    
    if commit:
        conn.commit()


def save_and_reinit_feature_utility_stats(
        id: int,
        feature_utility_stats: List[FeatureUtilityStats],
        idxs: List[int],
        step: int,
        distractor_tracker: DistractorTracker,
        conn: sqlite3.Connection,
    ):
    for idx in idxs:
        if feature_utility_stats[idx] is not None:
            feature_utility_stats[idx].pruned_step = step
            save_feature_utility_stats(feature_utility_stats[idx], conn, commit=False)  # No commit here
        
        is_distractor = distractor_tracker.distractor_mask[idx]
        feature_utility_stats[idx] = FeatureUtilityStats(
            id = id,
            utilities = [],
            feature_type = 'distractor' if is_distractor else 'real',
            creation_step = step,
        )
        id += 1
    
    return id


def run_experiment(
        cfg: DictConfig,
        task: NonlinearGEOFFTask,
        task_iterator: Iterator[Tuple[torch.Tensor, torch.Tensor]],
        model: MLP,
        criterion: nn.Module,
        optimizer: Optimizer,
        repr_optimizer: Optional[Optimizer],
        cbp_tracker: CBPTracker,
        distractor_tracker: DistractorTracker,
    ):
    # Distractor setup
    n_hidden_units = model.layers[-1].in_features
    distractor_tracker.process_new_features(list(range(n_hidden_units)))

    # Initialize database connection
    utility_stats_save_path = cfg.train.get('utility_stats_save_path', './utility_stats.db')
    
    # Remove existing database file if it exists to start fresh
    if os.path.exists(utility_stats_save_path):
        os.remove(utility_stats_save_path)
    
    db_conn = sqlite3.connect(utility_stats_save_path)
    
    try:
        # Initialize feature utility stats
        utility_feature_id = 0
        feature_utility_stats: List[Optional[FeatureUtilityStats]] = [None for _ in range(n_hidden_units)]
        utility_feature_id = save_and_reinit_feature_utility_stats(
            utility_feature_id, feature_utility_stats, list(range(n_hidden_units)), 0,
            distractor_tracker, db_conn,
        )

        # Training loop
        step = 0
        prev_pruned_idxs = set()
        prune_layer = model.layers[-2]
        pbar = tqdm(total=cfg.train.total_steps, desc='Training')
        
        # Flags
        log_utility_stats = cfg.train.get('log_utility_stats', False)
        log_pruning_stats = cfg.train.get('log_pruning_stats', False)
        log_model_stats = cfg.train.get('log_model_stats', False)

        # Initialize accumulators
        cumulant_stats = StandardizationStats(gamma=0.99)
        cumulative_loss = np.float128(0.0)
        loss_accum = 0.0
        mean_pred_loss_accum = 0.0
        pruned_accum = 0
        pruned_newest_feature_accum = 0
        n_steps_since_log = 0
        total_pruned = 0
        prune_thresholds = []
        target_buffer = []

        while step < cfg.train.total_steps:

            # Generate batch of data
            inputs, targets = next(task_iterator)

            # Add noise to targets
            if cfg.task.noise_std > 0:
                targets += torch.randn_like(targets) * cfg.task.noise_std
            
            with torch.no_grad():
                standardized_targets, cumulant_stats = standardize_targets(targets, cumulant_stats)
            
            if cfg.train.standardize_cumulants:
                targets = standardized_targets
            target_buffer.extend(targets.view(-1).tolist())
            
            features, targets = inputs.to(cfg.device), targets.to(cfg.device)

            # Reset weights and optimizer states for recycled features
            if cbp_tracker is not None:
                if log_pruning_stats:
                    pre_prune_utilities = cbp_tracker.get_statistics(prune_layer)['utility']

                pruned_idxs = cbp_tracker.prune_features()
                n_pruned = sum([len(idxs) for idxs in pruned_idxs.values()])
                total_pruned += n_pruned

                if prune_layer in pruned_idxs and len(pruned_idxs[prune_layer]) > 0:
                    new_feature_idxs = pruned_idxs[prune_layer].tolist()

                    # Turn some features into distractors
                    distractor_tracker.process_new_features(new_feature_idxs)

                    # Log pruning statistics
                    pruned_accum += len(new_feature_idxs)
                    n_new_pruned_features = len(set(new_feature_idxs).intersection(prev_pruned_idxs))
                    pruned_newest_feature_accum += n_new_pruned_features
                    prev_pruned_idxs = set(new_feature_idxs)
                    
                    # Save pruned feature and reinit utility stats for new features
                    utility_feature_id = save_and_reinit_feature_utility_stats(
                        utility_feature_id, feature_utility_stats, new_feature_idxs, step,
                        distractor_tracker, db_conn,
                    )
                    
                    if log_pruning_stats:
                        prune_thresholds.append(pre_prune_utilities[new_feature_idxs].max().item())
            
            # Forward pass
            outputs, param_inputs = model(
                features, distractor_tracker.replace_features)
            loss = criterion(outputs, targets)
            
            with torch.no_grad():
                if cfg.train.standardize_cumulants:
                    baseline_pred = torch.zeros_like(targets)
                else:
                    baseline_pred = cumulant_stats.running_mean.cpu().view(1, 1)
                mean_pred_loss = criterion(baseline_pred, targets)

            # Backward pass
            optimizer.zero_grad()
            if repr_optimizer is not None:
                repr_optimizer.zero_grad()
            
            if isinstance(optimizer, IDBD):
                # Mean over batch dimension
                param_inputs = {k: v.mean(dim=0) for k, v in param_inputs.items()}
                retain_graph = optimizer.version == 'squared_grads'
                loss.backward(retain_graph=retain_graph)
                optimizer.step(outputs, param_inputs)
            else:
                loss.backward()
                optimizer.step()
                
            if repr_optimizer is not None:
                repr_optimizer.step()

            # Update feature utility stats
            if step % 10 == 0:
                hidden_unit_utilities = cbp_tracker.get_statistics(prune_layer)['utility'].tolist()
                for idx in range(n_hidden_units):
                    if feature_utility_stats[idx] is None:
                        raise ValueError(f"Feature utility stats for feature {idx} was not initialized and is None!")
                    feature_utility_stats[idx].utilities.append(hidden_unit_utilities[idx])
            
            # Commit to DB every 100 steps
            if step % 100 == 0:
                db_conn.commit()
            
            # Accumulate metrics
            loss_accum += loss.item()
            cumulative_loss += loss.item()
            mean_pred_loss_accum += mean_pred_loss.item()
            n_steps_since_log += 1
            
            # Log metrics
            if step % cfg.train.log_freq == 0:
                n_distractors = distractor_tracker.distractor_mask.sum().item()
                n_real_features = distractor_tracker.distractor_mask.numel() - n_distractors
                metrics = {
                    'step': step,
                    'samples': step * cfg.train.batch_size,
                    'loss': loss_accum / n_steps_since_log,
                    'cumulative_loss': float(cumulative_loss),
                    'mean_prediction_loss': mean_pred_loss_accum / n_steps_since_log,
                    'squared_targets': torch.tensor(target_buffer).square().mean().item(),
                    'n_distractors': n_distractors,
                    'n_real_features': n_real_features,
                }

                if log_pruning_stats:
                    if pruned_accum > 0:
                        metrics['fraction_pruned_were_new'] = pruned_newest_feature_accum / pruned_accum
                        pruned_newest_feature_accum = 0
                        pruned_accum = 0
                    metrics['units_pruned'] = total_pruned
                    if len(prune_thresholds) > 0:
                        metrics['prune_threshold'] = np.mean(prune_thresholds)
                    prune_thresholds.clear()
                
                if log_utility_stats:
                    all_utilities = cbp_tracker.get_statistics(prune_layer)['utility']
                    distractor_mask = distractor_tracker.distractor_mask
                    real_utilities = all_utilities[~distractor_mask]
                    distractor_utilities = all_utilities[distractor_mask]
                    
                    cumulative_utility = all_utilities.sum().item()
                    metrics['cumulative_utility'] = cumulative_utility
                    
                    if len(real_utilities) > 0:
                        metrics['real_utility_median'] = real_utilities.median().item()
                        metrics['real_utility_25th'] = real_utilities.quantile(0.25).item()
                        metrics['real_utility_75th'] = real_utilities.quantile(0.75).item()
                    
                    if len(distractor_utilities) > 0:
                        metrics['distractor_utility_median'] = distractor_utilities.median().item()
                        metrics['distractor_utility_25th'] = distractor_utilities.quantile(0.25).item() 
                        metrics['distractor_utility_75th'] = distractor_utilities.quantile(0.75).item()

                # Add model statistics separately for real and distractor features
                if log_model_stats:
                    real_feature_masks = [
                        torch.ones(model.layers[0].weight.shape[1], dtype=torch.bool, device=model.layers[0].weight.device),
                        ~distractor_tracker.distractor_mask,
                    ]
                    metrics.update(get_model_statistics(
                        model, features, param_inputs, real_feature_masks, metric_prefix='real_'))
                    
                    distractor_feature_masks = [
                        real_feature_masks[0],
                        distractor_tracker.distractor_mask,
                    ]
                    metrics.update(get_model_statistics(
                        model, features, param_inputs, distractor_feature_masks, metric_prefix='distractor_'))

                log_metrics(metrics, cfg, step=step)
                
                pbar.set_postfix(loss=metrics['loss'])
                pbar.update(cfg.train.log_freq)
                
                # Reset accumulators
                loss_accum = 0.0
                mean_pred_loss_accum = 0.0
                n_steps_since_log = 0
                target_buffer = []

            step += 1

        pbar.close()
    
    finally:
        # Ensure database connection is closed
        db_conn.close()


@hydra.main(config_path='../conf', config_name='full_feature_search')
def main(cfg: DictConfig) -> None:
    """Run the feature recycling experiment."""
    assert cfg.model.n_layers == 2, "Only 2-layer models are supported!"

    cfg = init_experiment(cfg.project, cfg)

    task, task_iterator, model, criterion, optimizer, repr_optimizer, recycler, cbp_tracker = \
        prepare_ltu_geoff_experiment(cfg)
    model.forward = model_distractor_forward_pass.__get__(model)
    
    distractor_tracker = DistractorTracker(
        model,
        cfg.task.distractor_chance,
        tuple(cfg.task.distractor_mean_range),
        tuple(cfg.task.distractor_std_range),
        seed = seed_from_string(cfg.seed, 'distractor_tracker'),
    )
    
    run_experiment(
        cfg, task, task_iterator, model, criterion, optimizer,
        repr_optimizer, cbp_tracker, distractor_tracker,
    )
    
    finish_experiment(cfg)


if __name__ == '__main__':
    main()
