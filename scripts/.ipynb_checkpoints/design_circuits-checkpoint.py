#!/usr/bin/env python
# coding: utf-8
# %%

# %%
from __future__ import annotations

print("Importing modules")

#from utils5 import *
import argparse
import dgd.utils.utils5
import glob   
import pickle
import pandas as pd
import json
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
import networkx as nx
from tqdm.notebook import tqdm

import time
import random
import gymnasium as gym
from gymnasium import spaces
import subprocess


from datetime import datetime
import platform, sys

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3 import PPO, A2C

import torch as th
import torch.nn as nn
import torch_geometric.nn as geom_nn

import torch.nn.functional as F

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_checker import check_env

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from sb3_contrib.common.wrappers import ActionMasker

from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.logger import configure

import os
from pathlib import Path
from typing import List, Union

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CallbackList
#import environment
from dgd.environments.drl3env_loader4 import DRL3env
# ## Specify binary and biological inputs

from collections import deque
import math

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as geom_nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

'''
# # Load unoptimized graphs

# Load NIGs_unoptimized_library_3_input_1_output
with open('NIGs_unoptimized_library_3_input_1_output.pkl', 'rb') as file:
    NIGs_unoptimized_library_3_input_1_output = pickle.load(file)
'''

class GAT(BaseFeaturesExtractor):
    """
    Pre-norm GATv2 stack with residual projections, optional node-feature
    embedding and attention pooling.
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 128, *, embed_dim: 'int | None' = None):
        super().__init__(observation_space, features_dim)

        in_node_dim = observation_space["node_features"].shape[1]  # =4 (one-hot)
        self.node_embed: 'nn.Module | None' = None
        if embed_dim is not None:
            # one-hot (N,4) · W(4,embed_dim)  →  (N,embed_dim)
            self.node_embed = nn.Linear(in_node_dim, embed_dim, bias=False)
            in_node_dim = embed_dim                      # GNN now sees embed_dim


        layer_cfg = [(32, 4), (64, 4), (64, 4)]          # (out_dim, heads)
        self.gnn_layers = nn.ModuleList()
        cur_dim = in_node_dim

        for out_dim, heads in layer_cfg:
            out_channels = out_dim * heads               # concat=True
            conv = geom_nn.GATv2Conv(
                in_channels=cur_dim,
                out_channels=out_dim,
                heads=heads,
                concat=True,
                dropout=0.1,
            )
            norm = nn.LayerNorm(cur_dim)                 # pre-norm
            proj = (
                None
                if cur_dim == out_channels
                else nn.Linear(cur_dim, out_channels, bias=False)
            )          

            self.gnn_layers.append(nn.ModuleDict({"conv": conv, "norm": norm, "proj": proj}))
            cur_dim = out_channels


        self.global_pool = geom_nn.AttentionalAggregation(
            gate_nn=nn.Sequential(
                nn.Linear(cur_dim, cur_dim // 2),
                nn.ReLU(),
                nn.Linear(cur_dim // 2, 1),
            )
        )
        self.gcn_out_dim = cur_dim


        hidden_dim = 256                         # ← rename here
        self.combined_net = nn.Sequential(
            nn.Linear(self.gcn_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),   # ← and here
            nn.ReLU(),
            nn.Linear(hidden_dim, features_dim), # final projection seen by SB3
            nn.ReLU(),
        )

    def adj_matrix_to_edge_index(self, adj_matrix: th.Tensor) -> th.Tensor:
        row, col = (adj_matrix > 0).nonzero(as_tuple=True)
        edge_index = th.stack([row, col], dim=0).to(adj_matrix.device)
        return edge_index

    def encode_graph(self, x: th.Tensor, adj: th.Tensor) -> th.Tensor:
        if self.node_embed is not None:                  # optional embedding
            x = self.node_embed(x)

        edge_index = self.adj_matrix_to_edge_index(adj)

        for layer in self.gnn_layers:
            h = layer["conv"](layer["norm"](x), edge_index)   # pre-norm
            res = x if layer["proj"] is None else layer["proj"](x)
            x = F.relu(res + h)                               # residual + act

        batch = th.zeros(x.size(0), dtype=th.long, device=x.device)
        return self.global_pool(x, batch)

    def process_graph(self, node_features: th.Tensor, adj_matrix: th.Tensor) -> th.Tensor:
        if adj_matrix.dim() == 3:  # batched
            outs = [
                self.encode_graph(node_features[i], adj_matrix[i])
                for i in range(adj_matrix.size(0))
            ]
            return th.stack(outs)
        return self.encode_graph(node_features, adj_matrix)

    def forward(self, observation):
        device = next(self.parameters()).device

        node_features = th.as_tensor(observation["node_features"], dtype=th.float32, device=device)
        adj_matrix    = th.as_tensor(observation["adj_matrix"],    dtype=th.float32, device=device)

        graph_emb = self.process_graph(node_features, adj_matrix)
        if graph_emb.dim() == 3:                        # (B,1,D) -> (B,D)
            graph_emb = graph_emb.view(graph_emb.size(0), -1)

        return self.combined_net(graph_emb)

# Update policy_kwargs with the new class
policy_kwargs = dict(
    features_extractor_class=GAT,
    features_extractor_kwargs=dict(features_dim=128),
    net_arch=dict(pi=[100, 100, 100, 100, 100], vf=[100, 100, 100, 100, 100])
)

class PlottingCallback(BaseCallback):
    def __init__(self, check_freq, x_lim=None, y_lim=None):
        super(PlottingCallback, self).__init__()
        self.check_freq = check_freq
        self.x = []
        self.y_rewards = []
        self.total_steps = 0  # Explicit counter for total steps
        self.x_lim = x_lim  # Added x-axis limits
        self.y_lim = y_lim  # Added y-axis limits

    def _on_step(self) -> bool:
        # Increase total steps by 1 at each call
        self.total_steps += 1
        
        # Only record the data at the specified check frequency
        if self.total_steps % self.check_freq == 0:
            # Collect rewards from the buffer
            rewards = self.training_env.get_attr('rewards')
            # Flatten the list of rewards and calculate the mean reward
            y_reward = np.mean([reward for env_rewards in rewards for reward in env_rewards])
            self.y_rewards.append(y_reward)

            # Append the current total step count to x-axis data
            self.x.append(self.total_steps)
        
        return True

    def plot(self):
        fig, axs = plt.subplots(2, 1, figsize=(6, 6))

        # Plotting the average rewards
        axs[0].plot(self.x, self.y_rewards)
        axs[0].set_xlabel('Number of Steps')
        axs[0].set_ylabel('Average Reward')
        axs[0].set_title('Average Reward Over Time')

        # Set x-axis limits if provided
        if self.x_lim is not None:
            axs[0].set_xlim(self.x_lim)

        # Set y-axis limits if provided
        if self.y_lim is not None:
            axs[0].set_ylim(self.y_lim)

        plt.tight_layout()
        plt.show()
        

def save_full_registry(registry, lock, out_file):
    """Serialise the whole bucketed registry → pickle on disk."""
    out_file = Path(out_file)
    serialised = {}

    with (lock):
        for h, bucket in registry.items():          # bucket = list[tuple]
            serialised_bucket = []
            for canon, orig, e in bucket:
                serialised_bucket.append((nx.node_link_data(canon), nx.node_link_data(orig), e))
            serialised[h] = serialised_bucket      # could be length 1 or more

    with out_file.open("wb") as f:
        pickle.dump(serialised, f)

    print(f"[save] wrote {out_file}  (hashes {len(serialised)})")

def load_full_registry(pkl_file):
    mgr   = multiprocessing.Manager()
    reg   = mgr.dict()
    lock  = mgr.Lock()

    with open(pkl_file, "rb") as f:
        saved = pickle.load(f)           # { hash : [ (canon_nl, orig_nl, e), ... ] }

    with (lock or contextlib.nullcontext()):
        for h, bucket in saved.items():
            restored = []
            for canon_nl, orig_nl, e in bucket:
                canon = nx.node_link_graph(canon_nl)
                orig  = nx.node_link_graph(orig_nl)
                restored.append((canon, orig, e))
            reg[h] = restored

    return reg, lock          

class ProgressCallback(BaseCallback):
    """
    Prints every `log_every` steps:
        [  40 000 / 200 000 (20.0 %) ] 0 h 31 m 12 s elapsed  ETA 2 h 04 m 34 s  Ḡₜ = -10.42
    where Ḡₜ is the *moving-average* cumulative discounted return.
    You may smooth by *steps* (`smooth_steps`)
    **or** by log-lines (`smooth_lines`).  Only one should be >1.
    """

    def __init__(
        self,
        total_timesteps: int,
        gamma: float,
        log_every: int = 2_000,
        *,
        smooth_steps: int = 0,      # preferred – set e.g. 20_000
        smooth_lines: int = 1       # fallback if you want the old behaviour
    ):
        super().__init__()
        self.total_ts   = total_timesteps
        self.gamma      = gamma
        self.log_every  = log_every

        # decide which smoothing mode to use 
        if smooth_steps and smooth_lines != 1:
            raise ValueError("Specify *either* smooth_steps or smooth_lines, not both.")
        if smooth_steps < 0 or smooth_lines < 1:
            raise ValueError("Window sizes must be positive.")

        if smooth_steps:                           # --- steps mode ----------
            lines = max(1, math.ceil(smooth_steps / log_every))
        else:                                      # --- lines mode ----------
            lines = smooth_lines

        self._hist_g = deque(maxlen=lines)         # last N Ḡ snapshots


    def _on_training_start(self) -> None:
        self.t0 = time.time()
        n_envs  = self.model.get_env().num_envs
        self._G = np.zeros(n_envs, dtype=np.float32)  # discounted return per env


    def _on_step(self) -> bool:
        """
        Update running discounted returns and, every `log_every` steps,
        print timing plus a smoothed average of those returns (Ḡₜ).

        Key fix: we take a *snapshot* of self._G **before** zeroing out the
        entries whose episodes finished, so terminal rewards are included in
        the statistic that gets logged.
        """
        r_t   = np.asarray(self.locals["rewards"], dtype=np.float32)  # (n_envs,)
        dones = self.locals["dones"]                                  # (n_envs,)

        # update discounted returns -----------------------------------
        self._G = r_t + self.gamma * self._G

        # optional log line -------------------------------------------
        if self.num_timesteps % self.log_every == 0:
            G_snapshot = self._G.copy()                     # keep terminal reward
            g_now      = float(G_snapshot.mean())           # mean over all envs
            self._hist_g.append(g_now)
            g_avg      = sum(self._hist_g) / len(self._hist_g)

            # ----- time bookkeeping -------------------------------------
            elapsed = time.time() - self.t0
            sps     = self.num_timesteps / elapsed if elapsed else float("inf")
            eta     = (self.total_ts - self.num_timesteps) / sps if sps else float("inf")

            def _hms(sec):
                h, m = divmod(int(sec), 3600)
                m, s = divmod(m, 60)
                return f"{h:d} h {m:02d} m {s:02d} s"

            pct = 100 * self.num_timesteps / self.total_ts
            print(
                f"[ {self.num_timesteps:>7,d} / {self.total_ts:,}  ({pct:4.1f} %) ]  "
                f"{_hms(elapsed)} elapsed  ETA {_hms(eta)}   Ḡₜ = {g_avg: .2f}",
                flush=True,
            )

        # reset returns for envs whose episodes ended -----------------
        self._G[dones] = 0.0
        return True

class RegistrySnapshotCallback(BaseCallback):
    def __init__(self, registry, lock=None, every_rollouts=10,
                 dir_path="runs", name="shared_registry", verbose=0):
        super().__init__(verbose)

        self.registry = registry
        self.lock     = lock 
        self.every    = every_rollouts

        # make sure the directory exists once, up-front
        self.dir_path = Path(dir_path)
        self.dir_path.mkdir(parents=True, exist_ok=True)

        self.name    = name
        self.counter = 0

    def _on_step(self) -> bool:
        return True        
        
    # ------------------------------------------------------------------
    def _on_rollout_end(self):
        self.counter += 1
        if self.counter % self.every:
            return  # not this time

        fname = self.dir_path / f"{self.name}_{self.num_timesteps:_}.pkl"

        serialised = {}
        with self.lock:
            if self.verbose:
                total_graphs = sum(len(bucket) for bucket in self.registry.values())
                print(f"[snapshot] registry currently holds {total_graphs} graph(s)")

            for h, bucket in self.registry.items():
                if not bucket:          # skip empty buckets
                    continue
                serialised[h] = [
                    (nx.node_link_data(canon),
                     nx.node_link_data(orig),
                     e)
                    for canon, orig, e in bucket
                ]

        with fname.open("wb") as f:
            pickle.dump(serialised, f)

        if self.verbose:
            print(f"[snapshot] wrote {fname}  (hashes {len(serialised)})")





class PerEnvReturnCallback(BaseCallback):
    """Record discounted episode returns *per* sub‑env and persist them.

    Parameters
    ----------
    gamma : float
        Discount factor used to accumulate returns.
    save_dir : str or os.PathLike, optional
        Directory where ``env{idx}_returns.npy`` files will be written.
    save_every_episodes : int, optional
        Dump to disk every *n* completed episodes (per env). Set ``0`` to
        save only once at the end of training.  Defaults to ``0``.
    verbose : int, optional
        SB3 verbosity level.
    """

    def __init__(
        self,
        gamma: float,
        *,
        save_dir: Union[str, os.PathLike] = "per_env_returns",
        save_every_episodes: int = 0,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.gamma = gamma
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_every_episodes = save_every_episodes
        # filled at runtime
        self._running_returns: np.ndarray | None = None
        self._episode_counts: List[int] | None = None

    def _on_training_start(self) -> None:
        n_envs = self.model.get_env().num_envs
        self._running_returns = np.zeros(n_envs, dtype=np.float32)
        # one dynamic list per env → append on episode end
        self.episode_returns_per_env: List[List[float]] = [[] for _ in range(n_envs)]
        self._episode_counts = [0] * n_envs

    def _on_step(self) -> bool:
        rewards = self.locals["rewards"]  # shape: (n_envs,)
        dones   = self.locals["dones"]    # shape: (n_envs,)

        self._running_returns = rewards + self.gamma * self._running_returns

        for idx, done in enumerate(dones):
            if done:
                self.episode_returns_per_env[idx].append(self._running_returns[idx].item())
                self._running_returns[idx] = 0.0
                self._episode_counts[idx] += 1
                # periodic flush 
                if (
                    self.save_every_episodes
                    and self._episode_counts[idx] % self.save_every_episodes == 0
                ):
                    self._flush_env(idx)
        return True


    def _flush_env(self, env_idx: int):
        """Write the current list to ``env{idx}_returns.npy`` (overwrite)."""
        fname = self.save_dir / f"env{env_idx}_returns.npy"
        np.save(fname, np.asarray(self.episode_returns_per_env[env_idx], dtype=np.float32))
        if self.verbose:
            print(f"[PerEnvReturnCallback] wrote {fname} (n={len(self.episode_returns_per_env[env_idx])})")

    def _on_training_end(self) -> None:
        for idx in range(len(self.episode_returns_per_env)):
            self._flush_env(idx)
            
            
class PerEnvReturnAndLogCallback(PerEnvReturnCallback):
    """
    Inherits all saving behaviour from PerEnvReturnCallback **and**
    writes the most recent episode-return of every sub-env into the SB3
    logger so you can visualise it in TensorBoard/CSV.

    Parameters
    ----------
    gamma              : discount factor (same as parent)
    record_every_ep    : log every Nth episode per env (default: 1 → every ep)
    dump_every_steps   : flush logger every N env steps            (default: 2 000)
    other kwargs       : forwarded to the parent class
    """
    def __init__(self, gamma, *,
                 record_every_ep: int = 1,
                 dump_every_steps: int = 2_000,
                 **kwargs):
        super().__init__(gamma, **kwargs)
        self.record_every_ep  = record_every_ep
        self.dump_every_steps = dump_every_steps

    # --------------------------------------------------------------------- #
    def _on_step(self) -> bool:
        # run the parent logic first (updates running returns, saves to *.npy)
        super()._on_step()

        # now add the logging bit
        dones   = self.locals["dones"]
        for idx, done in enumerate(dones):
            if done and (self._episode_counts[idx] % self.record_every_ep == 0):
                latest_return = self.episode_returns_per_env[idx][-1]
                self.logger.record(f"env{idx}/episode_return", latest_return)

        # periodic flush so files stay in sync
        if self.num_timesteps % self.dump_every_steps == 0:
            self.logger.dump(self.num_timesteps)

        return True            
            

class BestEnergyLoggingCallback(BaseCallback):
    def __init__(self, shared_best_val):
        super().__init__()
        self.shared_best_val = shared_best_val

    def _on_training_start(self) -> None:
        # step 0
        self.logger.record("custom/best_energy", self.shared_best_val.value)
        self.logger.dump(0)

    def _on_step(self) -> bool:
        # record current value
        self.logger.record("custom/best_energy", self.shared_best_val.value)
        # **flush immediately – one point per env-step**
        self.logger.dump(self.num_timesteps)
        return True

def load_returns(dir_path: Union[str, os.PathLike]) -> List[np.ndarray]:
    """Load per‑env return arrays saved by :class:`PerEnvReturnCallback`."""
    dir_path = Path(dir_path)
    arrays = []
    for fname in sorted(dir_path.glob("env*_returns.npy")):
        arrays.append(np.load(fname))
    return arrays


def plot_returns(per_env_returns: List[np.ndarray], *, smoothing: int = 30):
    """Plot one curve per environment (optionally smoothed)."""
    plt.figure(figsize=(7, 4))
    for idx, arr in enumerate(per_env_returns):
        arr = np.asarray(arr, dtype=np.float32) 
        if arr.size == 0:
            continue
        y = arr.astype(np.float32)
        if len(y) >= smoothing:
            k = np.ones(smoothing) / smoothing
            y = np.convolve(y, k, mode="valid")
            x = np.arange(len(y)) + smoothing
        else:
            x = np.arange(len(y))
        plt.plot(x, y, label=f"env {idx}")
    plt.xlabel("Episode")
    plt.ylabel("Discounted return")
    plt.title("Learning curve per environment")
    plt.legend()
    plt.tight_layout()
    plt.show()


def load_graph_pickle(filename):
    """
    Load a graph from a pickle file and convert back to NetworkX format.
    
    Args:
        filename (str): Pickle file to load
        
    Returns:
        nx.DiGraph: Reconstructed graph
    """
    # Load the list from pickle
    with open(filename, 'rb') as f:
        graph_list = pickle.load(f)
    
    # Extract components
    num_nodes, edges, node_attrs = graph_list
    
    # Create new graph
    G = nx.DiGraph()
    
    # Add nodes with attributes
    for node, attr in node_attrs.items():
        if attr is not None:
            G.add_node(node, type=attr)
        else:
            G.add_node(node)
    
    # Add edges
    G.add_edges_from(edges)
    
    return G


# %%


# Get the number of CPU cores
num_cores = multiprocessing.cpu_count()

print(f"Number of CPU cores available: {num_cores}")
print(f"CUDA available: {th.cuda.is_available()}")
print(f"Number of GPUs: {th.cuda.device_count()}")


print("Finished with initial setup and going to main function")

if __name__ == '__main__':
    
    
    print("Importing and parsing arguments")
    
    # Parsing CLI commands 
    parser = argparse.ArgumentParser(description="Train DRL model on NIG circuits")
    
    sel_group = parser.add_mutually_exclusive_group(required=True)

    parser.add_argument('--n_envs', type=int, default=80, help='Number of parallel environments')
    parser.add_argument('--learning_rate', type=float, default=0.0003, help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=10, help='Maximum steps per episode')
    parser.add_argument('--n_steps', type=int, default=20, help='Steps per rollout (per env)')
    parser.add_argument('--n_epochs', type=int, default=4, help='Number of epochs per update')
    parser.add_argument('--batch_size', type=int, default=160, help='Minibatch size')
    parser.add_argument('--folder_name', type=str, default='runs/default_run', help='Output folder name')
    parser.add_argument('--total_timesteps', type=int, default=1600*120, help='Total training steps')
    
    sel_group.add_argument("--target_hex", type=str, nargs='+', help="Hex identifiers of circuits to train on, e.g. 0x3AC7 0x1A2B 0x0FD5")
    sel_group.add_argument("--n_random", type=int, help="Number of logic functions to ramdomly sample")
    
    # Registry flags
    parser.add_argument('--use_registry', action='store_true',  default=False, help='Enable the shared solution registry during training')
    parser.add_argument('--store_every_new_graph', action='store_true',  default=False, help='Store every new design found when flag included, otherwise store only on tie or improvement')
    parser.add_argument('--registry_sampling', action='store_true',  default=False, help='Sample from registry as initial states when flag included')
    parser.add_argument('--strict_iso_check', action='store_true',  default=False, help='Use Networkx is_isomorphic instead of hashing for uniqueness check')  
    parser.add_argument('--max_registry_size', type=int, default=None, help='Limit on the number of designs in registry (None is unlimited)')
    parser.add_argument('--initial_state_sampling_factor', type=float, default=3, help='Factor for implementing weighter sampling of initial states')
    
    args = parser.parse_args() 
      
    
    # Discount function 
    discount = 0.99
    
    # Learning rate 
    learning_rate = args.learning_rate
    
    # Number of environments
    n_envs = args.n_envs
    
    # Max steps per episode
    max_steps = args.max_steps
    
    # Steps to run before training step
    n_steps = args.n_steps
    
    # Epochs during training 
    n_epochs = args.n_epochs
    
    # batch during training 
    batch_size = args.batch_size
    
    # Output folder to log in the files
    folder_name = args.folder_name
    
    # Total time steps during the entire learning
    total_timesteps = args.total_timesteps
    
    # Hex circuit identifier 
    target_hex = args.target_hex 

    print("Selecting logic functions")
    
    # Path to the directory containing all the circuit files
    directory_path = '/home/gridsan/spalacios/DRL1/supercloud-testing/ABC-and-PPO-testing1/Verilog_files_for_all_4_input_1_output_truth_tables_as_NIGs/'

    '''
    # Load NIGs_unoptimized_library_3_input_1_output
    with open('NIGs_unoptimized_library_3_input_1_output.pkl', 'rb') as file:
        NIGs_unoptimized_library_3_input_1_output = pickle.load(file)    
    '''
    
    all_files = glob.glob(os.path.join(directory_path, '*_NIG_unoptimized.pkl'))

    #if specific logic functions are given
    if args.target_hex:
        selected_files = []
        for hex_id in args.target_hex:
            pattern = os.path.join(directory_path, f'{hex_id}_NIG_unoptimized.pkl')
            hits    = glob.glob(pattern)
            if not hits:
                raise RuntimeError(f'No circuit file found for {hex_id}!')
            selected_files.extend(hits)
    #if no specific logic function, select randomly 
    else:
        if args.n_random > len(all_files):
            raise ValueError(f'Requested {args.n_random} circuits but only {len(all_files)} available.')
        selected_files = random.sample(all_files, args.n_random)
        
    print(f'Selected {len(selected_files)} circuit(s)')

    
    # make sure the folder exists
    Path(folder_name).mkdir(parents=True, exist_ok=True)

    # Write selected logic functions to log
    chosen_log  = os.path.join(folder_name, f"selected_circuits.txt")
    with open(chosen_log, "w") as f:
        for fp in selected_files:
            print("Selected", fp, file=f)   
            print("Selected", fp)           

    print(f"Wrote {len(selected_files)} selected circuits to {chosen_log}", flush=True)
    
    meta = {
        "cli_args": vars(args),

        "system": {
            "timestamp_utc" : datetime.utcnow().isoformat(timespec="seconds"),
            "hostname"      : platform.node(),
            "python"        : sys.version.split()[0],
            "cuda_available": th.cuda.is_available(),
            "num_gpus"      : th.cuda.device_count(),
            "num_cpus"      : multiprocessing.cpu_count(),
        }
    }    
    
    meta_file = Path(folder_name) / "run_meta.json"   # ← define before use
    meta_file.write_text(json.dumps(meta, indent=2, sort_keys=True))   # ← add )
    print(f"Wrote {meta_file}")

    G_initial_states = []
    # Loop over the selected logic functions
    for circuit_file in selected_files:
        # Extract the circuit from the filename
        filename = os.path.basename(circuit_file)
        circuit_name = filename.split('_')[0]  

        # Load the graph
        G = load_graph_pickle(circuit_file)

        # Add the states to our collection
        G_initial_states.append(G.copy())

        # Optional: Print progress
        print(f"Processed {circuit_name}")

    print(f"Processed {len(selected_files)} total files")
    print(f"Total initial states generated: {len(G_initial_states)}")
    
    # Set the circuit name(s)
    if args.target_hex is not None:              
        circuit_name = "_".join(args.target_hex)  
    else:                                         
        circuit_name = f"{args.n_random}_random" 

    
    # Set the environments
    if args.use_registry:
        
        print("Setting up vectorized environments using registry")
        
        manager   = multiprocessing.Manager()
        registry_across_workers = manager.dict()          
        multiprocessing_lock  = manager.Lock()
        best_energy_across_workers  = manager.Value('d', math.inf)      
        
        def make_env():
            return DRL3env(
                max_nodes=100,
                graphs=G_initial_states,
                circuit_name=circuit_name,
                enable_full_graph_replacement=True,
                show_plots=False,
                log_info=False,
                max_steps=max_steps, 
                shared_registry=registry_across_workers,   
                registry_lock=multiprocessing_lock,     
                best_energy_across_workers = best_energy_across_workers,
                store_every_new_graph = args.store_every_new_graph,
                sampling_from_shared_registry = args.registry_sampling,
                max_registry_size = args.max_registry_size,
                strict_iso_check = args.strict_iso_check,
                initial_state_sampling_factor = args.initial_state_sampling_factor,
            )
    else:
        print("Setting up vectorized environments")
        def make_env():
                return DRL3env(
                    max_nodes=100,
                    graphs=G_initial_states,
                    circuit_name=circuit_name,
                    enable_full_graph_replacement=True,
                    show_plots=False,
                    log_info=False,
                    strict_iso_check=False,
                    max_steps=max_steps, 
                )

    
    env = make_vec_env(make_env, n_envs=n_envs, vec_env_cls=SubprocVecEnv)

    
    # Set up the model 
    print("Setting up model")
    
    entropy_coef = 0.005
    maximum_grad_norm =0.5
    
    model = MaskablePPO(
        "MultiInputPolicy", 
        env, 
        gamma=discount,
        learning_rate=learning_rate,
        policy_kwargs=policy_kwargs, 
        n_steps=n_steps, 
        n_epochs=n_epochs, 
        batch_size = batch_size, 
        ent_coef=entropy_coef,
        max_grad_norm=maximum_grad_norm,
        device="cuda",
    )
    
    stable_baselines3_logger_path = Path(folder_name)
    
    stable_baselines3_logger = configure(
        folder        = str(stable_baselines3_logger_path),           
        format_strings= ["stdout", "json", "csv", "tensorboard"]
    )
    
    model.set_logger(stable_baselines3_logger) 
    
    
    # Start timing before training
    start_time = time.time()
    
    
    '''
    if args.use_registry:
        registry_progress_callback = RegistrySnapshotCallback(registry_across_workers, multiprocessing_lock,
                                           every_rollouts=1,  # every 1600 steps
                                           dir_path=folder_name, verbose=1)  
    else:
        registry_progress_callback = None
    '''
    
    
    # ---------------- callbacks -----------------
    callbacks: list[BaseCallback] = []


    if args.use_registry:
        callbacks.append(
            RegistrySnapshotCallback(
                registry_across_workers, multiprocessing_lock,
                every_rollouts=1, dir_path=folder_name, verbose=1
            )
        )
        callbacks.append(
            BestEnergyLoggingCallback(best_energy_across_workers)
        )

    callbacks.append(
        PerEnvReturnAndLogCallback(
            gamma=discount,
            save_dir=folder_name + "/per_env_returns",
            record_every_ep=1,
            dump_every_steps=10000,
            verbose=0
        )
    )

    callback = CallbackList(callbacks) if callbacks else None
    
    
    print("Starting design process")
    
    model.learn(total_timesteps , progress_bar=False, callback=callback)

    # End timing after training
    end_time = time.time()
    
    training_time = end_time - start_time

    print("Done")
    
    # Convert to hours, minutes, seconds
    hours = int(training_time // 3600)
    minutes = int((training_time % 3600) // 60)
    seconds = int(training_time % 60)
    
    print(f"Completed in {hours}h {minutes}m {seconds}s")

    # Save the model
    model.save(Path(folder_name) / "trained_model")
  
    if args.use_registry:
        save_full_registry(registry_across_workers, multiprocessing_lock, Path(folder_name) / "final_shared_registry.pkl")
    





