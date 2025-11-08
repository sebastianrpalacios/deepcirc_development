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
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
import subprocess

#import gymnasium as gym

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

import os
from pathlib import Path
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
#import environment
from dgd.environments.drl3env_loader4 import DRL3env
# ## Specify binary and biological inputs

from collections import deque
import math, numpy as np

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as geom_nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

'''
input_signals_list_small_molecules = [
    {0: 0.0278, 1: 0.0022, 2: 0.0042},  # First set of input signals
    {0: 0.0278, 1: 0.0022, 2: 2.0082},  # Second set of input signals
    {0: 0.0278, 1: 5.0543, 2: 0.0042},  # Third set of input signals
    {0: 0.0278, 1: 5.0543, 2: 2.0082},  # First set of input signals
    {0: 3.9239, 1: 0.0022, 2: 0.0042},  # Second set of input signals
    {0: 3.9239, 1: 0.0022, 2: 2.0082},  # Third set of input signals    
    {0: 3.9239, 1: 5.0543, 2: 0.0042},  # Second set of input signals
    {0: 3.9239, 1: 5.0543, 2: 2.0082},  # Third set of input signals  
]    



input_signals_list_binary = [
    {0: 0, 1: 0, 2: 0},  # First set of input signals
    {0: 0, 1: 0, 2: 1},  # Second set of input signals
    {0: 0, 1: 1, 2: 0},  # Third set of input signals
    {0: 0, 1: 1, 2: 1},  # First set of input signals
    {0: 1, 1: 0, 2: 0},  # Second set of input signals
    {0: 1, 1: 0, 2: 1},  # Third set of input signals    
    {0: 1, 1: 1, 2: 0},  # Second set of input signals
    {0: 1, 1: 1, 2: 1},  # Third set of input signals  
]   


# ## Postech's library characterization

# %%


# Data
repressor_data = {
    "Repressor": ["AmeR", "AmtR", "BetI", "BM3R1", "BM3R1", "BM3R1", "HlyIIR", "IcaRA", "LitR", "LmrA", 
                  "PhlF", "PhlF", "PhlF", "PsrA", "QacR", "QacR", "SrpR", "SrpR", "SrpR", "SrpR"],
    "RBS": ["F1", "A1", "E1", "B1", "B2", "B3", "H1", "I1", "L1", "N1", 
            "P1", "P2", "P3", "R1", "Q1", "Q2", "S1", "S2", "S3", "S4"],
    "ymaxa": [3.835, 5.036, 3.065, 0.543, 0.822, 0.704, 2.462, 3.558, 4.296, 2.152,
              3.901, 6.505, 6.794, 6.489, 3.744, 3.778, 1.216, 2.556, 2.547, 3.314],
    "ymina": [1.06, 0.091, 1e-14, 0.002, 1e-14, 1e-14, 0.057, 1e-14, 0.074, 0.183,
              0.01, 1e-14, 1e-14, 1e-14, 1e-14, 1e-14, 1e-14, 1e-14, 1e-14, 1e-14],
    "Ka": [0.122, 0.022, 0.363, 0.051, 0.285, 0.261, 1e-14, 0.186, 1e-14, 1e-14,
          1e-14, 0.16, 0.167, 0.305, 0.126, 0.456, 1e-14, 0.224, 0.4, 0.108],
    "n": [1.306, 1.308, 1.833, 2.042, 1.685, 1.768, 2.596, 1.085, 1.696, 2.095,
          4, 2.237, 2.01, 1.54, 1.706, 1.979, 1.66, 1.337, 1.569, 1.344],
    "Toxicity (RPU)": ["-", "-", "-", "-", "-", "-", 4.06525, 0.47094, "-", "-", 
                       "-", "-", "-", "-", 4.06525, "-", "-", "-", "-", "-"]
}

# Create DataFrame
cello_v1_hill_function_parameters = pd.DataFrame(repressor_data)

cello_v1_hill_function_parameters


# ## Postech's toxicity data

# %%

# Load the JSON file
file_path = '/home/gridsan/spalacios/DRL1/supercloud-testing/ABC-and-PPO-testing1/gate_toxicity_POSTECH.json'
with open(file_path, 'r') as file:
    data = json.load(file)

# Extract all gate toxicity data
gate_toxicity_data = [item for item in data if item.get("collection") == "gate_toxicity"]

# Convert the list of dictionaries into a DataFrame
gate_toxicity_df = pd.DataFrame(gate_toxicity_data)

gate_toxicity_df.head()


# # Load unoptimized graphs

# Load NIGs_unoptimized_library_3_input_1_output
with open('NIGs_unoptimized_library_3_input_1_output.pkl', 'rb') as file:
    NIGs_unoptimized_library_3_input_1_output = pickle.load(file)
'''


class CustomGCNN(BaseFeaturesExtractor):
    """
    Pre-norm GATv2 stack with residual projections, optional node-feature
    embedding and attention pooling.
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 128, *, embed_dim: 'int | None' = None):
        super().__init__(observation_space, features_dim)

        # ---------- embedding (optional) ----------------------------------
        in_node_dim = observation_space["node_features"].shape[1]  # =4 (one-hot)
        self.node_embed: 'nn.Module | None' = None
        if embed_dim is not None:
            # one-hot (N,4) · W(4,embed_dim)  →  (N,embed_dim)
            self.node_embed = nn.Linear(in_node_dim, embed_dim, bias=False)
            in_node_dim = embed_dim                      # GNN now sees embed_dim

        # ------------------- GATv2 stack (pre-norm) ---------------------
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

        # ------------------- Global Attention pooling -------------------
        self.global_pool = geom_nn.AttentionalAggregation(
            gate_nn=nn.Sequential(
                nn.Linear(cur_dim, cur_dim // 2),
                nn.ReLU(),
                nn.Linear(cur_dim // 2, 1),
            )
        )
        self.gcn_out_dim = cur_dim


        # ------------------- Fusion head --------------------------------
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

    # ------------------------------------------------------------------
    def adj_matrix_to_edge_index(self, adj_matrix: th.Tensor) -> th.Tensor:
        row, col = (adj_matrix > 0).nonzero(as_tuple=True)
        edge_index = th.stack([row, col], dim=0).to(adj_matrix.device)
        return edge_index

    # ------------------------------------------------------------------
    def _encode_graph(self, x: th.Tensor, adj: th.Tensor) -> th.Tensor:
        if self.node_embed is not None:                  # optional embedding
            x = self.node_embed(x)

        edge_index = self.adj_matrix_to_edge_index(adj)

        for layer in self.gnn_layers:
            h = layer["conv"](layer["norm"](x), edge_index)   # pre-norm
            res = x if layer["proj"] is None else layer["proj"](x)
            x = F.relu(res + h)                               # residual + act

        batch = th.zeros(x.size(0), dtype=th.long, device=x.device)
        return self.global_pool(x, batch)

    # ------------------------------------------------------------------
    def process_graph(self, node_features: th.Tensor, adj_matrix: th.Tensor) -> th.Tensor:
        if adj_matrix.dim() == 3:  # batched
            outs = [
                self._encode_graph(node_features[i], adj_matrix[i])
                for i in range(adj_matrix.size(0))
            ]
            return th.stack(outs)
        return self._encode_graph(node_features, adj_matrix)

    # ------------------------------------------------------------------
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
    features_extractor_class=CustomGCNN,
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
        




def save_full_registry(registry, lock, out_file: str | Path):
    """Serialise the whole bucketed registry → pickle on disk."""
    out_file = Path(out_file)
    serialised = {}

    with (lock or contextlib.nullcontext()):
        for h, bucket in registry.items():          # bucket = list[tuple]
            serialised_bucket = []
            for canon, orig, e in bucket:
                serialised_bucket.append((
                    nx.node_link_data(canon),       # 0
                    nx.node_link_data(orig),        # 1
                    e                               # 2
                ))
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

        # decide which smoothing mode to use -------------------------------
        if smooth_steps and smooth_lines != 1:
            raise ValueError("Specify *either* smooth_steps or smooth_lines, not both.")
        if smooth_steps < 0 or smooth_lines < 1:
            raise ValueError("Window sizes must be positive.")

        if smooth_steps:                           # --- steps mode ----------
            lines = max(1, math.ceil(smooth_steps / log_every))
        else:                                      # --- lines mode ----------
            lines = smooth_lines

        self._hist_g = deque(maxlen=lines)         # last N Ḡ snapshots

    # ---------------------------------------------------------------------
    def _on_training_start(self) -> None:
        self.t0 = time.time()
        n_envs  = self.model.get_env().num_envs
        self._G = np.zeros(n_envs, dtype=np.float32)  # discounted return per env

    # ---------------------------------------------------------------------
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

        # 1) update discounted returns -----------------------------------
        self._G = r_t + self.gamma * self._G

        # 2) optional log line -------------------------------------------
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

        # 3) reset returns for envs whose episodes ended -----------------
        self._G[dones] = 0.0
        return True

class RegistrySnapshotCallback(BaseCallback):
    def __init__(self, registry, lock=None, every_rollouts=10,
                 dir_path="runs", name="shared_registry", verbose=0):
        super().__init__(verbose)

        self.registry = registry
        self.lock     = lock or contextlib.nullcontext()
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

    # ---------------------------------------------------------------------
    def _on_training_start(self) -> None:
        n_envs = self.model.get_env().num_envs
        self._running_returns = np.zeros(n_envs, dtype=np.float32)
        # one dynamic list per env → append on episode end
        self.episode_returns_per_env: List[List[float]] = [[] for _ in range(n_envs)]
        self._episode_counts = [0] * n_envs

    # ---------------------------------------------------------------------
    def _on_step(self) -> bool:
        rewards = self.locals["rewards"]  # shape: (n_envs,)
        dones   = self.locals["dones"]    # shape: (n_envs,)

        self._running_returns = rewards + self.gamma * self._running_returns

        for idx, done in enumerate(dones):
            if done:
                self.episode_returns_per_env[idx].append(self._running_returns[idx].item())
                self._running_returns[idx] = 0.0
                self._episode_counts[idx] += 1
                # periodic flush ------------------------------------------------
                if (
                    self.save_every_episodes
                    and self._episode_counts[idx] % self.save_every_episodes == 0
                ):
                    self._flush_env(idx)
        return True

    # ---------------------------------------------------------------------
    def _flush_env(self, env_idx: int):
        """Write the current list to ``env{idx}_returns.npy`` (overwrite)."""
        fname = self.save_dir / f"env{env_idx}_returns.npy"
        np.save(fname, np.asarray(self.episode_returns_per_env[env_idx], dtype=np.float32))
        if self.verbose:
            print(f"[PerEnvReturnCallback] wrote {fname} (n={len(self.episode_returns_per_env[env_idx])})")

    # ---------------------------------------------------------------------
    def _on_training_end(self) -> None:
        for idx in range(len(self.episode_returns_per_env)):
            self._flush_env(idx)



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


# %%


print(f"CUDA available: {th.cuda.is_available()}")
print(f"Number of GPUs: {th.cuda.device_count()}")


# %%

print("Will start main")

if __name__ == '__main__':
    
    
    print("Importing and parsing arguments")
    
    # Parsing CLI commands 
    parser = argparse.ArgumentParser(description="Train DRL model on NIG circuits")

    parser.add_argument('--n', type=int, default=10, help='Number of Boolean functions to train on')
    parser.add_argument('--n_envs', type=int, default=80, help='Number of parallel environments')
    parser.add_argument('--learning_rate', type=float, default=0.0003, help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=10, help='Maximum steps per episode')
    parser.add_argument('--n_steps', type=int, default=20, help='Steps per rollout (per env)')
    parser.add_argument('--n_epochs', type=int, default=4, help='Number of epochs per update')
    parser.add_argument('--batch_size', type=int, default=160, help='Minibatch size')
    parser.add_argument('--folder_name', type=str, default='runs/default_run', help='Output folder name')
    parser.add_argument('--total_timesteps', type=int, default=1600*120, help='Total training steps')

    args = parser.parse_args()   
    
    # Number of boolean functions to train on
    n = args.n
    
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

    # ---------------------------------------------
    manager   = multiprocessing.Manager()
    best_dict = manager.dict()          
    reg_lock  = manager.Lock()           

    print("Creating environments")
    
    
    
    '''
    # Define how many Boolean functions to train on
    n = 10
    graphs_per_boolean_function = 1
    
    
    discount = 0.99
    learning_rate = 0.0003
    n_envs = 80
    max_steps = 10    
    n_steps = 20 #20*80 = 1600
    folder_name = "runs/20250528_1_multiinput_not_pretrained_no_registry_env4_10_boolean_functions_CPU"
    total_timesteps = 1600*120
    n_epochs = 4
    batch_size = 160
    
    
    rollout_size   = n_envs * n_steps       # 1_600 in your setup
    save_every_k   = 10                    # number of rollouts between snapshots
    save_freq      = rollout_size * save_every_k   # 16 000 time-steps
    '''

    '''
    checkpoint_cb = CheckpointCallback(
        save_freq     = save_freq,               # <-- timesteps, not gradient steps
        save_path     = folder_name,             # “runs/20250425_multi_circuit_1”
        name_prefix   = "ppo_snapshot",
        save_replay_buffer = False,              # PPO has no replay buffer
        save_vecnormalize  = False,
    )    
    
    
    per_env_cb = PerEnvReturnCallback(discount, save_dir=folder_name, save_every_episodes=20)      
    
    progress_cb = ProgressCallback(total_timesteps, gamma = discount, log_every=100, smooth_steps=10)
    
    snap_cb = RegistrySnapshotCallback(best_dict, reg_lock,
                                       every_rollouts=1,  # every 1600 steps
                                       dir_path=folder_name, verbose=1)        
    '''
    
    # Path to the directory containing all the circuit files
    directory_path = '/home/gridsan/spalacios/DRL1/supercloud-testing/ABC-and-PPO-testing1/Verilog_files_for_all_4_input_1_output_truth_tables_as_NIGs/'

    # Get all .pkl files in the directory that match the pattern
    all_circuit_files = glob.glob(os.path.join(directory_path, '*_NIG_unoptimized.pkl'))

    # ---------- Exclude 0x3AC7 -------------------------------------------------
    target_hex = "0xmulticircuit"
    exclude_hex = "0x3AC7"
    all_circuit_files = [
        fp for fp in all_circuit_files
        if not os.path.basename(fp).startswith(f"{exclude_hex}_")
    ]
    if not all_circuit_files:
        raise RuntimeError(f"After excluding {exclude_hex}, no circuits remain!")
    print(f"[info] {len(all_circuit_files)} candidate circuits "
          f"after excluding {exclude_hex}", flush=True)
    # ---------------------------------------------------------------------------    

    # Randomly select n files (or all if there are fewer than n files)
    n = min(n, len(all_circuit_files))  # Make sure n is not larger than the total files available
    selected_circuit_files = random.sample(all_circuit_files, n)
    
    
    # ------------------------------------------------------------------
    # Persist the list of circuits picked for this run
    # make sure the folder exists
    Path(folder_name).mkdir(parents=True, exist_ok=True)

    timestamp   = time.strftime("%Y%m%d_%H%M%S")
    chosen_log  = os.path.join(folder_name, f"selected_circuits_{timestamp}.txt")

    with open(chosen_log, "w") as f:
        for fp in selected_circuit_files:
            print("[chosen]", fp, file=f)   # → file
            print("[chosen]", fp)           # → stdout / Slurm log

    print(f"[info] wrote {len(selected_circuit_files)} selected circuits to {chosen_log}", flush=True)
    # ------------------------------------------------------------------


    G_initial_states = []
    # Loop over the randomly selected circuit files
    for circuit_file in selected_circuit_files:
        # Extract the circuit hex from the filename
        filename = os.path.basename(circuit_file)
        circuit_hex = filename.split('_')[0]  # Assumes the format is 'HEXVALUE_NIG_unoptimized.pkl'

        # Load the graph
        G = load_graph_pickle(circuit_file)

        # Add the states to our collection
        G_initial_states.append(G.copy())

        # Optional: Print progress
        print(f"Processed {circuit_hex}")

    print(f"Processed {n} random files out of {len(all_circuit_files)} total files")
    print(f"Total initial states generated: {len(G_initial_states)}")
    
    circuit_hex = target_hex     # "0x3AC7"

    
    def make_env():
        # any per-worker initialisation
        return DRL3env(
            100,
            G_initial_states,
            circuit_name=circuit_hex,
            enable_full_graph_replacement=True,
            show_plots=False,
            log_info=False,
            shared_registry=best_dict,   # <<<<<<<<<<<<<<<<<<<<<<
            registry_lock=reg_lock,      # (same object for all)
        )
    '''
    def make_env():
        # any per-worker initialisation
        return DRL3env(
            100,
            G_initial_states,
            circuit_name=circuit_hex,
            enable_full_graph_replacement=True,
            show_plots=False,
            log_info=False,
            strict_iso_check=True,
        )
    '''
    
    env = make_vec_env(make_env, n_envs=n_envs, vec_env_cls=SubprocVecEnv)


    model = MaskablePPO(
        "MultiInputPolicy", 
        env, 
        gamma=discount,
        learning_rate=learning_rate,
        policy_kwargs=policy_kwargs, 
        n_steps=n_steps, 
        n_epochs=n_epochs, 
        batch_size = batch_size, 
        ent_coef=0.005,
        #ent_coef=0.1,
        max_grad_norm=0.5,
        device="cuda",
        tensorboard_log=folder_name
    )
    
    # Start timing before training
    start_time = time.time()

    registry_progress_callback = RegistrySnapshotCallback(best_dict, reg_lock,
                                       every_rollouts=1,  # every 1600 steps
                                       dir_path=folder_name, verbose=1)        
    

    model.learn(total_timesteps , progress_bar=False, callback = registry_progress_callback)
    
    # End timing after training
    end_time = time.time()
    
    training_time = end_time - start_time

    print("Done training")
    
    # Convert to hours, minutes, seconds
    hours = int(training_time // 3600)
    minutes = int((training_time % 3600) // 60)
    seconds = int(training_time % 60)
    
    print(f"Training completed in {hours}h {minutes}m {seconds}s")

    # Save the model
    model.save(Path(folder_name) / "trained_model")
  
    save_full_registry(best_dict, reg_lock, Path(folder_name) / "final_shared_registry.pkl")
    





