#!/usr/bin/env python
# coding: utf-8

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
from stable_baselines3.common.callbacks import StopTrainingOnNoModelImprovement

#import environment
from dgd.environments.drl3env_loader6_v2 import DRL3env
from dgd.environments.drl3env_loader6_v2 import _canonical_graph_transform as _apply_implicit_or
from dgd.environments.drl3env_loader6_v2 import _compute_hash, _compute_truth_key, _canon_and_energy

from dgd.utils.utils5 import energy_score, energy_score_general, check_implicit_OR_existence_v3
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

#from dgd.models.GAT_feature_extractor import GAT
from dgd.models.GAT_and_MLP_feature_extractor_with_scalars import GAT

# Update policy_kwargs with the new class
policy_kwargs = dict(
    features_extractor_class=GAT,
    features_extractor_kwargs=dict(features_dim=128),
    net_arch=dict(pi=[100, 100, 100, 100, 100], vf=[100, 100, 100, 100, 100])
) 

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
                print(f" Registry currently holds {total_graphs} graph(s)")

            for h, bucket in self.registry.items():
                if not bucket:          # skip empty buckets
                    continue
                serialised[h] = [(nx.node_link_data(canon), nx.node_link_data(orig),  e) for canon, orig, e in bucket]

        with fname.open("wb") as f:
            pickle.dump(serialised, f)

        if self.verbose:
            print(f" Wrote {fname}  (hashes {len(serialised)})")


class PerEnvReturnCallback(BaseCallback):
    """Record returns.

    Parameters
    ----------
    gamma : float
        Discount factor used to accumulate returns.
        Dump to disk every n completed episodes (per env). Set "0" to
        save only once at the end of training.  Defaults to "0".
    verbose : int, optional
        SB3 verbosity level.
    """

    def __init__(self,  gamma: float,  *,   verbose: int = 0):
        super().__init__(verbose)
        
        self.gamma = gamma
        
        self._running_returns: np.ndarray | None = None
        self._episode_counts: List[int] | None = None

    def _on_training_start(self) -> None:
        n_envs = self.model.get_env().num_envs
        
        # Running rewards in episode
        self._running_returns = np.zeros(n_envs, dtype=np.float32)
        
        # Running discounted rewards in episode
        self._running_discounted     = np.zeros(n_envs, dtype=np.float32)
        
        # Discount 
        self._gamma_pow       = np.ones (n_envs, dtype=np.float32)
        
        # one dynamic list per env -> append on episode end
        
        # Reward in each episde per environment
        self.episode_returns_per_env= [[] for _ in range(n_envs)]
        
        # Dicounted reward in each episde per environment
        self.episode_returns_per_env_disc = [[] for _ in range(n_envs)]
        
        # Episodes per environment
        self._episode_counts = [0] * n_envs

    def _on_step(self) -> bool:
        rewards = self.locals["rewards"]  # shape: (n_envs,)
        dones   = self.locals["dones"]    # shape: (n_envs,)

        #cummulative return
        self._running_returns += rewards
        self._running_discounted    += self._gamma_pow * rewards
        self._gamma_pow       *= self.gamma

        for idx, done in enumerate(dones):
            if done:
                self.episode_returns_per_env[idx].append(self._running_returns[idx].item())
                self.episode_returns_per_env_disc[idx].append(self._running_discounted   [idx].item())
                
                self._running_returns[idx] = 0.0
                self._running_discounted  [idx] = 0.0            
                self._gamma_pow     [idx] = 1.0                      
                
                self._episode_counts[idx] += 1

        return True

            
class PerEnvReturnAndLogCallback(PerEnvReturnCallback):
    """
    Inherits all saving behaviour from PerEnvReturnCallback and
    writes the most recent episode-return of every sub-env into the SB3
    logger to visualise it in TensorBoard/CSV.

    Parameters
    ----------
    gamma              : discount factor (same as parent)
    record_every_ep    : log every Nth episode per env (default: 1 → every ep)
    dump_every_steps   : flush logger every N env steps            (default: 2 000)
    other kwargs       : forwarded to the parent class
    """
    def __init__(self, gamma, *, record_every_ep: int = 1, dump_every_steps: int = 2_000,  **kwargs):
        super().__init__(gamma, **kwargs)
        self.record_every_ep  = record_every_ep
        self.dump_every_steps = dump_every_steps

    def _on_step(self) -> bool:
        # run the parent logic first (updates running returns, saves to *.npy)
        super()._on_step()

        # now add the logging bit
        dones   = self.locals["dones"]
        for idx, done in enumerate(dones):
            if done and (self._episode_counts[idx] % self.record_every_ep == 0):

                self.logger.record(f"env{idx}/episode_return", self.episode_returns_per_env[idx][-1])
                self.logger.record(f"env{idx}/episode_return_disc",   self.episode_returns_per_env_disc[idx][-1])              

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
        # flush immediately – one point per env-step
        self.logger.dump(self.num_timesteps)
        return True
    
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
    
    # Start timing 
    t_total_start = time.perf_counter()
    #start_time = time.perf_counter()
    
    print("Importing and parsing arguments")
    
    # Parsing CLI commands 
    parser = argparse.ArgumentParser(description="Train DRL model on NIG circuits")

    # General model and environment parameters 
    parser.add_argument('--n_envs', type=int, default=80, help='Number of parallel environments')
    parser.add_argument('--learning_rate', type=float, default=0.0003, help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=10, help='Maximum steps per episode')
    parser.add_argument('--n_steps', type=int, default=20, help='Steps per rollout (per env)')
    parser.add_argument('--n_epochs', type=int, default=4, help='Number of epochs per update')
    parser.add_argument('--batch_size', type=int, default=160, help='Minibatch size')
    parser.add_argument('--total_timesteps', type=int, default=1600*120, help='Total training steps')
    parser.add_argument('--entropy_coef', type=float, default=0.005, help='Entropy coeficient')
    parser.add_argument('--maximum_grad_norm', type=float, default=0.5, help='maximum_grad_norm')
    parser.add_argument('--discount', type=float, default=0.99, help='discount')
    parser.add_argument('--max_nodes', type=int, default=100, help='max_nodes in environment')
    parser.add_argument('--global_seed', type=int, default=None, help='global seed')
    
    parser.add_argument('--output_folder_name', type=str, default='runs/no_folder_name', help='Output folder name')

    # Include of fine-tuning will be performed 
    parser.add_argument('--fine_tuning', type=str, default=None, help='Path to model to use duriing fine tuning')
    
    # Use mode
    sel_group = parser.add_mutually_exclusive_group(required=True)
    sel_group.add_argument("--target_hex", type=str, nargs='+', help="Hex identifiers of circuits to use from the datasets, e.g. 0x3AC7 0x1A2B 0x0FD5")
    sel_group.add_argument("--n_random", action='store_true', default=False, help="Ramdomly sample from the datasets")
    sel_group.add_argument("--nig_path", metavar="FILE", default=None, help="Full path to a single NIG. Provided as .pkl file")  
    sel_group.add_argument("--load_registry", metavar="FILE", default=None, help="Full path to registry. It will initialize the shared registry.")
    parser.add_argument('--n_random_3', type=int, default=None, help= 'Number of random graphs to draw from the 3-input library')
    parser.add_argument('--n_random_4', type=int, default=None, help= 'Number of random graphs to draw from the 3-input library')       
       
    # Registry flags
    parser.add_argument('--use_registry', action='store_true',  default=False, help='Enable the shared solution registry')
    parser.add_argument('--store_every_new_graph', action='store_true',  default=False, help='Store every new design found when flag included, otherwise store only on tie or improvement')
    parser.add_argument('--registry_sampling', action='store_true',  default=False, help='Sample from registry as initial states when flag included')
    parser.add_argument('--strict_iso_check', action='store_true',  default=False, help='Use Networkx is_isomorphic instead of hashing for uniqueness check')  
    parser.add_argument('--max_registry_size', type=int, default=None, help='Limit on the number of designs in registry (No limit by default)')
    parser.add_argument('--initial_state_sampling_factor', type=float, default=3, help='Factor for implementing weighter sampling of initial states')
    parser.add_argument('--registry_read_only', action="store_true", help='Make registry read only')
    
    #other flags
    parser.add_argument('--early_stop', action="store_true", help='Perform early stopping')
        
    parser.add_argument('--checkpoints', dest='do_checkpoints', action='store_true',   help='Enable periodic model checkpoints (opt-in)')
    parser.add_argument('--eval', dest='do_eval', action='store_true', help='Enable evaluation during training (opt-in)')
    parser.add_argument('--periodic_registry', dest='periodic_registry', action='store_true', help='Enable periodic registry snapshots (opt-in)')
    args = parser.parse_args() 
    
    # Seeding from CLI parameter 
    if args.global_seed is not None:
        print(f"Using provided seed {args.global_seed}") 
        random.seed(args.global_seed)            
        np.random.seed(args.global_seed)
        th.manual_seed(args.global_seed)    

    # Select a random seed if None in CLI parameter       
    if args.global_seed is None:
        args.global_seed = int.from_bytes(os.urandom(4), "little")
        print(f"Generated random seed {args.global_seed}")
        random.seed(args.global_seed)            
        np.random.seed(args.global_seed)
        th.manual_seed(args.global_seed)              
      
    # Number of environments
    n_envs = args.n_envs
    
    # Output folder to log in the files
    output_folder_name = args.output_folder_name   
    
    Path(output_folder_name).mkdir(parents=True, exist_ok=True) 

    print("Selecting initial states")
    
    # If a specific NIG file was requested
    if args.nig_path:
        print("Using provided path to NIG file")
        nig_custom_file = Path(args.nig_path).expanduser().resolve()
        if not nig_custom_file.is_file():
            raise FileNotFoundError(f"NIG file not found: {nig_custom_file}")

        selected_files = [str(nig_custom_file)]
        directories    = [str(nig_custom_file.parent)]
        print(f"Using NIG file: {nig_custom_file}")

        
        chosen_log = os.path.join(output_folder_name, "selected_graphs.txt")
        with open(chosen_log, "w") as f:
            print("Selected", nig_custom_file, file=f)
        print("Selected", nig_custom_file)
        print(f"Wrote 1 selected circuit to {chosen_log}", flush=True)

    # If loading a shared registry
    elif args.load_registry:
        print("A registry will be loaded, so setting selected_files empty")
        selected_files = []
    
    # If selecting from the libraries of graphs across logic functions
    else:   
        print("I will be loading from the 3-input and 4-input datasets")
    
        LIB_DIR_3 = "/home/gridsan/spalacios/Designing complex biological circuits with deep neural networks/dgd/data/NIGs_3_inputs/"
        LIB_DIR_4 = "/home/gridsan/spalacios/Designing complex biological circuits with deep neural networks/dgd/data/NIGs_4_inputs/"

        def auto_dir_from_hex(hex_list: list[str]) -> str:
            h = hex_list[0].lower().removeprefix("0x")
            return LIB_DIR_3 if len(h) <= 2 else LIB_DIR_4

        # If --target_hex was used, infer from the first ID
        if args.target_hex:
            print("I will load a sepecific graph from the datasets")
            directories = [auto_dir_from_hex(args.target_hex)]
            directory_path = directories[0]
            selected_files = []
            for hex_id in args.target_hex:
                pattern = os.path.join(directory_path, f'{hex_id}_NIG_unoptimized.pkl')
                hits    = glob.glob(pattern)
                if not hits:
                    raise RuntimeError(f'No circuit file found for {hex_id}!')
                selected_files.extend(hits)
        #if no specific logic function, select randomly 
        else:
            print("I will be load graphs selected randomly from the datasets")
            # gather all files once
            all_files_3 = glob.glob(os.path.join(LIB_DIR_3, "*_NIG_unoptimized.pkl"))
            all_files_4 = glob.glob(os.path.join(LIB_DIR_4, "*_NIG_unoptimized.pkl"))

            selected_files = []        

            if args.n_random_3:
                print("I will be load 3-input graphs selected randomly from the datasets")
                if args.n_random_3 > len(all_files_3):
                    raise ValueError(f"Requested {args.n_random_3} graphs from 3-input library "
                                    f"but only {len(all_files_3)} available.")
                selected_files += random.sample(all_files_3, args.n_random_3)

            if args.n_random_4:
                print("I will be load 4-input graphs selected randomly from the datasets")
                if args.n_random_4 > len(all_files_4):
                    raise ValueError(f"Requested {args.n_random_4} graphs from 4-input library "
                                    f"but only {len(all_files_4)} available.")
                selected_files += random.sample(all_files_4, args.n_random_4)

            random.shuffle(selected_files)
            
        print(f'Selected {len(selected_files)} circuit(s)')
        
        # make sure the folder exists
        #Path(output_folder_name).mkdir(parents=True, exist_ok=True)

        # Write selected logic functions to log
        chosen_log  = os.path.join(output_folder_name, f"selected_graphs.txt")
        with open(chosen_log, "w") as f:
            for fp in selected_files:
                print("Selected", fp, file=f)   
                print("Selected", fp)           

        print(f"Wrote {len(selected_files)} selected circuits to {chosen_log}", flush=True)
        
    G_initial_states = []
    if args.load_registry:
        # Loading a registry as the initial state pool, so G_initial_states will not be used
        print("A registry will be loaded, so G_initial_states it left empty") 
        pass
    else:
        print("Setting G_initial_states")    
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

    # existing keys in G_initial_states
    existing_keys = {_compute_truth_key(g) for g in G_initial_states}
    print(f"Logic function keys in G_initial_states are {existing_keys}") 
    
    # Set the environments
    if args.use_registry:
        
        print("Setting up vectorized environments using registry")        

        manager   = multiprocessing.Manager()
        registry_across_workers = manager.dict()          
        multiprocessing_lock  = manager.Lock()
        best_energy_across_workers  = manager.Value('d', math.inf)  
        
        # if loading a previous registry
        if args.load_registry:
            reg_path = Path(args.load_registry).expanduser()
            print(f"Loading registry at {reg_path}")

            with reg_path.open("rb") as f:
                reg = pickle.load(f)       

                for h, bucket in reg.items():
                    registry_across_workers[h] = [
                        (nx.node_link_graph(canon_nl),
                        nx.node_link_graph(orig_nl),
                        e)
                        for canon_nl, orig_nl, e in bucket
                    ]
                    for _c, _o, e in registry_across_workers[h]:
                        if e < best_energy_across_workers.value:
                            best_energy_across_workers.value = e


            print(f"Loaded registry with {len(registry_across_workers)} hash buckets"
                f"best Energy = {best_energy_across_workers.value:.3f}")     
            
            #Use the existing keys in the loaded registry instead
            reg_items  = [item for bucket in registry_across_workers.values() for item in bucket]
            pool     = [orig for _, orig, _ in reg_items]   
            existing_keys = {_compute_truth_key(g) for g in pool}
            print(f"Using the existing keys in the loaded registry instead. The keys are: {existing_keys}")
            #More efficient keys calculation
            #existing_keys = {_compute_truth_key(orig)  for bucket in registry_across_workers.values() for _, orig, _ in bucket}         

        #else, add seed graphs to registry 
        else:
            print("Adding graphs in G_initial_states to shared registry") 
            #seed_energies = [energy_score(g, check_implicit_OR_existence_v3)[0] for g in G_initial_states]
            seed_energies = [_canon_and_energy(g)[0] for g in G_initial_states]
            best_energy_across_workers.value = min(seed_energies)
            
            for g_seed in G_initial_states:                      
                canon  = _apply_implicit_or(g_seed)
                h      = _compute_hash(canon)
                #e      = energy_score(g_seed, check_implicit_OR_existence_v3)[0]
                e      = energy_score_general(canon)[0]

                bucket = registry_across_workers.setdefault(h, [])
                bucket.append((canon, g_seed.copy(), e))                
                registry_across_workers[h] = bucket  
                
            print(f"Registry seeded with {len(registry_across_workers)} hash buckets, "f"best energy={best_energy_across_workers.value:.3f}")
            

        def make_env():
            return DRL3env(
                max_nodes=args.max_nodes,
                graphs=G_initial_states, #they have been seeded in the registry
                enable_full_graph_replacement=True,
                show_plots=False,
                log_info=False,
                max_steps=args.max_steps, 
                shared_registry=registry_across_workers,   
                registry_lock=multiprocessing_lock,     
                best_energy_across_workers = best_energy_across_workers,
                store_every_new_graph = args.store_every_new_graph,
                sampling_from_shared_registry = args.registry_sampling,
                max_registry_size = args.max_registry_size,
                strict_iso_check = args.strict_iso_check,
                initial_state_sampling_factor = args.initial_state_sampling_factor,
                registry_read_only = args.registry_read_only,
                existing_keys = existing_keys,
            )
    else:
        #existing_keys = {_compute_truth_key(g) for g in G_initial_states}
        print("Setting up vectorized environments without shared registry")
        def make_env():
            return DRL3env(
                max_nodes=args.max_nodes,
                graphs=G_initial_states,
                enable_full_graph_replacement=True,
                show_plots=False,
                log_info=False,
                strict_iso_check=False,
                max_steps=args.max_steps,
                existing_keys = existing_keys,
            )
    
    print("Creating vectorized environments")
    env = make_vec_env(make_env, n_envs=n_envs, vec_env_cls=SubprocVecEnv, seed = args.global_seed)
    
    # Definitions for eval environments
    if args.use_registry:
        #print("Setting up evaluation environments with registry")
        def make_eval_env():
            return DRL3env(
                max_nodes=args.max_nodes,
                graphs=G_initial_states, #they have been seeded in the registry
                enable_full_graph_replacement=True,
                show_plots=False,
                log_info=False,
                max_steps=args.max_steps, 
                shared_registry=registry_across_workers,   
                registry_lock=multiprocessing_lock,     
                best_energy_across_workers = best_energy_across_workers,
                store_every_new_graph = False,
                sampling_from_shared_registry = args.registry_sampling,
                max_registry_size = args.max_registry_size,
                strict_iso_check = args.strict_iso_check,
                initial_state_sampling_factor = args.initial_state_sampling_factor,
                registry_read_only = True,
                existing_keys = existing_keys,
            )
    else:
        def make_eval_env():
            #print("Setting up evaluation environments without registry")
            return DRL3env(
                max_nodes=args.max_nodes,
                graphs=G_initial_states,
                enable_full_graph_replacement=True,
                show_plots=False,
                log_info=False,
                strict_iso_check=False,
                max_steps=args.max_steps, 
                registry_read_only = True,
                existing_keys = existing_keys,
            )

    
    # Set up the model 
    print("Creating model")
    
    entropy_coef = args.entropy_coef
    maximum_grad_norm = args.maximum_grad_norm
    
    if (args.fine_tuning == None):
        print(f"Initializing a new model")
        model = MaskablePPO(
            "MultiInputPolicy", 
            env, 
            gamma=args.discount,
            learning_rate=args.learning_rate,
            policy_kwargs=policy_kwargs, 
            n_steps=args.n_steps, 
            n_epochs=args.n_epochs, 
            batch_size = args.batch_size, 
            ent_coef=entropy_coef,
            max_grad_norm=maximum_grad_norm,
            device="cuda",
            seed = args.global_seed,
        )
    else:
        print(f"Fine-tunning. Initializing to model {args.fine_tuning}")
        model = MaskablePPO.load(
            args.fine_tuning,
            env=env,
            n_steps=args.n_steps,                  
            device="cuda",
            seed = args.global_seed,           
        )    
    
    stable_baselines3_logger_path = Path(output_folder_name)
    
    stable_baselines3_logger = configure(
        folder        = str(stable_baselines3_logger_path),           
        format_strings= ["stdout", "json", "csv", "tensorboard"]
    )
    
    model.set_logger(stable_baselines3_logger) 
    
    print("Setting up callbacks")
    # Callbacks
    callbacks = []
    
    #Currently this would be every 160,000 steps (100 * n_steps * n_envs)
    registry_save_freq = 100
    if args.use_registry:
        if args.periodic_registry:
            callbacks.append(             
                RegistrySnapshotCallback(
                    registry_across_workers, multiprocessing_lock,
                    every_rollouts=registry_save_freq, dir_path=output_folder_name, verbose=1
                )
            )
        else:
            print("Default: skipping intermediate registry snapshots; final save only.")            
        
        callbacks.append(BestEnergyLoggingCallback(best_energy_across_workers))
        
    callbacks.append(
        PerEnvReturnAndLogCallback(
            gamma=args.discount,
            record_every_ep=1,
            dump_every_steps=100000,
            verbose=0
        )
    )


    
    #Check point Callback
    save_freq_steps = 1600*10
    
    if args.do_checkpoints:
        
        ckpt_dir = Path(output_folder_name) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_callback = CheckpointCallback(
            save_freq = int(save_freq_steps // args.n_envs),             # env-steps between checkpoints
            save_path = ckpt_dir,              # folder to drop .zip files in
            name_prefix = "ppo_masked"         # produces files like ppo_masked_100000_steps.zip
        )            
        callbacks.append(checkpoint_callback)
    else:
        print("Default: checkpoints disabled.")            
    
    
    
    
    #Evaluation Callback
    if args.do_eval:
        print("Creating evaluation environments")
        eval_env = make_vec_env(make_eval_env, n_envs   = min(8, n_envs), vec_env_cls = DummyVecEnv, seed = args.global_seed)
            
        #For multiple circuits, long train
        #eval_freq_steps = 1600*10
        
        #For single circuit, short train
        eval_freq_steps = 1600*1
        eval_freq = eval_freq_steps // args.n_envs

        max_no_improvement_evals_early_stop   = 5 
        min_evals_early_stop = 0      
                               
        if args.early_stop:
            stop_train_callback = StopTrainingOnNoModelImprovement(
                max_no_improvement_evals = max_no_improvement_evals_early_stop,  # stop after this many evaluation rounds
                min_evals                = min_evals_early_stop,                 # perfom this min evaluations before starting to count 
                verbose                  = 1
            )    
        else:
            stop_train_callback = None 
            
        eval_callback = MaskableEvalCallback(
            eval_env,
            eval_freq=eval_freq,    
            n_eval_episodes=100,                         # average over 100 episodes
            deterministic=False,                          
            render=False,
            log_path=str(Path(output_folder_name) / "eval_logs"),
            best_model_save_path=str(Path(output_folder_name) / "best_model"),
            verbose=1,
            callback_after_eval  = stop_train_callback
        )
        callbacks.append(eval_callback)       
    else:
        print("Default: evaluation disabled.")
        if args.early_stop:
            print("Warning: --early_stop requested but evaluation is disabled; skipping early stopping.")        
         
    
    callback = CallbackList(callbacks) if callbacks else None       
   
    print("Start learning")
    t_train_start = time.perf_counter()

    model.learn(args.total_timesteps, progress_bar=False, callback=callback)

    # Save the model
    model.save(Path(output_folder_name) / "trained_model")
    
    t_train_end = time.perf_counter()
  
    if args.use_registry:
        save_full_registry(registry_across_workers, multiprocessing_lock, Path(output_folder_name) / "final_shared_registry.pkl")
    
    
    # End timing
    t_total_end = time.perf_counter()
    
    training_time = t_train_end - t_train_start
        
    total_time = t_total_end - t_total_start

    print("Done")
    
    # Convert to hours, minutes, seconds
    hours_training_time = int(training_time // 3600)
    minutes_training_time = int((training_time % 3600) // 60)
    seconds_training_time = int(training_time % 60)
    
    hours_total_time = int(total_time// 3600)
    minutes_total_time = int((total_time % 3600) // 60)
    seconds_total_time = int(total_time % 60)
    
    print(f"Training time: {hours_training_time}h {minutes_training_time}m {seconds_training_time}s")
    
    print(f"Total time: {hours_total_time}h {minutes_total_time}m {seconds_total_time}s")


    print("Saving run metadata")   
    
    # Policy metadata
    feature_extractor_class = policy_kwargs["features_extractor_class"]
    feature_extractor_kwargs = policy_kwargs.get("features_extractor_kwargs", {})
    feature_extractor_name = f"{feature_extractor_class.__module__}.{feature_extractor_class.__qualname__}"

    # Add callback/training frequencies to metadata ---
    checkpoint_cfg = {
        "enabled": bool(args.do_checkpoints),
    }
    if args.do_checkpoints:
        checkpoint_cfg.update({
            "save_freq_steps": int(save_freq_steps),
            "save_freq_per_env": int(save_freq_steps // args.n_envs),
        })

    eval_cfg = {"enabled": bool(args.do_eval)}
    if args.do_eval:
        eval_cfg.update({
            "eval_freq_steps": int(eval_freq_steps),
            "eval_freq_per_env": int(eval_freq),
        })
    
    if args.early_stop and args.do_eval:
        early_stopping_cfg = {
            "enabled": True,
            "max_no_improvement_evals": int(max_no_improvement_evals_early_stop),
            "min_evals": int(min_evals_early_stop),
        }
    else:
        early_stopping_cfg = {
            "enabled": False,
            "max_no_improvement_evals": None,
            "min_evals": None,
        }     
        
     
    meta = {
        "cli_args": vars(args),
        "reward_fn": "reward = 100 / best_energy_in_episode  (terminal step)",
        "system": {
            "timestamp_utc" : datetime.utcnow().isoformat(timespec="seconds"),
            "hostname"      : platform.node(),
            "python"        : sys.version.split()[0],
            "cuda_available": th.cuda.is_available(),
            "num_gpus"      : th.cuda.device_count(),
            "num_cpus"      : multiprocessing.cpu_count(),
        },
        "callbacks": {
            "checkpoint": checkpoint_cfg,
            "evaluation": eval_cfg,
            "early_stopping": early_stopping_cfg,
        },
        "environment": {
            "env_name": f"{DRL3env.__module__}.{DRL3env.__name__}",
        },
        "policy": {
            "features_extractor": feature_extractor_name,          # e.g., dgd.models.GAT_feature_extractor.GAT
            "features_extractor_kwargs": feature_extractor_kwargs,
            "net_arch": policy_kwargs.get("net_arch"),
        },
        "timing": {
            "total_seconds"   : round(total_time,    3),
            "training_seconds": round(training_time, 3),
            "t_total_start"   : t_total_start,
            "t_train_start"   : t_train_start,
            "t_train_end"     : t_train_end,
            "t_total_end"     : t_total_end,
        },       
    }   

    meta_file = Path(output_folder_name) / "run_meta.json"   # define before use
    meta_file.write_text(json.dumps(meta, indent=2, sort_keys=True))  
    print(f"Wrote {meta_file}")    
