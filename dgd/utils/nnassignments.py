import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from dgd.utils.utils5 import *
import pickle
import pandas as pd
import json
import networkx as nx
from tqdm.notebook import tqdm
import time
import subprocess
import multiprocessing
import concurrent.futures
import os
import random
from collections import defaultdict
import socket
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from itertools import cycle
from pathlib import Path
import time
from sklearn.metrics import mean_squared_error, r2_score
import csv
import math

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.distance import euclidean
import networkx as nx
import pandas as pd
import random
import copy
import pickle
import itertools
from itertools import combinations
from itertools import product
import math
import scipy.sparse as sp

import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from IPython.display import display, Markdown
import os
import shutil
import networkx as nx
from typing import Dict, Any, Tuple, Callable
import os, sys, shutil, re
from pathlib import Path
import networkx as nx
import csv
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm
from copy import deepcopy



def small_topology_plot(graph_networkx):
    pos = nx.spring_layout(graph_networkx)
    color = 'lightblue'
    plt.figure(figsize=(3, 3))
    nx.draw(graph_networkx, pos, with_labels=False, node_color=color, node_size=250, font_size=10, font_weight='bold')
    nx.draw_networkx_labels(graph_networkx, pos, font_size=8, font_color='black')
    plt.show()   

def topology_plot_with_attrs(G, node_attr = None, edge_attr = None, seed = 42):

    if node_attr is None:
        first_node_attrs = next(iter(G.nodes(data=True)), (None, {}))[1]
        node_attr = next(iter(first_node_attrs), None)

    if edge_attr is None:
        for _, _, d in G.edges(data=True):
            if d:
                edge_attr = next(iter(d))
                break

    pos = nx.spring_layout(G, seed=seed)
    plt.figure(figsize=(5, 5)); plt.axis("off")

    node_labels = {}
    default_colour = "lightblue"          
    node_colors   = default_colour

    if node_attr is not None:
        values = [G.nodes[n].get(node_attr) for n in G.nodes()]


        if all(isinstance(v, (int, float)) for v in values if v is not None):
            norm = mcolors.Normalize(vmin=min(values), vmax=max(values))
            cmap = cm.get_cmap("viridis")
            node_colors = [cmap(norm(v)) for v in values]


        else:
            present_vals = {v for v in values if v is not None}

            unique_vals  = sorted(present_vals, key=lambda x: str(x))

            palette     = cycle(cm.tab20.colors)
            colour_map  = {val: next(palette) for val in unique_vals}
            missing_col = "#d3d3d3"                     # grey for None
            node_colors = [colour_map.get(v, missing_col) for v in values]


        for n in G.nodes():
            val = G.nodes[n].get(node_attr)
            node_labels[n] = f"{n}\n{node_attr}={val}"

    else:
        node_labels = {n: n for n in G.nodes()}


    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=400)
    nx.draw_networkx_edges(G, pos, arrows=G.is_directed())
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)

    if node_attr and not all(isinstance(v, (int, float)) for v in values if v is not None):
        handles = [
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=colour_map[val], markersize=8,
                       label=str(val))
            for val in unique_vals
        ]
        if any(v is None for v in values):
            handles.append(
                plt.Line2D([0], [0], marker="o", color="w",
                           markerfacecolor=missing_col, markersize=8,
                           label="None")
            )
        plt.legend(title=node_attr, handles=handles,
                   bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout(); plt.show()

def permute_graph(graph, permutation, input_nodes):
    mapping = {old: new for old, new in zip(input_nodes, permutation)}
    permuted_graph = nx.relabel_nodes(graph, mapping)
    return permuted_graph

def save_accumulated_results(accumulated_results, file_number, output_dir):
    """
    Save accumulated results to a file
    
    Parameters:
    -----------
    accumulated_results : list
        List of results to save
    file_number : int
        Current file number for naming
    output_dir : str
        Directory to save the file
        
    Returns:
    --------
    list
        Empty list to reset accumulator
    """
    output_file = os.path.join(output_dir, f"results_large_batch_{file_number}.pkl")
    try:
        with open(output_file, 'wb') as f:
            pickle.dump(accumulated_results, f)
        print(f"Successfully saved {len(accumulated_results):,} solutions to {output_file}")
    except Exception as e:
        print(f"Error saving file {output_file}: {str(e)}")
    return []  # Return empty list to reset accumulator

def process_permutation_batch(batch_data, roadblocking_inputs = [0, 2], specific_repressors = {'PhlF', 'SrpR', 'BM3R1', 'QacR'}):
    """
    Process a single batch of permutations
    """
    batch_id, adj_matrix, experimental_params, input_signals_small, input_signals_binary, gate_toxicity_df, permutations = batch_data
    
    batch_results = []
    for current_solution in permutations:
        # Create test graph
        Gtest = assign_representations_with_io_nodes_3(
            adj_matrix,
            experimental_params,
            current_solution
        )
        
        # Calculate toxicity
        toxicity_score, _ = calculate_toxicity_score(
            input_signals_small,
            Gtest,
            gate_toxicity_df
        )
        
        # Simulate logic
        actual_logic = simulate_signal_propagation(
            Gtest,
            input_signals_small
        )
        expected_logic = simulate_signal_propagation_binary(
            Gtest,
            input_signals_binary
        )
        
        # Calculate scores
        current_score = calculate_circuit_score(expected_logic, actual_logic)
        roadblocking_flag = is_roadblocking(Gtest, roadblocking_inputs = roadblocking_inputs, specific_repressors = specific_repressors)
        
        batch_results.append({
            'circuit_score': current_score,
            'toxicity_score': toxicity_score,
            'roadblocking_flag': roadblocking_flag[0],
            'permutation': current_solution
        })
    
    return batch_id, batch_results

'''
def worker_init():
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except Exception:
        pass
    try:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass
'''

def parallel_process_circuits(adj_matrix, valid_permutations, experimental_params,
                            input_signals_small, input_signals_binary,
                            gate_toxicity_df, output_dir,
                            num_samples=2000000, batch_size=1000,
                            solutions_per_file=None, num_cores = None):
    """
    Process a random subset of circuit solutions using all available cores
    and save results in configurable large chunks
    
    Parameters:
    -----------
    ... [previous parameters] ...
    solutions_per_file : int, optional
        Number of solutions to accumulate before saving to a file.
        If None, automatically sets to num_samples/10 (default)
        Min value is batch_size, max value is num_samples
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set default solutions_per_file if not specified
    if solutions_per_file is None:
        solutions_per_file = num_samples // 10  # Default to 10 files total
    else:
        # Ensure solutions_per_file is between batch_size and num_samples
        solutions_per_file = max(batch_size, min(solutions_per_file, num_samples))
    
    if num_cores == None:
        CPUs_alloc = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
        num_cores = CPUs_alloc
    
    print(f"Utilizing all {num_cores} cores for parallel processing")
    print(f"Solutions per file: {solutions_per_file:,}")
    print(f"Expected number of files: {num_samples // solutions_per_file + bool(num_samples % solutions_per_file)}")
    
    # Generate random indices for the subset
    print(f"Selecting {num_samples:,} random solutions from {len(valid_permutations):,} valid permutations")
    random_indices = random.sample(range(len(valid_permutations)), num_samples)
    selected_permutations = [valid_permutations[i] for i in random_indices]
    
    # Find the highest file number already saved
    existing_files = [f for f in os.listdir(output_dir) 
                     if f.startswith("results_large_batch_") and f.endswith(".pkl")]
    current_file_number = 0
    if existing_files:
        max_file_num = max(int(f.split('_')[-1].split('.')[0]) for f in existing_files)
        current_file_number = max_file_num + 1
    
    # Calculate total solutions already processed
    solutions_processed = current_file_number * solutions_per_file
    if solutions_processed >= num_samples:
        print("All solutions have already been processed")
        return    
  
    # Prepare common data
    common_data = (adj_matrix, experimental_params, input_signals_small, 
                  input_signals_binary, gate_toxicity_df)
    
    # Create batches for remaining solutions
    remaining_permutations = selected_permutations[solutions_processed:]
    total_batches = (len(remaining_permutations) + batch_size - 1) // batch_size
    
    # Prepare all batch data
    batch_data_list = []
    for batch_number in range(total_batches):
        start_idx = batch_number * batch_size
        end_idx = min(start_idx + batch_size, len(remaining_permutations))
        current_batch_permutations = remaining_permutations[start_idx:end_idx]
        
        batch_data = (batch_number,) + common_data + (current_batch_permutations,)
        batch_data_list.append(batch_data)
    
    start_time = time.perf_counter()
    progress_log = [(0.0, solutions_processed)]
    
    print(f"Processing {len(batch_data_list)} batches using {num_cores} cores")
    
    accumulated_results = []
    processed_solutions = solutions_processed
    
    # Process all batches using all cores
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = [executor.submit(process_permutation_batch, batch_data) 
                  for batch_data in batch_data_list]
        
        with tqdm(total=len(batch_data_list), desc="Processing batches", 
                 unit="batch") as pbar:
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    batch_id, batch_results = future.result()
                    accumulated_results.extend(batch_results)
                    processed_solutions += len(batch_results)
                    
                    elapsed_time = time.perf_counter() - start_time
                    progress_log.append((elapsed_time, processed_solutions))
                    
                    
                    # Save when we've accumulated enough solutions
                    if len(accumulated_results) >= solutions_per_file:
                        print(f"\nSaving file {current_file_number} with {len(accumulated_results):,} solutions...")
                        save_accumulated_results(
                            accumulated_results, 
                            current_file_number,
                            output_dir
                        )
                        accumulated_results = []  # Reset accumulator
                        current_file_number += 1
                    
                    # Update progress
                    elapsed_time = time.perf_counter() - start_time
                    remaining_batches = len(batch_data_list) - pbar.n
                    time_per_batch = elapsed_time / (pbar.n + 1)
                    remaining_time = remaining_batches * time_per_batch
                    
                    pbar.set_postfix({
                        'Solutions': f'{processed_solutions:,}/{num_samples:,}',
                        'Elapsed': f'{elapsed_time/60:.1f}min',
                        'Remaining': f'{remaining_time/60:.1f}min',
                        'Solutions/sec': f'{processed_solutions/elapsed_time:.1f}'
                    })
                    pbar.update(1)
                    
                except Exception as e:
                    print(f"\nError processing batch {batch_id}: {str(e)}")
                    continue
    
    # Save any remaining results
    if accumulated_results:
        print(f"\nSaving final file with {len(accumulated_results):,} solutions...")
        save_accumulated_results(accumulated_results, current_file_number, output_dir)
        current_file_number += 1
    
    total_time = time.perf_counter() - start_time
    
    if not progress_log or progress_log[-1][1] < processed_solutions:
        progress_log.append((total_time, processed_solutions))    
    
    
    print("\nProcessing completed!")
    print(f"Total solutions processed: {processed_solutions:,}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Average processing speed: {processed_solutions/total_time:.1f} solutions/second")
    print(f"Total files created: {current_file_number }")
    
    return {
        'total_solutions_processed': processed_solutions,
        'total_time': total_time,
        'output_directory': output_dir,
        'solutions_per_second': processed_solutions/total_time,
        'total_files': current_file_number,
        'progress': progress_log,
    }
    
def load_circuit_results(output_dir):
    """
    Load all circuit results from batch files in the specified directory.
    
    Parameters:
    -----------
    output_dir : str
        Directory containing the batch result files
        
    Returns:
    --------
    tuple
        (all_permutations, all_toxicity_scores, all_circuit_scores, all_roadblocking_flags)
    """
    # Get list of all result files (now using the correct filename pattern)
    files = [f for f in os.listdir(output_dir) 
             if f.startswith("results_large_batch_") and f.endswith(".pkl")]
    files.sort(key=lambda x: int(x[len("results_large_batch_"):-len(".pkl")]))
    
    # Initialize lists to store the data
    all_permutations = []
    all_toxicity_scores = []
    all_circuit_scores = []
    all_roadblocking_flags = []
    
    # Count total number of results for statistics
    total_results = 0
    
    # Load all results with a progress bar
    print(f"Found {len(files)} batch files to load")
    
    if len(files) == 0:
        print(f"No files found in directory: {output_dir}")
        print("Make sure the directory is correct and contains files with pattern 'results_large_batch_*.pkl'")
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    for file in tqdm(files, desc="Loading results"):
        try:
            with open(os.path.join(output_dir, file), 'rb') as f:
                results = pickle.load(f)
                for result in results:
                    all_permutations.append(result['permutation'])
                    all_toxicity_scores.append(result['toxicity_score'])
                    all_circuit_scores.append(result['circuit_score'])
                    all_roadblocking_flags.append(result['roadblocking_flag'])
                total_results += len(results)
                
            # Print progress for large files
            if (total_results % 100000) == 0:
                print(f"\nProcessed {total_results:,} solutions so far...")
                
        except Exception as e:
            print(f"Error loading file {file}: {str(e)}")
            continue
    
    if total_results == 0:
        print("No results were successfully loaded")
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    # Convert lists to numpy arrays for better memory efficiency
    all_permutations = np.array(all_permutations)
    all_toxicity_scores = np.array(all_toxicity_scores)
    all_circuit_scores = np.array(all_circuit_scores)
    all_roadblocking_flags = np.array(all_roadblocking_flags)
    
    # Print summary statistics
    print("\nLoading completed:")
    print(f"Total results loaded: {total_results:,}")
    print(f"Unique permutations: {len(np.unique(all_permutations, axis=0)):,}")
    print(f"Average circuit score: {np.mean(all_circuit_scores):.3f}")
    print(f"Average toxicity score: {np.mean(all_toxicity_scores):.3f}")
    print(f"Roadblocking percentage: {(np.sum(all_roadblocking_flags) / len(all_roadblocking_flags) * 100):.1f}%")
    
    return (all_permutations, all_toxicity_scores, all_circuit_scores, all_roadblocking_flags)


# Training Loop
def train_model(model, X_train, y_train, X_test, y_test, optimizer, criterion, num_epochs):
    train_losses = []
    test_losses = []
    pbar = tqdm(range(num_epochs), desc="Training Progress")
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()
        
        outputs = model(X_train)
        train_loss = criterion(outputs, y_train)
        
        train_loss.backward()
        optimizer.step()
        
        train_losses.append(train_loss.item())
        
        model.eval()
        with th.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test)
        
        test_losses.append(test_loss.item())
        
        pbar.set_postfix({'Train Loss': train_loss.item(), 'Test Loss': test_loss.item()})
    
    return train_losses, test_losses



def train_model_minibatch(model, X_train, y_train, X_eval, y_eval, optimizer, criterion, num_epochs,
                batch_size=256, shuffle=True, sampler=None, seed=42, grad_clip=None, evaluate_validation_set = True):
    """
    X_eval/y_eval can be your validation set; keep test truly untouched for the final report.
    """
    device = next(model.parameters()).device
    g = th.Generator(device='cpu')
    g.manual_seed(seed)

    # Build loaders. If the function passes a sampler, shuffle must be False.
    train_ds = TensorDataset(X_train, y_train)
    use_shuffle = (sampler is None) and shuffle
    train_loader = DataLoader(train_ds,
                              batch_size=batch_size,
                              shuffle=use_shuffle,
                              sampler=sampler,
                              drop_last=False,        # set True if you use BatchNorm
                              num_workers=0,
                              pin_memory=(device.type == 'cuda' and X_train.device.type == 'cpu'),
                              generator=g)

    eval_loader = DataLoader(TensorDataset(X_eval, y_eval),
                             batch_size=max(1024, batch_size),
                             shuffle=False,
                             num_workers=0,
                             pin_memory=(device.type == 'cuda' and X_eval.device.type == 'cpu'))

    train_losses, eval_losses = [], []
    epoch_train_loss = None
    epoch_eval_loss = None
    
    pbar = tqdm(range(num_epochs), desc="Training Progress")
    for epoch in pbar:
        # train 
        model.train()
        running = 0.0
        nobs = 0
        for xb, yb in train_loader:
            if xb.device != device: xb = xb.to(device, non_blocking=True)
            if yb.device != device: yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            if grad_clip is not None:
                th.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            bs = xb.size(0)
            running += loss.item() * bs
            nobs += bs
        epoch_train_loss = running / max(1, nobs)
        train_losses.append(epoch_train_loss)

        # eval 
        if evaluate_validation_set:
            model.eval()
            running = 0.0
            nobs = 0
            with th.no_grad():
                for xb, yb in eval_loader:
                    if xb.device != device: xb = xb.to(device, non_blocking=True)
                    if yb.device != device: yb = yb.to(device, non_blocking=True)
                    pred = model(xb)
                    loss = criterion(pred, yb)
                    bs = xb.size(0)
                    running += loss.item() * bs
                    nobs += bs
            epoch_eval_loss = running / max(1, nobs)
            eval_losses.append(epoch_eval_loss)

        pbar.set_postfix({
            'Train': f'{epoch_train_loss:.4f}' if epoch_train_loss is not None else 'None',
            'Eval': f'{epoch_eval_loss:.4f}' if epoch_eval_loss is not None else 'None'
        })

    return train_losses, eval_losses

def train_model_minibatch_early_stoping(model, X_train, y_train, X_eval, y_eval, optimizer, criterion, num_epochs,
                batch_size=256, shuffle=True, sampler=None, seed=42, grad_clip=None, evaluate_validation_set=True,
                # optional early stopping 
                early_stop_patience=None,           # e.g., 10; None disables
                early_stop_min_delta=0.0,           # required improvement to reset patience
                restore_best_on_stop=True):         # load best weights when stopping
    """
    X_eval/y_eval is validation set; keep test truly untouched for the final report.
    """
    device = next(model.parameters()).device
    g = th.Generator(device='cpu')
    g.manual_seed(seed)

    # Build loaders. If the function passes a sampler, shuffle must be False.
    train_ds = TensorDataset(X_train, y_train)
    use_shuffle = (sampler is None) and shuffle
    train_loader = DataLoader(train_ds,
                              batch_size=batch_size,
                              shuffle=use_shuffle,
                              sampler=sampler,
                              drop_last=False,        # set True if you use BatchNorm
                              num_workers=0,
                              pin_memory=(device.type == 'cuda' and X_train.device.type == 'cpu'),
                              generator=g)

    eval_loader = DataLoader(TensorDataset(X_eval, y_eval),
                             batch_size=max(1024, batch_size),
                             shuffle=False,
                             num_workers=0,
                             pin_memory=(device.type == 'cuda' and X_eval.device.type == 'cpu'))

    train_losses, eval_losses = [], []
    epoch_train_loss = None
    epoch_eval_loss = None

    # early stopping state 
    best_metric = float('inf')
    bad_epochs = 0
    best_state = None

    pbar = tqdm(range(num_epochs), desc="Training Progress")
    for epoch in pbar:
        # train 
        model.train()
        running = 0.0
        nobs = 0
        for xb, yb in train_loader:
            if xb.device != device: xb = xb.to(device, non_blocking=True)
            if yb.device != device: yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            if grad_clip is not None:
                th.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            bs = xb.size(0)
            running += loss.item() * bs
            nobs += bs
        epoch_train_loss = running / max(1, nobs)
        train_losses.append(epoch_train_loss)

        # eval 
        if evaluate_validation_set:
            model.eval()
            running = 0.0
            nobs = 0
            with th.no_grad():
                for xb, yb in eval_loader:
                    if xb.device != device: xb = xb.to(device, non_blocking=True)
                    if yb.device != device: yb = yb.to(device, non_blocking=True)
                    pred = model(xb)
                    loss = criterion(pred, yb)
                    bs = xb.size(0)
                    running += loss.item() * bs
                    nobs += bs
            epoch_eval_loss = running / max(1, nobs)
            eval_losses.append(epoch_eval_loss)

        # early stopping 
        if early_stop_patience is not None:
            metric = epoch_eval_loss if evaluate_validation_set else epoch_train_loss
            if metric < (best_metric - early_stop_min_delta):
                best_metric = metric
                bad_epochs = 0
                if restore_best_on_stop:
                    # lightweight CPU copy with best weights
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad_epochs += 1
                if bad_epochs >= early_stop_patience:
                    if restore_best_on_stop and best_state is not None:
                        model.load_state_dict(best_state)
                        model.to(device)
                    print(f"[EarlyStop] Stopping at epoch {epoch}. Best loss = {best_metric:.6f}")
                    break

        pbar.set_postfix({
            'Train': f'{epoch_train_loss:.4f}' if epoch_train_loss is not None else 'None',
            'Eval': f'{epoch_eval_loss:.4f}' if epoch_eval_loss is not None else 'None',
            'ES': f'{bad_epochs}/{early_stop_patience or "-"}'
        })

    return train_losses, eval_losses


def weighted_circuit_score_sampler(y_train, qs=(0.7, 0.9, 0.97), class_weights=(1.0, 2.0, 4.0, 8.0), seed=42):
    
    # y: 1D CPU tensor
    y = y_train.detach().float().flatten().to('cpu')

    # cut by quantiles
    q = th.quantile(y, th.tensor(qs))
    bins = th.bucketize(y, q)            # values in {0,1,2,3} for 3 cuts

    # class weights per bin (last bin = rarest/highest)
    cw = th.tensor(class_weights, dtype=th.float32)
    w = cw[bins]

    # normalize for stability
    w = w / w.mean()

    # deterministic generator for reproducibility
    gen = th.Generator(device='cpu')
    gen.manual_seed(seed)

    return WeightedRandomSampler(weights=w.to(th.double),
                                 num_samples=len(w),
                                 replacement=True,
                                 generator=gen)
    
def save_model(model, filepath):
    # If the model is wrapped in DataParallel, we need to save the underlying model's state_dict
    if isinstance(model, nn.DataParallel):
        th.save(model.module.state_dict(), filepath)
    else:
        th.save(model.state_dict(), filepath)
        
# Load the state dictionaries
def load_model(model, filepath):
    state_dict = th.load(filepath, map_location=th.device('cpu'))
    model.load_state_dict(state_dict)
    
def train_multitask_minibatch(
    model,
    X_train, y_circuit_train, y_toxicity_train,
    X_val, y_circuit_val, y_toxicity_val,
    num_epochs=10, batch_size=4096, lr=1e-3, alpha=None, beta=None,
    grad_clip=None, seed=42, device=None
):
    th.manual_seed(seed)
    device = device or th.device("cuda" if th.cuda.is_available() else "cpu")
    model.to(device)

    train_ds = TensorDataset(X_train, y_circuit_train, y_toxicity_train)
    val_ds   = TensorDataset(X_val,  y_circuit_val,  y_toxicity_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, drop_last=False)

    criterion = nn.MSELoss()
    optim_all = optim.Adam(model.parameters(), lr=lr)

    # If tasks are on different scales, balance by inverse variance (keeps your "no input scaling" preference)
    if alpha is None or beta is None:
        var_c = th.var(y_circuit_train, unbiased=False).item() + 1e-8
        var_t = th.var(y_toxicity_train, unbiased=False).item() + 1e-8
        alpha = 1.0 / var_c
        beta  = 1.0 / var_t

    history = {
        "train_total": [], "val_total": [],
        "train_circuit": [], "train_toxicity": [],
        "val_circuit": [], "val_toxicity": []
    }

    for epoch in range(num_epochs):
        # ---- Train ----
        model.train()
        tr_loss = tr_c = tr_t = 0.0
        for xb, ycb, ytb in train_loader:
            xb  = xb.to(device)
            ycb = ycb.to(device)
            ytb = ytb.to(device)

            yhat_c, yhat_t = model(xb)
            loss_c = criterion(yhat_c, ycb)
            loss_t = criterion(yhat_t, ytb)
            loss = alpha * loss_c + beta * loss_t

            optim_all.zero_grad()
            loss.backward()
            if grad_clip is not None:
                th.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optim_all.step()

            tr_loss += loss.item() * xb.size(0)
            tr_c    += loss_c.item() * xb.size(0)
            tr_t    += loss_t.item() * xb.size(0)

        ntr = len(train_ds)
        tr_loss /= ntr; tr_c /= ntr; tr_t /= ntr

        # ---- Validate ----
        model.eval()
        va_loss = va_c = va_t = 0.0
        with th.no_grad():
            for xb, ycb, ytb in val_loader:
                xb  = xb.to(device)
                ycb = ycb.to(device)
                ytb = ytb.to(device)
                yhat_c, yhat_t = model(xb)
                loss_c = criterion(yhat_c, ycb)
                loss_t = criterion(yhat_t, ytb)
                loss = alpha * loss_c + beta * loss_t

                va_loss += loss.item() * xb.size(0)
                va_c    += loss_c.item() * xb.size(0)
                va_t    += loss_t.item() * xb.size(0)

        nva = len(val_ds)
        va_loss /= nva; va_c /= nva; va_t /= nva

        history["train_total"].append(tr_loss)
        history["val_total"].append(va_loss)
        history["train_circuit"].append(tr_c)
        history["train_toxicity"].append(tr_t)
        history["val_circuit"].append(va_c)
        history["val_toxicity"].append(va_t)

        print(f"Epoch {epoch+1:02d} | "
              f"train: total={tr_loss:.5f}, circuit={tr_c:.5f}, tox={tr_t:.5f}  ||  "
              f"val: total={va_loss:.5f}, circuit={va_c:.5f}, tox={va_t:.5f}")

    return history

from typing import Optional
from dataclasses import dataclass

@dataclass
class TrainJob:
    name: str
    gpu_id: Optional[int]
    seed: int
    # model config
    input_size: int
    hidden_size: int
    output_size: int
    num_layers: int
    dropout: float
    lr: float
    # data & training
    X_train: th.Tensor
    y_train: th.Tensor
    X_val: th.Tensor
    y_val: th.Tensor
    num_epochs: int
    batch_size: int
    evaluate_validation_set: bool
    early_stop_patience: int
    early_stop_min_delta: float
    # final model output path
    out_path: str

def run_one_job(job: TrainJob, result_dict):
    # device selection 
    if job.gpu_id is not None and th.cuda.is_available():
        th.cuda.set_device(job.gpu_id)
        device = th.device(f"cuda:{job.gpu_id}")
    else:
        device = th.device("cpu")
    device_str = str(device)

    # seeding 
    try:
        import random, numpy as np
        random.seed(job.seed); np.random.seed(job.seed)
    except Exception:
        pass
    th.manual_seed(job.seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(job.seed)

    # model and optimizer 
    model = RegressionNN(job.input_size, job.hidden_size, job.output_size, num_layers=job.num_layers, dropout=job.dropout, activation=nn.ReLU).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=job.lr)

    # data (left on CPU; function moves per batch) ---
    Xtr, ytr = job.X_train, job.y_train
    Xva, yva = job.X_val,   job.y_val

    # train 
    train_losses, eval_losses = train_model_minibatch_early_stoping(
        model,
        Xtr, ytr, Xva, yva,
        optimizer, criterion,
        num_epochs=job.num_epochs,
        batch_size=job.batch_size,
        shuffle=True, sampler=None, seed=job.seed, grad_clip=None,
        evaluate_validation_set=job.evaluate_validation_set,                 
        early_stop_patience=job.early_stop_patience,
        early_stop_min_delta=job.early_stop_min_delta,
        restore_best_on_stop=False
    )

    # ensure output dir 
    out_path: Path = Path(job.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # save FINAL weights (CPU-portable) without moving the live model 
    cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    th.save(cpu_state, out_path)

    # save losses + metadata next to the model 
    losses_path: Path = out_path.with_name(out_path.stem + "_losses.pt")
    th.save({
        "name": job.name,
        "seed": job.seed,
        "num_epochs_requested": job.num_epochs,
        "train_losses": train_losses,
        "eval_losses": eval_losses,   # empty if evaluate_validation_set=False
        "config": {
            "input_size": job.input_size,
            "hidden_size": job.hidden_size,
            "output_size": job.output_size,
            "num_layers": job.num_layers,
            "dropout": job.dropout,
            "lr": job.lr,
            "batch_size": job.batch_size,
            "early_stop_patience": job.early_stop_patience,
            "early_stop_min_delta": job.early_stop_min_delta,
        }
    }, losses_path)

    # summarize metrics 
    def _last_float(lst):
        return float(lst[-1]) if isinstance(lst, (list, tuple)) and len(lst) > 0 else float("nan")

    final_train = _last_float(train_losses)
    final_eval  = _last_float(eval_losses)
    final_metric = final_eval if not (final_eval != final_eval) else final_train  # prefer val if available

    # return to parent 
    result_dict[job.name] = {
        "device": device_str,
        "final_metric": final_metric,
        "final_train": final_train,
        "final_eval": final_eval,
        "model_path": str(out_path),
        "losses_path": str(losses_path),
    }




class RegressionNN(nn.Module):
    """
    Fully-connected MLP with configurable depth and width.
    num_layers = number of HIDDEN layers (not counting the output layer).
    """
    def __init__(self, input_size, hidden_size, output_size=1,
                 num_layers=5, dropout=0.1, activation=nn.ReLU):
        super().__init__()
        assert num_layers >= 1, "num_layers must be >= 1"

        layers = []
        act = activation  # pass a class like nn.ReLU or nn.GELU

        # First hidden layer: input -> hidden
        layers += [nn.Linear(input_size, hidden_size), act()]
        if dropout and dropout > 0:
            layers += [nn.Dropout(dropout)]

        # Remaining hidden layers: hidden -> hidden
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), act()]
            if dropout and dropout > 0:
                layers += [nn.Dropout(dropout)]

        # Output layer: hidden -> output
        layers += [nn.Linear(hidden_size, output_size)]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # X is one-hot with shape [B, num_parts, num_classes]
        # flatten to [B, num_parts * num_classes]
        x = x.flatten(1)
        return self.model(x)
    
    
def load_losses(losses_path):
    """
    Returns (train_losses, eval_losses, meta_dict)
    """
    blob = th.load(losses_path, map_location="cpu")
    train_losses = blob.get("train_losses", [])
    eval_losses  = blob.get("eval_losses", [])
    meta = {k: v for k, v in blob.items() if k not in ("train_losses", "eval_losses")}
    return train_losses, eval_losses, meta


class Timer:
    def __init__(self):
        self._t0 = time.perf_counter()
        self.marks = [("start", self._t0)]

    def mark(self, label):
        now = time.perf_counter()
        self.marks.append((label, now))
        return now

    def report(self):
        out = []
        for (l0, t0), (l1, t1) in zip(self.marks, self.marks[1:]):
            out.append(f"{l0}→{l1}: {t1 - t0:.6f}s")
        out.append(f"total: {self.marks[-1][1] - self.marks[0][1]:.6f}s")
        return "\n".join(out)
    
    
## SA benchmarking code


def getRandomGateFromUnassignedGroup(current_solution, gate_library):
    # Define the groups of gates
    group_AmeR = [0]
    group_AmtR = [1]
    group_BetI = [2]
    group_BM3R1 = [3, 4, 5]
    group_HlyIIR = [6]
    group_IcaRA = [7]
    group_LitR = [8]
    group_LmrA = [9]
    group_PhlF = [10, 11, 12]
    group_PsrA = [13]
    group_QacR = [14, 15]
    group_SrpR = [16, 17, 18, 19]
    
    # Step 1: Create a list of all groups for easier processing
    all_groups = [
        group_AmeR, group_AmtR, group_BetI, group_BM3R1, 
        group_HlyIIR, group_IcaRA, group_LitR, group_LmrA,
        group_PhlF, group_PsrA, group_QacR, group_SrpR
    ]
    
    # Step 2: Rule out groups that have integers in current solution
    available_groups = []
    for group in all_groups:
        # Check if any number from this group is in current_solution
        group_is_available = True
        for gate in group:
            if gate in current_solution:
                group_is_available = False
                break
        # If no number from this group is in current_solution, add it to available_groups
        if group_is_available:
            available_groups.append(group)
    
    # Step 3: Get a random group from available groups
    if not available_groups:  # Check if there are any available groups
        return None  # Return None if no groups are available
    random_group = random.choice(available_groups)
    
    # Step 4: Get a random integer from the selected group
    random_gate = random.choice(random_group)
    
    # Step 5: Return the randomly selected gate
    return random_gate


def getRandomGateFromUnassignedGroup_with_max_inputs_restriction(current_solution, gate_library, G, gate_max_incoming_signals_df):
    # Define the groups of gates
    group_AmeR = [0]
    group_AmtR = [1]
    group_BetI = [2]
    group_BM3R1 = [3, 4, 5]
    group_HlyIIR = [6]
    group_IcaRA = [7]
    group_LitR = [8]
    group_LmrA = [9]
    group_PhlF = [10, 11, 12]
    group_PsrA = [13]
    group_QacR = [14, 15]
    group_SrpR = [16, 17, 18, 19]
    
    # Step 0: Remove gates that cannot be assigned based on incoming signals
    # Use the validate_max_incoming_signals function
    all_groups = [
        group_AmeR, group_AmtR, group_BetI, group_BM3R1, 
        group_HlyIIR, group_IcaRA, group_LitR, group_LmrA,
        group_PhlF, group_PsrA, group_QacR, group_SrpR
    ]
    
    filtered_groups = []
    for group in all_groups:
        filtered_group = []
        for gate in group:
            # Check if the gate can be assigned using validate_max_incoming_signals
            can_assign = validate_max_incoming_signals(
                graph=G,
                dataframe=gate_library,
                repressor_index=gate,
                gate_max_incoming_signals_df=gate_max_incoming_signals_df
            )
            if can_assign:
                filtered_group.append(gate)
        if filtered_group:
            filtered_groups.append(filtered_group)
    
    # If no gates are left after filtering, return None
    if not filtered_groups:
        return None
    
    # Step 1: Use the filtered groups for further processing
    all_groups = filtered_groups

    # Step 2: Rule out groups that have gates already in current_solution
    available_groups = []
    for group in all_groups:
        # Check if any gate from this group is in current_solution
        group_is_available = True
        for gate in group:
            if gate in current_solution:
                group_is_available = False
                break
        # If no gate from this group is in current_solution, add it to available_groups
        if group_is_available:
            available_groups.append(group)
    
    # Step 3: Get a random group from available groups
    if not available_groups:  # Check if there are any available groups
        return None  # Return None if no groups are available
    random_group = random.choice(available_groups)
    
    # Step 4: Get a random gate from the selected group
    random_gate = random.choice(random_group)
    
    # Step 5: Return the randomly selected gate
    return random_gate

def simulated_annealing_cello2(
    adj_matrix, initial_solution, MAXTEMP, MINTEMP, steps, T0_steps,
    input_signals_list_small_molecules, input_signals_list_binary,
    gate_library, gate_toxicity_df, D_GROWTH_THRESHOLD=0.75, print_statements = False, plot = False
):
    """
    This function implements the simulated annealing algorithm in Cello 2.

    Parameters:
    - df: DataFrame containing gate data.
    - adj_matrix: adjacency matrix of the circuit.
    - initial_solution: initial assignment of gates to nodes (list of indices in df).
    - MAXTEMP, MINTEMP: initial and final temperatures.
    - steps: number of steps before reaching Tmin.
    - T0_steps: number of steps at Tmin.
    - input_signals_list_small_molecules: list of input signals for simulation.
    - input_signals_list_binary: list of expected binary outputs.
    - gate_toxicity_df: DataFrame containing toxicity data.
    - D_GROWTH_THRESHOLD: growth threshold for toxicity evaluation.
    """

    # Initialize current solution
    G_initial = assign_representations_with_io_nodes_3(adj_matrix, gate_library, initial_solution)    
    expected_logic_initial = simulate_signal_propagation_binary(G_initial, input_signals_list_binary)
    actual_logic_initial = simulate_signal_propagation(G_initial, input_signals_list_small_molecules)
    circuit_score_initial = calculate_circuit_score(expected_logic_initial, actual_logic_initial)
    toxicity_score_initial, detailed_results_initial = calculate_toxicity_score(input_signals_list_small_molecules, G_initial, gate_toxicity_df)
    
    optimal_score = circuit_score_initial
    optimal_toxicty = toxicity_score_initial
    optimal_solution = initial_solution.copy()
    
    # Initialize current solution
    current_solution = initial_solution.copy()
    
    # Temperature parameters
    LOGMAX = math.log10(MAXTEMP)
    LOGMIN = math.log10(MINTEMP)
    LOGINC = (LOGMAX - LOGMIN) / steps

    total_steps = steps + T0_steps

  
    # Lists to track scores for visualization
    iteration_numbers = []
    optimal_scores = []


    for j in tqdm(range(total_steps), desc="Optimizing", leave=True):
        logTemperature = LOGMAX - j * LOGINC 
        temperature = math.pow(10, logTemperature)
        
        if j >= steps:
            temperature = 0.0
        
        
        print("temp",  temperature) if print_statements else None       
        print("log(temp)", logTemperature) if print_statements else None

        G_current = assign_representations_with_io_nodes_3(adj_matrix, gate_library, current_solution)
        expected_logic_current = simulate_signal_propagation_binary(G_current, input_signals_list_binary)
        actual_logic_current = simulate_signal_propagation(G_current, input_signals_list_small_molecules)
        circuit_score_current = calculate_circuit_score(expected_logic_current, actual_logic_current)
        toxicity_score_current, detailed_results = calculate_toxicity_score(input_signals_list_small_molecules, G_current, gate_toxicity_df)
        
        #rejectImmediately = False;
        #tandemSwap = False; not used    
        
        # Decide whether to perform tandem swap or gate swap
        r = random.random()
        
        #if (r < thresh):
        #    tandemSwap = True;
        
        print("Current solution: ", current_solution) if print_statements else None    
        
           
        # If gate <-> library swap
        gateA = getRandomGateFromUnassignedGroup(current_solution, gate_library);
        #gateA_types = getGateTypeinLibrary(gateA)
        if (gateA == None):
            gateA = random.choice(current_solution)
        
        while True:
            gateB = random.choice(current_solution);
            #if (gateA != gateB and gateA_type == gateBtype):
            if (gateA != gateB):
                break     
                
        #if not tandem swap
        #if !tandemSwap: Not used
        roadblock_flags_before = is_roadblocking(G_current)
        numBlockedBefore = len(roadblock_flags_before[1])

        if gateA in current_solution and gateB in current_solution:
            new_solution = swap_within_circuit(current_solution, gateA, gateB)
        else:
            circuit_gate = gateB
            library_gate = gateA
            new_solution = swap_with_library(current_solution, circuit_gate, library_gate)

        G_new = assign_representations_with_io_nodes_3(adj_matrix, gate_library, new_solution)
        roadblock_flags_after = is_roadblocking(G_new)
        numBlockedAfter = len(roadblock_flags_after[1])

        if (numBlockedAfter > numBlockedBefore):
            
            # roadblocking is worse, so reject
            iteration_numbers.append(j)
            optimal_scores.append(optimal_score)
            continue #go to the next iteration
          
        # evaluate
        actual_logic_new = simulate_signal_propagation(G_new, input_signals_list_small_molecules)
        expected_logic_new = simulate_signal_propagation_binary(G_new, input_signals_list_binary)
        circuit_score_new = calculate_circuit_score(expected_logic_new, actual_logic_new)
        toxicity_score_new, detailed_results_new = calculate_toxicity_score(input_signals_list_small_molecules, G_new, gate_toxicity_df)

        if (toxicity_score_current < D_GROWTH_THRESHOLD):
            if (toxicity_score_new > toxicity_score_current):
                print("Accept immediately -- already below mimimum growth threshold, and this swap helps.") if print_statements else None
                
                current_solution = new_solution.copy()
                
                if circuit_score_new > optimal_score and (toxicity_score_new >= D_GROWTH_THRESHOLD):
                    optimal_score = circuit_score_new
                    optimal_toxicty = toxicity_score_new
                    optimal_solution = new_solution.copy()
                
                iteration_numbers.append(j)
                optimal_scores.append(optimal_score)
                
                continue
            else:
                # undo
                #rejectImmediately = True
                print("Reject immediately -- already below mimimum growth threshold, and this swap does not help.") if print_statements else None
                
                iteration_numbers.append(j)
                optimal_scores.append(optimal_score)
                
                continue

        elif (toxicity_score_new < D_GROWTH_THRESHOLD):
        # // undo
            #rejectImmediately = True;
            print("Reject immediately -- below minimum growth threshold.") if print_statements else None
            
            iteration_numbers.append(j)
            optimal_scores.append(optimal_score)
            
            continue

        # Not used    
        #undo
        #if (rejectImmediately):
        #    #Undo solution
        #    pass
        
        #acept or reject 
        before = circuit_score_current
        after = circuit_score_new
        #probability = math.exp((after - before) / temperature)
        #probability = safe_exp((after - before) / temperature)
        
        if temperature == 0:
            # Only accept improvements when temperature is 0
            probability = 1.0 if after > before else 0.0
        else:
            probability = safe_exp((after - before) / temperature)          
        
        ep = random.random()  # Returns a float between 0.0 and 1.0

        if (ep < probability):
            #accept
            print("Accept") if print_statements else None
            current_solution = new_solution.copy()
            
            if circuit_score_new > optimal_score and (toxicity_score_new >= D_GROWTH_THRESHOLD):
                optimal_score = circuit_score_new
                optimal_toxicty = toxicity_score_new
                optimal_solution = new_solution.copy()
            
        else:
            #reject swap
            pass
        
        iteration_numbers.append(j)
        optimal_scores.append(optimal_score)

    if (plot):    
        # Plotting iterations vs highest score
        plt.figure(figsize=(8, 4))
        plt.plot(iteration_numbers, optimal_scores, marker='o', markersize=3, linestyle='--', color='b')
        plt.title('Iterations vs Highest Score')
        plt.xlabel('Iteration')
        plt.ylabel('Circuit Score')
        plt.show()

    # Optionally, draw the final network
    #draw_network_with_colors_and_labels_from_G(G)

    return optimal_score, optimal_toxicty, optimal_solution

def safe_exp(x):
    try:
        return math.exp(x)
    except OverflowError:
        if x > 0:
            return float('inf')
        return 0