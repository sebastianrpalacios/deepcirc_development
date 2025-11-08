#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Runs three evaluation tracks (trained, untrained-same-arch, random) with an
ActionMasker-wrapped DRL3env and a shared registry.

CHANGELOG (relative to your original script):
- best_energy_in_episode in the CSV is now read directly from the env's
  per-episode tracker (`_best_energy_in_episode`), i.e., independent of what
  was (or wasn't) stored in the registry in that episode.
- During policy rollout, we attempt to pass `action_masks` to MaskablePPO.predict()
  (with a safe fallback for sb3-contrib versions that don't accept it).
"""

import json
import argparse
import csv
import pickle
import multiprocessing
from pathlib import Path

import numpy as np
import torch as th
import networkx as nx

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

# env + utils from your repo
from dgd.environments.drl3env_loader6_with_trajectories import (
    DRL3env,
    _compute_truth_key,
    _compute_hash,
    _apply_implicit_or,
)
from dgd.utils.utils5 import load_graph_pickle, energy_score, check_implicit_OR_existence_v3


def save_episode_trajectory(env, out_dir: Path, episode: int, track_name: str):
    """Dump the current episode's trajectory from the underlying env to JSON."""
    get_traj = None
    try:
        get_traj = env.get_wrapper_attr("get_trajectory")
    except Exception:
        pass
    if get_traj is None:
        # Not wrapped, maybe it's the base env
        get_traj = getattr(env, "get_trajectory", None)

    if get_traj is None:
        return  # no-op if env doesn't support it

    steps = get_traj()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ep_{episode:04d}.json"
    payload = {"episode": int(episode), "track": track_name, "steps": steps}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def mask_fn(env):
    return env.action_masks()


def _to_int_action(action):
    if isinstance(action, np.ndarray):
        if action.shape == ():
            return int(action.item())
        return int(action.reshape(-1)[0])
    return int(action)


def _to_graph(x):
    """Return a NetworkX graph whether x is already a graph or a node_link JSON dict."""
    if isinstance(x, (nx.Graph, nx.DiGraph)):
        return x
    if isinstance(x, dict):
        return nx.node_link_graph(x)
    try:
        return nx.node_link_graph(x)
    except Exception:
        raise TypeError(f"Cannot convert object of type {type(x)} to NetworkX graph")


def _get_best_energy_in_episode(env):
    """
    Fetch the env's per-episode minimum energy (tracked internally by DRL3env)
    after an episode finishes.
    """
    try:
        val = env.get_wrapper_attr("_best_energy_in_episode")
        if val is not None:
            return float(val)
    except Exception:
        pass
    try:
        return float(env.unwrapped._best_energy_in_episode)
    except Exception:
        return None

def save_run_meta(out_dir=None, filename="run_metadata_simple.json"):
    """
    Writes a tiny JSON receipt with (1) UTC timestamp, (2) ALL CLI flags,
    and (3) imported top-level modules (+version when available).

    Usage:
        save_run_meta(out_dir=getattr(args, "output_folder_name", "."))
        # or just: save_run_meta()  # auto-uses --output_folder_name if present
    """
    import sys, json
    from datetime import datetime, timezone
    from pathlib import Path

    # ---- timestamp
    ts = datetime.now(timezone.utc).isoformat()

    # ---- capture CLI (raw + schema-free parse)
    argv = sys.argv[1:]
    flags, positionals = {}, []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok.startswith("--"):
            key, val = tok[2:], True
            if "=" in key:
                key, val = key.split("=", 1)
            elif i + 1 < len(argv) and not argv[i + 1].startswith("-"):
                val = argv[i + 1]; i += 1
            flags.setdefault(key, []).append(val)
        elif tok.startswith("-") and len(tok) > 1:
            rest = tok[1:]
            if "=" in rest:
                k, val = rest.split("=", 1); flags.setdefault(k, []).append(val)
            elif len(rest) == 1 and i + 1 < len(argv) and not argv[i + 1].startswith("-"):
                flags.setdefault(rest, []).append(argv[i + 1]); i += 1
            else:
                for ch in rest: flags.setdefault(ch, []).append(True)
        else:
            positionals.append(tok)
        i += 1

    # ---- imports snapshot (top-level names, with version if available)
    imps, seen = [], set()
    for name, mod in list(sys.modules.items()):
        if not mod or getattr(mod, "__file__", None) is None:  # skip builtins/frozen
            continue
        top = name.split(".", 1)[0]
        if top in seen: 
            continue
        seen.add(top)
        ver = getattr(mod, "__version__", None)
        if ver is None:
            base = sys.modules.get(top)
            ver = getattr(base, "__version__", None) if base else None
        imps.append({"name": top, **({"version": str(ver)} if ver else {})})
    imps.sort(key=lambda x: x["name"])

    # ---- choose output dir
    if out_dir is None:
        out_dir = (flags.get("output_folder_name", ["."])[-1]) if "output_folder_name" in flags else "."
    out_path = Path(out_dir); out_path.mkdir(parents=True, exist_ok=True)
    out_file = out_path / filename

    # ---- write
    payload = {
        "timestamp_iso_utc": ts,
        "cli": {"raw": argv, "flags": flags, "positionals": positionals},
        "imports": imps,
    }
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[meta] wrote {out_file}")
    return str(out_file)

# ---------------- Registry helpers ----------------

def init_shared_registry(manager, seed_graphs, lock):
    """
    Create a Manager dict and pre-seed it with the initial graphs.
    LIVE registry stores GRAPHS (not JSON), as tuples: (canon_graph, orig_graph, energy_float).
    """
    reg = manager.dict()
    with lock:
        for G in seed_graphs:
            canon = _apply_implicit_or(G.copy())
            e, _ = energy_score(G, check_implicit_OR_existence_v3)
            key = _compute_hash(canon)
            bucket = reg.get(key, [])
            bucket.append((canon, G.copy(), float(e)))
            reg[key] = bucket
    return reg


def save_registry_pickle(registry, lock, path: Path):
    """Write the shared registry to a pickle safely: convert graphs → node_link JSON."""
    plain = {}
    with lock:
        for k, bucket in registry.items():
            json_bucket = []
            for canon_g, orig_g, e in bucket:
                canon_g = _to_graph(canon_g)
                orig_g = _to_graph(orig_g)
                json_bucket.append((nx.node_link_data(canon_g), nx.node_link_data(orig_g), float(e)))
            plain[k] = json_bucket
    with path.open("wb") as f:
        pickle.dump(plain, f)


def save_registry_summary_csv(registry, lock, path: Path):
    """
    Write a CSV summary with one row per registry entry (NOT per-episode).
    Columns: hash, energy, size (nodes) of ORIGINAL graph.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["hash", "energy", "size"])
        with lock:
            for h, bucket in registry.items():
                for _, orig_item, e in bucket:
                    G = _to_graph(orig_item)
                    w.writerow([h, float(e), G.number_of_nodes()])


def _snapshot_registry_counts(registry, lock):
    """Map hash -> count of items currently in that bucket."""
    with lock:
        return {h: len(bucket) for h, bucket in registry.items()}


def _best_new_energy_since(registry, lock, snapshot_counts):
    """
    Among entries added to the registry after the snapshot, return:
    (min_new_energy, new_items_count). If nothing new, returns (None, 0).
    """
    min_e = None
    new_items = 0
    with lock:
        for h, bucket in registry.items():
            start = snapshot_counts.get(h, 0)
            for _, _, e in bucket[start:]:
                new_items += 1
                min_e = e if (min_e is None or e < min_e) else min_e
    return min_e, new_items


def _update_running_best(prev_best, best_in_ep):
    """Fast running min that ignores None values."""
    if best_in_ep is None:
        return prev_best
    return best_in_ep if (prev_best is None or best_in_ep < prev_best) else prev_best

def _global_best_energy(registry, lock):
    """Minimum energy seen anywhere in the registry at call time."""
    best = None
    with lock:
        for _, bucket in registry.items():
            for _, _, e in bucket:
                best = e if (best is None or e < best) else best
    return best


def _append_episode_metrics_row(path: Path, episode: int, best_in_ep, best_so_far, new_graphs: int):
    """
    Append one row to episode_metrics.csv with:
    episode, best_energy_in_episode, best_energy_so_far, num_new_graphs
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["episode", "best_energy_in_episode", "best_energy_so_far", "num_new_graphs"])
        w.writerow([
            episode,
            "" if best_in_ep is None else float(best_in_ep),
            "" if best_so_far is None else float(best_so_far),
            "" if new_graphs is None else int(new_graphs),
        ])


# ---------------- Rollout loops ----------------

def run_episode_with_policy(model, env, deterministic, ep_seed=None):
    # reproducible per-episode reset
    try:
        obs, _ = env.reset(seed=ep_seed)
    except TypeError:
        obs, _ = env.reset()

    done = False
    while not done:
        # use action mask if available; safe fallback for older sb3-contrib
        mask = None
        try:
            mask = env.get_wrapper_attr("action_masks")()
        except Exception:
            pass
        if mask is not None:
            try:
                action, _ = model.predict(obs, deterministic=deterministic, action_masks=mask)
            except TypeError:
                action, _ = model.predict(obs, deterministic=deterministic)
        else:
            action, _ = model.predict(obs, deterministic=deterministic)

        action = _to_int_action(action)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated


def run_episode_random_masked(env, ep_seed=None):
    try:
        obs, _ = env.reset(seed=ep_seed)
    except TypeError:
        obs, _ = env.reset()

    rng = np.random.default_rng(ep_seed)
    done = False
    while not done:
        mask = env.get_wrapper_attr("action_masks")()  # boolean mask
        valid = np.flatnonzero(mask)
        action = int(rng.choice(valid))
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated


# ---------------- Env factory ----------------

def build_env_with_registry(seed_graphs, max_nodes, max_steps, existing_keys,
                            registry, registry_lock, best_energy_across_workers,
                            registry_sampling=True, initial_state_sampling_factor=0):
    base_env = DRL3env(
        max_nodes=max_nodes,
        graphs=seed_graphs,
        shared_registry=registry,
        registry_lock=registry_lock,
        store_every_new_graph=True,
        sampling_from_shared_registry=registry_sampling,
        registry_read_only=False,
        max_steps=max_steps,
        enable_full_graph_replacement=True,
        show_plots=False,
        log_info=False,
        strict_iso_check=False,
        initial_state_sampling_factor=initial_state_sampling_factor,
        existing_keys=existing_keys,
        best_energy_across_workers=best_energy_across_workers,  # required when using a shared registry
    )
    
    try:
        base_env.start_trajectory_logging(True)
    except Exception:
        pass
    return ActionMasker(base_env, mask_fn)


# ---------------- Main ----------------

def main():
    p = argparse.ArgumentParser(
        "Trained + Random(masked) + Untrained(masked) using a shared registry (single worker). "
        "Also logs per-episode best energy (from env) and running best to CSV."
    )
    p.add_argument("--model_path", required=True, help="Path to trained_model.zip")
    p.add_argument("--seed_files", nargs="+", required=True, help="One or more seed .pkl graph files")
    p.add_argument("--output_folder_name", required=True, help="Base output folder")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--max_nodes", type=int, default=100)
    p.add_argument("--max_steps", type=int, default=10)
    p.add_argument('--initial_state_sampling_factor', type=float, default=0,
                   help='Factor for implementing weighted sampling of initial states')
    # default=False => omit -> stochastic; add -> greedy (applies to trained & untrained)
    p.add_argument("--deterministic", action="store_true", default=False)
    p.add_argument("--seed", type=int, default=123, help="Base RNG seed for reproducibility")
    args = p.parse_args()

    base_out = Path(args.output_folder_name)
    (base_out / "trained_masked").mkdir(parents=True, exist_ok=True)
    (base_out / "trained_masked" / "trajectories").mkdir(parents=True, exist_ok=True)
    (base_out / "untrained_masked").mkdir(parents=True, exist_ok=True)
    (base_out / "untrained_masked" / "trajectories").mkdir(parents=True, exist_ok=True)
    (base_out / "random_masked").mkdir(parents=True, exist_ok=True)
    (base_out / "random_masked" / "trajectories").mkdir(parents=True, exist_ok=True)

    # global seeding
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    th.manual_seed(args.seed)

    # load seeds
    seed_paths = [str(Path(s)) for s in args.seed_files]
    print(f"[INFO] Using {len(seed_paths)} seed file(s):")
    for fp in seed_paths:
        print(" -", fp)
    G_initial_states = [load_graph_pickle(fp) for fp in seed_paths]
    existing_keys = {_compute_truth_key(g) for g in G_initial_states}
    print(f"[INFO] existing_keys initialized with {len(existing_keys)} key(s)")

    device = "cuda" if th.cuda.is_available() else "cpu"

    # ========== TRACK 1: TRAINED (masked, registry) ==========
    out_dir1 = base_out / "trained_masked"

    ( out_dir1 / "trajectories" ).mkdir(parents=True, exist_ok=True)

    mgr1 = multiprocessing.Manager()
    lock1 = mgr1.Lock()
    best1 = mgr1.Value('d', float('inf'))  # shared "best energy" (used by env)
    registry1 = init_shared_registry(mgr1, G_initial_states, lock1)  # pre-seed with initial graphs

    env1 = build_env_with_registry(
        G_initial_states, args.max_nodes, args.max_steps,
        existing_keys, registry1, lock1, best1,
        registry_sampling=True, initial_state_sampling_factor=args.initial_state_sampling_factor
    )
    trained = MaskablePPO.load(args.model_path, env=env1, device=device)

    metrics_csv1 = out_dir1 / "trained_episode_metrics.csv"
    running_best1 = None
    for ep in range(args.episodes):
        #snap = _snapshot_registry_counts(registry1, lock1)

        run_episode_with_policy(trained, env1, args.deterministic, ep_seed=args.seed + ep)
        save_episode_trajectory(env1, (base_out / "trained_masked" / "trajectories"), ep, "trained_masked")

        # independent of registry: read env's best energy for this episode
        best_in_ep = _get_best_energy_in_episode(env1)

        # still collect "new graphs added" for this episode
        #_, new_graphs = _best_new_energy_since(registry1, lock1, snap)
        new_graphs = None

        # keep your previous definition of "best_so_far" as the registry-wide best
        #best_so_far = _global_best_energy(registry1, lock1)
        running_best1 = _update_running_best(running_best1, best_in_ep)
        best_so_far = running_best1

        _append_episode_metrics_row(metrics_csv1, ep, best_in_ep, best_so_far, new_graphs)

    save_registry_pickle(registry1, lock1, out_dir1 / "trained_final_shared_registry.pkl")
    save_registry_summary_csv(registry1, lock1, out_dir1 / "trained_registry_summary.csv")
    print(f"[INFO] Trained(masked) registry + metrics saved to {out_dir1}")

    '''
    # ========== TRACK 2: UNTRAINED (masked, registry) ==========
    out_dir2 = base_out / "untrained_masked"

    ( out_dir2 / "trajectories" ).mkdir(parents=True, exist_ok=True)

    mgr2 = multiprocessing.Manager()
    lock2 = mgr2.Lock()
    best2 = mgr2.Value('d', float('inf'))
    registry2 = init_shared_registry(mgr2, G_initial_states, lock2)

    env2 = build_env_with_registry(
        G_initial_states, args.max_nodes, args.max_steps,
        existing_keys, registry2, lock2, best2,
        registry_sampling=True, initial_state_sampling_factor=args.initial_state_sampling_factor
    )

    # Build same-arch random-init policy
    tmp = MaskablePPO.load(args.model_path, env=env2, device=device)
    policy_class = tmp.policy.__class__
    policy_kwargs = getattr(tmp, "policy_kwargs", None)
    untrained = MaskablePPO(
        policy=policy_class,
        env=env2,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4, n_steps=16, batch_size=16, n_epochs=1,
        device=device, verbose=0,
    )
    del tmp

    metrics_csv2 = out_dir2 / "untrained_episode_metrics.csv"
    for ep in range(args.episodes):
        snap = _snapshot_registry_counts(registry2, lock2)

        run_episode_with_policy(untrained, env2, args.deterministic, ep_seed=args.seed + ep)
        save_episode_trajectory(env2, (base_out / "untrained_masked" / "trajectories"), ep, "untrained_masked")

        best_in_ep = _get_best_energy_in_episode(env2)
        _, new_graphs = _best_new_energy_since(registry2, lock2, snap)
        best_so_far = _global_best_energy(registry2, lock2)

        _append_episode_metrics_row(metrics_csv2, ep, best_in_ep, best_so_far, new_graphs)

    save_registry_pickle(registry2, lock2, out_dir2 / "untrained_final_shared_registry.pkl")
    save_registry_summary_csv(registry2, lock2, out_dir2 / "untrained_registry_summary.csv")
    print(f"[INFO] Untrained(masked) registry + metrics saved to {out_dir2}")
    '''
    # ========== TRACK 3: RANDOM (masked, registry) ==========
    out_dir3 = base_out / "random_masked"

    ( out_dir3 / "trajectories" ).mkdir(parents=True, exist_ok=True)

    mgr3 = multiprocessing.Manager()
    lock3 = mgr3.Lock()
    best3 = mgr3.Value('d', float('inf'))
    registry3 = init_shared_registry(mgr3, G_initial_states, lock3)

    env3 = build_env_with_registry(
        G_initial_states, args.max_nodes, args.max_steps,
        existing_keys, registry3, lock3, best3,
        registry_sampling=True, initial_state_sampling_factor=args.initial_state_sampling_factor
    )

    metrics_csv3 = out_dir3 / "random_episode_metrics.csv"
    running_best3 = None
    for ep in range(args.episodes):
        #snap = _snapshot_registry_counts(registry3, lock3)

        run_episode_random_masked(env3, ep_seed=args.seed + ep)
        save_episode_trajectory(env3, (base_out / "random_masked" / "trajectories"), ep, "random_masked")

        best_in_ep = _get_best_energy_in_episode(env3)
        #_, new_graphs = _best_new_energy_since(registry3, lock3, snap)
        new_graphs = None
        
        #best_so_far = _global_best_energy(registry3, lock3)
        running_best3 = _update_running_best(running_best3, best_in_ep)
        best_so_far = running_best3        

        _append_episode_metrics_row(metrics_csv3, ep, best_in_ep, best_so_far, new_graphs)

    save_registry_pickle(registry3, lock3, out_dir3 / "random_final_shared_registry.pkl")
    save_registry_summary_csv(registry3, lock3, out_dir3 / "random_registry_summary.csv")
    print(f"[INFO] Random(masked) registry + metrics saved to {out_dir3}")

    print(f"[DONE] All registries & per-episode metrics saved under: {base_out}")
    
    save_run_meta(out_dir=base_out, filename="run_metadata.json")
    

if __name__ == "__main__":
    main()
