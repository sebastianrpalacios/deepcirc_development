# vector_eval.py
import json
import argparse
import csv
import pickle
import multiprocessing
from pathlib import Path

import numpy as np
import torch as th
import networkx as nx
import random
import os
import time
from typing import Optional

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

# env + utils from your repo
from dgd.environments.drl3env_loader6_with_trajectories import (
    DRL3env,
    _compute_truth_key,
    _compute_hash,
    _apply_implicit_or,
)
from dgd.utils.utils5 import load_graph_pickle, energy_score, check_implicit_OR_existence_v3


# -------------------------
# I/O helpers
# -------------------------
def save_episode_trajectory_from_vec(vec_env, env_idx: int, out_dir: Path, episode: int, track_name: str):
    """Dump the finished sub-env's trajectory to JSON (works with VecEnv via env_method)."""
    steps_list = vec_env.env_method("get_trajectory", indices=[env_idx])
    steps = steps_list[0] if steps_list else None
    if steps is None:
        return
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ep_{episode:06d}.json"  # global episode counter
    payload = {"episode": int(episode), "track": track_name, "env_index": int(env_idx), "steps": steps}
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
    if isinstance(x, (nx.Graph, nx.DiGraph)):
        return x
    if isinstance(x, dict):
        return nx.node_link_graph(x)
    try:
        return nx.node_link_graph(x)
    except Exception:
        raise TypeError(f"Cannot convert object of type {type(x)} to NetworkX graph")

def _get_best_energy_in_episode_from_vec(vec_env, env_idx: int) -> Optional[float]:
    """Fetch per-episode minimum energy tracked inside the given sub-env."""
    # try via get_wrapper_attr (works if wrapped); else try unwrapped
    try:
        val = vec_env.env_method("get_wrapper_attr", "_best_energy_in_episode", indices=[env_idx])[0]
        if val is not None:
            return float(val)
    except Exception:
        pass
    try:
        val = vec_env.env_method(lambda e: getattr(e.unwrapped, "_best_energy_in_episode", None), indices=[env_idx])[0]
        return None if val is None else float(val)
    except Exception:
        return None

def save_run_meta(out_dir=None, filename="run_metadata_simple.json", elapsed_seconds: Optional[float] = None):
    import sys
    from datetime import datetime, timezone, timedelta
    ts = datetime.now(timezone.utc).isoformat()
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

    imps, seen = [], set()
    for name, mod in list(sys.modules.items()):
        if not mod or getattr(mod, "__file__", None) is None:
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

    if out_dir is None:
        out_dir = (flags.get("output_folder_name", ["."])[-1]) if "output_folder_name" in flags else "."
    out_path = Path(out_dir); out_path.mkdir(parents=True, exist_ok=True)

    payload = {
        "timestamp_iso_utc": ts,
        "cli": {"raw": argv, "flags": flags, "positionals": positionals},
        "imports": imps,
    }
    # Only record track 1 runtime if provided
    if elapsed_seconds is not None:
        payload["track1_runtime_seconds"] = round(float(elapsed_seconds), 3)
        payload["track1_runtime_hms"] = str(timedelta(seconds=int(round(elapsed_seconds))))

    with (out_path / filename).open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[meta] wrote {(out_path / filename)}")
    return str(out_path / filename)


# -------------------------
# Registry helpers
# -------------------------
def init_shared_registry(manager, seed_graphs, lock):
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["hash", "energy", "size"])
        with lock:
            for h, bucket in registry.items():
                for _, orig_item, e in bucket:
                    G = _to_graph(orig_item)
                    w.writerow([h, float(e), G.number_of_nodes()])

# (Unused now in the rollout loop; kept for optional debug/compat)
def _snapshot_registry_counts(registry, lock):
    with lock:
        return {h: len(bucket) for h, bucket in registry.items()}

def _best_new_energy_since(registry, lock, snapshot_counts):
    min_e = None
    new_items = 0
    with lock:
        for h, bucket in registry.items():
            start = snapshot_counts.get(h, 0)
            for _, _, e in bucket[start:]:
                new_items += 1
                min_e = e if (min_e is None or e < min_e) else min_e
    return min_e, new_items

def _global_best_energy(registry, lock):
    best = None
    with lock:
        for _, bucket in registry.items():
            for _, _, e in bucket:
                best = e if (best is None or e < best) else best
    return best

def _append_episode_metrics_row(path: Path, episode: int, best_in_ep, best_so_far, new_graphs: Optional[int]):
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

def _read_global_best(best_scalar) -> Optional[float]:
    """
    Fast path: read global best from a multiprocessing.Value('d', ...).
    Returns None if Value is missing or still at +inf.
    """
    if best_scalar is None:
        return None
    try:
        v = float(best_scalar.value)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return None


# -------------------------
# VecEnv builders
# -------------------------
def make_env_thunk(seed_graphs, max_nodes, max_steps, existing_keys,
                   registry, registry_lock, best_energy_value,
                   registry_sampling=True, initial_state_sampling_factor=0.0,
                   masked=True, start_trajectory=True, env_seed: Optional[int]=None):
    """
    Returns a function that builds ONE environment (for SubprocVecEnv).
    """
    def _thunk():
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
            best_energy_across_workers=best_energy_value,
        )
        if start_trajectory:
            try:
                base_env.start_trajectory_logging(True)
            except Exception:
                pass
        env = base_env if not masked else ActionMasker(base_env, mask_fn)
        if env_seed is not None:
            try:
                env.reset(seed=env_seed)
            except TypeError:
                pass
        return env
    return _thunk

def build_vec_env(n_envs: int, *thunk_args, **thunk_kwargs):
    thunks = []
    # pop once (or just get without popping)
    env_seed_base = thunk_kwargs.pop("env_seed_base", None)
    for i in range(n_envs):
        seed_i = int(env_seed_base) + i if env_seed_base is not None else None
        thunks.append(make_env_thunk(*thunk_args, env_seed=seed_i, **thunk_kwargs))
    return SubprocVecEnv(thunks) if n_envs > 1 else DummyVecEnv(thunks)


# -------------------------
# Vec helpers for rollouts
# -------------------------
def get_vec_action_masks(vec_env) -> np.ndarray:
    """Return (n_env, action_dim) boolean mask array from sub-envs."""
    masks_list = vec_env.env_method("action_masks")
    masks = [np.asarray(m, dtype=bool) for m in masks_list]
    return np.stack(masks, axis=0)

def run_parallel_episodes_with_policy(model, vec_env, deterministic: bool, total_episodes: int,
                                      out_dir: Path, track_name: str,
                                      registry, registry_lock, metrics_csv: Path,
                                      best_scalar=None, save_trajectories: bool = True):
    """
    Drives a VecEnv until 'total_episodes' terminations have been observed (across all workers).
    Saves trajectories and per-episode metrics on each 'done'.

    NOTE: Avoids registry scans in the loop. best_so_far is read from shared scalar.
    """
    n_envs = vec_env.num_envs
    global_ep = 0
    obs = vec_env.reset()

    while global_ep < total_episodes:
        # Mask-aware action selection
        try:
            masks = get_vec_action_masks(vec_env)
            actions, _ = model.predict(obs, deterministic=deterministic, action_masks=masks)
        except Exception:
            actions, _ = model.predict(obs, deterministic=deterministic)

        # Ensure int actions per env
        if isinstance(actions, np.ndarray):
            actions = actions.reshape(-1)
            actions = np.array([_to_int_action(a) for a in actions], dtype=int)
        else:
            actions = np.array([_to_int_action(actions)] * n_envs, dtype=int)

        obs, rewards, dones, infos = vec_env.step(actions)

        # Handle episode ends per sub-env
        done_indices = np.flatnonzero(dones).tolist()
        for idx in done_indices:
            # Save trajectory if enabled
            if save_trajectories:
                save_episode_trajectory_from_vec(vec_env, idx, out_dir / "trajectories", global_ep, track_name)

            # Fast episode metrics
            best_in_ep = _get_best_energy_in_episode_from_vec(vec_env, idx)
            best_so_far = _read_global_best(best_scalar)
            new_graphs = None  # intentionally blank to avoid registry scans

            _append_episode_metrics_row(metrics_csv, global_ep, best_in_ep, best_so_far, new_graphs)

            global_ep += 1
            if global_ep >= total_episodes:
                break  # allow clean exit even if other envs also finished this step

def run_parallel_episodes_random_masked(vec_env, total_episodes: int,
                                        out_dir: Path, track_name: str,
                                        registry, registry_lock, metrics_csv: Path,
                                        rng: np.random.Generator,
                                        best_scalar=None, save_trajectories: bool = True):
    """
    Random masked policy runner without registry reads inside the loop.
    """
    n_envs = vec_env.num_envs
    global_ep = 0
    obs = vec_env.reset()

    while global_ep < total_episodes:
        masks = get_vec_action_masks(vec_env)  # (n_env, action_dim)
        actions = []
        for i in range(n_envs):
            valid = np.flatnonzero(masks[i])
            a = int(rng.choice(valid))
            actions.append(a)
        actions = np.asarray(actions, dtype=int)
        obs, rewards, dones, infos = vec_env.step(actions)

        done_indices = np.flatnonzero(dones).tolist()
        for idx in done_indices:
            if save_trajectories:
                save_episode_trajectory_from_vec(vec_env, idx, out_dir / "trajectories", global_ep, track_name)

            best_in_ep = _get_best_energy_in_episode_from_vec(vec_env, idx)
            best_so_far = _read_global_best(best_scalar)
            new_graphs = None  # intentionally blank

            _append_episode_metrics_row(metrics_csv, global_ep, best_in_ep, best_so_far, new_graphs)

            global_ep += 1
            if global_ep >= total_episodes:
                break


# -------------------------
# Main
# -------------------------
def main():
    p = argparse.ArgumentParser("Vectorized evaluation with SubprocVecEnv across multiple tracks.")
    p.add_argument("--model_path", required=True, help="Path to trained_model.zip")
    p.add_argument("--seed_files", nargs="+", required=True, help="One or more seed .pkl graph files")
    p.add_argument("--output_folder_name", required=True, help="Base output folder")
    p.add_argument("--episodes", type=int, default=100, help="Total episodes to collect (across all envs)")
    p.add_argument("--n_envs", type=int, default=8, help="Number of parallel env workers")
    p.add_argument("--max_nodes", type=int, default=100)
    p.add_argument("--max_steps", type=int, default=10)
    p.add_argument('--initial_state_sampling_factor', type=float, default=0.0,
                   help='Weighted sampling of initial states (used by default trained tracks)')
    p.add_argument("--deterministic", action="store_true", default=False)
    p.add_argument("--seed", type=int, default=None, help="Base RNG seed for reproducibility")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument(
        "--no_trajectories",
        action="store_true",
        help="Disable saving episode trajectories and env-side trajectory collection"
    )
    p.add_argument(
        "--no_ablation",
        action="store_true",
        help="Run only TRACK 1 (trained, masked, shared registry)"
    )
    args = p.parse_args()

    # Toggle for trajectories (saving + env-side logging)
    save_trajectories = not args.no_trajectories

    base_out = Path(args.output_folder_name)

    # Global seeding
    if args.seed is None:
        args.seed = int.from_bytes(os.urandom(4), "little")
        print(f"Generated random seed {args.seed}")
    else:
        print(f"Using provided seed {args.seed}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # Load seed graphs
    seed_paths = [str(Path(s)) for s in args.seed_files]
    print(f"[INFO] Using {len(seed_paths)} seed file(s):")
    for fp in seed_paths:
        print(" -", fp)
    G_initial_states = [load_graph_pickle(fp) for fp in seed_paths]
    existing_keys = {_compute_truth_key(g) for g in G_initial_states}
    print(f"[INFO] existing_keys initialized with {len(existing_keys)} key(s)")

    # Device
    if args.device == "auto":
        device = "cuda" if th.cuda.is_available() else "cpu"
    else:
        device = args.device

    # -------------------------------------------------------
    # TRACK 1: TRAINED (masked, shared registry)
    # -------------------------------------------------------
    

    out_dir1 = base_out / "trained_masked"
    out_dir1.mkdir(parents=True, exist_ok=True)
    if save_trajectories:
        (out_dir1 / "trajectories").mkdir(parents=True, exist_ok=True)

    mgr1 = multiprocessing.Manager()
    lock1 = mgr1.Lock()
    best1 = mgr1.Value('d', float('inf'))
    registry1 = init_shared_registry(mgr1, G_initial_states, lock1)

    env1 = build_vec_env(
        args.n_envs,
        G_initial_states, args.max_nodes, args.max_steps, existing_keys,
        registry1, lock1, best1,
        registry_sampling=True,
        initial_state_sampling_factor=args.initial_state_sampling_factor,
        masked=True,
        start_trajectory=save_trajectories,
        env_seed_base=args.seed  # distinct per sub-env
    )
    trained_1 = MaskablePPO.load(args.model_path, env=env1, device=device)

    metrics_csv1 = out_dir1 / "trained_episode_metrics.csv"
    
    t1_start = time.perf_counter()
    
    run_parallel_episodes_with_policy(
        trained_1, env1, args.deterministic, args.episodes,
        out_dir1, "trained_masked",
        registry1, lock1, metrics_csv1,
        best_scalar=best1,
        save_trajectories=save_trajectories,
    )
    
    t1_elapsed = time.perf_counter() - t1_start    
    
    save_registry_pickle(registry1, lock1, out_dir1 / "trained_final_shared_registry.pkl")
    save_registry_summary_csv(registry1, lock1, out_dir1 / "trained_registry_summary.csv")
    print(f"[INFO] Trained(masked) registry + metrics saved to {out_dir1}")

    # Record and persist ONLY Track 1 runtime
    
    print(f"[TIMER] Track 1 run time: {t1_elapsed:.2f}s")
    save_run_meta(out_dir=base_out, filename="run_metadata.json", elapsed_seconds=t1_elapsed)

    # If only Track 1 requested, end here.
    if args.no_ablation:
        print(f"[DONE] Track 1 completed. Outputs saved under: {out_dir1}")
        return

    # -------------------------------------------------------
    # TRACK 3: TRAINED (masked, WRITE-ONLY REGISTRY, NO SAMPLING)
    # -------------------------------------------------------
    out_dir3 = base_out / "trained_masked_no_registry"
    out_dir3.mkdir(parents=True, exist_ok=True)
    if save_trajectories:
        (out_dir3 / "trajectories").mkdir(parents=True, exist_ok=True)

    mgr3 = multiprocessing.Manager()
    lock3 = mgr3.Lock()
    best3 = mgr3.Value('d', float('inf'))
    registry3 = init_shared_registry(mgr3, G_initial_states, lock3)

    env3 = build_vec_env(
        args.n_envs,
        G_initial_states, args.max_nodes, args.max_steps, existing_keys,
        registry3, lock3, best3,
        registry_sampling=False,  # write-only, no sampling from registry
        initial_state_sampling_factor=args.initial_state_sampling_factor,
        masked=True,
        start_trajectory=save_trajectories,
        env_seed_base=args.seed + 10_000
    )
    trained_3 = MaskablePPO.load(args.model_path, env=env3, device=device)

    metrics_csv3 = out_dir3 / "trained_no_registry_episode_metrics.csv"
    run_parallel_episodes_with_policy(
        trained_3, env3, args.deterministic, args.episodes,
        out_dir3, "trained_masked_no_registry",
        registry3, lock3, metrics_csv3,
        best_scalar=best3,
        save_trajectories=save_trajectories,
    )
    save_registry_pickle(registry3, lock3, out_dir3 / "trained_no_sampling_final_shared_registry.pkl")
    save_registry_summary_csv(registry3, lock3, out_dir3 / "trained_no_sampling_registry_summary.csv")
    print(f"[INFO] Trained(masked, write-only registry, no sampling) saved to {out_dir3}")

    # -------------------------------------------------------
    # TRACK 4: TRAINED (UNMASKED inference, shared registry)
    # -------------------------------------------------------
    out_dir4 = base_out / "trained_unmasked"
    out_dir4.mkdir(parents=True, exist_ok=True)
    if save_trajectories:
        (out_dir4 / "trajectories").mkdir(parents=True, exist_ok=True)

    mgr4 = multiprocessing.Manager()
    lock4 = mgr4.Lock()
    best4 = mgr4.Value('d', float('inf'))
    registry4 = init_shared_registry(mgr4, G_initial_states, lock4)

    env4 = build_vec_env(
        args.n_envs,
        G_initial_states, args.max_nodes, args.max_steps, existing_keys,
        registry4, lock4, best4,
        registry_sampling=True,
        initial_state_sampling_factor=args.initial_state_sampling_factor,
        masked=False,  # UNMASKED
        start_trajectory=save_trajectories,
        env_seed_base=args.seed + 20_000
    )
    trained_4 = MaskablePPO.load(args.model_path, env=env4, device=device)

    metrics_csv4 = out_dir4 / "trained_unmasked_episode_metrics.csv"
    run_parallel_episodes_with_policy(
        trained_4, env4, args.deterministic, args.episodes,
        out_dir4, "trained_unmasked",
        registry4, lock4, metrics_csv4,
        best_scalar=best4,
        save_trajectories=save_trajectories,
    )
    save_registry_pickle(registry4, lock4, out_dir4 / "trained_unmasked_final_shared_registry.pkl")
    save_registry_summary_csv(registry4, lock4, out_dir4 / "trained_unmasked_registry_summary.csv")
    print(f"[INFO] Trained(UNMASKED inference) saved to {out_dir4}")

    # -------------------------------------------------------
    # TRACK 5: RANDOM (masked, shared registry)
    # -------------------------------------------------------
    out_dir5 = base_out / "random_masked"
    out_dir5.mkdir(parents=True, exist_ok=True)
    if save_trajectories:
        (out_dir5 / "trajectories").mkdir(parents=True, exist_ok=True)

    mgr5 = multiprocessing.Manager()
    lock5 = mgr5.Lock()
    best5 = mgr5.Value('d', float('inf'))
    registry5 = init_shared_registry(mgr5, G_initial_states, lock5)

    env5 = build_vec_env(
        args.n_envs,
        G_initial_states, args.max_nodes, args.max_steps, existing_keys,
        registry5, lock5, best5,
        registry_sampling=True,
        initial_state_sampling_factor=args.initial_state_sampling_factor,
        masked=True,
        start_trajectory=save_trajectories,
        env_seed_base=args.seed + 30_000
    )

    metrics_csv5 = out_dir5 / "random_episode_metrics.csv"
    run_parallel_episodes_random_masked(
        env5, args.episodes,
        out_dir5, "random_masked",
        registry5, lock5, metrics_csv5,
        rng,
        best_scalar=best5,
        save_trajectories=save_trajectories,
    )
    save_registry_pickle(registry5, lock5, out_dir5 / "random_final_shared_registry.pkl")
    save_registry_summary_csv(registry5, lock5, out_dir5 / "random_registry_summary.csv")
    print(f"[INFO] Random(masked) saved to {out_dir5}")

    print(f"[DONE] All outputs saved under: {base_out}")
    # Note: no additional save_run_meta call here to avoid overwriting Track 1 runtime


if __name__ == "__main__":
    # Make 'spawn' default on platforms where 'fork' can cause CUDA / process-safety issues.
    try:
        import multiprocessing as mp
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
