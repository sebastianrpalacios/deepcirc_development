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

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

# --- envs + utils ---
# Trained track: ORIGINAL env (no backtrack action)
from dgd.environments.drl3env_loader6_with_trajectories import (
    DRL3env as DRL3envTrained,
    _compute_truth_key,
    _compute_hash,
    _apply_implicit_or,
)
# Random track: BACKTRACKING env (adds one extra action)
from dgd.environments.drl3env_loader6_with_trajectories_and_backtracking import (
    DRL3env as DRL3envRandom,
)

from dgd.utils.utils5 import load_graph_pickle, energy_score, check_implicit_OR_existence_v3

# ----------------------- I/O helpers -----------------------
def save_episode_trajectory(env, out_dir: Path, episode: int, track_name: str):
    """Dump the current episode's trajectory from the underlying env to JSON."""
    get_traj = None
    try:
        get_traj = env.get_wrapper_attr("get_trajectory")
    except Exception:
        pass
    if get_traj is None:
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

def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

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
    """Fetch the env's per-episode minimum energy after an episode finishes."""
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
    """Tiny JSON receipt with UTC timestamp, CLI flags, and imported top-level modules (+version)."""
    import sys
    from datetime import datetime, timezone

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
    out_file = out_path / filename

    payload = {
        "timestamp_iso_utc": ts,
        "cli": {"raw": argv, "flags": flags, "positionals": positionals},
        "imports": imps,
    }
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[meta] wrote {out_file}")
    return str(out_file)


# ----------------------- Registry helpers -----------------------
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

def _append_episode_metrics_row(path: Path, episode, best_in_ep, best_so_far, new_graphs):
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
            int(new_graphs),
        ])


# ----------------------- Episode runners -----------------------
def run_episode_with_policy(model, env, deterministic, ep_seed=None):
    """Masked inference for the trained policy (passes action masks)."""
    try:
        obs, _ = env.reset(seed=ep_seed)
    except TypeError:
        obs, _ = env.reset()
    done = False
    while not done:
        mask = env.get_wrapper_attr("action_masks")()
        action, _ = model.predict(obs, deterministic=deterministic, action_masks=mask)
        action = _to_int_action(action)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

def run_episode_random_masked(env, ep_seed=None):
    """Uniform random over env.action_masks()."""

    obs, _ = env.reset(seed=ep_seed)

    rng = np.random.default_rng(ep_seed)
    done = False
    while not done:
        mask = env.get_wrapper_attr("action_masks")()
        valid = np.flatnonzero(mask)
        action = int(rng.choice(valid))
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated


# ----------------------- Env factories -----------------------
def build_env_trained_with_registry(seed_graphs, max_nodes, max_steps, existing_keys,
                                    registry, registry_lock, best_energy_across_workers,
                                    registry_sampling=True, initial_state_sampling_factor=0.0):
    """Use ORIGINAL env (no backtrack) → matches PPO checkpoint action space."""
    base_env = DRL3envTrained(
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
        best_energy_across_workers=best_energy_across_workers,
    )
    try:
        base_env.start_trajectory_logging(True)
    except Exception:
        pass
    return ActionMasker(base_env, mask_fn)

def build_env_random_store_only(seed_graphs, max_nodes, max_steps, existing_keys,
                                registry, registry_lock, best_energy_across_workers,
                                initial_state_sampling_factor=0.0):
    """Use BACKTRACK env; store in registry but do NOT sample from it."""
    base_env = DRL3envRandom(
        max_nodes=max_nodes,
        graphs=seed_graphs,
        shared_registry=registry,
        registry_lock=registry_lock,
        store_every_new_graph=True,
        sampling_from_shared_registry=False,  # do NOT sample for random track
        registry_read_only=False,
        max_steps=max_steps,
        enable_full_graph_replacement=True,
        show_plots=False,
        log_info=False,
        strict_iso_check=False,
        initial_state_sampling_factor=initial_state_sampling_factor,
        existing_keys=existing_keys,
        best_energy_across_workers=best_energy_across_workers,
    )
    try:
        base_env.start_trajectory_logging(True)
    except Exception:
        pass
    return ActionMasker(base_env, mask_fn)


# ----------------------- Main -----------------------
def main():
    p = argparse.ArgumentParser("Trained(masked) + Random(masked: 1 long trajectory)")
    p.add_argument("--model_path", required=True, help="Path to trained_model.zip")
    p.add_argument("--seed_files", nargs="+", required=True, help="One or more seed .pkl graph files")
    p.add_argument("--output_folder_name", required=True, help="Base output folder")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--max_nodes", type=int, default=100)
    p.add_argument("--max_steps", type=int, default=10)
    p.add_argument("--deterministic", action="store_true", default=False)
    p.add_argument("--seed", type=int, default=None, help="Base RNG seed for reproducibility")
    p.add_argument("--initial_state_sampling_factor", type=float, default=0.0,
                   help="Weighted sampling factor for initial states (trained track only)")
    args = p.parse_args()

    base_out = Path(args.output_folder_name)
    (base_out / "trained_masked" / "trajectories").mkdir(parents=True, exist_ok=True)
    (base_out / "random_masked" / "trajectories").mkdir(parents=True, exist_ok=True)

    # Global seeding
    if args.seed is not None:
        print(f"Using provided seed {args.seed}")
        random.seed(args.seed)
        np.random.seed(args.seed)
        th.manual_seed(args.seed)
    else:
        args.seed = int.from_bytes(os.urandom(4), "little")
        print(f"Generated random seed {args.seed}")
        random.seed(args.seed)
        np.random.seed(args.seed)
        th.manual_seed(args.seed)

    # Load seeds
    seed_paths = [str(Path(s)) for s in args.seed_files]
    print(f"[INFO] Using {len(seed_paths)} seed file(s):")
    for fp in seed_paths:
        print(" -", fp)
    G_initial_states = [load_graph_pickle(fp) for fp in seed_paths]
    existing_keys = {_compute_truth_key(g) for g in G_initial_states}
    print(f"[INFO] existing_keys initialized with {len(existing_keys)} key(s)")

    device = "cuda" if th.cuda.is_available() else "cpu"

    # ========= TRAINED (masked, shared registry; may sample from registry) =========
    out_tr = base_out / "trained_masked"
    mgr1 = multiprocessing.Manager()
    lock1 = mgr1.Lock()
    best1 = mgr1.Value('d', float('inf'))
    registry1 = init_shared_registry(mgr1, G_initial_states, lock1)

    env_tr = build_env_trained_with_registry(
        G_initial_states, args.max_nodes, args.max_steps,
        existing_keys, registry1, lock1, best1,
        registry_sampling=True,
        initial_state_sampling_factor=args.initial_state_sampling_factor
    )
    trained = MaskablePPO.load(args.model_path, env=env_tr, device=device)

    metrics_csv_tr = out_tr / "trained_episode_metrics.csv"
    for ep in range(args.episodes):
        snap = _snapshot_registry_counts(registry1, lock1)
        run_episode_with_policy(trained, env_tr, args.deterministic, ep_seed=args.seed + ep)
        save_episode_trajectory(env_tr, out_tr / "trajectories", ep, "trained_masked")
        best_in_ep = _get_best_energy_in_episode(env_tr)
        _, new_graphs = _best_new_energy_since(registry1, lock1, snap)
        best_so_far = _global_best_energy(registry1, lock1)
        _append_episode_metrics_row(metrics_csv_tr, ep, best_in_ep, best_so_far, new_graphs)

    save_registry_pickle(registry1, lock1, out_tr / "trained_final_shared_registry.pkl")
    save_registry_summary_csv(registry1, lock1, out_tr / "trained_registry_summary.csv")
    print(f"[INFO] Trained(masked) registry + metrics saved to {out_tr}", flush=True)

    # ========= RANDOM (masked, ONE long trajectory = episodes * max_steps; store-only) =========
    out_rs = base_out / "random_masked"
    mgr2 = multiprocessing.Manager()
    lock2 = mgr2.Lock()
    best2 = mgr2.Value('d', float('inf'))
    registry2 = init_shared_registry(mgr2, G_initial_states, lock2)

    # Set the random env's max_steps to the full budget
    random_budget_steps = int(args.episodes) * int(args.max_steps)
    env_rs = build_env_random_store_only(
        G_initial_states, args.max_nodes, random_budget_steps,
        existing_keys, registry2, lock2, best2,
        initial_state_sampling_factor=0.0
    )

    # Run ONE long random episode
    print(f"[INFO] Random single-trajectory budget = {random_budget_steps} steps "
          f"(episodes={args.episodes} × max_steps={args.max_steps})", flush=True)
    run_episode_random_masked(env_rs, ep_seed=args.seed + 10_000)

    # Save the single trajectory as ep_0000.json
    save_episode_trajectory(env_rs, out_rs / "trajectories", 0, "random_masked")

    # Derive per-episode metrics from the single trajectory and write CSV
    traj = env_rs.get_wrapper_attr("get_trajectory")()
    # Build dict: step_index -> energy
    step_to_energy = {}
    for s in traj:
        try:
            step_to_energy[int(s["step"])] = float(s["energy"])
        except Exception:
            pass

    # Sanity: should have steps 0..random_budget_steps
    # Compute per-pseudo-episode metrics
    csv_path_rs = out_rs / "random_episode_metrics.csv"
    with csv_path_rs.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "best_energy_in_episode", "best_energy_so_far", "num_new_graphs"])

        best_so_far = None
        for ep in range(args.episodes):
            # Segment [start, end] in step indices; include step 0 in ep0
            start_step = ep * args.max_steps
            end_step = (ep + 1) * args.max_steps
            seg_energies = []
            for t in range(start_step, end_step + 1):
                if t in step_to_energy:
                    seg_energies.append(step_to_energy[t])

            best_in_ep = float(np.min(seg_energies)) if seg_energies else None
            if best_in_ep is not None:
                best_so_far = best_in_ep if (best_so_far is None or best_in_ep < best_so_far) else best_so_far

            w.writerow([
                ep,
                "" if best_in_ep is None else float(best_in_ep),
                "" if best_so_far is None else float(best_so_far),
                0,  # num_new_graphs not tracked per segment in single long episode
            ])

    save_registry_pickle(registry2, lock2, out_rs / "random_final_shared_registry.pkl")
    save_registry_summary_csv(registry2, lock2, out_rs / "random_registry_summary.csv")
    print(f"[INFO] Random(masked, single trajectory) registry + metrics saved to {out_rs}", flush=True)

    print(f"[DONE] All outputs saved under: {base_out}")
    save_run_meta(out_dir=base_out, filename="run_metadata.json")


if __name__ == "__main__":
    main()
