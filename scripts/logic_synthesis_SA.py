import json
import argparse
import csv
import pickle
import multiprocessing
from pathlib import Path
import time 

import numpy as np
import networkx as nx
import random
import os
import math
from typing import Optional

from sb3_contrib.common.wrappers import ActionMasker

# --- envs + utils ---
from dgd.environments.drl3env_loader6_with_trajectories_and_backtracking import (
    _compute_truth_key,
    _compute_hash,
    _apply_implicit_or,
)

from dgd.environments.drl3env_loader6_with_trajectories_and_backtracking import (
    DRL3env as DRL3envBacktrack,
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


# ----------------------- Rich run metadata -----------------------
def save_run_meta(out_dir: Path, filename: str, args_namespace, extra: Optional[dict] = None):
    """
    Save a comprehensive JSON receipt with:
      - UTC timestamp
      - Raw CLI argv
      - Parsed args (resolved values)
      - Derived params (via `extra`)
      - Imported top-level modules (+versions)
    """
    import sys
    from datetime import datetime, timezone

    ts = datetime.now(timezone.utc).isoformat()

    # Gather imports (+ versions)
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

    payload = {
        "timestamp_iso_utc": ts,
        "cli": {"raw": sys.argv[1:]},
        "args": vars(args_namespace) if args_namespace is not None else {},
        "derived": extra or {},
        "imports": imps,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / filename
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


# ----------------------- SA helpers -----------------------
def _get_backtrack_index(env) -> Optional[int]:
    """
    Discover the backtrack action index from the underlying env.
    If unavailable, return None; SA will still run but cannot revert on reject.
    """
    for name in ("BACKTRACK_ID",):
        try:
            attr = env.get_wrapper_attr(name)
            if callable(attr):
                val = attr()
                if isinstance(val, (int, np.integer)):
                    return int(val)
            elif isinstance(attr, (int, np.integer)):
                return int(attr)
        except Exception:
            pass
        try:
            val = getattr(env.unwrapped, name)
            if callable(val):
                val = val()
            if isinstance(val, (int, np.integer)):
                return int(val)
        except Exception:
            pass
    return None

def _current_energy_from_info_or_env(env, info) -> float:
    """
    Compute energy exclusively from env.current_solution.
    Raises a RuntimeError if current_solution is missing.
    """
    try:
        cur = env.get_wrapper_attr("current_solution")
    except Exception:
        cur = getattr(getattr(env, "unwrapped", env), "current_solution", None)

    if cur is None:
        raise RuntimeError("env.current_solution is missing or None")

    G = _to_graph(cur)  # accepts nx.Graph/DiGraph or node-link dict
    e, _ = energy_score(G, check_implicit_OR_existence_v3)
    return float(e)

def _temperature(step, total_steps, t0, tmin, schedule="exp"):
    step = max(0, int(step))
    total_steps = max(1, int(total_steps))
    if schedule == "linear":
        return max(tmin, t0 - (t0 - tmin) * (step / total_steps))
    alpha = (tmin / t0) ** (step / total_steps)
    return max(tmin, t0 * alpha)

def run_episode_sa_masked(
    env,
    ep_seed=None,
    total_budget_steps: Optional[int] = None,
    t0: float = 1.0,
    tmin: float = 0.01,
    schedule: str = "exp",
    prefer_non_backtrack: bool = True,
    verbose: bool = True,
    log_every_steps: int = 1000,
):
    """
    Simulated Annealing over masked actions with progress logging.
    Progress counts ACTUAL env steps (including backtracks) and obeys total_budget_steps.
    """
    try:
        obs, _ = env.reset(seed=ep_seed)
    except TypeError:
        obs, _ = env.reset()

    rng = np.random.default_rng(ep_seed)
    done = False
    env_steps = 0  # counts every env.step() call

    prev_energy = None
    best_energy_so_far = None

    start_time = time.time()

    back_idx = _get_backtrack_index(env)
    if verbose and back_idx is None:
        print("[SA] Backtrack index not found; will not be able to revert on reject.", flush=True)

    if total_budget_steps is None:
        total_budget_steps = getattr(env.unwrapped, "max_steps", None) or 1_000_000

    # Initial progress line
    if log_every_steps is not None and log_every_steps >= 0:
        print(f"[SA] Starting: budget={total_budget_steps} steps, T0={t0}, Tmin={tmin}, schedule={schedule}",
              flush=True)

    # === Step-budget guard added here ===
    while not done and env_steps < total_budget_steps:
        # Temperature based on true step count
        T = _temperature(env_steps, total_steps=total_budget_steps, t0=t0, tmin=tmin, schedule=schedule)

        mask = env.get_wrapper_attr("action_masks")()
        valid = np.flatnonzero(mask)
        if valid.size == 0:
            if verbose:
                print("[SA] No valid actions; terminating early.", flush=True)
            break

        cand = valid
        if prefer_non_backtrack and back_idx is not None and cand.size > 1:
            cand = cand[cand != back_idx]
            if cand.size == 0:
                cand = valid
        action = int(rng.choice(cand))

        # Execute proposal
        next_obs, reward, terminated, truncated, info = env.step(action)
        env_steps += 1  # count it
        done = terminated or truncated

        # Energy at new state
        new_energy = _current_energy_from_info_or_env(env, info)

        # Accept/reject
        accept = True
        if prev_energy is None:
            prev_energy = new_energy
        else:
            dE = (new_energy - prev_energy)
            if dE <= 0:
                prev_energy = new_energy
            else:
                p = math.exp(-dE / max(T, 1e-12))
                if rng.random() < p:
                    prev_energy = new_energy
                else:
                    accept = False

        # Track best-so-far of accepted states
        if prev_energy is not None and (best_energy_so_far is None or prev_energy < best_energy_so_far):
            best_energy_so_far = prev_energy

        # Immediate backtrack on reject (respect the budget)
        if (not accept) and (not done) and (back_idx is not None) and (env_steps < total_budget_steps):
            mask2 = env.get_wrapper_attr("action_masks")()
            valid2 = np.flatnonzero(mask2)
            if back_idx in valid2:
                _, _, terminated2, truncated2, _ = env.step(back_idx)
                env_steps += 1  # count backtrack
                done = done or terminated2 or truncated2
                if verbose:
                    print(f"[SA] Rejected proposal -> BACKTRACK (action {back_idx}) at env_step {env_steps}", flush=True)

        # Progress print
        if log_every_steps and log_every_steps > 0:
            if (env_steps == 1) or (env_steps % log_every_steps == 0) or done or (env_steps >= total_budget_steps):
                elapsed = time.time() - start_time
                pct = 100.0 * min(env_steps, total_budget_steps) / max(1, total_budget_steps)
                print(f"[SA] step {env_steps}/{total_budget_steps} "
                      f"({pct:5.1f}%)  T={T:.4g}  bestE={best_energy_so_far}  elapsed={elapsed:.1f}s",
                      flush=True)

        obs = next_obs

    elapsed = time.time() - start_time
    print(f"[SA] Finished {env_steps} steps. Best energy={best_energy_so_far}. Total time={elapsed:.1f}s",
          flush=True)


def build_env_sa_store_only(seed_graphs, max_nodes, max_steps, existing_keys,
                            registry, registry_lock, best_energy_across_workers,
                            initial_state_sampling_factor=0.0):
    """Use BACKTRACK env; store in registry but do NOT sample from it."""
    base_env = DRL3envBacktrack(
        max_nodes=max_nodes,
        graphs=seed_graphs,
        shared_registry=registry,
        registry_lock=registry_lock,
        store_every_new_graph=True,
        sampling_from_shared_registry=False,  # SA track only
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
    p = argparse.ArgumentParser("SA(masked: single long trajectory)")
    p.add_argument("--seed_files", nargs="+", required=True, help="One or more seed .pkl graph files")
    p.add_argument("--output_folder_name", required=True, help="Base output folder")
    p.add_argument("--steps", type=int, required=True, help="Total number of environment steps (budget)")
    p.add_argument("--max_nodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=None, help="Base RNG seed for reproducibility")
    p.add_argument("--initial_state_sampling_factor", type=float, default=0.0,
                   help="Weighted sampling factor for initial states")
    # SA params
    p.add_argument("--sa_T0", type=float, default=1.0, help="Initial temperature")
    p.add_argument("--sa_Tmin", type=float, default=0.01, help="Minimum temperature")
    p.add_argument("--sa_schedule", choices=["exp", "linear"], default="exp", help="Cooling schedule")
    p.add_argument("--sa_verbose", action="store_true", default=True, help="Verbose SA logs")
    p.add_argument("--log_every_steps", type=int, default=1000, help="Print progress every N steps (0 to disable).")
    args = p.parse_args()

    base_out = Path(args.output_folder_name)
    (base_out / "sa_masked" / "trajectories").mkdir(parents=True, exist_ok=True)

    # Global seeding
    if args.seed is not None:
        print(f"Using provided seed {args.seed}")
        random.seed(args.seed)
        np.random.seed(args.seed)
    else:
        args.seed = int.from_bytes(os.urandom(4), "little")
        print(f"Generated random seed {args.seed}")
        random.seed(args.seed)
        np.random.seed(args.seed)

    # Load seeds
    seed_paths = [str(Path(s)) for s in args.seed_files]
    print(f"[INFO] Using {len(seed_paths)} seed file(s):")
    for fp in seed_paths:
        print(" -", fp)
    G_initial_states = [load_graph_pickle(fp) for fp in seed_paths]
    existing_keys = {_compute_truth_key(g) for g in G_initial_states}
    print(f"[INFO] existing_keys initialized with {len(existing_keys)} key(s)")

    # ========= SA (masked, ONE long trajectory; store-only) =========
    out_sa = base_out / "sa_masked"
    mgr = multiprocessing.Manager()
    lock = mgr.Lock()
    best_shared = mgr.Value('d', float('inf'))
    registry = init_shared_registry(mgr, G_initial_states, lock)

    # Set the SA env's max_steps to the full user-provided budget
    sa_budget_steps = int(args.steps)
    env_sa = build_env_sa_store_only(
        G_initial_states, args.max_nodes, sa_budget_steps,
        existing_keys, registry, lock, best_shared,
        initial_state_sampling_factor=args.initial_state_sampling_factor
    )

    print(f"[INFO] SA single-trajectory budget = {sa_budget_steps} steps", flush=True)

    prefer_non_backtrack = True

    run_episode_sa_masked(
        env_sa,
        ep_seed=args.seed + 10_000,
        total_budget_steps=sa_budget_steps,
        t0=args.sa_T0,
        tmin=args.sa_Tmin,
        schedule=args.sa_schedule,
        prefer_non_backtrack=prefer_non_backtrack,
        verbose=args.sa_verbose,
        log_every_steps=args.log_every_steps,  
    )

    # Save the single trajectory as ep_0000.json
    save_episode_trajectory(env_sa, out_sa / "trajectories", 0, "sa_masked")

    # Per-step metrics: step, energy, best_energy_so_far
    traj = env_sa.get_wrapper_attr("get_trajectory")()
    csv_path_sa = out_sa / "sa_step_metrics.csv"
    best_so_far = None
    with csv_path_sa.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "energy", "best_energy_so_far"])
        for s in traj:
            try:
                t = int(s["step"])
                e = float(s["energy"])
                if (best_so_far is None) or (e < best_so_far):
                    best_so_far = e
                w.writerow([t, e, best_so_far])
            except Exception:
                # Skip steps that don't have the expected fields
                pass

    # Persist registry artifacts
    save_registry_pickle(registry, lock, out_sa / "sa_final_shared_registry.pkl")
    save_registry_summary_csv(registry, lock, out_sa / "sa_registry_summary.csv")
    print(f"[INFO] SA(masked, single trajectory) registry + metrics saved to {out_sa}", flush=True)

    # Rich run metadata (args + derived params)
    save_run_meta(
        out_dir=base_out,
        filename="run_metadata.json",
        args_namespace=args,
        extra={
            "sa_budget_steps": sa_budget_steps,
            "num_seed_files": len(seed_paths),
            "seed_paths": seed_paths,
            "prefer_non_backtrack": prefer_non_backtrack,
            "output_dirs": {
                "base": str(base_out),
                "sa_masked": str(out_sa),
                "trajectories": str(out_sa / "trajectories"),
            },
        },
    )

    print(f"[DONE] All outputs saved under: {base_out}")


if __name__ == "__main__":
    main()
