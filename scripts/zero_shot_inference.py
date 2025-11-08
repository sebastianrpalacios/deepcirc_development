#!/usr/bin/env python
import argparse, csv, pickle
from pathlib import Path
import numpy as np
import torch as th
import networkx as nx

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

# env + utils from your repo
from dgd.environments.drl3env_loader4 import DRL3env, _compute_truth_key
from dgd.utils.utils5 import load_graph_pickle, energy_score, check_implicit_OR_existence_v3

def mask_fn(env):
    return env.action_masks()

def _to_int_action(action):
    if isinstance(action, np.ndarray):
        if action.shape == ():
            return int(action.item())
        return int(action.reshape(-1)[0])
    return int(action)

def run_episode_with_policy(model, env, deterministic=True, track_every_step=True, ep_seed=None):
    # reproducible per-episode reset
    try:
        obs, _ = env.reset(seed=ep_seed)
    except TypeError:
        obs, _ = env.reset()

    done = False
    traj = []
    best_e = np.inf
    best_g = None

    # capture initial state
    g0 = env.unwrapped.current_solution.copy()
    e0, _ = energy_score(g0, check_implicit_OR_existence_v3)
    traj.append((nx.node_link_data(g0), float(e0)))
    best_e, best_g = float(e0), g0.copy()

    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        action = _to_int_action(action)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        g = env.unwrapped.current_solution.copy()
        e, _ = energy_score(g, check_implicit_OR_existence_v3)
        if track_every_step:
            traj.append((nx.node_link_data(g), float(e)))
        if e < best_e:
            best_e, best_g = float(e), g.copy()

    return traj, best_g, best_e

def run_episode_random_masked(env, track_every_step=True, ep_seed=None):
    # reproducible per-episode reset
    try:
        obs, _ = env.reset(seed=ep_seed)
    except TypeError:
        obs, _ = env.reset()

    rng = np.random.default_rng(ep_seed)
    done = False
    traj = []
    best_e = np.inf
    best_g = None

    # capture initial state
    g0 = env.unwrapped.current_solution.copy()
    e0, _ = energy_score(g0, check_implicit_OR_existence_v3)
    traj.append((nx.node_link_data(g0), float(e0)))
    best_e, best_g = float(e0), g0.copy()

    while not done:
        # get current mask and sample uniformly among valid actions
        mask = env.get_wrapper_attr("action_masks")()
        valid = np.flatnonzero(mask)
        action = int(rng.choice(valid))

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        g = env.unwrapped.current_solution.copy()
        e, _ = energy_score(g, check_implicit_OR_existence_v3)
        if track_every_step:
            traj.append((nx.node_link_data(g), float(e)))
        if e < best_e:
            best_e, best_g = float(e), g.copy()

    return traj, best_g, best_e

def save_results(out_dir: Path, all_trajs, best_rows):
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "trajectories.pkl").open("wb") as f:
        pickle.dump(all_trajs, f)
    with (out_dir / "episode_best_summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "best_energy", "best_pickle"])
        w.writerows(best_rows)

def build_env(seed_graphs, max_nodes, max_steps, existing_keys):
    base_env = DRL3env(
        max_nodes=max_nodes,
        graphs=seed_graphs,
        shared_registry=None,
        registry_lock=None,
        store_every_new_graph=False,
        sampling_from_shared_registry=False,
        registry_read_only=True,
        max_steps=max_steps,
        enable_full_graph_replacement=True,
        show_plots=False,
        log_info=False,
        strict_iso_check=False,
        initial_state_sampling_factor=0.0,
        existing_keys=existing_keys,
    )
    return ActionMasker(base_env, mask_fn)

def main():
    p = argparse.ArgumentParser("Trained + Random(masked) + Untrained(masked) zero-shot evaluation")
    p.add_argument("--model_path", required=True, help="Path to trained_model.zip")
    p.add_argument("--seed_files", nargs="+", required=True, help="One or more seed .pkl graph files")
    p.add_argument("--output_folder_name", required=True, help="Base output folder")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--max_nodes", type=int, default=100)
    p.add_argument("--max_steps", type=int, default=10)
    # default=False => omit flag -> stochastic; add flag -> deterministic (applies to trained & untrained)
    p.add_argument("--deterministic", action="store_true", default=False)
    p.add_argument("--seed", type=int, default=123, help="Base RNG seed for reproducibility")
    p.add_argument("--track_every_step", action="store_true", default=True)
    args = p.parse_args()

    base_out = Path(args.output_folder_name)
    (base_out / "trained_masked").mkdir(parents=True, exist_ok=True)
    (base_out / "untrained_masked").mkdir(parents=True, exist_ok=True)
    (base_out / "random_masked").mkdir(parents=True, exist_ok=True)

    # global seeding
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    th.manual_seed(args.seed)

    # load seed graphs
    seed_paths = [str(Path(s)) for s in args.seed_files]
    print(f"[INFO] Using {len(seed_paths)} seed file(s):")
    for fp in seed_paths:
        print(" -", fp)
    G_initial_states = [load_graph_pickle(fp) for fp in seed_paths]
    existing_keys = {_compute_truth_key(g) for g in G_initial_states}
    print(f"[INFO] existing_keys initialized with {len(existing_keys)} key(s)")

    # build three independent envs so RNG/state don’t cross-contaminate
    env_trained    = build_env(G_initial_states, args.max_nodes, args.max_steps, existing_keys)
    env_untrained  = build_env(G_initial_states, args.max_nodes, args.max_steps, existing_keys)
    env_random     = build_env(G_initial_states, args.max_nodes, args.max_steps, existing_keys)

    device = "cuda" if th.cuda.is_available() else "cpu"

    # ===== Trained model (masked) =====
    trained = MaskablePPO.load(args.model_path, env=env_trained, device=device)

    # ===== Untrained model (masked) =====
    # Reuse architecture from the trained model to guarantee same policy net, random-init weights.
    tmp = MaskablePPO.load(args.model_path, env=env_untrained, device=device)
    policy_class = tmp.policy.__class__
    policy_kwargs = getattr(tmp, "policy_kwargs", None)
    untrained = MaskablePPO(
        policy=policy_class,
        env=env_untrained,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4, n_steps=16, batch_size=16, n_epochs=1,
        device=device, verbose=0,
    )
    del tmp  # free

    # ---------- RUN 1: TRAINED ----------
    out_dir = base_out / "trained_masked"
    all_trajs, best_rows = [], []
    for ep in range(args.episodes):
        ep_seed = args.seed + ep
        traj, best_g, best_e = run_episode_with_policy(
            model=trained, env=env_trained, deterministic=args.deterministic,
            track_every_step=args.track_every_step, ep_seed=ep_seed
        )
        all_trajs.append(traj)
        best_path = out_dir / f"episode_{ep:04d}_best.pkl"
        with best_path.open("wb") as f:
            pickle.dump(nx.node_link_data(best_g), f)
        best_rows.append([ep, best_e, best_path.name])
    save_results(out_dir, all_trajs, best_rows)
    print(f"[INFO] Trained(masked) saved to {out_dir}")

    # ---------- RUN 2: UNTRAINED ----------
    out_dir2 = base_out / "untrained_masked"
    all_trajs2, best_rows2 = [], []
    for ep in range(args.episodes):
        ep_seed = args.seed + ep
        traj, best_g, best_e = run_episode_with_policy(
            model=untrained, env=env_untrained, deterministic=args.deterministic,
            track_every_step=args.track_every_step, ep_seed=ep_seed
        )
        all_trajs2.append(traj)
        best_path = out_dir2 / f"episode_{ep:04d}_best.pkl"
        with best_path.open("wb") as f:
            pickle.dump(nx.node_link_data(best_g), f)
        best_rows2.append([ep, best_e, best_path.name])
    save_results(out_dir2, all_trajs2, best_rows2)
    print(f"[INFO] Untrained(masked) saved to {out_dir2}")

    # ---------- RUN 3: RANDOM ----------
    out_dir3 = base_out / "random_masked"
    all_trajs3, best_rows3 = [], []
    for ep in range(args.episodes):
        ep_seed = args.seed + ep
        traj, best_g, best_e = run_episode_random_masked(
            env=env_random, track_every_step=args.track_every_step, ep_seed=ep_seed
        )
        all_trajs3.append(traj)
        best_path = out_dir3 / f"episode_{ep:04d}_best.pkl"
        with best_path.open("wb") as f:
            pickle.dump(nx.node_link_data(best_g), f)
        best_rows3.append([ep, best_e, best_path.name])
    save_results(out_dir3, all_trajs3, best_rows3)
    print(f"[INFO] Random(masked) saved to {out_dir3}")

    print(f"[DONE] All results in: {base_out}")

if __name__ == "__main__":
    main()
