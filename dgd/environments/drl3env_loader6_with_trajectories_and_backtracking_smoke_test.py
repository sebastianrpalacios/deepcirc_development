# drl3env_loader6_with_trajectories_and_backtracking_smoke_test.py
import argparse
import numpy as np

from dgd.environments.drl3env_loader6_with_trajectories_and_backtracking import (
    DRL3env, _compute_hash, _compute_truth_key
)
from dgd.utils.utils5 import load_graph_pickle, energy_score, check_implicit_OR_existence_v3


DEFAULTS = {
    "seed_pickle": "/home/gridsan/spalacios/Designing complex biological circuits with deep neural networks/dgd/data/NIGs_4_inputs/0x5F01_NIG_unoptimized.pkl",
    "max_steps": 6,
    "max_nodes": 128,
    "seed": 123,
}


def first_forward_action(mask, term_id):
    """Return the first forward (non-terminate/non-backtrack) action, or None."""
    m = np.asarray(mask, dtype=bool).ravel()
    fwd = np.flatnonzero(m[:term_id])
    return int(fwd[0]) if fwd.size > 0 else None


def run_smoke(seed_pickle: str, max_steps: int, max_nodes: int, seed: int):
    # Load a real seed
    G = load_graph_pickle(seed_pickle)

    # Ensure this seed satisfies the env's lower bound for best_energy
    be, _ = energy_score(G, check_implicit_OR_existence_v3)
    if be < 1.0:
        raise RuntimeError(
            f"Seed energy is {be} < 1.0; pick another seed with energy >= 1.0 for this smoke test."
        )

    env = DRL3env(
        max_nodes=max_nodes,
        graphs=[G],
        shared_registry=None,
        registry_lock=None,
        store_every_new_graph=False,
        sampling_from_shared_registry=False,
        registry_read_only=False,
        max_steps=max_steps,
        enable_full_graph_replacement=True,
        show_plots=False,
        log_info=False,
        strict_iso_check=False,
        initial_state_sampling_factor=0.0,
        existing_keys={_compute_truth_key(G)},  # keep input-permutation shuffle valid
    )

    # Reset
    obs, info = env.reset(seed=seed)
    term_id = env.TERMINATE_ID
    back_id = env.BACKTRACK_ID

    # Root snapshot hash
    h_root = _compute_hash(env.current_solution)

    # Root mask
    mask0 = env.action_masks()
    assert mask0.shape[0] == env.action_space.n, "mask length != action_space.n"

    # If no forward action at root:
    if not bool(mask0[:term_id].any()):
        # TERMINATE must be available at root, BACKTRACK must be masked (depth==0)
        assert bool(mask0[term_id]), "TERMINATE should be available at root when no forwards exist"
        assert not bool(mask0[back_id]), "BACKTRACK should be masked at root (depth==0)"

        # Simulate depth>0 by pushing a snapshot (for mask check)
        env._state_stack.append(env._copy_graph(env.current_solution))
        mask1 = env.action_masks()
        assert bool(mask1[back_id]), "BACKTRACK should be available when depth>0"

        # Take BACKTRACK: (1) step++ (2) restore graph (3) not done yet
        before = env.current_step_in_episode
        obs, reward, done, truncated, info = env.step(back_id)
        assert env.current_step_in_episode == before + 1, "BACKTRACK must consume a step"
        assert _compute_hash(env.current_solution) == h_root, "Backtrack did not restore previous graph"
        assert not done, "Episode should not be done yet"

        # BACKTRACK masked again at root
        mask_after = env.action_masks()
        assert not bool(mask_after[back_id]), "BACKTRACK must be masked at root after backtrack"

        # Spam BACKTRACK at root to ensure episode ends by max_steps (no-ops but count steps)
        while not done:
            obs, reward, done, truncated, info = env.step(back_id)
            assert _compute_hash(env.current_solution) == h_root, "Root backtrack must not change the graph"

        assert env.current_step_in_episode == env.max_steps, "Episode did not end at max_steps"
        print("✅ Smoke test passed (root had no forward actions).")
        return

    # Else: there is a forward action at root — take one, then backtrack it
    a = first_forward_action(mask0, term_id)
    assert a is not None, "Expected a forward action at root"

    before = env.current_step_in_episode
    h_before = _compute_hash(env.current_solution)

    # Forward step
    obs, reward, done, truncated, info = env.step(a)
    assert env.current_step_in_episode == before + 1, "Forward step must increment step counter"
    assert not done, "Episode ended unexpectedly after one forward step (increase max_steps if needed)"

    # Depth>0: BACKTRACK should be available
    mask1 = env.action_masks()
    assert bool(mask1[back_id]), "BACKTRACK should be available after going deeper"

    # Backtrack to previous hash
    mid = env.current_step_in_episode
    obs, reward, done, truncated, info = env.step(back_id)
    assert env.current_step_in_episode == mid + 1, "BACKTRACK must consume a step"
    assert _compute_hash(env.current_solution) == h_before, "Backtrack did not restore the previous graph"

    # At root again: BACKTRACK should be masked unless no forward actions exist
    mask_after = env.action_masks()
    if bool(mask_after[:term_id].any()):
        assert not bool(mask_after[back_id]), "BACKTRACK should be masked at root when forwards exist"
    else:
        assert bool(mask_after[term_id]), "TERMINATE should be available at root if no forwards exist"

    print("✅ DRL3env backtrack smoke test passed.")


if __name__ == "__main__":
    # Defaults let you run with no CLI; you can still override via flags.
    parser = argparse.ArgumentParser("Backtrack smoke test for DRL3env")
    parser.add_argument("--seed_pickle", default=DEFAULTS["seed_pickle"])
    parser.add_argument("--max_steps", type=int, default=DEFAULTS["max_steps"])
    parser.add_argument("--max_nodes", type=int, default=DEFAULTS["max_nodes"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    args = parser.parse_args()

    print("[Smoke] Using parameters:")
    print(f"  seed_pickle = {args.seed_pickle}")
    print(f"  max_steps   = {args.max_steps}")
    print(f"  max_nodes   = {args.max_nodes}")
    print(f"  seed        = {args.seed}")

    run_smoke(
        seed_pickle=args.seed_pickle,
        max_steps=args.max_steps,
        max_nodes=args.max_nodes,
        seed=args.seed,
    )
