from __future__ import annotations
import itertools
import time 
import os
import pickle
import random
from collections import defaultdict
from typing import DefaultDict, Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import networkx as nx
from networkx.algorithms import weisfeiler_lehman_graph_hash as wl_hash
import multiprocessing           
#import contextlib 

from dgd.utils.utils5 import (
    calculate_truth_table_v2,
    generate_one_hot_features_from_adj,
    resize_matrix,
    energy_score,
    check_implicit_OR_existence_v2,
    add_implicit_OR_to_dag_v2,
    exhaustive_cut_enumeration_dag,
    is_fanout_free_standalone,
    generate_subgraph,
    substitute_subgraph, 
)

MOTIFS_PATH = "action_motifs.pkl"
with open(MOTIFS_PATH, "rb") as f:
    action_motifs = pickle.load(f)

UNIQUE_GRAPHS = action_motifs["graphs"]         
TTABLE_TO_ACTIONS = action_motifs["lookup"]
NUM_ACTIONS = len(UNIQUE_GRAPHS)

print(f"Loaded {NUM_ACTIONS} motifs from {MOTIFS_PATH}")

def canon_hash(g):
    return wl_hash(g, node_attr=None, edge_attr=None, iterations=3, digest_size=16)

def _truth_key(g):
    tt = calculate_truth_table_v2(g)
    bits = "".join(str(o[0]) for _, o in sorted(tt.items()))
    return (len(tt).bit_length() - 1, int(bits, 2))

def _permute_and_match(motif, key, aid,  *, first_only = False):
    inputs = [n for n in motif if motif.in_degree(n) == 0]
    if len(inputs) != key[0]:
        return None
    matches = []
    for perm in itertools.permutations(inputs):
        g2 = nx.relabel_nodes(motif, dict(zip(inputs, perm)), copy=True)
        if _truth_key(g2) == key:
            if first_only:
                return g2
            matches.append(g2)
    if not matches:
        return None
    chosen = random.choice(matches)    
    return chosen

def _apply_implicit_or(G, fanin_size: int = 2):
    G_copy = G.copy()
    output_nodes = [n for n in G_copy if G_copy.out_degree(n) == 0]
    if not output_nodes:
        return G_copy
    output_node = output_nodes[0]
    results_check_implicit_OR_existence = check_implicit_OR_existence_v2(G_copy, output_node, fanin_size)
    best_node_reduction_found, best_node_reduction_found_key = 0, None
    for key, value in results_check_implicit_OR_existence.items():
        if value["is_there_an_implicit_OR"] and value["number_of_nodes_available_for_removal"] > best_node_reduction_found:
            best_node_reduction_found_key, best_node_reduction_found = key, value["number_of_nodes_available_for_removal"]
    if best_node_reduction_found_key is None:
        return G_copy
    cut = results_check_implicit_OR_existence[best_node_reduction_found_key]["cut"]
    cone = results_check_implicit_OR_existence[best_node_reduction_found_key]["cone"]
    return add_implicit_OR_to_dag_v2(G_copy, output_node, cut, cone)


# def get_global_optima(registry):
#    """Return [(original_graph_copy, energy), …] for all entries in the registry."""
#    flat_items = [item
#                  for bucket in registry.values()
#                  for item   in bucket]
#    return [(orig.copy(), e) for _, orig, e in flat_items]


class DRL3env(gym.Env):

    def __init__(
        self,
        max_nodes: int,
        graphs,
        *,
        shared_registry: "dict[str, list[tuple[nx.DiGraph, nx.DiGraph, float]]] | None" = None,
        registry_lock = None,
        sampling_from_shared_registry = True,
        max_steps: int = 10,
        circuit_name = None,
        enable_full_graph_replacement = True,
        show_plots = False,
        log_info = False,
        strict_iso_check = False,
        initial_state_sampling_weight = 3,
    ):
        super().__init__()
        if not graphs:
            raise ValueError("Initial graphs list cannot be empty")

        self._shared_registry = shared_registry      
        self._registry_lock   = registry_lock 
        self.initial_state_sampling_weight = initial_state_sampling_weight
        self.sampling_from_shared_registry = sampling_from_shared_registry        
        self.strict_iso_check = strict_iso_check           
        self.circuit_name = circuit_name or "unnamed"
        self._owner_pid = os.getpid()
        self.show_plots = show_plots and (os.getpid() == self._owner_pid)
        self.log_info   = log_info  and (os.getpid() == self._owner_pid)
        self._best_energy        = float("inf")
        self._global_best_energy = float("inf")  
        self.initial_states   = [g.copy() for g in graphs]
        self._seed_energies = [energy_score(g, check_implicit_OR_existence_v2)[0] for g in self.initial_states]        
        self._populate_registry_with_seeds()         
        self.current_solution = random.choice(self.initial_states).copy()
        self.enable_full_graph_replacement = enable_full_graph_replacement
        self.max_nodes   = max_nodes
        self.max_steps   = max_steps
        self.current_step = 0
        self.all_episodes_current_step = 0
        self.TERMINATE_ID = NUM_ACTIONS          
        self.action_space = spaces.Discrete(NUM_ACTIONS + 1)

        self.observation_space = spaces.Dict(
            {
                "node_features": spaces.Box(0, 1, shape=(max_nodes, 4), dtype=np.float32),
                "adj_matrix":    spaces.Box(0, 1, shape=(max_nodes, max_nodes), dtype=np.float32),
            }
        )
        self._mask_cache: np.ndarray | None = None
        self._subgraphs_for_action: Dict[int, List[Tuple[int, Tuple[int, ...], Tuple[int, int]]]] = defaultdict(list)

#    def _seen_hashes(self) -> set[str]:
#        if self._shared_registry is not None:
#            return set(self._shared_registry.keys())
#        return {h for h, _, _, _ in self.optimal_graphs}

    def _canonical_graph(self, g: nx.DiGraph) -> nx.DiGraph:
        return _apply_implicit_or(g)       


#    def _save_optimal_graphs(self):
#        pid   = os.getpid()
#        fname = f"optimal_{self.circuit_name}_{pid}_{int(time.time())}.pkl"
#        payload = [(orig.copy(), e) for _, _, orig, e in self.optimal_graphs]   
#        with open(fname, "wb") as f:
#            pickle.dump(payload, f)
#        if self.log_info:
#            print(f"[save] wrote {fname} (n={len(payload)})")

    def _populate_registry_with_seeds(self):
        """Insert initial seed graphs into the shared registry (bucket layout)."""
        if self._shared_registry is None:
            return
        
        if self._registry_lock is None:
            raise ValueError("Registry lock is required to use the shared solutions across workers")       

        with (self._registry_lock):
            
            # Only populate if the registry is empty
            if len(self._shared_registry):
                return       
                      
            for g_seed in self.initial_states:
                canon = self._canonical_graph(g_seed)
                h     = canon_hash(canon)

                bucket = self._shared_registry.setdefault(h, [])

                bucket.append(
                    (canon,                    # keep canon reference
                     g_seed.copy(),            # copy of original seed graph
                     energy_score(g_seed, check_implicit_OR_existence_v2)[0])
                )
                
                self._shared_registry[h] = bucket            
            

    # -------------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        
        self.current_step = 0
        self._mask_cache  = None
        
        if self.log_info:
            t0 = time.perf_counter()        
        


        # --- build pool  ---------------------------------
        if self.sampling_from_shared_registry and self._shared_registry is not None:
            # sample only from registry (already contains the seeds)
            reg_items  = [item for bucket in self._shared_registry.values() for item in bucket]
            pool     = [orig for _, orig, _ in reg_items]
            energies = [e    for _, _,    e in reg_items]
            
            if not pool:          # this should never happen
                raise RuntimeError(
                    "[reset] Shared registry is empty ‒ expected at least the initial graph(s). "
                )
                

        else:
            # sample only from the original seed set (no registry sampling)
            pool     = self.initial_states
            energies = self._seed_energies   
            
            if not pool:
                raise RuntimeError("[reset] initial_states is empty — nothing to sample from.")            


        p   = self.initial_state_sampling_weight   # exponent (>1 → stronger bias)
        eps = 1e-6                                 # avoids division by zero           
        # reciprocal-power weights: w = 1 / (energy + eps)^p
        weights = [1.0 / ((e + eps) ** p) for e in energies]

        # draw a graph proportional to those weights
        self.current_solution = random.choices(pool, weights=weights, k=1)[0].copy()

        # ------------------------------------------------------------
        # 2) house-keeping
        # ------------------------------------------------------------

        self._best_energy, _ = energy_score(self.current_solution, check_implicit_OR_existence_v2)

        ##CHECK THIS:
        self._global_best_energy    = min(self._global_best_energy, self._best_energy)        
        
        if self.log_info:
            print("[reset] took", (time.perf_counter() - t0)*1e3, "ms")

        return self._build_obs(), {}



    # ----------------------------------------------------------------------
    def action_masks(self):
        if self.log_info:
            t_all = time.perf_counter()             # → total timer


        mask = np.zeros(NUM_ACTIONS + 1, dtype=bool)
        self._subgraphs_for_action.clear()
        tt_counter: DefaultDict[Tuple[int, int], list] = defaultdict(list)


        # primary-inputs
        source_nodes = {n for n, deg in self.current_solution.in_degree() if deg == 0}

        # outputs (terminal nodes)
        output_nodes = {n for n, deg in self.current_solution.out_degree() if deg == 0}

        # predecessors of the output node(s) – skipped **only** when full-graph replacement is disabled
        pred_of_output = set()
        if not self.enable_full_graph_replacement:
            for o in output_nodes:
                pred_of_output.update(self.current_solution.predecessors(o))

        if self.log_info:
            print(f"Source nodes   : {source_nodes}")
            print(f"Output nodes   : {output_nodes}")
            print(f"Skip-before-out: {pred_of_output}")


        if self.log_info:
            t_enum = time.perf_counter()

        for tgt in self.current_solution.nodes():

            if (
                tgt in source_nodes          # skip primary-inputs
                or tgt in output_nodes       # skip outputs
                or tgt in pred_of_output     # skip predecesors of output node when full-graph replacement is disabled
            ):
                continue


            for cut in exhaustive_cut_enumeration_dag(self.current_solution, 4, tgt, filter_redundant=True):
                if not is_fanout_free_standalone(self.current_solution, tgt, cut):
                    continue

                sg = generate_subgraph(self.current_solution, tgt, cut, draw=False)
                if len([n for n in sg if sg.in_degree(n) == 0]) != len(cut):
                    continue

                key = _truth_key(sg)
                for aid in TTABLE_TO_ACTIONS.get(key, []):
                    mask[aid] = True
                    self._subgraphs_for_action[aid].append((tgt, tuple(cut), key))
                tt_counter[key].append((tgt, tuple(cut)))

        if self.log_info:
            dt = time.perf_counter() - t_enum
            print(f"[timer] enumeration & mapping took {dt:.3f} s")


        if self.log_info:
            t_dbg = time.perf_counter()

            uniq = sum(1 for v in tt_counter.values() if len(v) == 1)
            for (n_in, tt_int), items in tt_counter.items():
                if len(items) > 1:
                    print(f"[action_masks] TT {tt_int} ({n_in}-in) appears {len(items)} times.")
            print(f"[action_masks] {uniq} truth tables appear once among {len(tt_counter)} candidates.")

            dt = (time.perf_counter() - t_dbg) * 1e3
            print(f"[timer] stats-print took {dt:.2f} ms")



        if mask[: self.TERMINATE_ID].sum() == 0:
            if self.log_info:
                print("Nothing left to rewrite")
            mask[self.TERMINATE_ID] = 1

        self._mask_cache = mask

        if self.log_info:
            dt = time.perf_counter() - t_all
            print(f"[timer] action_masks() TOTAL {dt:.3f} s")

        
        if not mask.any():
            raise RuntimeError("Action mask with no valid actions")
        
        return mask


    # -------------------------------------------------------------------------
    def step(self, action):
        """
        One environment step.
        * Applies the chosen rewrite (if valid).
        * Tracks both episode‑best and global‑best energies.
        * Gives a reward only at episode end:   100 / best_energy_in_episode.
        """
        
        
        self.current_step    += 1
        self.all_episodes_current_step += 1



        if self._mask_cache is None:
            self.action_masks()           


        if action == self.TERMINATE_ID:            

            done    = True
            reward  = (100.0 / self._best_energy) if self._best_energy != 0 else 0.0
            
                        
            if self.log_info:
                print(f"Terminate action selected  self._best_energy={self._best_energy:.3f} "
                      f"Reward={reward:.5f}")
            return self._build_obs(), reward, done, False, {}
        


        if action not in self._subgraphs_for_action:      # <-- guard
            # should not happen with a correct mask
            if self.log_info:
                print(f"The action is not in self._subgraphs_for_action. Inspect the masking.")                
            return self._build_obs(), 0.0, False, False, {}        


        # Apply the rewrite 
        tt_before = calculate_truth_table_v2(self.current_solution)
        tgt, cut, key = random.choice(self._subgraphs_for_action[action])

        repl = _permute_and_match(UNIQUE_GRAPHS[action], key, action)
        if repl is None:
            if self.log_info:
                print(f"Permutation failed in the environment. Inspect the masking.") 
            return self._build_obs(), 0.0, False, False, {}

        cand    = generate_subgraph(self.current_solution, tgt, cut, draw=False)
        new_sol = substitute_subgraph(self.current_solution, cand, repl)
        
        
        # size guard 
        new_size = len(new_sol)
        if new_size > self.max_nodes:
            if self.log_info:
                print(f"Rewrite skipped because size {new_size} > limit {self.max_nodes}")
            # (could return a small negative reward instead)
            return self._build_obs(), 0.0, False, False, {}
                
        if calculate_truth_table_v2(new_sol) != tt_before:
            raise ValueError("The logic function changed after rewrite")

        # Accept the rewrite 
        self.current_solution = new_sol 
        
        # Invalidate for next step
        self._mask_cache      = None           

        # Compute new current energy 
        current_energy, _ = energy_score(self.current_solution, check_implicit_OR_existence_v2)

        # Update best energy in the episode
        self._best_energy = min(self._best_energy, current_energy)


        # Case A: no shared registry exists
        if self._shared_registry is None:
            if current_energy < self._global_best_energy:
                self._global_best_energy = current_energy

        # Case B shared registry exists 
        else:
            # process only if the energy ties or beats the global best
            if current_energy <= self._global_best_energy:

                # canonicalise once and WL-hash
                canon = self._canonical_graph(self.current_solution)
                h     = canon_hash(canon)

                with (self._registry_lock):
                    bucket: list[tuple[nx.DiGraph, nx.DiGraph, float]] = self._shared_registry.setdefault(h, [])

                    # ── B-1 : strict iso check → avoid storing isomorphic graphs -----
                    if self.strict_iso_check:
                        already_there = any(nx.is_isomorphic(canon, cg_prev) for cg_prev, _, _ in bucket)
                        if not already_there:
                            bucket.append(
                                (canon.copy(),
                                 self.current_solution.copy(),
                                 current_energy)
                            )
                            self._shared_registry[h] = bucket

                    # ── B-2 : no strict iso check → rely purely on WL hash ----------
                    else:
                        # store only once per WL hash
                        if not bucket:                       # bucket empty ⇒ first entry
                            bucket.append((canon.copy(), self.current_solution.copy(), current_energy))
                            self._shared_registry[h] = bucket
                        # if bucket already populated we assume same topology→do nothing
                        
                    if self.log_info:           # optional debug print
                        print("registry size =", sum(len(b) for b in self._shared_registry.values()))                        

                # promote the scalar only on a strict improvement
                if current_energy < self._global_best_energy:
                    self._global_best_energy = current_energy
            # else: energy worse than global best → nothing to do



                    
        if self.show_plots:
            self._draw_graph(f"Step {self.current_step}: size={len(self.current_solution)}")

        # 4) Reward + termination ----------------------------------------------
        done  = self.current_step >= self.max_steps        
        
        reward = (100.0 / self._best_energy) if done and self._best_energy != 0 else 0.0

        if self.log_info:
            print(
                f"[step] step={self.current_step} done={done} "
                f"energy={current_energy:.3f} "
                f"best_ep={self._best_energy:.3f} "
                f"best_global={self._global_best_energy:.3f} "
                f"reward={reward:.5f}"
            )
            
        if self.log_info and self._shared_registry is not None:
            print("registry size =", sum(len(b) for b in self._shared_registry.values()))
            
            
        #if done:
        #    self._save_optimal_graphs()            

        return self._build_obs(), reward, done, False, {}

    # ---------------------------- drawing helper -----------------------------
    def _draw_graph(self, title: str):
        if not self.show_plots:
            return
        plt.figure(figsize=(4, 4))
        pos = nx.spring_layout(self.current_solution, seed=42)
        nx.draw(self.current_solution, pos, with_labels=True, node_size=300, font_size=8)
        plt.title(title)
        plt.show()

    # ---------------------------- observation --------------------------------
    def _build_obs(self):        
       
        nodes = list(nx.topological_sort(self.current_solution))
        adj_dense = nx.adjacency_matrix(self.current_solution, nodelist=nodes).toarray()
        adj = np.asarray(resize_matrix(adj_dense, self.max_nodes), dtype=np.float32)
        feats = np.asarray(
            generate_one_hot_features_from_adj(
                nx.adjacency_matrix(self.current_solution, nodelist=nodes),
                pad_size=self.max_nodes,
            ),
            dtype=np.float32,
        )
        
       
        # ---------- shape guards ------------------------------------------------
        assert adj.shape   == (self.max_nodes, self.max_nodes), \
            f"adj shape {adj.shape} ≠ ({self.max_nodes}, {self.max_nodes})"
        assert feats.shape == (self.max_nodes, 4), \
            f"feats shape {feats.shape} ≠ ({self.max_nodes}, 4)"

        # ---------- adidtional checks for dtype & value range ------------------------
        assert adj.dtype   == np.float32 and feats.dtype == np.float32
        assert np.isin(adj,  (0.0, 1.0)).all(), "adjacency contains non-binary entries"
        assert np.isin(feats, (0.0, 1.0)).all(), "node features not one-hot/padded"     
        
        
        return {"node_features": feats, "adj_matrix": adj}

    def close(self):
        # make sure we always dump the best graphs, even on SIGINT/normal close
        super().close()
        #self._save_optimal_graphs()
        

    # ---------------------------------------------------------------------
    def render(self):
        if self.show_plots:
            self._draw_graph("Current topology")
