from __future__ import annotations
import itertools
import time 
import os
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import networkx as nx
from networkx.algorithms import weisfeiler_lehman_graph_hash        

from dgd.utils.utils5 import (
    calculate_truth_table_v2,
    generate_one_hot_features_from_adj,
    resize_matrix,
    energy_score,
    check_implicit_OR_existence_v3,
    add_implicit_OR_to_dag_v2,
    exhaustive_cut_enumeration_dag,
    is_fanout_free_standalone,
    generate_subgraph,
    substitute_subgraph, 
)

MOTIFS_PATH = "/home/gridsan/spalacios/Designing complex biological circuits with deep neural networks/scripts/action_motifs.pkl"
with open(MOTIFS_PATH, "rb") as f:
    action_motifs = pickle.load(f)

UNIQUE_GRAPHS = action_motifs["graphs"]         
TTABLE_TO_ACTIONS = action_motifs["lookup"]
NUM_ACTIONS = len(UNIQUE_GRAPHS)

print(f"Done loading action motifs. There are {NUM_ACTIONS} unique motifs.")

def _compute_hash(g):
    return weisfeiler_lehman_graph_hash(g, node_attr=None, edge_attr=None, iterations=30, digest_size=16)

def _compute_truth_key(g):
    tt = calculate_truth_table_v2(g)
    bits = "".join(str(o[0]) for _, o in sorted(tt.items()))
    return (len(tt).bit_length() - 1, int(bits, 2))

def _apply_implicit_or(G, fanin_size: int = 2):
    G_copy = G.copy()
    output_nodes = [n for n in G_copy if G_copy.out_degree(n) == 0]
    if not output_nodes:
        return G_copy
    output_node = output_nodes[0]
    results_check_implicit_OR_existence = check_implicit_OR_existence_v3(G_copy, output_node, fanin_size)
    best_node_reduction_found, best_node_reduction_found_key = 0, None
    for key, value in results_check_implicit_OR_existence.items():
        if value["is_there_an_implicit_OR"] and value["number_of_nodes_available_for_removal"] > best_node_reduction_found:
            best_node_reduction_found_key, best_node_reduction_found = key, value["number_of_nodes_available_for_removal"]
    if best_node_reduction_found_key is None:
        return G_copy
    cut = results_check_implicit_OR_existence[best_node_reduction_found_key]["cut"]
    cone = results_check_implicit_OR_existence[best_node_reduction_found_key]["cone"]
    return add_implicit_OR_to_dag_v2(G_copy, output_node, cut, cone)

class DRL3env(gym.Env):

    def __init__(
        self,
        max_nodes: int,
        graphs = None,
        *,
        shared_registry = None,
        registry_lock = None,
        store_every_new_graph = False,
        best_energy_across_workers = None,
        sampling_from_shared_registry = True,
        registry_read_only = False,
        max_registry_size = None,
        max_steps: int = 10,
        enable_full_graph_replacement = True,
        show_plots = False,
        log_info = True,
        strict_iso_check = False,
        initial_state_sampling_factor = 0,
        existing_keys = None,
        one_graph_per_hash_mode = True #If initial_state_sampling_factor is 0, and no has collisions 
    ):
        super().__init__()
        

        self.existing_keys = existing_keys
        self._shared_registry = shared_registry      
        self._registry_lock   = registry_lock
        self._max_registry_size = max_registry_size
        self._global_best_energy_across_workers = best_energy_across_workers
        self._one_graph_per_hash_mode = one_graph_per_hash_mode
        
        self._registry_read_only = registry_read_only
        
        #If shared_registry is used, check that there is a lock and a global value
        if self._shared_registry is not None:
            if self._registry_lock is None:
                raise ValueError("Registry lock is required to use the shared solutions across workers") 

            if self._global_best_energy_across_workers is None:
                raise ValueError("best_energy_across_workers is required to use the shared solutions across workers")   
        
        self.initial_state_sampling_factor = initial_state_sampling_factor
        self.sampling_from_shared_registry = sampling_from_shared_registry
        self.store_every_new_graph = store_every_new_graph
        self.strict_iso_check = strict_iso_check           
        self._owner_pid = os.getpid()
        self.show_plots = show_plots
        self.log_info   = log_info  
        
        self._best_energy_in_episode = float("inf")
        
        self._global_best_energy_in_worker = float("inf")          
        self.current_solution = None
        
        if self._shared_registry is not None:
            #shared registry, so graphs were already seeded to registry in main process
            
            if self.sampling_from_shared_registry:
                # sampling from the registry, so do not copy to self.initial_states
                self.initial_states = []
                self._seed_energies = []
                
            else:
                # No sampling from the registry, so we need to sample from self.initial_states
                self.initial_states   = [g.copy() for g in graphs]
                self._seed_energies = [energy_score(g, check_implicit_OR_existence_v3)[0] for g in self.initial_states]
                #self._global_best_energy_in_worker = min(self._seed_energies)             
        
        else:
            #No shared registry, so copying all graphs to initial states
            self.initial_states   = [g.copy() for g in graphs]
            self._seed_energies = [energy_score(g, check_implicit_OR_existence_v3)[0] for g in self.initial_states]
            #self._global_best_energy_in_worker = min(self._seed_energies)

        # Determine the logic functions used in this run 
        # if using shared registry, it was seeded already. 
        '''
        if self._shared_registry is not None:
            with (self._registry_lock):
                # sample only from registry (already contains the seeds)
                reg_items  = [item for bucket in self._shared_registry.values() for item in bucket]
                pool     = [orig for _, orig, _ in reg_items]

        else:
            pool = self.initial_states      
        self.existing_keys = {_compute_truth_key(g) for g in pool}
        print(f"Logic function keys in this run are {self.existing_keys}")   
        '''
                  
        self.enable_full_graph_replacement = enable_full_graph_replacement
        self.max_nodes   = max_nodes
        self.max_steps   = max_steps
        self.current_step_in_episode = 0
        self.all_episodes_in_worker_current_step = 0
        self.TERMINATE_ID = NUM_ACTIONS          
        self.action_space = spaces.Discrete(NUM_ACTIONS + 1)

        self.observation_space = spaces.Dict(
            {
                "node_features": spaces.Box(0, 1, shape=(max_nodes, 4), dtype=np.float32),
                "adj_matrix":    spaces.Box(0, 1, shape=(max_nodes, max_nodes), dtype=np.float32),
                "best_energy":   spaces.Box(1.0, np.inf, (1,), np.float32),
                "steps_left":    spaces.Box(0.0,float(self.max_steps),(1,), np.float32),                 
            }
        )
        
        self._mask_cache = None
        
        # Dict[int, List[Tuple[int, Tuple[int, ...], Tuple[int, int]]]] 
        self._subgraphs_for_action = defaultdict(list)
   
    def _permute_and_match(self, motif, key, *, first_only = False):
        inputs = [n for n in motif if motif.in_degree(n) == 0]
        if len(inputs) != key[0]:
            return None
        matches = []
        for perm in itertools.permutations(inputs):
            g2 = nx.relabel_nodes(motif, dict(zip(inputs, perm)), copy=True)
            if _compute_truth_key(g2) == key:
                if first_only:
                    return g2
                matches.append(g2)
        if not matches:
            return None
        # previous implementation: chosen = random.choice(matches)
        idx = self.np_random.integers(len(matches))
        chosen =  matches[idx]                 
        return chosen

    def _canonical_graph_transform(self, g):
        return _apply_implicit_or(g)          
    
    def _populate_registry_with_seeds(self):
        """Insert initial seed graphs into the shared registry (bucket layout)."""
        if self._shared_registry is None:
            return       
            
        with (self._registry_lock):
            
            # Only populate if the registry is empty
            if len(self._shared_registry):                
                return
            else:                
                self._global_best_energy_across_workers.value = min(self._seed_energies)
                for g_seed in self.initial_states:                    
                    canon = self._canonical_graph_transform(g_seed)
                    h     = _compute_hash(canon)
                    bucket = self._shared_registry.setdefault(h, [])
                    bucket.append((canon, g_seed.copy(), energy_score(g_seed, check_implicit_OR_existence_v3)[0]))
                    self._shared_registry[h] = bucket            

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        
        self.current_step_in_episode = 0
        self._mask_cache  = None
        
        if self.log_info:
            t0 = time.perf_counter()     
        
        # exponent (>1 biases towards smaller designs)
        p   = self.initial_state_sampling_factor        
        
        if p == 0 and self._one_graph_per_hash_mode:
            # Build sampling pool 
            if self.sampling_from_shared_registry and self._shared_registry is not None:
                
                keys = tuple(self._shared_registry.keys())
                h = keys[int(self.np_random.integers(len(keys)))]
                bucket = self._shared_registry[h]
                canon, orig, e = bucket[int(self.np_random.integers(len(bucket)))] 
                self.current_solution = orig.copy()                                 

            else:
                # sample only from the original seed set (no registry sampling)
                pool     = self.initial_states
                energies = self._seed_energies   
                
                if not pool:
                    raise RuntimeError("Initial_states, expected at the initial graph(s).")  
            
                idx     = self.np_random.choice(len(pool))
                self.current_solution = pool[idx].copy()    
        
        if p == 0 and not self._one_graph_per_hash_mode:
            # Build sampling pool 
            if self.sampling_from_shared_registry and self._shared_registry is not None:
                # sample only from registry (already contains the seeds)
                reg_items  = [item for bucket in self._shared_registry.values() for item in bucket]
                pool     = [orig for _, orig, _ in reg_items]
                energies = [e    for _, _,    e in reg_items]
                
                if not pool:          # this should never happen
                    raise RuntimeError(
                        "Shared registry is empty, expected at least the initial graph(s)."
                    )                    

            else:
                # sample only from the original seed set (no registry sampling)
                pool     = self.initial_states
                energies = self._seed_energies   
                
                if not pool:
                    raise RuntimeError("Initial_states, expected at the initial graph(s).")  
            
            # This one can be faster but not much, because the slowdown is from flattening the multiprocessing dict
            #idx = self.np_random.choice(len(pool))
            #self.current_solution = pool[idx].copy()  

            weights = [1.0 / (e ** p) for e in energies]
            # self.np_random draw proportional to the weights
            #self.current_solution = random.choices(pool, weights=weights, k=1)[0].copy()
            weights = np.asarray(weights, dtype=np.float64)
            prob    = weights / weights.sum()          # must sum to 1 for NumPy
            idx     = self.np_random.choice(len(pool), p=prob)
            self.current_solution = pool[idx].copy()            

        elif p > 0:
            # Build sampling pool 
            if self.sampling_from_shared_registry and self._shared_registry is not None:
                # sample only from registry (already contains the seeds)
                reg_items  = [item for bucket in self._shared_registry.values() for item in bucket]
                pool     = [orig for _, orig, _ in reg_items]
                energies = [e    for _, _,    e in reg_items]
                
                if not pool:          # this should never happen
                    raise RuntimeError(
                        "Shared registry is empty, expected at least the initial graph(s)."
                    )
            else:
                # sample only from the original seed set (no registry sampling)
                pool     = self.initial_states
                energies = self._seed_energies   
                
                if not pool:
                    raise RuntimeError("Initial_states, expected at the initial graph(s).")  
            
            # avoids division by zero in case energy changes meaning in the future
            #epsilon = 1e-6                                         
            # reciprocal-power weights
            weights = [1.0 / (e ** p) for e in energies]

            # self.np_random draw proportional to the weights
            #self.current_solution = random.choices(pool, weights=weights, k=1)[0].copy()
            weights = np.asarray(weights, dtype=np.float64)
            prob    = weights / weights.sum()          # must sum to 1 for NumPy
            idx     = self.np_random.choice(len(pool), p=prob)
            self.current_solution = pool[idx].copy()
        
        # randomize input permutation when multi-boolean
        #existing_keys = {_compute_truth_key(g) for g in pool}
        inputs = [n for n in self.current_solution if self.current_solution.in_degree(n) == 0]       
        perms = list(itertools.permutations(inputs))
        self.np_random.shuffle(perms)
        for perm in perms:                     # iterate without replacement
            mapping  = dict(zip(inputs, perm))
            g_perm   = nx.relabel_nodes(self.current_solution, mapping, copy=True)
            
            if _compute_truth_key(g_perm) in self.existing_keys: 
                self.current_solution = g_perm
                break # stop at the first valid permutation                                

        #Set the initial energy for this episode 
        self._best_energy_in_episode, _ = energy_score(self.current_solution, check_implicit_OR_existence_v3)

        #This is for best one this worker has used or found, separate from the global best across workers
        self._global_best_energy_in_worker    = min(self._global_best_energy_in_worker, self._best_energy_in_episode) 

        if self.log_info:
            print("Reset function took", (time.perf_counter() - t0)*1e3, "ms")

        return self._build_obs(), {}


    def action_masks(self):
        if self.log_info:
            t_all = time.perf_counter()            


        mask = np.zeros(NUM_ACTIONS + 1, dtype=bool)
        self._subgraphs_for_action.clear()
        tt_counter: DefaultDict[Tuple[int, int], list] = defaultdict(list)

        # primary-inputs
        source_nodes = {n for n, deg in self.current_solution.in_degree() if deg == 0}

        # outputs (terminal nodes)
        output_nodes = {n for n, deg in self.current_solution.out_degree() if deg == 0}

        # predecessors of the output node(s), to be skiped when full-graph replacement is disabled
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

            if (tgt in source_nodes) or (tgt in output_nodes) or (tgt in pred_of_output):
                continue

            for cut in exhaustive_cut_enumeration_dag(self.current_solution, 4, tgt, filter_redundant=True):
                if not is_fanout_free_standalone(self.current_solution, tgt, cut):
                    continue

                sg = generate_subgraph(self.current_solution, tgt, cut, draw=False)
                if len([n for n in sg if sg.in_degree(n) == 0]) != len(cut):
                    continue

                key = _compute_truth_key(sg)
                for action_id in TTABLE_TO_ACTIONS.get(key, []):
                    mask[action_id] = True
                    self._subgraphs_for_action[action_id].append((tgt, tuple(cut), key))
                tt_counter[key].append((tgt, tuple(cut)))

        if self.log_info:
            dt = time.perf_counter() - t_enum
            print(f"Cuts computation and mapping took {dt:.3f} s")

        if self.log_info:
            t_dbg = time.perf_counter()

            uniq = sum(1 for v in tt_counter.values() if len(v) == 1)
            for (n_in, tt_int), items in tt_counter.items():
                if len(items) > 1:
                    print(f" TT {tt_int} ({n_in}-in) appears {len(items)} times.")
            print(f"{uniq} truth tables appear once among {len(tt_counter)} candidates.")

            dt = (time.perf_counter() - t_dbg) * 1e3
            print(f"Stats-print took {dt:.2f} ms")



        if mask[: self.TERMINATE_ID].sum() == 0:
            if self.log_info:
                print("Nothing left to rewrite")
            mask[self.TERMINATE_ID] = 1

        self._mask_cache = mask

        if self.log_info:
            dt = time.perf_counter() - t_all
            print(f"action_masks()took {dt:.3f} s")

        
        if not mask.any():
            raise RuntimeError("Action mask with no valid actions")
        
        return mask
    

    def step(self, action):
        """
        One environment step.
        - Applies the chosen rewrite (if valid).
        - Tracks both episode-best and global-best energies.
        - Gives a reward only at episode end:   100 / best_energy_in_episode.
        """
        
        
        self.current_step_in_episode    += 1
        self.all_episodes_in_worker_current_step += 1
        
        # Termination 
        done  = self.current_step_in_episode >= self.max_steps   

        if self._mask_cache is None:
            self.action_masks()           

        # Terminate episode
        if action == self.TERMINATE_ID:            

            # Terminate now
            done    = True
            reward  = (100.0 / self._best_energy_in_episode) if self._best_energy_in_episode != 0 else 0.0
            
                        
            if self.log_info:
                print(f"Terminate action selected  self._best_energy_in_episode={self._best_energy_in_episode:.3f} "
                      f"Reward={reward:.5f}")
            return self._build_obs(), reward, done, False, {}
        
        # Terminate step
        if action not in self._subgraphs_for_action:     
            # should not happen with a correct mask
            if self.log_info:
                print(f"The action is not in self._subgraphs_for_action. Inspect the masking.")                
            return self._build_obs(), 0.0, done, False, {}       

        # Apply the rewrite 
        tt_before = calculate_truth_table_v2(self.current_solution)
        
        #Previous: tgt, cut, key = random.choice(self._subgraphs_for_action[action])
        idx = self.np_random.integers(len(self._subgraphs_for_action[action]))
        tgt, cut, key = self._subgraphs_for_action[action][idx]

        # Terminate step
        repl = self._permute_and_match(UNIQUE_GRAPHS[action], key)
        if repl is None:
            if self.log_info:
                print(f"Permutation unsolved in the environment. Inspect the masking.") 
            return self._build_obs(), 0.0, done, False, {}

        cand    = generate_subgraph(self.current_solution, tgt, cut, draw=False)
        new_sol = substitute_subgraph(self.current_solution, cand, repl)        
        
        # Terminate step
        new_size = len(new_sol)
        if new_size > self.max_nodes:
            if self.log_info:
                print(f"Rewrite skipped because size {new_size} > limit {self.max_nodes}")
            # could return a small negative reward instead
            return self._build_obs(), 0.0, done, False, {}
                
        if calculate_truth_table_v2(new_sol) != tt_before:
            raise ValueError("The logic function changed after rewrite")

        # Accept the rewrite 
        self.current_solution = new_sol 
        
        # Invalidate for next step
        self._mask_cache = None           
        
        # Compute new current energy 
        current_energy, _ = energy_score(self.current_solution, check_implicit_OR_existence_v3)

        # Update best energy in the episode (always needed)
        self._best_energy_in_episode = min(self._best_energy_in_episode, current_energy)
        
        # Update global best across episodes
        self._global_best_energy_in_worker    = min(self._global_best_energy_in_worker, current_energy) 

        # Case A: no shared registry used
        if self._shared_registry is None:
            #Optional for routines when no shared registry is used
            pass
        # Case B: shared registry used 
        else:

             with (self._registry_lock):
        
                run_store_graph_routine_flag = ((self.store_every_new_graph or current_energy <= self._global_best_energy_across_workers.value) and (not self._registry_read_only))
            
                # Check that the registry does not exceed the size specified 
                if (run_store_graph_routine_flag and self._max_registry_size is not None and len(self._shared_registry) >= self._max_registry_size):
                    run_store_graph_routine_flag = False
        
                if run_store_graph_routine_flag:              
                    
                    # Compute canonical graph and compute hash
                    canon = self._canonical_graph_transform(self.current_solution)
                    h     = _compute_hash(canon)

                    bucket = self._shared_registry.setdefault(h, [])

                    # Case B.1: strict iso check 
                    if self.strict_iso_check:
                        already_there = any(nx.is_isomorphic(canon, cg_prev) for cg_prev, _, _ in bucket)
                        if not already_there:
                            bucket.append((canon.copy(),self.current_solution.copy(),current_energy))
                            self._shared_registry[h] = bucket
                            
                            # Update global best value across workers as needed
                            if current_energy < self._global_best_energy_across_workers.value:
                                self._global_best_energy_across_workers.value = current_energy                                 
                            

                    # Case B.2: no strict iso check (but still pretty good) 
                    else:
                        # store only once per hash
                        if not bucket:                       # bucket empty, so first entry
                            bucket.append((canon.copy(), self.current_solution.copy(), current_energy))
                            self._shared_registry[h] = bucket                            
                                    
                            # Update global best value across workers as needed
                            if current_energy < self._global_best_energy_across_workers.value:
                                self._global_best_energy_across_workers.value = current_energy                                 

                        if self.log_info:          
                            print("registry size =", sum(len(b) for b in self._shared_registry.values()))                        

                # Update local worker with the best enery across workers --> because it has access to the shared registry                
                #self._global_best_energy_in_worker = self._global_best_energy_across_workers.value

        if self.show_plots:
            self._draw_graph(f"Step {self.current_step_in_episode}: size={len(self.current_solution)}")   
        
        reward = (100.0 / self._best_energy_in_episode) if done and self._best_energy_in_episode != 0 else 0.0

        if self.log_info:
            print(
                f"step={self.current_step_in_episode} done={done} "
                f"energy={current_energy:.3f} "
                f"best_ep={self._best_energy_in_episode:.3f} "
                f"reward={reward:.5f}"
            )
            
        if self.log_info and self._shared_registry is not None:
            print("registry size =", sum(len(b) for b in self._shared_registry.values()))
            
            
        #if done:
        #    self._save_optimal_graphs()            

        return self._build_obs(), reward, done, False, {}


    def _draw_graph(self, title: str):
        if not self.show_plots:
            return
        plt.figure(figsize=(4, 4))
        pos = nx.spring_layout(self.current_solution, seed=111)
        nx.draw(self.current_solution, pos, with_labels=True, node_size=300, font_size=8)
        plt.title(title)
        plt.show()


    def _build_obs(self):        
       
        #nodes = list(nx.topological_sort(self.current_solution))
        nodes = sorted(self.current_solution.nodes())
        adj_dense = nx.adjacency_matrix(self.current_solution, nodelist=nodes).toarray()
        adj = np.asarray(resize_matrix(adj_dense, self.max_nodes), dtype=np.float32)
        feats = np.asarray(
            generate_one_hot_features_from_adj(
                nx.adjacency_matrix(self.current_solution, nodelist=nodes),
                pad_size=self.max_nodes,
            ),
            dtype=np.float32,
        )
        
        # shape checks 
        assert adj.shape   == (self.max_nodes, self.max_nodes), f"adj shape {adj.shape} ≠ ({self.max_nodes}, {self.max_nodes})"
        assert feats.shape == (self.max_nodes, 4), f"feats shape {feats.shape} ≠ ({self.max_nodes}, 4)"

        # checks for dtype and value range 
        assert adj.dtype   == np.float32 and feats.dtype == np.float32
        assert np.isin(adj,  (0.0, 1.0)).all(), "adjacency contains non-binary entries"
        assert np.isin(feats, (0.0, 1.0)).all(), "node features not one-hot/padded"     
        
        #scalars computation        
        be = self._best_energy_in_episode
        assert np.isfinite(be), "best_energy is not finite"
        assert be >= 1.0, f"best_energy {be} violates Box lower bound 1.0"
        best_energy_in_episode_arr = np.asarray([np.float32(be)], dtype=np.float32)
        
        # steps_left must be in [0, max_steps]
        steps_left_val = self.max_steps - self.current_step_in_episode
        assert 0 <= steps_left_val <= self.max_steps, f"steps_left out of bounds: {steps_left_val}"
        steps_left_arr = np.asarray([np.float32(steps_left_val)], dtype=np.float32)
                  
        #return {"node_features": feats, "adj_matrix": adj}
        return {
            "node_features": feats,
            "adj_matrix":    adj,
            "best_energy":   best_energy_in_episode_arr,
            "steps_left":    steps_left_arr,
        }        

    def close(self):
        # make sure we always dump the best graphs, even on SIGINT/normal close
        super().close()
        #self._save_optimal_graphs()
        
    def render(self):
        if self.show_plots:
            self._draw_graph("Current topology")
