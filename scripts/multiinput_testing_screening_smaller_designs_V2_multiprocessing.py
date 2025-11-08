# %%
from dgd.utils.utils5 import *
import pickle
import pandas as pd
import json
import numpy as np
import time
import matplotlib.pyplot as plt
import pickle
import csv
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import functools, pickle, traceback, multiprocessing as mp

# %%
# ---------- worker.py (or just a top-level function in the same file) ----------
@functools.lru_cache(maxsize=1)
def lib4():
    path = "/home/gridsan/spalacios/DRL1/supercloud-testing/ABC-and-PPO-testing1/4_input_precomputed_graphs/graphs_library_4_input_ABC.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def optimise_single_key(value_decimal):
    try:
        G = lib4()[value_decimal]          # loaded once per process
        initial = G.number_of_nodes() - 5

        final_graphs = iterative_optimization_without_persisting_OR(
            G.copy(), 10, 1000, 10,
            precomputed_graphs_1_input, graphs_library_1_input,
            precomputed_graphs_2_input, graphs_library_2_input,
            precomputed_graphs_3_input, graphs_library_3_input,
            precomputed_graphs_4_input, graphs_library_4_input,
            plot=False,
        )

        lowest = min(g.number_of_nodes() - 5 for g in final_graphs)
        payload = {
            "initial_num_gates": initial,
            "lowest_gates":      lowest,
            "final_graphs":      final_graphs,
        }
        return (value_decimal, payload, None)   # success

    except Exception as e:
        return (value_decimal, None, traceback.format_exc())


def is_fanout_free(G, target_node, cut):
    """Check if the cut is fanout-free"""

    #Get all the nodes in G that are both ancestors of target_node and decendants of nodes 'cut'. The name of that set of nodes is 'cone'. 
    # Step 1: Find ancestors of target_node
    ancestors_of_target = nx.ancestors(G, target_node)
    # Step 2: Find descendants of all nodes in 'cut'
    descendants_of_cut = set()
    for node in cut:
        descendants_of_cut.update(nx.descendants(G, node))        
    # Step 3: Determine the 'cone'
    cone = ancestors_of_target.intersection(descendants_of_cut)

    #In G, check that nodes in 'cone' have no edges to a node not in the cone, but allow edges to target_node.
    # Step 4: Check that nodes in 'cone' are only directly connected to nodes within 'cone', but allow edges to target_node.
    fanout_free = True
    for node in cone:
        # Check outgoing edges from each node in the cone
        for successor in G.successors(node):
            if successor not in cone and successor != target_node:
                # If a successor is not in the cone and is not the target_node, the cut is not fanout-free
                fanout_free = False
                #print(f"Node {node} has an outgoing edge to node {successor}, which is outside the cone and is not the target_node.")
                return fanout_free  # Can immediately return as we found a violation

    return fanout_free

def generate_random_initial_states_multi_input(G_list, number_of_states, precomputed_graphs_1_input, graphs_library_1_input, precomputed_graphs_2_input, graphs_library_2_input, precomputed_graphs_3_input, graphs_library_3_input, precomputed_graphs_4_input, graphs_library_4_input):
    #random_initial_states_list = [G.copy()]
    random_initial_states_list = G_list
    
    # Initialize the progress bar
    pbar = tqdm(total=number_of_states, desc="Generating Random Initial States", unit="state")
    
    while len(random_initial_states_list) < number_of_states + 1:
        current_solution = random.choice(random_initial_states_list)
    
        # Choose a random target node, excluding specified nodes
        excluded_nodes = [0, 1, 2, 3]  # Assuming these are input nodes
        target_node = random.choice([node for node in current_solution.nodes() if node not in excluded_nodes])

        select_cut_size_n = random.choice([1, 2, 3, 4])

        #feasible_cuts = find_feasible_cuts(current_solution, target_node, max_cut_size=select_cut_size_n, filter_redundant=True)
        feasible_cuts = exhaustive_cut_enumeration_dag(current_solution, select_cut_size_n, target_node = target_node, filter_redundant=True)
        feasible_cuts_of_size_n = [cut for cut in feasible_cuts if len(cut) == select_cut_size_n]
        

        feasible_cuts_of_size_n = [cut for cut in feasible_cuts_of_size_n if is_fanout_free(current_solution, target_node, cut) == True]

        if not feasible_cuts_of_size_n:
            continue  # Skip iteration if no suitable cut found

        cut = random.choice(feasible_cuts_of_size_n)
        subgraph = generate_subgraph(current_solution, target_node, cut, draw=False)
        
        # Filter out the trivial cut (subgraph is only cut + target_node)
        cut_set = set(cut)
        cut_set.add(target_node)
        if len(subgraph.nodes()) == len(cut_set):
            continue

        # Calculate truth table and attempt to get a replacement graph from library
        if len([node for node in subgraph.nodes() if subgraph.in_degree(node) == 0]) == len(cut):
            truth_table = calculate_truth_table(subgraph)
            binary_str = ''.join(str(output[0]) for inputs, output in sorted(truth_table.items()))
            truth_table_int = int(binary_str, 2)

            if select_cut_size_n == 1:
                # Check if the truth table integer is in the graphs library
                if truth_table_int not in precomputed_graphs_1_input:
                    continue  # Skip the rest of the iteration if no replacement graph is found

                associated_graphs = precomputed_graphs_1_input[truth_table_int]
                selected_graph_index = random.choice(associated_graphs)
                replacement_graph = graphs_library_1_input[selected_graph_index]  

            elif select_cut_size_n == 2:
                # Check if the truth table integer is in the graphs library
                if truth_table_int not in precomputed_graphs_2_input:
                    continue  # Skip the rest of the iteration if no replacement graph is found

                associated_graphs = precomputed_graphs_2_input[truth_table_int]
                selected_graph_index = random.choice(associated_graphs)
                replacement_graph = graphs_library_2_input[selected_graph_index]  

            elif select_cut_size_n == 3:
                #print("select_cut_size_n: ", select_cut_size_n)
                # Check if the truth table integer is in the graphs library
                if truth_table_int not in precomputed_graphs_3_input:
                    continue  # Skip the rest of the iteration if no replacement graph is found

                associated_graphs = precomputed_graphs_3_input[truth_table_int]
                selected_graph_index = random.choice(associated_graphs)
                replacement_graph = graphs_library_3_input[selected_graph_index]        
                

            elif select_cut_size_n == 4:
                #print("select_cut_size_n: ", select_cut_size_n)
                # Check if the truth table integer is in the graphs library
                if truth_table_int not in precomputed_graphs_4_input:
                    continue  # Skip the rest of the iteration if no replacement graph is found

                associated_graphs = precomputed_graphs_4_input[truth_table_int]
                selected_graph_index = random.choice(associated_graphs)
                replacement_graph = graphs_library_4_input[selected_graph_index]                           
                

            if calculate_truth_table(subgraph) != calculate_truth_table(replacement_graph):
                # If the truth tables do not match, raise an exception
                raise ValueError("The truth tables of subgraph and replacement do not match.")

            new_solution = substitute_subgraph(current_solution, subgraph, replacement_graph)

            node_with_more_than_two_incoming_edges = any(new_solution.in_degree(node) > 2 for node in new_solution.nodes()) #check if any gate has more than 2 inputs

            if node_with_more_than_two_incoming_edges:
                print("There is at least one node with more than two incoming edges.")
                print("Nodes with more than 2 incoming edges:", node_with_more_than_two_incoming_edges)
                print("target_node", target_node)
                visualize_graph_rewriting(current_solution, highlight_nodes=cut, title="current_solution")
                visualize_graph_rewriting(new_solution, highlight_nodes=cut, title="new_solution")
                visualize_graph_rewriting(subgraph, highlight_nodes=cut, title="subgraph")
                visualize_graph_rewriting(replacement_graph, highlight_nodes=cut, title="replacement_graph")

            # Check if new solution is 3-input 1-output
            entry_nodes = [node for node in new_solution.nodes() if new_solution.in_degree(node) == 0]
            exit_nodes = [node for node in new_solution.nodes() if new_solution.out_degree(node) == 0]

            # Check if the counts match the specified criteria
            has_three_entry_nodes = len(entry_nodes) == 4
            has_one_exit_node = len(exit_nodes) == 1

            if not has_three_entry_nodes or not has_one_exit_node:
                print("BUG!")
                print("entry_nodes: ", entry_nodes)
                print("exit_nodes: ", exit_nodes)

            if calculate_truth_table_v2(current_solution) != calculate_truth_table_v2(new_solution):
                # If the truth tables do not match, raise an exception
                print("target_node", target_node)
                print("cut", cut)
                raise ValueError("The truth tables before and after rewriting do not match.")

            # Append new_solution only if it is not isomorphic to any graph in the list
            if not any(nx.is_isomorphic(new_solution, existing_graph) for existing_graph in random_initial_states_list):
                random_initial_states_list.append(new_solution)
                pbar.update(1)  # Update the progress bar
    
    pbar.close()  # Close the progress bar when done
    return random_initial_states_list  # Return the list of random initial states



#This is a large library with pruning, all ABC graphs, no additional keys
with open('/home/gridsan/spalacios/DRL1/supercloud-testing/ABC-and-PPO-testing1/graphs_library_4_input_4_3_pruned.pkl', 'rb') as file:
    graphs_library_4_input = pickle.load(file)

# Load precomputed_graphs_4_input
with open('/home/gridsan/spalacios/DRL1/supercloud-testing/ABC-and-PPO-testing1/precomputed_graphs_4_input_4_3_pruned.pkl', 'rb') as file:
    precomputed_graphs_4_input = pickle.load(file)  
        

# Load graphs_library_3_input
with open('/home/gridsan/spalacios/DRL1/supercloud-testing/ABC-and-PPO-testing1/graphs_library_3_input_3_7.pkl', 'rb') as file:
    graphs_library_3_input = pickle.load(file)

# Load precomputed_graphs_3_input
with open('/home/gridsan/spalacios/DRL1/supercloud-testing/ABC-and-PPO-testing1/precomputed_graphs_3_input_3_7.pkl', 'rb') as file:
    precomputed_graphs_3_input = pickle.load(file)


# Load graphs_library_2_input
with open('/home/gridsan/spalacios/DRL1/supercloud-testing/ABC-and-PPO-testing1/graphs_library_2_input.pkl', 'rb') as file:
    graphs_library_2_input = pickle.load(file)

# Load precomputed_graphs_2_input
with open('/home/gridsan/spalacios/DRL1/supercloud-testing/ABC-and-PPO-testing1/precomputed_graphs_2_input.pkl', 'rb') as file:
    precomputed_graphs_2_input = pickle.load(file)


# Load graphs_library_1_input
with open('/home/gridsan/spalacios/DRL1/supercloud-testing/ABC-and-PPO-testing1/graphs_library_1_input.pkl', 'rb') as file:
    graphs_library_1_input = pickle.load(file)

# Load precomputed_graphs_1_input
with open('/home/gridsan/spalacios/DRL1/supercloud-testing/ABC-and-PPO-testing1/precomputed_graphs_1_input.pkl', 'rb') as file:
    precomputed_graphs_1_input = pickle.load(file)


# Load NIGs_unoptimized_library_3_input_1_output
with open('/home/gridsan/spalacios/DRL1/supercloud-testing/ABC-and-PPO-testing1/NIGs_unoptimized_library_3_input_1_output.pkl', 'rb') as file:
    NIGs_unoptimized_library_3_input_1_output = pickle.load(file)


# %%
def print_graph_structure(G):
    """
    Print adjacency list and attributes of a NetworkX graph.
    
    Parameters:
    G (nx.Graph): A NetworkX graph object
    """
    # Print adjacency list
    print("\n=== Adjacency List ===")
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        print(f"Node {node} -> {neighbors}")
    
    # Print node attributes
    print("\n=== Node Attributes ===")
    for node, attrs in G.nodes(data=True):
        print(f"Node {node}:", attrs if attrs else "No attributes")
    
    # Print edge attributes
    print("\n=== Edge Attributes ===")
    for u, v, attrs in G.edges(data=True):
        print(f"Edge {u}-{v}:", attrs if attrs else "No attributes")

def plot_graph(G, layout='spring', node_color='lightblue', node_size=500, 
               with_labels=True, font_size=10, edge_color='gray', 
               title='Network Graph'):
    """
    Plot a NetworkX graph with customizable options.
    
    Parameters:
    G (nx.Graph): NetworkX graph
    layout (str): Layout type ('spring', 'circular', 'random', 'shell')
    node_color (str): Color of nodes
    node_size (int): Size of nodes
    with_labels (bool): Whether to show node labels
    font_size (int): Size of label font
    edge_color (str): Color of edges
    title (str): Title of the plot
    """
    # Set up the figure
    plt.figure(figsize=(10, 8))
    
    # Choose layout
    if layout == 'spring':
        pos = nx.spring_layout(G)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'random':
        pos = nx.random_layout(G)
    elif layout == 'shell':
        pos = nx.shell_layout(G)
    else:
        pos = nx.spring_layout(G)  # default to spring layout
    
    # Draw the graph
    nx.draw(G, pos,
            node_color=node_color,
            node_size=node_size,
            with_labels=with_labels,
            font_size=font_size,
            edge_color=edge_color,
            font_weight='bold')
    
    # Draw edge labels if they exist
    edge_labels = nx.get_edge_attributes(G, 'weight')
    if edge_labels:
        nx.draw_networkx_edge_labels(G, pos, edge_labels)
    
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    
    return plt        
        
def plot_directed_graph(G, **kwargs):
    plt = plot_graph(G, **kwargs)
    # Get the axes object
    ax = plt.gca()
    # Clear the current plot
    ax.clear()
    # Draw the directed graph with arrows
    nx.draw(G, nx.spring_layout(G), 
           with_labels=True,
           node_color=kwargs.get('node_color', 'lightblue'),
           node_size=kwargs.get('node_size', 500),
           arrows=True,  # This adds the direction arrows
           arrowsize=20)
    plt.title(kwargs.get('title', 'Directed Graph'))
    return plt

#print_graph_structure(G)
#plot_directed_graph(G)


# %% [markdown]
# ### 4-input

# %%
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

# %% [markdown]
# ### Top performing approach ABC many initial graphs

# %%


# %%
# Load graphs_library_4_input_ABC
#with open('/home/gridsan/spalacios/DRL1/supercloud-testing/ABC-and-PPO-testing1/4_input_precomputed_graphs/graphs_library_4_input_ABC.pkl', 'rb') as file:
    graphs_library_4_input_ABC = pickle.load(file)

# Verify the loaded data
#print(type(graphs_library_4_input_ABC))  # Should be a dict
#print(len(graphs_library_4_input_ABC))  # Number of elements in the dictionary

# %%
import networkx as nx
from tqdm import tqdm
import matplotlib.pyplot as plt

def iterative_optimization_without_persisting_OR(
    G,
    number_of_iterations,
    number_of_graphs_per_iteration,
    N_best,
    precomputed_graphs_1_input,
    graphs_library_1_input,
    precomputed_graphs_2_input,
    graphs_library_2_input,
    precomputed_graphs_3_input,
    graphs_library_3_input,
    precomputed_graphs_4_input,
    graphs_library_4_input,
    plot = True
):
    """
    Perform multiple iterations of the optimization flow:
      1) Generate random states from the current set of graphs.
      2) Check for implicit OR rewriting (only for computing energy).
      3) Compute energy, store the best N *original* graphs (unrewritten).
      4) Repeat.

    We do NOT feed the rewritten graphs back into generate_random_initial_states_multi_input.
    We only do rewriting to measure potential improvement in "energy".

    Additionally, track and plot the best energy at each iteration.

    Parameters
    ----------
    G : nx.DiGraph
        The initial directed acyclic graph (DAG).
    number_of_iterations : int
        How many times to repeat the optimization cycle.
    number_of_graphs_per_iteration : int
        How many new graphs to generate in each iteration.
    N_best : int
        How many best (lowest-energy) graphs to keep for the next iteration.
    precomputed_graphs_... : dict
        Libraries for subgraph replacement.
    graphs_library_... : dict
        Corresponding stored graphs for subgraph replacement.

    Returns
    -------
    list
        The final set of best original graphs (unrewritten).
    """

    # Start with a single-graph list
    current_graphs = [G.copy()]

    # List to store the best energy at each iteration for plotting
    best_energies = []

    for iteration in range(number_of_iterations):
        print(f"\n=== Iteration {iteration+1} / {number_of_iterations} ===")

        # 1) Generate new states from the current set
        #    NOTE: your function should already return the seeds plus new solutions
        all_new_graphs = generate_random_initial_states_multi_input(
            current_graphs,
            number_of_graphs_per_iteration,
            precomputed_graphs_1_input,
            graphs_library_1_input,
            precomputed_graphs_2_input,
            graphs_library_2_input,
            precomputed_graphs_3_input,
            graphs_library_3_input,
            precomputed_graphs_4_input,
            graphs_library_4_input
        )

        # Because generate_random_initial_states_multi_input already includes
        # the original 'current_graphs', we do NOT need to add them again.
        candidate_graphs_original = all_new_graphs

        # 2) For each candidate, apply OR rewriting *only* to compute energy
        energy_evaluations = []
        #for g_candidate in tqdm(candidate_graphs_original, desc="Evaluating Implicit OR"):
        for g_candidate in candidate_graphs_original:    

            # Identify output node (assuming 1)
            exit_nodes = [n for n in g_candidate.nodes() if g_candidate.out_degree(n) == 0]
            if not exit_nodes:
                energy_evaluations.append((g_candidate, float('inf')))
                continue

            output_node = exit_nodes[0]
            size_input_to_OR_gate = 2

            # Check potential implicit OR
            implicit_OR_results = check_implicit_OR_existence_v2(
                g_candidate, output_node, size_input_to_OR_gate
            )

            max_removal = 0
            max_implicit_OR_key = None

            for key, value in implicit_OR_results.items():
                if (value['is_there_an_implicit_OR'] and 
                    value['number_of_nodes_available_for_removal'] > max_removal):
                    max_removal = value['number_of_nodes_available_for_removal']
                    max_implicit_OR_key = key

            # Temporarily rewrite (for energy measurement only)
            if max_implicit_OR_key is not None:
                cut  = implicit_OR_results[max_implicit_OR_key]['cut']
                cone = implicit_OR_results[max_implicit_OR_key]['cone']
                g_rewritten = add_implicit_OR_to_dag_v2(
                    g_candidate, output_node, cut, cone
                )
                # example energy measure: number_of_nodes() minus 4 input + 1 output
                energy_value = len(g_rewritten.nodes()) - 4 - 1
            else:
                energy_value = len(g_candidate.nodes()) - 4 - 1

            # Store (original graph, energy_of_temporary_rewritten)
            energy_evaluations.append((g_candidate, energy_value))

        # 3) Sort by energy and keep the best N original graphs
        energy_evaluations.sort(key=lambda x: x[1])  # ascending by energy
        best_pairs = energy_evaluations[:N_best]

        # Record best energy for plotting
        best_energy_this_iter = best_pairs[0][1]
        best_energies.append(best_energy_this_iter)

        # Replace current_graphs with the best N for the next iteration
        current_graphs = [p[0] for p in best_pairs]

        
    # Finally, plot best energy vs. iteration
    if plot:
        plt.figure(figsize=(6,4))
        plt.plot(range(1, number_of_iterations+1), best_energies, marker='o')
        plt.xlabel('Iteration')
        plt.ylabel('Best Energy')
        plt.title('Best Graph Energy vs. Iteration')
        plt.grid(True)
        plt.show()

    # Return final best set
    return current_graphs
        



# %%
#load 
with open("/home/gridsan/spalacios/Designing complex biological circuits with deep neural networks/dgd/tests/one_index_per_permutation_class_in_graphs_library_4_input_ABC.pkl", "rb") as f:
    n_values = pickle.load(f)
    
# %%

if __name__ == "__main__":
    mp.set_start_method("fork", force=True)     # safe on Linux

    ids = [n for n in n_values if n > 0]
    workers = min(30, mp.cpu_count())           # match SBATCH cpus
    results, errors = {}, {}

    with ProcessPoolExecutor(max_workers=workers) as pool:
        fut_to_id = {pool.submit(optimise_single_key, vid): vid for vid in ids}
        for fut in as_completed(fut_to_id):
            vid, payload, err = fut.result()
            if err:
                errors[vid] = err
                print(f"[!] key {vid} failed")
            else:
                results[vid] = payload
                if len(results) % 10 == 0:
                    print(f"[{len(results)}/{len(ids)}] done", flush=True)

    print(f"✓ completed {len(results)} keys   ✗ {len(errors)} errors")
    for vid, tb in errors.items():
        print(f"--- key {vid} traceback ---\n{tb}")

    # ---------- 1) Pickle the full `results` dict ---------- #
    out_dir = Path("screening_for_keys_where_ABC_is_outperformed")
    out_dir.mkdir(exist_ok=True)          # create a folder if it isn’t there
    pickle_path = out_dir / "screening_for_keys_where_ABC_is_outperformed_initial_num_gates_all_keys_1perpermutationclass.pkl"

    with open(pickle_path, "wb") as f:
        pickle.dump(results, f)

    print(f"[✓] Saved full results dict → {pickle_path.resolve()}")

    # ---------- 2) Write the summary CSV, now with Δ column ---------- #
    csv_path = out_dir / "screening_for_keys_where_ABC_is_outperformed_initial_num_gates6_all_keys_1perpermutationclass.csv"
    fieldnames = ["value_decimal", "initial_num_gates", "lowest_gates", "delta_gates"]  # added delta_gates

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for value_decimal, payload in results.items():
            initial_g = payload["initial_num_gates"]
            lowest_g  = payload["lowest_gates"]
            delta_g   = initial_g - lowest_g           # positive means improvement

            writer.writerow(
                {
                    "value_decimal":   value_decimal,
                    "initial_num_gates": initial_g,
                    "lowest_gates":     lowest_g,
                    "delta_gates":      delta_g,
                }
            )

    print(f"[✓] Saved summary CSV     → {csv_path.resolve()}")




