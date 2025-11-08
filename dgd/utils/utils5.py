# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

#
# This is a significant update, updating the system to handle an arbitrary number of inputs
#

# +
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
import json


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

def generate_random_initial_states_multi_input(G, number_of_states, precomputed_graphs_1_input, graphs_library_1_input, precomputed_graphs_2_input, graphs_library_2_input, precomputed_graphs_3_input, graphs_library_3_input):
    random_initial_states_list = [G.copy()]
    
    # Initialize the progress bar
    pbar = tqdm(total=number_of_states, desc="Generating Random Initial States", unit="state")
    
    while len(random_initial_states_list) < number_of_states + 1:
        current_solution = random.choice(random_initial_states_list)
    
        # Choose a random target node, excluding specified nodes
        excluded_nodes = [0, 1, 2, 3]  # Assuming these are input nodes
        target_node = random.choice([node for node in current_solution.nodes() if node not in excluded_nodes])

        select_cut_size_n = random.choice([1, 2, 3])

        feasible_cuts = find_feasible_cuts(current_solution, target_node, max_cut_size=select_cut_size_n, filter_redundant=True)
        feasible_cuts_of_size_n = [cut for cut in feasible_cuts if len(cut) == select_cut_size_n]

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
                # Check if the truth table integer is in the graphs library
                if truth_table_int not in precomputed_graphs_3_input:
                    continue  # Skip the rest of the iteration if no replacement graph is found

                associated_graphs = precomputed_graphs_3_input[truth_table_int]
                selected_graph_index = random.choice(associated_graphs)
                replacement_graph = graphs_library_3_input[selected_graph_index]                

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

            if calculate_truth_table(current_solution) != calculate_truth_table(new_solution):
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

def generate_random_initial_states_multi_input_v2(G, number_of_states, precomputed_graphs_1_input, graphs_library_1_input, precomputed_graphs_2_input, graphs_library_2_input, precomputed_graphs_3_input, graphs_library_3_input, precomputed_graphs_4_input, graphs_library_4_input):
    random_initial_states_list = [G.copy()]
    #random_initial_states_list = G_list
    
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




def resize_matrix(adj_matrix, final_size=200):
    """
    Resize a given matrix to a specific size, filling in with zeroes.

    :param adj_matrix: List of lists representing the original matrix.
    :param final_size: The desired size of the matrix (final_size x final_size).
    :return: Resized matrix as a list of lists.
    """
    # Create a final_size x final_size matrix filled with zeroes
    resized_matrix = [[0 for _ in range(final_size)] for _ in range(final_size)]

    # Copy the values from the original matrix to the new matrix
    for i in range(min(len(adj_matrix), final_size)):
        for j in range(min(len(adj_matrix[i]), final_size)):
            resized_matrix[i][j] = adj_matrix[i][j]

    return resized_matrix


def generate_one_hot_features_from_adj(adj_matrix_sparse, pad_size=None):
    # Convert the sparse adjacency matrix to a dense format
    adj_matrix = adj_matrix_sparse.toarray()
    
    # Number of nodes
    num_nodes = adj_matrix.shape[0]
    
    # Initialize a feature matrix with zeros
    num_classes = 4  # Four classes as described
    features = np.zeros((num_nodes, num_classes), dtype=np.float32)
    
    # Calculate in-degree (sum of each column) and out-degree (sum of each row)
    in_degree = np.sum(adj_matrix, axis=0)  # In-degree is the column sum
    out_degree = np.sum(adj_matrix, axis=1)  # Out-degree is the row sum

    for idx in range(num_nodes):
        # Assign class based on the input/output criteria
        if out_degree[idx] == 0:
            features[idx, 3] = 1  # Class 4: No outputs
        elif in_degree[idx] == 0:
            features[idx, 2] = 1  # Class 3: No inputs
        elif in_degree[idx] == 1 and out_degree[idx] > 0:
            features[idx, 0] = 1  # Class 1: One input and at least one output
        elif in_degree[idx] == 2 and out_degree[idx] > 0:
            features[idx, 1] = 1  # Class 2: Two inputs and at least one output

    # Padding logic
    if pad_size is not None and pad_size > num_nodes:
        # Pad the matrix with zeros to the desired pad_size
        padded_features = np.zeros((pad_size, num_classes), dtype=np.float32)
        padded_features[:num_nodes, :] = features
        return padded_features
    else:
        # Return original matrix if no padding is needed or pad_size <= num_nodes
        return features


def assign_one_hot_encoded_features(G, one_hot_encoding_features, node_representation_indices):
    """
    This function takes in G generated using assign_representations_with_io_nodes_3 and adds one more atrtibute, which is 
    a one-hot encoded vector according to one_hot_encoding_features
    """
    # Create the DAG from the adjacency matrix
    #G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
    
    # Remove disconnected nodes (nodes with neither incoming nor outgoing edges)
    #disconnected_nodes = [node for node in G.nodes() if G.in_degree(node) == 0 and G.out_degree(node) == 0]
    #G.remove_nodes_from(disconnected_nodes)
    
    # Identify input and output nodes
    input_nodes = [node for node in G.nodes() if G.in_degree(node) == 0]
    output_nodes = [node for node in G.nodes() if G.out_degree(node) == 0]
    
    # Filter nodes that are neither inputs nor outputs
    non_io_nodes = [node for node in G.nodes() if node not in input_nodes and node not in output_nodes]
    
    # Ensure the length of representation indices matches the number of non-io nodes
    if len(non_io_nodes) != len(node_representation_indices):
        raise ValueError("Length of node_representation_indices does not match the number of non-input/output nodes.")
    
    # Assign attributes to non-input/output nodes based on the specified representations
    for node, rep_index in zip(non_io_nodes, node_representation_indices):
        row_index = rep_index
        G.nodes[node]['feature'] = one_hot_encoding_features[row_index]
    
    # Set a specific attribute for input and output nodes to distinguish them
    for node in input_nodes:
        G.nodes[node]['type'] = 'input'
        G.nodes[node]['feature'] = one_hot_encoding_features[node + 20]
        
    for node in output_nodes:
        G.nodes[node]['type'] = 'output'
        G.nodes[node]['feature'] = one_hot_encoding_features[23]
    
    return G


def smooth(scalars, weight):
    """Smooth a list of scalars using exponential moving average with the specified weight."""
    smoothed = [scalars[0]]  # Initialize with the first value
    
    for point in scalars[1:]:
        smoothed_val = smoothed[-1] * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        
    return smoothed


def sum_every_n_steps(lst, n):
    return [sum(lst[i:i+n]) for i in range(0, len(lst), n)]


def cello2_toxicity_score(gate_name, cumulative_input, gate_toxicity_df, update_interpolation = False):
    # Filter the DataFrame for the given gate_name
    row = gate_toxicity_df[gate_toxicity_df['gate_name'] == gate_name]
    #print("row:", row)
    # Check if the gate_name was found
    if not row.empty:
        
        #print("Gate Name: ", gate_name)
        # Extract 'input' and 'growth' as lists
        input_values = np.array(row.iloc[0]['input'])
        #print("input_values", input_values)
        growth_values = np.array(row.iloc[0]['growth'])
        
        # Input value
        x = cumulative_input
        #print("Cumulative input to the gate: ", x)

       
        index_min_inputs_a = -1
        growth_a = float('inf')
        index_min_inputs_b = -1
        growth_b = float('inf')

        for i, record in enumerate(input_values):
            distance = euclidean([x], [record])

            if distance < growth_a:
                index_min_inputs_b = index_min_inputs_a
                growth_b = growth_a
                index_min_inputs_a = i
                growth_a = distance
            elif distance < growth_b:
                index_min_inputs_b = i
                growth_b = distance

        value_a = growth_values[index_min_inputs_a]
        value_b = growth_values[index_min_inputs_b]
        
        
        if update_interpolation:
            #This interpolation is better
            rtn = (value_b * growth_a + value_a * growth_b) / (growth_a + growth_b)
        else:
            #This is how Cello 2.0 does it
            rtn = (value_a * growth_a + value_b * growth_b) / (growth_a + growth_b) 
        
        
        interpolated_growth = rtn
        
              
        if interpolated_growth > 1.0:
            interpolated_growth = 1.0
            
        if interpolated_growth < 0.01:
            interpolated_growth = 0.01
            
        #print("Interpolated_growth: ", interpolated_growth)
        
        return interpolated_growth
    else:
        print("No toxicity data for this gate was found: ", gate_name)
        return 1.0 # Return 1 if no data was found

   
def cello2_propagate_signals_and_calculate_toxixity(G, input_signals, gate_toxicity_df):
    """
    Dynamically propagate signals through a Directed Acyclic Graph (DAG), where each node
    performs a computation based on its configured function—except for nodes marked as 'output',
    which directly pass through the sum of signals they receive.

    Inputs:
    - G (nx.DiGraph): The graph representing the network, with nodes marked as 'output' where applicable.
    - input_signals (dict): A mapping from node indices (inputs) to their respective initial signals.
    - gate_toxicity_df (Data frame): Data from UFC file containing gate toxocity data for each gate.

    Outputs:
    - A dictionary mapping node indices labeled as 'output' to their respective final output signals.
    """
    
    c = 0.4 # conversion factor that we need for the output plasmid
    
    outputs = {node: 0 for node in G.nodes()}  # Initialize output signals for all nodes
    intermediate_inputs = []  # Initialize a list to store inputs at intermediate nodes
    growth_scores = [] # Initialize a list to store grwoth scores 
    
    # Assign initial input signals
    for node, signal in input_signals.items():
        outputs[node] = signal
        #print([node])

    # Process nodes in topological order
    for node in nx.topological_sort(G):
        # For nodes with the 'type' attribute set to 'output', sum the signals from all predecessors
        if G.nodes[node].get('type') == 'output':
            outputs[node] = c*(sum(outputs[predecessor] for predecessor in G.predecessors(node)))
        elif node not in input_signals:  # Processing nodes
            #print("node: ", node)
            #print("input_signals: ", input_signals)
            # Accumulate input from all predecessors
            cumulative_input = sum(outputs[predecessor] for predecessor in G.predecessors(node)) 
            intermediate_inputs.append((node, cumulative_input))  # Store inputs at intermediate nodes
            
            

            repressor_name = G.nodes[node].get('Repressor') # Get the repressor name
            
            
            
            RBS_name = G.nodes[node].get('RBS') # Get the RBS name
            gate_name = RBS_name + "_" + repressor_name # Generate the gate_name
            growth = cello2_toxicity_score(gate_name, cumulative_input, gate_toxicity_df)
            #print(growth.item())
            
            #if gate_name == "E1_BetI":
            #    print(growth)
            
            if isinstance(growth, np.generic):
                growth_scores.append((node, growth.item()))  # Convert numpy scalar to Python scalar
            else:
                growth_scores.append((node, growth))  # Directly use the Python float


            # Retrieve the node's computation parameters
            ymax = G.nodes[node].get('ymaxa', 0)
            ymin = G.nodes[node].get('ymina', 0)
            Ka = G.nodes[node].get('Ka', 1)
            n = G.nodes[node].get('n', 1)

            # Store the signal after processing by the node's Hill function
            outputs[node] = hill_function(cumulative_input, ymax, ymin, Ka, n)

    # Filter the outputs to include only those for nodes marked as 'output'
    output_signals = {node: signal for node, signal in outputs.items() if G.nodes[node].get('type') == 'output'}
    
    # Extract just the scores (the second element of each tuple)
    scores_array = np.array([score for _, score in growth_scores])

    # Multiply growth scores
    multiplied_growth_scores = np.prod(scores_array)

    return output_signals, intermediate_inputs, growth_scores, multiplied_growth_scores    

    
def cello2_calculate_toxicity_score(input_signals_list, G, gate_toxicity_df):
    """
    Simulates signal propagation through a biochemical network and calculates toxicity.
    
    Parameters:
    - input_signals_list (list of dict): Each dictionary represents a set of input signals for simulation.
    - G (nx.DiGraph): The graph representing the biochemical network.
    - gate_toxicity_df (DataFrame): A pandas DataFrame containing toxicity data for each gate.
    
    Returns:
    - toxicity_score (float): The calculated toxicity score based on growth scores.
    - detailed_results (dict): A dictionary containing detailed simulation results.
    """
    
    # Initialize a list to hold all output results
    all_outputs = []
    all_intermediates = []
    all_growth_scores = []
    all_multiplied_growth_scores = []

    for i, input_signals in enumerate(input_signals_list):
        #print(f"\nTesting input set {i+1}: {input_signals}")

        # Convert binary string inputs to integer signals for each node
        #input_signals_dict = {node: int(signal) for node, signal in enumerate(input_signals)}

        # Propagate signals through the graph
        outputs, intermediates, growth_scores, multiplied_growth_scores = cello2_propagate_signals_and_calculate_toxixity(G, input_signals, gate_toxicity_df)

        # Collect the output for the current input set
        all_outputs.append(outputs)
        all_intermediates.append(intermediates)
        all_growth_scores.append(growth_scores)
        all_multiplied_growth_scores.append(multiplied_growth_scores)
    
    toxicity_score = min(all_multiplied_growth_scores)
    
    # Prepare detailed results
    detailed_results = {
        "outputs": all_outputs,
        "intermediates": all_intermediates,
        "growth_scores": all_growth_scores,
        "multiplied_growth_scores": all_multiplied_growth_scores
    }
    
    return toxicity_score, detailed_results    

# + jupyter={"outputs_hidden": true}
def is_roadblocking(graph, roadblocking_inputs =[0, 2], specific_repressors = {'PhlF', 'SrpR', 'BM3R1', 'QacR'}):
    
    #Road blocking rules 
    #These cannot connect to the same NOR gate
    #pTac (A), pBad (C), PhlF, SrpR, BM3R1, QacR
    
    #Does not check the output, because we do not use a tandem architecture

    # In G, the attributes of the nodes refer to a repressor. Check whether a node has two connections coming from 2 of the following: 2, 0, PhlF, SrpR, BM3R1, QacR. If yes, return True. 
    invalid_nodes = []
    
    found = False

    # Iterate through each node
    for node in graph.nodes(data=True):
        if node[1].get('type') not in ['input', 'output']:
        #if node[1].get('type') not in ['input']:    
            # Count the incoming edges from specific nodes/repressors
            count = 0
            for edge in graph.in_edges(node[0]):
                source_node, _ = edge
                source_data = graph.nodes[source_node]
                
                # Check if the source node is one of the specific input nodes
                if source_node in roadblocking_inputs:
                    count += 1
                
                # Check if the source node has one of the specific repressors
                elif 'Repressor' in source_data and source_data['Repressor'] in specific_repressors:
                    count += 1
            
            # If the node has at least two incoming edges that match the criteria, add to list
            if count >= 2:
                invalid_nodes.append(node[0])
                found = True
                
    return (found, invalid_nodes)

def add_implicit_OR_to_dag(G):
    """
    Returns a modified copy of the given DAG by removing the node without outgoing edges and its predecessor.
    Then, assigns type 'output' to the new node without outgoing edges.

    Args:
    - G: A NetworkX DiGraph object representing the DAG.

    Returns:
    - A copy of the modified DAG, leaving the original DAG unchanged.
    """
    # Create a deep copy of G to ensure the original graph is not modified
    G_copy = G.copy()
    
    # Ensure the graph is a DAG
    if not nx.is_directed_acyclic_graph(G_copy):
        raise ValueError("The provided graph is not a DAG")

    # Identify the sink node (node without outgoing edges)
    sink_nodes = [node for node in G_copy.nodes() if G_copy.out_degree(node) == 0]
    
    # For simplicity, assume there is only one sink node to be removed
    if sink_nodes:
        sink_node = sink_nodes[0]
        
        # Identify the predecessors of the sink node
        predecessors = list(G_copy.predecessors(sink_node))
        
        # For simplicity, let's remove the first predecessor if there are any
        if predecessors:
            predecessor_node = predecessors[0]
            
            # Delete the sink node and its predecessor
            G_copy.remove_node(sink_node)
            G_copy.remove_node(predecessor_node)
            
            # After removal, identify the new sink nodes
            new_sink_nodes = [node for node in G_copy.nodes() if G_copy.out_degree(node) == 0]
            
            # Add type 'output' to the new sink nodes
            for node in new_sink_nodes:
                G_copy.nodes[node]['type'] = 'output'

    return G_copy


# +
def log_interpolate_growth(gate_name, cumulative_input, gate_toxicity_df):
    # Filter the DataFrame for the given gate_name
    row = gate_toxicity_df[gate_toxicity_df['gate_name'] == gate_name]
    #print("row:", row)
    
    # Check if the gate_name was found
    if not row.empty:
        
        #print("Gate Name: ", gate_name)
        
        # Extract 'input' and 'growth' as lists
        input_values = np.array(row.iloc[0]['input'])
        #print("input_values", input_values)
        
        growth_values = np.array(row.iloc[0]['growth'])
        
        # Input value
        #print("Cumulative input to the gate: ", cumulative_input)

        incoming_rpu = cumulative_input

        tox_score = 1.0

        # if incoming rpu is below the first titration rpu
        if incoming_rpu < input_values[0]:
            tox_score = growth_values[0]

        # if incoming rpu is above the last titration rpu
        elif incoming_rpu > input_values[-1]:
            tox_score = growth_values[-1]

        # if incoming rpu is in the titration range, use weighted average of the two surrounding titration points
        else:
            # search titrations until titration > incoming rpu
            for t in range(len(input_values)):
                if incoming_rpu < input_values[t]:
                    lower_rpu = math.log10(input_values[t - 1])
                    upper_rpu = math.log10(input_values[t])

                    lower_tox = growth_values[t - 1]
                    upper_tox = growth_values[t]

                    weight = (math.log10(incoming_rpu) - lower_rpu) / (upper_rpu - lower_rpu)

                    weighted_avg = (lower_tox * (1 - weight)) + (upper_tox * weight)

                    tox_score = weighted_avg

                    break

        if tox_score > 1.0:
            tox_score = 1.0
        if tox_score < 0.01:
            tox_score = 0.01


        return tox_score
        
    else:
        print("No toxicity data for this gate was found: ", gate_name)
        return 1.0 # Return 1 if no data was found


def propagate_signals_and_calculate_toxixity(G, input_signals, gate_toxicity_df):
    """
    Dynamically propagate signals through a Directed Acyclic Graph (DAG), where each node
    performs a computation based on its configured function—except for nodes marked as 'output',
    which directly pass through the sum of signals they receive.

    Inputs:
    - G (nx.DiGraph): The graph representing the network, with nodes marked as 'output' where applicable.
    - input_signals (dict): A mapping from node indices (inputs) to their respective initial signals.
    - gate_toxicity_df (Data frame): Data from UFC file containing gate toxocity data for each gate.

    Outputs:
    - A dictionary mapping node indices labeled as 'output' to their respective final output signals.
    """
    
    c = 0.4 # conversion factor that we need for the output plasmid
    
    outputs = {node: 0 for node in G.nodes()}  # Initialize output signals for all nodes
    intermediate_inputs = []  # Initialize a list to store inputs at intermediate nodes
    growth_scores = [] # Initialize a list to store grwoth scores 
    
    # Assign initial input signals
    for node, signal in input_signals.items():
        outputs[node] = signal
        #print([node])

    # Process nodes in topological order
    for node in nx.topological_sort(G):
        # For nodes with the 'type' attribute set to 'output', sum the signals from all predecessors
        if G.nodes[node].get('type') == 'output':
            outputs[node] = c*(sum(outputs[predecessor] for predecessor in G.predecessors(node)))
        elif node not in input_signals:  # Processing nodes
            #print("node: ", node)
            #print("input_signals: ", input_signals)
            # Accumulate input from all predecessors
            cumulative_input = sum(outputs[predecessor] for predecessor in G.predecessors(node)) 
            intermediate_inputs.append((node, cumulative_input))  # Store inputs at intermediate nodes
            
            

            repressor_name = G.nodes[node].get('Repressor') # Get the repressor name
            
            
            
            RBS_name = G.nodes[node].get('RBS') # Get the RBS name
            gate_name = RBS_name + "_" + repressor_name # Generate the gate_name
            growth = log_interpolate_growth(gate_name, cumulative_input, gate_toxicity_df)
            #print(growth.item())
            
            #if gate_name == "E1_BetI":
            #    print(growth)
            
            if isinstance(growth, np.generic):
                growth_scores.append((node, growth.item()))  # Convert numpy scalar to Python scalar
            else:
                growth_scores.append((node, growth))  # Directly use the Python float


            # Retrieve the node's computation parameters
            ymax = G.nodes[node].get('ymaxa', 0)
            ymin = G.nodes[node].get('ymina', 0)
            Ka = G.nodes[node].get('Ka', 1)
            n = G.nodes[node].get('n', 1)

            # Store the signal after processing by the node's Hill function
            outputs[node] = hill_function(cumulative_input, ymax, ymin, Ka, n)

    # Filter the outputs to include only those for nodes marked as 'output'
    output_signals = {node: signal for node, signal in outputs.items() if G.nodes[node].get('type') == 'output'}
    
    
    # Extract just the scores (the second element of each tuple)
    scores_array = np.array([score for _, score in growth_scores])

    # Multiply growth scores
    multiplied_growth_scores = np.prod(scores_array)

    return output_signals, intermediate_inputs, growth_scores, multiplied_growth_scores    

    
def calculate_toxicity_score(input_signals_list, G, gate_toxicity_df):
    """
    Simulates signal propagation through a biochemical network and calculates toxicity.
    
    Parameters:
    - input_signals_list (list of dict): Each dictionary represents a set of input signals for simulation.
    - G (nx.DiGraph): The graph representing the biochemical network.
    - gate_toxicity_df (DataFrame): A pandas DataFrame containing toxicity data for each gate.
    
    Returns:
    - toxicity_score (float): The calculated toxicity score based on growth scores.
    - detailed_results (dict): A dictionary containing detailed simulation results.
    """
    
    # Initialize a list to hold all output results
    all_outputs = []
    all_intermediates = []
    all_growth_scores = []
    all_multiplied_growth_scores = []

    for i, input_signals in enumerate(input_signals_list):
        #print(f"\nTesting input set {i+1}: {input_signals}")

        # Convert binary string inputs to integer signals for each node
        #input_signals_dict = {node: int(signal) for node, signal in enumerate(input_signals)}

        # Propagate signals through the graph
        outputs, intermediates, growth_scores, multiplied_growth_scores = propagate_signals_and_calculate_toxixity(G, input_signals, gate_toxicity_df)

        # Collect the output for the current input set
        all_outputs.append(outputs)
        all_intermediates.append(intermediates)
        all_growth_scores.append(growth_scores)
        all_multiplied_growth_scores.append(multiplied_growth_scores)
    
    toxicity_score = min(all_multiplied_growth_scores)
    
    # Prepare detailed results
    detailed_results = {
        "outputs": all_outputs,
        "intermediates": all_intermediates,
        "growth_scores": all_growth_scores,
        "multiplied_growth_scores": all_multiplied_growth_scores
    }
    
    return toxicity_score, detailed_results    


# +
### import math


def simulated_annealing_cello_v3(df, adj_matrix, initial_solution, Tmax, Tmin, iterations_before_Tmin, input_signals_list_small_molecules, input_signals_list_binary, iterations_at_Tmin=10000):

    #    input_signals_list_small_molecules = [
    #        {0: 0.008, 1: 0.001, 2: 0.003},  # First set of input signals
    #        {0: 0.008, 1: 0.001, 2: 2.8},  # Second set of input signals
    #        {0: 0.008, 1: 4.4, 2: 0.003},  # Third set of input signals
    #        {0: 0.008, 1: 4.4, 2: 2.8},  # First set of input signals
    #        {0: 2.5, 1: 0.001, 2: 0.003},  # Second set of input signals
    #        {0: 2.5, 1: 0.001, 2: 2.8},  # Third set of input signals    
    #        {0: 2.5, 1: 4.4, 2: 0.003},  # Second set of input signals
    #        {0: 2.5, 1: 4.4, 2: 2.8},  # Third set of input signals  
    #    ]

    #Postech data, roadblocking and toxicity 

    #pTac 0.0042 2.0082 
    #pTet 0.0022 5.0543 
    #pBAD 0.0278 3.9239 

    #pBaD pTet pTac

    current_solution = initial_solution

    # Assuming the presence of these functions and variables:
    # assign_representations_with_io_nodes_3, simulate_signal_propagation, simulate_signal_propagation_binary, calculate_circuit_score, perform_action
    
    G = assign_representations_with_io_nodes_3(adj_matrix, df, current_solution)
    actual_logic = simulate_signal_propagation(G, input_signals_list_small_molecules)
    expected_logic = simulate_signal_propagation_binary(G, input_signals_list_binary)
    current_score = calculate_circuit_score(expected_logic, actual_logic)
    optimal_score = current_score
    optimal_assignment = current_solution

    LOGMAX = math.log10(Tmax)
    LOGMIN = math.log10(Tmin)
    LOGINC = (LOGMAX - LOGMIN) / iterations_before_Tmin
    
    temperature = Tmax
    i = 0
    no_improve = 0
    iterations_since_Tmin = 0

    total_iterations = iterations_before_Tmin + iterations_at_Tmin
    pbar = tqdm(total=total_iterations, desc="Optimizing", leave=True)

    # Initialize a list to track optimal scores
    optimal_scores = []
    iteration_numbers = []

    while True:
        new_solution = perform_action(current_solution, df)
        
        G = assign_representations_with_io_nodes_3(adj_matrix, df, new_solution)
        actual_logic = simulate_signal_propagation(G, input_signals_list_small_molecules)
        expected_logic = simulate_signal_propagation_binary(G, input_signals_list_binary)
        
        new_score = calculate_circuit_score(expected_logic, actual_logic)
        
        copied_new_solution = new_solution.copy()
        
        #print("new_solution: ", new_solution)
        #print("new_score: ", new_score)

        if acceptance_probability(current_score, new_score, temperature) > random.random():
            current_solution = new_solution
            current_score = new_score
            if current_score > optimal_score:
                optimal_score = new_score
                optimal_assignment = copied_new_solution
                #no_improve = 0                
            else:
                pass
                #if (temperature <= Tmin):
                #    no_improve += 1
        else:
            pass
            #if (temperature <= Tmin):
            #    no_improve += 1

        # Update the list with the current optimal score and iteration number
        optimal_scores.append(optimal_score)
        iteration_numbers.append(i)

        if i >= total_iterations:
            break  # Break the loop if no improvement is observed for max_no_improve iterations

        if temperature > Tmin:
            logTemperature = LOGMAX - i * LOGINC
            temperature = math.pow(10, logTemperature)
        else:
            temperature = Tmin  # Keep temperature at Tmin, do not decrease further
            iterations_since_Tmin += 1
        i += 1
        
        
        
        

        #print("iteration: ", i)
        #print("iterations_since_Tmin: ", iterations_since_Tmin)
        #print("temperature: ", temperature)
        
        pbar.update(1)

    pbar.close()

    # Plotting the iterations versus the optimal score

    # Set global font to Arial
    plt.rcParams['font.family'] = 'DejaVu Sans'

    plt.figure(figsize=(8, 4))
    plt.plot(iteration_numbers, optimal_scores, marker='o', markersize=3, linestyle='--', color='b')
    plt.title('Iterations vs Highest Score', fontdict={'fontname': 'DejaVu Sans'})
    plt.xlabel('Iteration', fontdict={'fontname': 'DejaVu Sans'})
    plt.ylabel('Circuit Score', fontdict={'fontname': 'DejaVu Sans'})
    # plt.grid(True)  # Keep this commented out to maintain a background without lines
    plt.show()
    
    draw_network_with_colors_and_labels_from_G(G)

    print("optimal_assignment: ", optimal_assignment)
    return optimal_score, optimal_assignment  # Return both optimal score and solution

# +

def generate_random_initial_states(G, number_of_states, precomputed_graphs_1_input, graphs_library_1_input, precomputed_graphs_2_input, graphs_library_2_input, precomputed_graphs_3_input, graphs_library_3_input):
    random_initial_states_list = [G.copy()]
    
    # Initialize the progress bar
    pbar = tqdm(total=number_of_states, desc="Generating Random Initial States", unit="state")
    
    while len(random_initial_states_list) < number_of_states + 1:
        current_solution = random.choice(random_initial_states_list)
    
        # Choose a random target node, excluding specified nodes
        excluded_nodes = [0, 1, 2]  # Assuming these are input nodes
        target_node = random.choice([node for node in current_solution.nodes() if node not in excluded_nodes])

        select_cut_size_n = random.choice([1, 2, 3])

        feasible_cuts = find_feasible_cuts(current_solution, target_node, max_cut_size=select_cut_size_n, filter_redundant=True)
        feasible_cuts_of_size_n = [cut for cut in feasible_cuts if len(cut) == select_cut_size_n]

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
                # Check if the truth table integer is in the graphs library
                if truth_table_int not in precomputed_graphs_3_input:
                    continue  # Skip the rest of the iteration if no replacement graph is found

                associated_graphs = precomputed_graphs_3_input[truth_table_int]
                selected_graph_index = random.choice(associated_graphs)
                replacement_graph = graphs_library_3_input[selected_graph_index]                

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
            has_three_entry_nodes = len(entry_nodes) == 3
            has_one_exit_node = len(exit_nodes) == 1

            if not has_three_entry_nodes or not has_one_exit_node:
                print("BUG!")
                print("entry_nodes: ", entry_nodes)
                print("entry_nodes: ", entry_nodes)

            if calculate_truth_table(current_solution) != calculate_truth_table(new_solution):
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


# -

def covert_to_nor_not(adj_matrix):
    """
    
    Replace AND gates with NOR/NOT gates 
    
    Specifically, For nodes with exactly 2 incoming edges in a DAG represented by an adjacency matrix,
    adds one node in place of each incoming edge.
    
    Parameters:
    - adj_matrix (np.ndarray): The original adjacency matrix.

    Returns:
    - np.ndarray: The modified adjacency matrix with added nodes.
    """
    n = adj_matrix.shape[0]  # Original number of nodes
    incoming_edges = np.sum(adj_matrix, axis=0)  # Count incoming edges for each node

    # Iterate over each node to check if it has exactly 2 incoming edges
    for target_node in range(n):
        if incoming_edges[target_node] == 2:
            # Find the nodes with incoming edges to the target node
            source_nodes = np.where(adj_matrix[:, target_node] == 1)[0]
            
            for source_node in source_nodes:
                # Add a new node for each incoming edge
                new_node_index = adj_matrix.shape[0]
                adj_matrix = np.pad(adj_matrix, ((0, 1), (0, 1)), 'constant', constant_values=0)
                
                # Redirect the original edge to the new node
                adj_matrix[source_node, target_node] = 0  # Remove the original edge
                adj_matrix[source_node, new_node_index] = 1  # Edge from source to new node
                adj_matrix[new_node_index, target_node] = 1  # Edge from new node to target
                
                # Update the incoming edges count after adding a new node
                incoming_edges[target_node] = 1  # The target now has a new incoming edge
                incoming_edges = np.append(incoming_edges, 0)  # New node has no incoming edges yet

    return adj_matrix


def remove_redundant_not_with_exclusions(adj_matrix):
    """
    Simplifies a DAG represented by an adjacency matrix by identifying pairs of directly
    connected nodes that each have only 1 incoming edge and are not isolated (having both
    incoming and outgoing edges), and replacing them with a single edge.

    Parameters:
    - adj_matrix (np.ndarray): The original adjacency matrix.

    Returns:
    - np.ndarray: The modified adjacency matrix.
    """
    n = adj_matrix.shape[0]  # Number of nodes
    incoming_edges = np.sum(adj_matrix, axis=0)  # Count incoming edges for each node
    outgoing_edges = np.sum(adj_matrix, axis=1)  # Count outgoing edges for each node

    # Attempt to simplify until no more simplifications can be made
    while True:
        simplified = False
        for i in range(n):
            if incoming_edges[i] == 0 or outgoing_edges[i] == 0:
                # Skip nodes that have either 0 incoming or 0 outgoing edges
                continue
            
            successors = np.where(adj_matrix[i, :] == 1)[0]
            for successor in successors:
                if incoming_edges[successor] == 0 or outgoing_edges[successor] == 0:
                    # Also skip the successor if it has either 0 incoming or 0 outgoing edges
                    continue
                
                # Check if both nodes have exactly one incoming edge
                if incoming_edges[i] == 1 and incoming_edges[successor] == 1:
                    predecessor = np.where(adj_matrix[:, i] == 1)[0]
                    successor_successors = np.where(adj_matrix[successor, :] == 1)[0]
                    
                    if predecessor.size == 1:  # Ensure i has exactly one predecessor
                        # Directly connect the predecessor of i to the successor(s) of the direct successor
                        for s_successor in successor_successors:
                            adj_matrix[predecessor, s_successor] = 1
                        # Remove i and its direct successor from the graph
                        adj_matrix = np.delete(adj_matrix, [i, successor], axis=0)
                        adj_matrix = np.delete(adj_matrix, [i, successor], axis=1)
                        incoming_edges = np.delete(incoming_edges, [i, successor])
                        outgoing_edges = np.delete(outgoing_edges, [i, successor])
                        
                        simplified = True
                        n -= 2  # Update the number of nodes
                        break  # Break to restart the search as the matrix has changed
            if simplified:
                break  # Break to restart the outer loop as well

        if not simplified:
            break  # Exit if no simplifications were made in the last iteration

    return adj_matrix


def replace_negative_edges(adj_matrix):
    """
    For each -1 in the adjacency matrix, replaces it by adding a new node
    with incoming and outgoing edges of value 1. 
    
    Overall, generates a new graph where 1 incoming edge is a NOT gate, and 2 incoming edges is AND gate

    Parameters:
    - adj_matrix (np.ndarray): The original adjacency matrix.

    Returns:
    - np.ndarray: The modified adjacency matrix.
    """
    n = adj_matrix.shape[0]  # Original number of nodes
    nodes_to_add = np.argwhere(adj_matrix == -1)  # Find all -1 entries
    
    # Process each -1 entry
    for i, j in nodes_to_add:
        # Add new node: increase the size of the adjacency matrix
        new_node_index = adj_matrix.shape[0]  # Index of the new node
        adj_matrix = np.pad(adj_matrix, ((0, 1), (0, 1)), 'constant', constant_values=(0))
        
        # Set incoming edge to new node and outgoing edge from new node
        adj_matrix[i, new_node_index] = 1  # Incoming edge to new node
        adj_matrix[new_node_index, j] = 1  # Outgoing edge from new node
        
        # Remove the original -1 edge
        adj_matrix[i, j] = 0
    
    return adj_matrix


def aiger_ascii_to_adj_matrix(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
    if not lines:  # Check if the list is empty
        raise ValueError(f"The file {file_path} is empty or could not be read.")


    # Parse the header to get the number of gates, inputs, and outputs
    header = lines[0].strip().split(' ')
    num_gates = int(header[5])
    #print("num_gates: "); print(num_gates)
    num_inputs = int(header[2])
    #print("num_inputs: "); print(num_inputs)
    num_outputs = int(header[4])
    #print("num_outputs: "); print(num_outputs)

    # Adjust the index for gates, considering the inputs and outputs
    index_offset = num_inputs
    adj_matrix_size = num_gates + index_offset + num_outputs
    adj_matrix = [[0 for _ in range(adj_matrix_size)] for _ in range(adj_matrix_size)]

    # Parse the file to fill the adjacency matrix
    for line in lines:
        if line.startswith('aag') or line.startswith('c'):
            # Comments
            continue
            
        if line.startswith('i'):
            # Inputs 
            continue
            
        if line.startswith('o'):
            # Outputs            
            continue
            
        parts = line.strip().split(' ')
        #print(parts)
        

        if len(parts) == 3:
            # This is an AND gate
            gate, in1, in2 = map(int, parts)
            gate = gate // 2
            #print("gate: "); print(gate)

            # Check for inversion and adjust the input node indices
            value_in1 = -1 if in1 % 2 else 1
            node_in1 = in1 // 2
            #print("node_in1: "); print(node_in1)

            value_in2 = -1 if in2 % 2 else 1
            node_in2 = in2 // 2
            #print("node_in2: "); print(node_in2)

            adj_matrix[node_in1-1][gate-1] = value_in1  # Connect input 1 to gate
            adj_matrix[node_in2-1][gate-1] = value_in2  # Connect input 2 to gate
            


    #Attach the output. Note: This code below only works for 3-input 1-output circuits
    #print("OUTPUT")
    #print(lines[4])
    output_int = int(lines[4])
    output_node = output_int // 2
    value_node = -1 if output_int % 2 else 1   
    adj_matrix[output_node-1][-1] = value_node
    
    adj_matrix = np.array(adj_matrix)

    return adj_matrix


# +
def check_for_valid_or_gate_at_output(adj_matrix):
    """
    Searches for nodes without outgoing edges and checks specified conditions for their predecessors.

    Parameters:
    - adj_matrix (np.ndarray): The adjacency matrix of the DAG.

    Returns:
    - list: A list of nodes (0-indexed) without outgoing edges that meet the specified conditions.
    """
    # Nodes without outgoing edges
    nodes_without_outgoing = np.where(np.sum(adj_matrix, axis=1) == 0)[0]
    
#    print(nodes_without_outgoing)
    
    # Initialize an empty list to store nodes that meet the conditions
    valid_nodes = []
    
    for node in nodes_without_outgoing:
        # Find the predecessors of the current node
        predecessors = np.where(adj_matrix[:, node] == 1)[0]
        
        # Condition 1: The node must have exactly one predecessor with 1 incoming and 1 outgoing edge
        if len(predecessors) == 1:
            pred = predecessors[0]
            if np.sum(adj_matrix[:, pred]) == 1 and np.sum(adj_matrix[pred, :]) == 1:
                # Find the predecessors of the predecessor
                pred_predecessors = np.where(adj_matrix[:, pred] == 1)[0]
                
                # Condition 2: The predecessor of the predecessor must have exactly 2 incoming and 1 outgoing edge
                if len(pred_predecessors) == 1:
                    pred_pred = pred_predecessors[0]
                    if np.sum(adj_matrix[:, pred_pred]) == 2 and np.sum(adj_matrix[pred_pred, :]) == 1:
                        valid_nodes.append(node)
    
    return valid_nodes



# -

def check_for_valid_or_gate_at_output(adj_matrix):
    """
    Searches for nodes without outgoing edges and checks specified conditions for their predecessors.

    Parameters:
    - adj_matrix (np.ndarray): The adjacency matrix of the DAG.

    Returns:
    - list: A list of nodes (0-indexed) without outgoing edges that meet the specified conditions.
    """
    # Nodes without outgoing edges
    nodes_without_outgoing = np.where(np.sum(adj_matrix, axis=1) == 0)[0]
    
    print(nodes_without_outgoing)
    
    # Initialize an empty list to store nodes that meet the conditions
    valid_nodes = []
    
    for node in nodes_without_outgoing:
        # Find the predecessors of the current node
        predecessors = np.where(adj_matrix[:, node] == 1)[0]
        
        # Condition 1: The node must have exactly one predecessor with 1 incoming and 1 outgoing edge
        if len(predecessors) == 1:
            pred = predecessors[0]
            if np.sum(adj_matrix[:, pred]) == 1 and np.sum(adj_matrix[pred, :]) == 1:
                # Find the predecessors of the predecessor
                pred_predecessors = np.where(adj_matrix[:, pred] == 1)[0]
                
                # Condition 2: The predecessor of the predecessor must have exactly 2 incoming and 1 outgoing edge
                if len(pred_predecessors) == 1:
                    pred_pred = pred_predecessors[0]
                    if np.sum(adj_matrix[:, pred_pred]) == 2 and np.sum(adj_matrix[pred_pred, :]) == 1:
                        valid_nodes.append(node)
    
    return valid_nodes


def draw_dag_from_adj_matrix(adj_matrix, node_colors=None, ax=None):
    """
    Draws the DAG from the adjacency matrix with specified node colors on the given axis.

    Parameters:
    - adj_matrix: Adjacency matrix representing the DAG.
    - node_colors: Dictionary mapping node indices to colors. If a node is not in the dictionary, it defaults to 'lightblue'.
    - ax: matplotlib axis object to draw the graph on.
    """
    if ax is None:
        ax = plt.gca()  # Get current axis if none provided
    
    # Create a directed graph
    G = nx.DiGraph()

    # Adding nodes and edges from the adjacency matrix
    for i, row in enumerate(adj_matrix):
        for j, val in enumerate(row):
            if val != 0:
                edge_label = 'o' if val == -1 else ''
                G.add_edge(i, j, label=edge_label)

    pos = nx.spring_layout(G)  # positions for all nodes
    edge_labels = nx.get_edge_attributes(G, 'label')

    if node_colors is None:
        node_colors = {}
    colors = [node_colors.get(node, 'lightblue') for node in G.nodes()]

    # Draw the graph on the specified axis
    nx.draw(G, pos, with_labels=True, node_color=colors, node_size=500, edge_color='gray', linewidths=1, font_size=12, ax=ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', ax=ax)


def count_gates2(adj_matrix):
    """
    Counts the number of nodes in a DAG that have both incoming and outgoing edges,
    excluding nodes that are missing either type of edge.

    Parameters:
    - adj_matrix (np.ndarray): The adjacency matrix of the DAG.

    Returns:
    - int: The count of nodes with both incoming and outgoing edges.
    """
    # Identify nodes with incoming edges (sum columns)
    nodes_with_incoming = np.count_nonzero(adj_matrix, axis=0) 
    
    # Identify nodes with outgoing edges (sum rows)
    nodes_with_outgoing = np.count_nonzero(adj_matrix, axis=1) 
    
    # Count nodes with both incoming and outgoing edges
    nodes_with_both = np.logical_and(nodes_with_incoming, nodes_with_outgoing)
    
    # Return the count of nodes with both types of edges
    return np.sum(nodes_with_both)


# +
def delete_nodes_from_adj_matrix(adj_matrix, nodes_to_delete):
    """
    Creates a copy of the adjacency matrix and deletes nodes from it by setting their corresponding rows and columns to 0.

    Parameters:
    - adj_matrix (np.ndarray): The adjacency matrix of the DAG.
    - nodes_to_delete (list): List of nodes to delete.

    Returns:
    - np.ndarray: A new adjacency matrix after deletion.
    """
    # Create a copy of the adjacency matrix to avoid modifying the original
    adj_matrix_copy = adj_matrix.copy()
    for node in nodes_to_delete:
        adj_matrix_copy[node, :] = 0  # Set the node's outgoing edges to 0 in the copy
        adj_matrix_copy[:, node] = 0  # Set the node's incoming edges to 0 in the copy
    return adj_matrix_copy

def check_for_valid_or_gate_at_output_and_replace_for_or_gate(adj_matrix):
    """
    Searches for nodes without outgoing edges and checks specified conditions for their predecessors in a DAG.
    Deletes the node and its predecessor when conditions are met, and returns a new adjacency matrix.

    Parameters:
    - adj_matrix (np.ndarray): The adjacency matrix of the DAG.

    Returns:
    - np.ndarray: A new adjacency matrix after deletion, if conditions are met.
    """
    # Create a copy of the adjacency matrix to work with
    modified_adj_matrix = adj_matrix.copy()
    
    # Nodes without outgoing edges
    nodes_without_outgoing = np.where(np.sum(modified_adj_matrix, axis=1) == 0)[0]
    
    for node in nodes_without_outgoing:
        # Find the predecessors of the current node
        predecessors = np.where(modified_adj_matrix[:, node] == 1)[0]
        
        if len(predecessors) == 1:
            pred = predecessors[0]
            if np.sum(modified_adj_matrix[:, pred]) == 1 and np.sum(modified_adj_matrix[pred, :]) == 1:
                # Find the predecessors of the predecessor
                pred_predecessors = np.where(modified_adj_matrix[:, pred] == 1)[0]
                
                if len(pred_predecessors) == 1:
                    pred_pred = pred_predecessors[0]
                    if np.sum(modified_adj_matrix[:, pred_pred]) == 2 and np.sum(modified_adj_matrix[pred_pred, :]) == 1:
                        # If conditions are met, delete the node and its predecessor from the copy of the adjacency matrix
                        modified_adj_matrix = delete_nodes_from_adj_matrix(modified_adj_matrix, [node, pred])
    
    return modified_adj_matrix


# +
import random

def random_part_selection(df, length):
    """
    Randomly select unique indices from a pandas DataFrame without choosing the same repressor more than once.
    """
    unique_repressors = df['Repressor'].unique()  # Get unique repressors
    selected_indices = []

    while len(selected_indices) < length:
        index = random.randint(0, len(df) - 1)  # Randomly select an index
        repressor = df.loc[index, 'Repressor']
        if repressor in unique_repressors:
            selected_indices.append(index)
            unique_repressors = unique_repressors[unique_repressors != repressor]  # Remove selected repressor

    return selected_indices


# +
def assign_representations_with_io_nodes(adj_matrix, df, node_representation_indices):
    """
    Assign physical representations to nodes in a DAG from a pandas DataFrame,
    automatically identifying input and output nodes.
    """
    # Create the DAG from the adjacency matrix
    G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
    
    # Automatically identify input nodes as those with only outgoing edges
    input_nodes = [node for node in G.nodes() if G.in_degree(node) == 0 and G.out_degree(node) > 0]
    print(input_nodes)
    
    # Automatically identify output nodes as those with only incoming edges
    output_nodes = [node for node in G.nodes() if G.out_degree(node) == 0 and G.in_degree(node) > 0]
    print(output_nodes)
    
    # Assign attributes to each node based on the specified representations
    for node in G.nodes():
        if node not in input_nodes and node not in output_nodes:
            # Directly access the list by index for the node's representation index
            if node < len(node_representation_indices):
                row_index = node_representation_indices[node]
                # Assign all columns as attributes to the node
                for col in df.columns:
                    G.nodes[node][col] = df.iloc[row_index][col]
            # Optionally, handle the case where the node index is out of bounds for the list
            else:
                print(f"Warning: Node {node} is out of bounds for node_representation_indices.")
        else:
            # Set a specific attribute for input and output nodes to distinguish them
            G.nodes[node]['type'] = 'input' if node in input_nodes else 'output'
    
    return G

import networkx as nx
import pandas as pd

def assign_representations_with_io_nodes_2(adj_matrix, df, node_representation_indices):
    """
    Assign physical representations to nodes in a DAG from a pandas DataFrame,
    automatically identifying input and output nodes. Representation indices are assigned
    from left to right to each node that is not an input or an output, based on the provided list.
    """
    # Create the DAG from the adjacency matrix
    G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
    
    # Automatically identify input nodes as those with only outgoing edges
    input_nodes = [node for node in G.nodes() if G.in_degree(node) == 0 and G.out_degree(node) > 0]
    
    # Automatically identify output nodes as those with only incoming edges
    output_nodes = [node for node in G.nodes() if G.out_degree(node) == 0 and G.in_degree(node) > 0]
    
    # Initialize an index for accessing representation indices
    representation_index = 0
    
    # Assign attributes to each node based on the specified representations
    for node in G.nodes():
        if node not in input_nodes and node not in output_nodes:
            if representation_index < len(node_representation_indices):
                # Use the current representation index for the node
                row_index = node_representation_indices[representation_index]
                # Assign all columns as attributes to the node
                for col in df.columns:
                    G.nodes[node][col] = df.iloc[row_index][col]
                # Move to the next representation index
                representation_index += 1
            else:
                # Optionally handle the case where there are more non-input/output nodes than representations provided
                print(f"Warning: Not enough representation indices provided for all non-input/output nodes.")
        else:
            # Set a specific attribute for input and output nodes to distinguish them
            G.nodes[node]['type'] = 'input' if node in input_nodes else 'output'
    
    return G


def assign_representations_with_io_nodes_3(adj_matrix, df, node_representation_indices):
    
  
    """
    Assign physical representations to nodes in a DAG from a pandas DataFrame, automatically identifying
    input and output nodes. Representation indices are assigned from left to right to each node that is
    not an input or an output, based on the provided list. Disconnected nodes are excluded from the graph.
    """
    # Create the DAG from the adjacency matrix
    G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
    
    # Remove disconnected nodes (nodes with neither incoming nor outgoing edges)
    disconnected_nodes = [node for node in G.nodes() if G.in_degree(node) == 0 and G.out_degree(node) == 0]
    G.remove_nodes_from(disconnected_nodes)
    
    # Identify input and output nodes
    input_nodes = [node for node in G.nodes() if G.in_degree(node) == 0]
    output_nodes = [node for node in G.nodes() if G.out_degree(node) == 0]
    
    # Filter nodes that are neither inputs nor outputs
    non_io_nodes = [node for node in G.nodes() if node not in input_nodes and node not in output_nodes]
    
    # Ensure the length of representation indices matches the number of non-io nodes
    if len(non_io_nodes) != len(node_representation_indices):
        raise ValueError("Length of node_representation_indices does not match the number of non-input/output nodes.")
    
    # Assign attributes to non-input/output nodes based on the specified representations
    for node, rep_index in zip(non_io_nodes, node_representation_indices):
        row_index = rep_index
        # Assign all columns as attributes to the node
        for col in df.columns:
            G.nodes[node][col] = df.iloc[row_index][col]
    
    # Set a specific attribute for input and output nodes to distinguish them
    for node in input_nodes:
        G.nodes[node]['type'] = 'input'
    for node in output_nodes:
        G.nodes[node]['type'] = 'output'
    
    return G

def assign_representations_with_io_nodes_3_DiGraph(G, df, node_representation_indices, topological_sort = False):
    
  
    """
    Assign physical representations to nodes in a DAG from a pandas DataFrame, automatically identifying
    input and output nodes. Representation indices are assigned from left to right to each node that is
    not an input or an output, based on the provided list. Disconnected nodes are excluded from the graph.
    """
    
    G = G.copy()
        
    # Identify input and output nodes
    input_nodes = [node for node in G.nodes() if G.in_degree(node) == 0]
    output_nodes = [node for node in G.nodes() if G.out_degree(node) == 0]
    
    # Filter nodes that are neither inputs nor outputs
    non_io_nodes = [node for node in G.nodes() if node not in input_nodes and node not in output_nodes]
    
    # Ensure the length of representation indices matches the number of non-io nodes
    if len(non_io_nodes) != len(node_representation_indices):
        raise ValueError("Length of node_representation_indices does not match the number of non-input/output nodes.")
    
    if topological_sort:
        #non_io_nodes =  [n for n in nx.topological_sort(G) if n in non_io_nodes]
        ordering = []                      # empty list to collect nodes
        for n in nx.topological_sort(G):   # iterate in topo order
            if n in non_io_nodes:              # keep only the interior nodes
                ordering.append(n)  
        non_io_nodes = ordering                     
    else:
        non_io_nodes = sorted(non_io_nodes)
    
    # Assign attributes to non-input/output nodes based on the specified representations
    for node, rep_index in zip(non_io_nodes, node_representation_indices):
        row_index = rep_index
        # Assign all columns as attributes to the node
        for col in df.columns:
            G.nodes[node][col] = df.iloc[row_index][col]
    
    # Set a specific attribute for input and output nodes to distinguish them
    for node in input_nodes:
        G.nodes[node]['type'] = 'input'
    for node in output_nodes:
        G.nodes[node]['type'] = 'output'
    
    return G


def assign_io_nodes(adj_matrix):
    """
    Assign only input and outputs to the nodes. Disconnected nodes are excluded from the graph.
    """
    # Create the DAG from the adjacency matrix
    G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
    
    # Remove disconnected nodes (nodes with neither incoming nor outgoing edges)
    disconnected_nodes = [node for node in G.nodes() if G.in_degree(node) == 0 and G.out_degree(node) == 0]
    G.remove_nodes_from(disconnected_nodes)
    
    # Identify input and output nodes
    input_nodes = [node for node in G.nodes() if G.in_degree(node) == 0]
    output_nodes = [node for node in G.nodes() if G.out_degree(node) == 0]
       
    # Set a specific attribute for input and output nodes to distinguish them
    for node in input_nodes:
        G.nodes[node]['type'] = 'input'
    for node in output_nodes:
        G.nodes[node]['type'] = 'output'
    
    return G


# +


def count_nodes(adj_matrix):
    """
    Count the number of inputs, outputs, and the rest of the nodes in a DAG from an adjacency matrix.
    """
    # Create the DAG from the adjacency matrix
    G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
    
    # Remove disconnected nodes (nodes with neither incoming nor outgoing edges)
    disconnected_nodes = [node for node in G.nodes() if G.in_degree(node) == 0 and G.out_degree(node) == 0]
    G.remove_nodes_from(disconnected_nodes)
    
    # Identify input and output nodes
    input_nodes = [node for node in G.nodes() if G.in_degree(node) == 0]
    output_nodes = [node for node in G.nodes() if G.out_degree(node) == 0]
    
    # Count the number of non-input/output nodes
    non_io_nodes = len(G.nodes()) - len(input_nodes) - len(output_nodes)
    
    return len(input_nodes), len(output_nodes), non_io_nodes





# -



def hill_function(x, ymax, ymin, Ka, n):
    """
    Compute the output of a Hill function.

    Parameters:
    - x: Input signal to the node.
    - ymax: Maximum output level.
    - Ka: Affinity constant.
    - n: Hill coefficient.

    Returns:
    - Output signal of the node.
    """
    return ymin + (((Ka**n)*(ymax - ymin)) / (x**n + Ka**n))



# +
def propagate_signals_through_graph(G, input_signals):
    """
    Dynamically propagate signals through a Directed Acyclic Graph (DAG), where each node
    performs a computation based on its configured function—except for nodes marked as 'output',
    which directly pass through the sum of signals they receive.

    Inputs:
    - G (nx.DiGraph): The graph representing the network, with nodes marked as 'output' where applicable.
    - input_signals (dict): A mapping from node indices (inputs) to their respective initial signals.

    Outputs:
    - A dictionary mapping node indices labeled as 'output' to their respective final output signals.
    """
    
    c = 0.4 # conversion factor that we need for the output plasmid
    
    outputs = {node: 0 for node in G.nodes()}  # Initialize output signals for all nodes

    # Assign initial input signals
    for node, signal in input_signals.items():
        outputs[node] = signal

    # Process nodes in topological order
    for node in nx.topological_sort(G):
        # For nodes with the 'type' attribute set to 'output', sum the signals from all predecessors and multiply by c
        if G.nodes[node].get('type') == 'output':
            outputs[node] = c*(sum(outputs[predecessor] for predecessor in G.predecessors(node)))
        elif node not in input_signals:  # Processing nodes
            # Accumulate input from all predecessors
            cumulative_input = sum(outputs[predecessor] for predecessor in G.predecessors(node))
            
            # Retrieve the node's computation parameters
            ymax = G.nodes[node].get('ymaxa', 0)
            ymin = G.nodes[node].get('ymina', 0)
            Ka = G.nodes[node].get('Ka', 1)
            n = G.nodes[node].get('n', 1)

            # Store the signal after processing by the node's Hill function
            outputs[node] = hill_function(cumulative_input, ymax, ymin, Ka, n)

    # Filter the outputs to include only those for nodes marked as 'output'
    output_signals = {node: signal for node, signal in outputs.items() if G.nodes[node].get('type') == 'output'}

    return output_signals





# +
def propagate_signals_through_graph_logic(G, input_signals):
    """
    Dynamically propagate signals through a Directed Acyclic Graph (DAG), where each node
    performs NOR logic for 2-input nodes and NOT logic for 1-input nodes, except for nodes marked as 'output',
    which behave as a buffer if they have only one input edge, and behave like an OR gate if they have more than one input edge.

    Inputs:
    - G (nx.DiGraph): The graph representing the network, with nodes marked as 'output' where applicable.
    - input_signals (dict): A mapping from node indices (inputs) to their respective initial signals.

    Outputs:
    - A dictionary mapping node indices labeled as 'output' to their respective final output signals.
    """
    outputs = {node: 0 for node in G.nodes()}  # Initialize output signals for all nodes

    # Assign initial input signals
    for node, signal in input_signals.items():
        outputs[node] = signal

    # Process nodes in topological order
    for node in nx.topological_sort(G):
        # Output nodes logic
        if G.nodes[node].get('type') == 'output':
            predecessor_signals = [outputs[predecessor] for predecessor in G.predecessors(node)]
            if len(predecessor_signals) == 1:
                # Buffer behavior for one input
                outputs[node] = predecessor_signals[0]
            else:
                # OR gate behavior for more than one input
                outputs[node] = int(any(predecessor_signals))
        # Processing nodes (NOR and NOT logic)
        elif node not in input_signals:
            predecessor_signals = [outputs[predecessor] for predecessor in G.predecessors(node)]
            if len(predecessor_signals) == 1:
                # Implement NOT logic for 1-input nodes
                outputs[node] = 0 if predecessor_signals[0] else 1
            elif len(predecessor_signals) == 2:
                # Implement NOR logic for 2-input nodes
                outputs[node] = 0 if any(predecessor_signals) else 1
            else:
                # Handle nodes with unsupported number of inputs
                print(f"Warning: Node {node} has an unsupported number of inputs: {len(predecessor_signals)}.")

    # Filter to include only output node signals
    output_signals = {node: signal for node, signal in outputs.items() if G.nodes[node].get('type') == 'output'}

    return output_signals




# -

def simulate_signal_propagation(G, input_signals_list):
    """
    Simulate the propagation of signals through a graph for a list of input signal sets.

    Parameters:
    - G: A graph structure on which the signal propagation is simulated.
    - input_signals_list: A list of dictionaries, where each dictionary contains input signals for nodes.

    Returns:
    - A list containing the output signals for each input signal set after propagation through the graph.
    """
    all_outputs = []

    for input_signals in input_signals_list:
        # Propagate signals through the graph
        outputs = propagate_signals_through_graph(G, input_signals)
        
        # Collect the output for the current input set
        all_outputs.append(outputs)
    
    return all_outputs



def simulate_signal_propagation_binary(G, input_signals_list):
    """
    Simulate the propagation of signals through a graph for a list of input signal sets.

    Parameters:
    - G: A graph structure on which the signal propagation is simulated.
    - input_signals_list: A list of dictionaries, where each dictionary contains input signals for nodes.

    Returns:
    - A list containing the output signals for each input signal set after propagation through the graph.
    """
    all_outputs = []

    for input_signals in input_signals_list:
        # Propagate signals through the graph
        outputs = propagate_signals_through_graph_logic(G, input_signals)
        
        # Collect the output for the current input set
        all_outputs.append(outputs)
    
    return all_outputs



def compare_simulate_signal_propagation_binary_outout_to_hex(input_list, hex_number):
    # Step 1: Convert list of dictionaries into a binary number string
    # Dynamically use the first (and assumed only) key in each dictionary
    binary_string = ''.join(str(list(d.values())[0]) for d in input_list)
    
    # Step 2: Convert the binary number string to an integer
    binary_integer = int(binary_string, 2)
    
    # Convert the hex_number to an integer for comparison
    hex_integer = int(hex_number, 16)
    
    # Step 3 & 4: Compare and return the result
    return binary_integer == hex_integer


def count_nodes_with_no_incoming_edges_but_not_isolated(matrix):
    # Count of nodes with no incoming edges
    no_incoming_edges = np.sum(matrix, axis=0) == 0
    # Count of nodes with no outgoing edges
    no_outgoing_edges = np.sum(matrix, axis=1) == 0
    # Nodes that are not isolated (have at least one outgoing edge or at least one incoming edge)
    not_isolated = ~(no_incoming_edges & no_outgoing_edges)
    # Nodes with no incoming edges but are not isolated
    result = no_incoming_edges & not_isolated
    # Return the count of such nodes
    return np.sum(result)


def is_list_empty(lst):
    # Returns True if the list is empty, False otherwise
    return not lst


# # Functions for Simulated annealing 

# +
import math
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

def simulated_annealing_cello_v2(df, adj_matrix, initial_solution, Tmax, Tmin, iterations_before_Tmin, iterations_at_Tmin=10000):

    current_solution = initial_solution

    # Assuming the presence of these functions and variables:
    # assign_representations_with_io_nodes_3, simulate_signal_propagation, simulate_signal_propagation_binary, calculate_circuit_score, perform_action
    
    G = assign_representations_with_io_nodes_3(adj_matrix, df, current_solution)
    actual_logic = simulate_signal_propagation(G, input_signals_list_small_molecules)
    expected_logic = simulate_signal_propagation_binary(G, input_signals_list_binary)
    current_score = calculate_circuit_score(expected_logic, actual_logic)
    optimal_score = current_score
    optimal_solution = current_solution

    LOGMAX = math.log10(Tmax)
    LOGMIN = math.log10(Tmin)
    LOGINC = (LOGMAX - LOGMIN) / iterations_before_Tmin
    
    temperature = Tmax
    i = 0
    no_improve = 0
    iterations_since_Tmin = 0

    total_iterations = iterations_before_Tmin + iterations_at_Tmin
    pbar = tqdm(total=total_iterations, desc="Optimizing", leave=True)

    # Initialize a list to track optimal scores
    optimal_scores = []
    iteration_numbers = []

    while True:
        new_solution = perform_action(current_solution, df)
        
        G = assign_representations_with_io_nodes_3(adj_matrix, df, new_solution)
        actual_logic = simulate_signal_propagation(G, input_signals_list_small_molecules)
        expected_logic = simulate_signal_propagation_binary(G, input_signals_list_binary)
        
        new_score = calculate_circuit_score(expected_logic, actual_logic)

        if acceptance_probability(current_score, new_score, temperature) > random.random():
            current_solution = new_solution
            current_score = new_score
            if new_score > optimal_score:
                optimal_score = new_score
                optimal_solution = new_solution
                #no_improve = 0                
            else:
                pass
                #if (temperature <= Tmin):
                #    no_improve += 1
        else:
            pass
            #if (temperature <= Tmin):
            #    no_improve += 1

        # Update the list with the current optimal score and iteration number
        optimal_scores.append(optimal_score)
        iteration_numbers.append(i)

        if i >= total_iterations:
            break  # Break the loop if no improvement is observed for max_no_improve iterations

        if temperature > Tmin:
            logTemperature = LOGMAX - i * LOGINC
            temperature = math.pow(10, logTemperature)
        else:
            temperature = Tmin  # Keep temperature at Tmin, do not decrease further
            iterations_since_Tmin += 1
        i += 1
        
        #print("iteration: ", i)
        #print("iterations_since_Tmin: ", iterations_since_Tmin)
        #print("temperature: ", temperature)
        
        pbar.update(1)

    pbar.close()

    # Plotting the iterations versus the optimal score

    # Set global font to Arial
    plt.rcParams['font.family'] = 'DejaVu Sans'

    plt.figure(figsize=(8, 4))
    plt.plot(iteration_numbers, optimal_scores, marker='o', markersize=3, linestyle='--', color='b')
    plt.title('Iterations vs Highest Score', fontdict={'fontname': 'DejaVu Sans'})
    plt.xlabel('Iteration', fontdict={'fontname': 'DejaVu Sans'})
    plt.ylabel('Circuit Score', fontdict={'fontname': 'DejaVu Sans'})
    # plt.grid(True)  # Keep this commented out to maintain a background without lines
    plt.show()

    return optimal_score, optimal_solution  # Return both optimal score and solution


# -
def calculate_circuit_score(logical_list, physical_list):
    # Initialize variables to store the lowest physical value for True and
    # the highest physical value for False
    lowest_true = None
    highest_false = None

    # Iterate through the lists
    for logical_dict, physical_dict in zip(logical_list, physical_list):
        logical_key = list(logical_dict.keys())[0]
        logical_value = list(logical_dict.values())[0]
        physical_key = list(physical_dict.keys())[0]
        physical_value = list(physical_dict.values())[0]

        if logical_value:  # True
            # Check if lowest_true is None or the current physical value is lower
            if lowest_true is None or physical_value < lowest_true:
                lowest_true = physical_value
        else:  # False
            # Check if highest_false is None or the current physical value is higher
            if highest_false is None or physical_value > highest_false:
                highest_false = physical_value

    # Calculate the ratio if both lowest_true and highest_false are found
    if lowest_true is not None and highest_false is not None:
        ratio = lowest_true / highest_false
        return ratio
    else:
        return None  # Return None if either lowest_true or highest_false is not found


def perform_action_v2(selected_indices, repressor_data):
    # Randomly select an element from selected_indices
    
    """
    
    This version will  incorporate toxicity and roadblocking 
    
    """
    
    selection1 = random.choice(selected_indices)

    
    # Randomly select an action (1 or 2)
    action = random.randint(1, 2)
    
    if action == 1:
        # Action 1: Select another element in selected_indices
        #print("action 1")
        #print("selection 1: ", selection1)
        # Exclude selection1 from the choices
        available_selections = selected_indices.copy()
        available_selections.remove(selection1)
        selection2 = random.choice(available_selections)
        #print("selection 2: ", selection2)
        
        # Swap selection1 with selection2 in selected_indices
        index_selection1 = selected_indices.index(selection1)
        #print("index_selection1 : ", index_selection1)
        
        index_selection2 = selected_indices.index(selection2)
        #print("index_selection2 : ", index_selection2)
        
        selected_indices[index_selection1] = selection2
        selected_indices[index_selection2] = selection1
        
    else:
        # Action 2: Select an index in the pandas data frame
        #print("action 2")
        #print("selection 1: ", selection1)
        
        # Get the repressor at selection1
        selected_repressor = repressor_data.loc[selection1, 'Repressor']

        # Find indices of the same repressor as selection1, excluding selection1 itself
        same_repressor_indices = repressor_data.index[repressor_data['Repressor'] == selected_repressor].tolist()
        same_repressor_indices.remove(selection1)  # Remove selection1 from the list

        # Get repressors for selected indices
        selected_repressors = repressor_data.loc[selected_indices, 'Repressor'].unique()

        # Find indices with no overlapping repressors with selected_indices
        non_overlapping_indices = repressor_data.index[~repressor_data['Repressor'].isin(selected_repressors)].tolist()
        
        # Join same_repressor_indices and non_overlapping_indices
        available_selections = same_repressor_indices + non_overlapping_indices

        # Make second selection
        selection2 = random.choice(available_selections)
        #print("selection 2: ", selection2)
        
        #Make replacement
        index_selection1 = selected_indices.index(selection1)        
        selected_indices[index_selection1] = selection2
        
    return selected_indices


def swap_within_circuit(current_solution, gateA, gateB):

    current_solution_copy = current_solution.copy()    
    index_gateA = current_solution_copy.index(gateA) 
    index_gateB = current_solution_copy.index(gateB)    
    
    current_solution_copy[index_gateA] = gateB
    current_solution_copy[index_gateB] = gateA

    return current_solution_copy


def swap_with_library(current_solution, circuit_gate, library_gate):

    current_solution_copy = current_solution.copy()    
    index_circuit_gate = current_solution_copy.index(circuit_gate) 
    current_solution_copy[index_circuit_gate] = library_gate
    
    return current_solution_copy






def perform_action(selected_indices, repressor_data):
    # Randomly select an element from selected_indices
    selection1 = random.choice(selected_indices)
       
    # Randomly select an action (1 or 2)
    action = random.randint(1, 2)
    
    if action == 1:
        # Action 1: Select another element in selected_indices
        #print("action 1")
        #print("selection 1: ", selection1)
        # Exclude selection1 from the choices
        available_selections = selected_indices.copy()
        available_selections.remove(selection1)
        selection2 = random.choice(available_selections)
        #print("selection 2: ", selection2)
        
        # Swap selection1 with selection2 in selected_indices
        index_selection1 = selected_indices.index(selection1)
        #print("index_selection1 : ", index_selection1)
        
        index_selection2 = selected_indices.index(selection2)
        #print("index_selection2 : ", index_selection2)
        
        selected_indices[index_selection1] = selection2
        selected_indices[index_selection2] = selection1
        
    else:
        # Action 2: Select an index in the pandas data frame
        #print("action 2")
        #print("selection 1: ", selection1)
        
        # Get the repressor at selection1
        selected_repressor = repressor_data.loc[selection1, 'Repressor']

        # Find indices of the same repressor as selection1, excluding selection1 itself
        same_repressor_indices = repressor_data.index[repressor_data['Repressor'] == selected_repressor].tolist()
        same_repressor_indices.remove(selection1)  # Remove selection1 from the list

        # Get repressors for selected indices
        selected_repressors = repressor_data.loc[selected_indices, 'Repressor'].unique()

        # Find indices with no overlapping repressors with selected_indices
        non_overlapping_indices = repressor_data.index[~repressor_data['Repressor'].isin(selected_repressors)].tolist()
        
        # Join same_repressor_indices and non_overlapping_indices
        available_selections = same_repressor_indices + non_overlapping_indices

        # Make second selection
        selection2 = random.choice(available_selections)
        #print("selection 2: ", selection2)
        
        #Make replacement
        index_selection1 = selected_indices.index(selection1)        
        selected_indices[index_selection1] = selection2
        
    return selected_indices

# +
import math
import random

def acceptance_probability(old_score, new_score, temperature):
    """
    Calculate the acceptance probability of a new solution.
    Modify this to favor higher scores.
    """
    if new_score > old_score:
        return 1.0
    else:
        return math.exp((new_score - old_score) / temperature)

def simulated_annealing_cello_v1(df, adj_matrix, initial_solution, Tmax, C, max_no_improve=10000):
    current_solution = initial_solution
    
    G = assign_representations_with_io_nodes_3(adj_matrix, df, current_solution)
    actual_logic = simulate_signal_propagation(G, input_signals_list_small_molecules)
    expected_logic = simulate_signal_propagation_binary(G, input_signals_list_binary)
    current_score = calculate_circuit_score(expected_logic, actual_logic)

    temperature = Tmax
    i = 0
    no_improve = 0

    while temperature > 0 or no_improve < max_no_improve:
        new_solution = perform_action(current_solution, df)
        
        G = assign_representations_with_io_nodes_3(adj_matrix, df, new_solution)
        actual_logic = simulate_signal_propagation(G, input_signals_list_small_molecules)
        expected_logic = simulate_signal_propagation_binary(G, input_signals_list_binary)
        
        new_score = calculate_circuit_score(expected_logic, actual_logic)

        if acceptance_probability(current_score, new_score, temperature) > random.random():
            current_solution = new_solution
            if new_score > current_score:
                current_score = new_score
                no_improve = 0
                print(f"Iteration: {i}, Temperature: {temperature:.4f}, Improved Score: {current_score}")
            else:
                no_improve += 1
        else:
            no_improve += 1

        if temperature > 0:
            temperature = Tmax * math.exp(-C * i)
            i += 1
        elif no_improve >= max_no_improve:
            break

    return current_solution, current_score


# +
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import string


def draw_network_with_colors_and_labels_from_G(G, seed = 42):
    color_mapping = {
        'AmeR': 'lightblue',
        'HlyIIR': 'lightgreen',
        'PhlF': 'orange',
        'AmtR': 'blue',
        'IcaRA': 'lightpink',
        'PsrA': 'thistle',  # Valid color name
        'BetI': 'darkblue',
        'LitR': 'purple',
        'QacR': 'green',
        'BM3R1': 'red',
        'LmrA': 'darkorange',
        'SrpR': 'darkgreen'
    }

    # Prepare the edge colors and labels for each node
    edge_colors = []
    labels = {}
    
    # Identify input and output nodes
    input_nodes = [node for node in G.nodes() if G.in_degree(node) == 0]
    output_nodes = [node for node in G.nodes() if G.out_degree(node) == 0]
    
    # Sort input nodes in reverse order by their node number
    sorted_input_nodes = sorted(input_nodes, reverse=True)
    # Generate labels for input nodes using the alphabet
    #alphabet = list(string.ascii_uppercase)
    #input_labels = {node: alphabet[i] for i, node in enumerate(sorted_input_nodes)}
    
    #input_labels = {2: 'A, 2', 1: 'B, 1', 0: 'C, 0'}
    # This will label the first node in the sorted list as "A, <node>", second as "B, <node>", etc.
    input_labels = {node: f"{string.ascii_uppercase[i]}, {node}" for i, node in enumerate(sorted_input_nodes)}    
  
    for node, data in G.nodes(data=True):
        node_edge_color = 'gray'  # Default edge color
        node_label = 'N/A'  # Default label
        
        # Update label and edge color for input nodes
        if node in input_labels:
            node_label = input_labels[node]
            node_edge_color = 'yellow'  # Example color for input node edges
        # Label output nodes as "Y"
        elif node in output_nodes:
            node_label = "Y"
            node_edge_color = 'yellow'  # Example color for output node edges
        # For other nodes, use repressor and RBS for labeling
        elif 'Repressor' in data and 'RBS' in data:
            repressor = data['Repressor']
            rbs = data['RBS']
            node_label = f"{repressor}\n{rbs}"
            node_edge_color = color_mapping.get(repressor, 'y')
        
        labels[node] = node_label
        edge_colors.append(node_edge_color)

    # Draw the network
    pos = nx.spring_layout(G, seed=seed)
    nx.draw(G, pos, node_color='white', with_labels=False, node_size=800, 
            edgecolors=edge_colors, linewidths=2)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)

    plt.show()


# -

# ## ML-based desings 

# +
def visualize_cuts(G, node, cuts):
    pos = nx.spring_layout(G)
    
    # Identify PIs and POs
    PIs = [n for n in G.nodes() if G.in_degree(n) == 0]
    POs = [n for n in G.nodes() if G.out_degree(n) == 0]
    
    plt.figure(figsize=(10, 6))
    # Draw all nodes
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=700)
    # Highlight PIs and POs by drawing them with yellow borders and transparent fill
    nx.draw_networkx_nodes(G, pos, nodelist=PIs+POs, node_color='none', edgecolors='yellow', linewidths=2)
    plt.title('Original Graph')
    plt.show()
    
    for i, cut in enumerate(cuts, start=1):
        plt.figure(figsize=(10, 6))
        # Draw all nodes
        nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=700)
        # Highlight nodes in the cut
        nx.draw_networkx_nodes(G, pos, nodelist=cut, node_color='orange')
        # Highlight the target node
        nx.draw_networkx_nodes(G, pos, nodelist=[node], node_color='red')
        # Highlight PIs and POs by drawing them with yellow borders and transparent fill, again
        nx.draw_networkx_nodes(G, pos, nodelist=PIs+POs, node_color='none', edgecolors='yellow', linewidths=2)
        plt.title(f'Cut {i}: {cut}')
        plt.show()
        
        
def visualize_subgraphs_from_cuts(G, target_node, cuts):
    pos = nx.spring_layout(G)  # Generate positions for consistent layouts
    
    for i, cut in enumerate(cuts, start=1):
        plt.figure(figsize=(10, 6))

        # Draw the original graph in light blue to establish the background
        nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=700)

        # Highlight the cut nodes in orange
        nx.draw_networkx_nodes(G, pos, nodelist=cut, node_color='orange')

        # Remove nodes in the cut from a copy of the graph
        G_modified = G.copy()
        G_modified.remove_nodes_from(cut)

        # Draw the predecessor subgraph in green if the target node exists
        if target_node in G_modified:
            ancestors = nx.ancestors(G_modified, target_node) | {target_node}
            subgraph = G_modified.subgraph(ancestors)
            nx.draw(subgraph, pos, with_labels=True, node_color='green', edge_color='gray', node_size=700)
        
        # Highlight the target node in red
        if target_node in G:
            nx.draw_networkx_nodes(G, pos, nodelist=[target_node], node_color='red')

        # Identify and highlight PIs and POs with yellow borders for the entire graph
        PIs = [n for n in G.nodes() if G.in_degree(n) == 0]
        POs = [n for n in G.nodes() if G.out_degree(n) == 0]
        nx.draw_networkx_nodes(G, pos, nodelist=PIs+POs, node_color='none', edgecolors='yellow', linewidths=2)
        
        plt.title(f'Enhanced Visualization for Cut {i}: {cut}, around {target_node}')
        plt.show()
        


# +
def generate_subgraph(G, target_node, cut, draw = True):
    # Create a copy of G to work on
    '''
    This function creates a subgraph that includes:

        1. Target_node
        2. All the ancestors of target_node that are also decendants of any node in 'cut'
        3. The nodes in cut

    The function then removes edges where both nodes are in 'cut'
    '''    

    # Step 1 & 2: Find all ancestors of target_node and descendants of nodes in 'cut'
    ancestors_of_target = nx.ancestors(G, target_node)
    descendants_of_cut = set()
    for node in cut:
        descendants_of_cut.update(nx.descendants(G, node))
    # Intersection of ancestors and descendants, plus target_node, plus nodes in 'cut'
    required_nodes = ancestors_of_target.intersection(descendants_of_cut).union(cut, {target_node})

    # Create a subgraph with these nodes
    subgraph = G.subgraph(required_nodes).copy()

    # Step 3: Remove edges where both nodes are in 'cut'
    edges_to_remove = [(u, v) for u, v in subgraph.edges() if u in cut and v in cut]
    subgraph.remove_edges_from(edges_to_remove)    
    
    
    if (draw):
        # Visualize
        pos = nx.spring_layout(G)  # Positions for all nodes

        # Draw the original graph lightly
        nx.draw_networkx_nodes(G, pos, node_color="grey", alpha=0.2)
        nx.draw_networkx_edges(G, pos, edge_color="grey", alpha=0.2)
        nx.draw_networkx_labels(G, pos)

        # Highlight the subgraph
        nx.draw_networkx_nodes(subgraph, pos, nodelist=subgraph.nodes(), node_color="blue", alpha=0.9)
        nx.draw_networkx_edges(subgraph, pos, edgelist=subgraph.edges(), edge_color="blue", alpha=0.9)

        # Highlight the cut nodes
        nx.draw_networkx_nodes(G, pos, nodelist=cut, node_color="red", alpha=0.9)

        # Highlight the target node
        nx.draw_networkx_nodes(G, pos, nodelist=[target_node], node_color="green", alpha=0.9)

        plt.show()
    return subgraph


def visualize_result(G, target_node, cut):    

    

    subgraph_nodes = set([target_node]) | set(cut)    
    
    # Visualize
    pos = nx.spring_layout(G)  # Positions for all nodes

    # Draw the original graph lightly
    nx.draw_networkx_nodes(G, pos, node_color="grey", alpha=0.2)
    nx.draw_networkx_edges(G, pos, edge_color="grey", alpha=0.2)
    nx.draw_networkx_labels(G, pos)

    # Highlight the subgraph
    # nx.draw_networkx_nodes(subgraph, pos, nodelist=subgraph_nodes, node_color="blue", alpha=0.9)
    # nx.draw_networkx_edges(subgraph, pos, edgelist=subgraph.edges(), edge_color="blue", alpha=0.9)

    # Highlight the cut nodes
    nx.draw_networkx_nodes(G, pos, nodelist=cut, node_color="red", alpha=0.9)

    # Highlight the target node
    nx.draw_networkx_nodes(G, pos, nodelist=[target_node], node_color="green", alpha=0.9)

    plt.show()
    return subgraph


# +
#Modify this function to add this functionality. The node with 'output' behaves like an OR gate, and it can have 1 to n inputs.

def evaluate_node(G, node, node_values):
    """
    Recursively evaluates the signal at a node based on incoming signals
    and the logic gate behavior (NOT for 1 incoming edge, NOR for 2 incoming edges, Buffer for 'output' type).
    """
    if node in node_values:
        # Return the already computed signal value for this node
        return node_values[node]
    
    # Check if the node is a buffer ('output' type)
    if G.nodes[node].get('type') == 'output':
        # For a buffer, directly pass the signal from the first (and only) incoming edge
        source_node = list(G.in_edges(node))[0][0]
        node_values[node] = evaluate_node(G, source_node, node_values)
    else:
        incoming_edges = list(G.in_edges(node))
        if len(incoming_edges) == 1:
            # NOT gate logic
            source_node = incoming_edges[0][0]
            source_value = evaluate_node(G, source_node, node_values)
            node_values[node] = 1 - source_value  # NOT operation
        elif len(incoming_edges) == 2:
            # NOR gate logic
            source_node1 = incoming_edges[0][0]
            source_node2 = incoming_edges[1][0]
            source_value1 = evaluate_node(G, source_node1, node_values)
            source_value2 = evaluate_node(G, source_node2, node_values)
            node_values[node] = int(not (source_value1 or source_value2))  # NOR operation
        else:
            # Nodes with no incoming edges or more than 2 are not supported
            raise ValueError(f"Node {node} has an unsupported number of incoming edges: {len(incoming_edges)}")
    
    return node_values[node]

def calculate_truth_table(G):
    """
    Computes the truth table for a circuit represented by graph G,
    where nodes with 1 incoming edge act as NOT gates, and nodes with 2 incoming edges act as NOR gates.
    Output gates are treated a buffers.
    """
    # Identify input and output nodes
    input_nodes = sorted([node for node in G.nodes() if G.in_degree(node) == 0])
    output_nodes = sorted([node for node in G.nodes() if G.out_degree(node) == 0])
    
    # Initialize the truth table
    truth_table = {}

    # Iterate over all possible input combinations for the input nodes
    for inputs in itertools.product([0, 1], repeat=len(input_nodes)):
        node_values = dict(zip(input_nodes, inputs))  # Map each input node to its input value
        
        # Evaluate the output for each output node (assuming a single output node for simplicity)
        outputs = tuple(evaluate_node(G, node, node_values) for node in output_nodes)
        truth_table[inputs] = outputs

    return truth_table


def evaluate_node_v2(G, node, node_values):
    """
    Recursively evaluates the signal at a node based on incoming signals
    and the logic gate behavior (NOT for 1 incoming edge, NOR for 2 incoming edges,
    and OR for 'output' type with 1 to n inputs).
    """
    if node in node_values:
        # The value is already in node_values so simply return it. This will be true for input nodes.
        return node_values[node]

    incoming_edges = list(G.in_edges(node))
    
    # Check if the node is a buffer ('output' type), which now acts as an OR gate
    if G.nodes[node].get('type') == 'output':
        # For an 'output' node, compute the OR operation across all incoming edges
        or_result = 0
        for edge in incoming_edges:
            #edge is (source node, current node), so edge[0] is the source node 
            source_node = edge[0]
            source_value = evaluate_node_v2(G, source_node, node_values)
            or_result = or_result or source_value
        node_values[node] = or_result
    elif len(incoming_edges) == 1:
        # NOT gate logic
        source_node = incoming_edges[0][0]
        source_value = evaluate_node_v2(G, source_node, node_values)
        node_values[node] = 1 - source_value  # NOT operation
    elif len(incoming_edges) == 2:
        # NOR gate logic for 2 inputs, but can extend this to behave differently for >2 inputs if needed
        nor_result = 1
        for edge in incoming_edges:
            source_node = edge[0]
            source_value = evaluate_node_v2(G, source_node, node_values)
            nor_result = nor_result and (1 - source_value)  # NOR operation as a series of NOT(OR)
        node_values[node] = nor_result
    else:
        # This else branch could be for nodes with no incoming edges, which would be an error in most logic circuits
        raise ValueError(f"Node {node} cannot be evaluated. It has {len(incoming_edges)} incoming edges.")
    
    return node_values[node]

def calculate_truth_table_v2(G):
    """
    Computes the truth table for a circuit represented by graph G,
    where nodes with 1 incoming edge act as NOT gates, and nodes with 2 incoming edges act as NOR gates.
    Output gates are treated like OR
    """
    # Identify input and output nodes
    input_nodes = sorted([node for node in G.nodes() if G.in_degree(node) == 0])
    output_nodes = sorted([node for node in G.nodes() if G.out_degree(node) == 0])
    
    # Initialize the truth table
    truth_table = {}

    # Iterate over all possible input combinations for the input nodes
    for inputs in itertools.product([0, 1], repeat=len(input_nodes)):
        node_values = dict(zip(input_nodes, inputs))  # Map each input node to its input value
        
        # Evaluate the output for each output node (assuming a single output node for simplicity)
        #outputs = tuple(evaluate_node_v2(G, node, node_values) for node in output_nodes)
        results = []
        for output_node in output_nodes:
            value = evaluate_node_v2(G, output_node, node_values)
            results.append(value)
        outputs = tuple(results)
        truth_table[inputs] = outputs

    return truth_table


def write_tt_files(G, truth_table, folder: Path, basename="truth_table"):
    folder.mkdir(parents=True, exist_ok=True)

    # Determine IO names in the same way calculate_truth_table_v2 does
    input_nodes  = sorted([n for n in G.nodes() if G.in_degree(n) == 0])
    output_nodes = sorted([n for n in G.nodes() if G.out_degree(n) == 0])

    # Canonical input order: last input toggles fastest
    combos = list(itertools.product([0, 1], repeat=len(input_nodes)))
    header = [*(str(x) for x in input_nodes), *(str(y) for y in output_nodes)]

    # --- CSV ---
    csv_path = folder / f"{basename}.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for iv in combos:
            ov = truth_table.get(iv)
            if ov is None:  # skip if something is missing
                continue
            w.writerow([*iv, *ov])

    # --- Pretty text (matches what you’d print) ---
    lines = []
    lines.append(" | ".join(header))
    lines.append("-" * len(lines[0]))
    for iv in combos:
        ov = truth_table.get(iv)
        if ov is None:
            continue
        lines.append(" ".join(map(str, iv)) + " | " + " ".join(map(str, ov)))
    (folder / f"{basename}.txt").write_text("\n".join(lines) + "\n")


def truth_table_to_index(truth_table):
    # Converts truth table outputs to a binary number, then to decimal
    binary_string = ''.join(str(output[0]) for output in truth_table.values())  # Assumes single output node
    return int(binary_string, 2)

# +
def find_feasible_cuts(G, target_node, max_cut_size=3, filter_redundant=True):
    feasible_cuts = []

    # Find all ancestors of the target node. 
    predecessors = nx.ancestors(G, target_node)
    predecessors.add(target_node)  # Ensure the target node is considered

    # Determine Primary Inputs (PIs) - nodes with no predecessors
    PIs = [node for node in G.nodes() if G.in_degree(node) == 0 and node in predecessors]
    

    def is_fanout_free(cut):
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



    def is_valid_cut(cut, target):
        # Ensure the target is not in the cut
        if target in cut:
            return False

         
        #For each node in PIs, check the paths to target, and make sure it has at least one node in cut. 
        
        paths_meet_criteria = True  # Assume all paths meet the criteria initially

        for pi_node in PIs:
            # Find all paths from PI node to target
            all_paths = list(nx.all_simple_paths(G, source=pi_node, target=target))

            # Check that all paths include a node from cut
            path_includes_cut = all(set(path).intersection(cut) for path in all_paths)

            if not path_includes_cut:
                #print(f"No path from PI node {pi_node} to target node {target} includes any node from cut.")
                paths_meet_criteria = False
                return False  # Can break early if any PI node doesn't meet the criteria

        # Create a subgraph excluding the cut nodes, if they exist in G
        #subgraph_nodes = [node for node in G.nodes() if node not in cut]
        #modified_graph = G.subgraph(subgraph_nodes).copy()

        # Check if the target is still reachable from any PI
        #for pi in PIs:
        #    if pi in modified_graph and target in modified_graph:
        #        if nx.has_path(modified_graph, pi, target):
        #            return False

        # Check if the cut is fanout-free
        return is_fanout_free(cut)


    # Generate all combinations of predecessor nodes for cut sizes up to max_cut_size
    for cut_size in range(1, min(max_cut_size + 1, len(predecessors))): #Cut sizes, and deals with the case where predecesors is less than k (i.e., max_cut_size)
        for cut in combinations(predecessors, cut_size): #combination because the order does not matter. Need to just check each node in a potential cut
            if is_valid_cut(cut, target_node): #check if it is a valid cut
                feasible_cuts.append(set(cut))

    # Filter out redundant cuts if required
    if filter_redundant:
        non_redundant_cuts = []
        for cut in sorted(feasible_cuts, key=len):
            if not any(other_cut <= cut for other_cut in non_redundant_cuts):
                non_redundant_cuts.append(cut)
        feasible_cuts = [tuple(cut) for cut in non_redundant_cuts]
    else:
        feasible_cuts = [tuple(cut) for cut in feasible_cuts]

    return feasible_cuts





# -

# Rewrite

# +
def rename_nodes_by_mapping_input_nodes(dag_to_rename, mapping_source, mapping_target):
    
    #mapping_source is the subgraph's input nodes
    
    #mapping_target is the replacement graph's input nodes 
    
    
    if len(mapping_source) != len(mapping_target):
        raise ValueError("The mapping arrays must be of equal length.")

    # Identify nodes with incoming edges
    nodes_with_incoming_edges = [node for node in dag_to_rename.nodes() if dag_to_rename.in_degree(node) > 0]

    # Create a mapping for nodes to add a '*' for identified nodes, and directly map others according to original mapping
    intermediate_mapping = {}
    for node in dag_to_rename.nodes():
        if node in nodes_with_incoming_edges:
            # Append '*' to node IDs that have incoming edges
            intermediate_mapping[node] = f"{node}*"
        else:
            # For nodes without input edges, check if they are in the original mapping target list
            if node in mapping_target:
                # Map directly to the new label if in the target list, otherwise keep the original label
                intermediate_mapping[node] = mapping_source[mapping_target.index(node)]
            else:
                # If the node is not intended to be renamed, it retains its original label
                intermediate_mapping[node] = node

    # Applying the intermediate mapping to the DAG
    dag_copy = nx.relabel_nodes(dag_to_rename, intermediate_mapping, copy=True)

    return dag_copy


def rename_nodes_by_mapping_exit_points(dag_to_rename, mapping_source):

    # Identify all nodes in dag_to_rename without outgoing edges (exit points)
    exit_points = [node for node in dag_to_rename.nodes() if dag_to_rename.out_degree(node) == 0]
    
    # Ensure there is a matching number of exit points and source mappings
    if len(mapping_source) != len(exit_points):
        print("mapping_source: ", mapping_source)
        print("exit_points: ", exit_points)
        raise ValueError("The number of source mappings does not match the number of exit points.")
    
    # Prepare the mapping from exit points to the source mappings
    mapping = dict(zip(exit_points, mapping_source))
    
    # Apply the mapping to rename nodes
    dag_copy = nx.relabel_nodes(dag_to_rename, mapping, copy=True)
    
    return dag_copy
    
def substitute_subgraph(G, subgraph, replacement_graph, draw_relabeling = False):
    # Create a deep copy of G to work with
    G_copy = copy.deepcopy(G)
    
    replacement_graph_copy = copy.deepcopy(replacement_graph)
    
    #Deal with special case for subgraph having an output node. In this case, we do not remove the output node from replacement_graph
    has_output_type = any(subgraph.nodes[node].get('type') == 'output' for node in subgraph.nodes)

    if not has_output_type:
        # If False, remove the node in 'replacement_graph' of type 'output'
        nodes_to_remove = [node for node in replacement_graph_copy.nodes if replacement_graph_copy.nodes[node].get('type') == 'output']
        for node in nodes_to_remove:
            replacement_graph_copy.remove_node(node)
    
    if (draw_relabeling):
        visualize_graph_rewriting(replacement_graph_copy, title="Replacement graph before relabeling")
    
    # Identidy input nodes for both subgraph and replacement_graph and sort them
    subgraph_input_nodes = sorted([node for node in subgraph.nodes() if subgraph.in_degree(node) == 0])
    subgraph_output_nodes = sorted([node for node in subgraph.nodes() if subgraph.out_degree(node) == 0])
    
    replacement_graph_input_nodes = sorted([node for node in replacement_graph_copy.nodes() if replacement_graph_copy.in_degree(node) == 0])
    replacement_graph_output_nodes = sorted([node for node in replacement_graph_copy.nodes() if replacement_graph_copy.out_degree(node) == 0])
    
    # Assuming rename_nodes_by_mapping is defined to handle G_copy for conflict resolution
    # Relabel input nodes in replacement_graph to match subgraph input nodes (so the names will not change on the main graph G)
    mapping_source = subgraph_input_nodes
    mapping_target = replacement_graph_input_nodes
    replacement_graph_relabeled_input_nodes = rename_nodes_by_mapping_input_nodes(replacement_graph_copy, mapping_source, mapping_target)
    
    if (draw_relabeling):
        visualize_graph_rewriting(replacement_graph_relabeled_input_nodes, title="Replacement graph after entry point relabeling")
    
    # Relabel nodes in replacement_graph to match subgraph exit points
    mapping_source = subgraph_output_nodes
    #print("subgraph_output_nodes before relabel", subgraph_output_nodes)
    #mapping_target = replacement_graph_output_nodes #Not needed because it is only one and identifiable from lack of outgoing edges
    replacement_graph_relabeled_input_output_nodes = rename_nodes_by_mapping_exit_points(replacement_graph_relabeled_input_nodes, mapping_source)
    #print("subgraph_output_nodes after relabel", subgraph_output_nodes)

    if (draw_relabeling):    
        visualize_graph_rewriting(replacement_graph_relabeled_input_output_nodes, title="Replacement graph after entry and exit point relabeling")

    
    #Get the nodes in the cone 
    cone = copy.deepcopy(subgraph)
    cone.remove_nodes_from(subgraph_input_nodes)
    cone.remove_nodes_from(subgraph_output_nodes)

    #Remove edges from cut to target
    for cut_node in subgraph_input_nodes:
        for subgraph_output_node in subgraph_output_nodes:
            if G_copy.has_edge(cut_node, subgraph_output_node):  # Check if the edge exists before attempting to remove it
                G_copy.remove_edge(cut_node, subgraph_output_node)

    #Delete the input attrubute of the nodes in the replacement graph 
    for node in replacement_graph_relabeled_input_nodes:
        if node in replacement_graph_relabeled_input_output_nodes.nodes:
            # Check if the 'type' attribute exists for the node, then delete it
            if 'type' in replacement_graph_relabeled_input_output_nodes.nodes[node]:
                del replacement_graph_relabeled_input_output_nodes.nodes[node]['type']
    
    # Add the replacement graph to G_copy
    G_copy = nx.compose(G_copy, replacement_graph_relabeled_input_output_nodes)

    # Remove the previous nodes 
    G_copy.remove_nodes_from(cone.nodes())
    
    # Finally, remove the node IDs with *, and generate new integer labels not in G
    
    # Scan G_copy for nodes with a * in the Node ID
    nodes_with_star = [node for node in G_copy.nodes() if '*' in str(node)]
                
    # Generate new unique integer IDs for these nodes
    max_id = max([node for node in G_copy.nodes() if isinstance(node, int)], default=0)
    new_ids = iter(range(max_id + 1, max_id + 1 + len(nodes_with_star)))
    
    # Create a mapping of old IDs (with *) to new integer IDs
    star_to_int_mapping = {old_id: next(new_ids) for old_id in nodes_with_star}
    
    # Relabel nodes in G_copy according to the mapping
    G_copy = nx.relabel_nodes(G_copy, star_to_int_mapping, copy=False)
            
            
            
    return G_copy


# +
# This is a visualization helper function
def visualize_graph_rewriting(G, highlight_nodes=None, highlight_color='red', title="Graph"):
    plt.figure(figsize=(3, 3))  # Smaller figure size
    pos = nx.spring_layout(G)
    
    # Draw the entire graph
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=200, edge_color='gray')  # Further reduced node_size to 200
    
    # Highlight specific nodes if provided
    if highlight_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=highlight_nodes, node_color=highlight_color, node_size=200)  # Reduced node_size to 200
    
    plt.title(title)
    plt.show()



# -



# +
#Need to update this to account for output node

def check_implicit_OR_existence(G):
    """
    This can miss some 2-input OR because it only looks for 1 OR structure
    """

    G_copy = copy.deepcopy(G)
    
    # Identify nodes of type 'output'
    nodes_to_remove = [node for node in G_copy.nodes if G_copy.nodes[node].get('type') == 'output']
    
    # Remove identified nodes from the graph
    G_copy.remove_nodes_from(nodes_to_remove)
    
    
    # Find all output nodes (nodes without outgoing edges)
    output_nodes = [node for node in G_copy.nodes() if G_copy.out_degree(node) == 0]
    
    for output_node in output_nodes:
        # Check if the output node has only one incoming edge
        if G_copy.in_degree(output_node) != 1:
            #print(f"Output node {output_node} does not have exactly one incoming edge.")
            return False  # Condition 1 is violated
        
        # Get the predecessor of the output node (assuming exactly one due to previous check)
        predecessor = next(G_copy.predecessors(output_node))
        
        # Check if the predecessor has exactly two incoming edges
        if G_copy.in_degree(predecessor) != 2:
            #print(f"Predecessor node {predecessor} of output node {output_node} does not have exactly two incoming edges.")
            return False  # Condition 2 is violated

    # If none of the output nodes violate the conditions
    #print("All output nodes and their predecessors meet the specified conditions.")
    return True

  

# -

def add_implicit_OR_to_dag_v2(G, output_node, cut, cone):

    G_copy = copy.deepcopy(G)
    
    # Delete edges from nodes in 'cut' to 'output_node'
    for cut_node in cut:
        if G_copy.has_edge(cut_node, output_node):
            G_copy.remove_edge(cut_node, output_node)
    
    # Delete nodes in 'cone'
    for node in cone:
        if node in G_copy.nodes():
            G_copy.remove_node(node)
    
    # Add 1 edge from each node in 'cut' to 'output_node'
    for cut_node in cut:
        G_copy.add_edge(cut_node, output_node)
    
    return G_copy


#TO DO: udpdate this to use the faster version of find_feasible_cuts
def check_implicit_OR_existence_v2(G, output_node, size_input_to_OR_gate):
    
    """
    This function uses a feasible cut rooted at output_node to check for any structure that
    computes an OR. 

    size_input_to_OR_gate is the input size to the implicit OR

    Can be used for multi-output circuits.  

    Returns in a list of tuple for each possible OR
    
    is_there_an_implicit_OR: Boolean representing whether it exisits
    number_of_nodes_available_for_removal: nodes that can be removed if changes is made
    cut: nodes in the cut
    code: nodes in the cone (subgraph - cut - target node)
    """        

    if size_input_to_OR_gate > 3:
        raise ValueError("Input size larger than 3 currently not supported")
    if size_input_to_OR_gate < 2:
        raise ValueError("Input size smaller than 2 is not allowed")

    G_copy = copy.deepcopy(G)
    results = {}

    feasible_cuts = find_feasible_cuts(G_copy, output_node, size_input_to_OR_gate, filter_redundant=True)
    
    feasible_cuts_of_size_size_input_to_OR_gate = [cut for cut in feasible_cuts if len(cut) == size_input_to_OR_gate]
    
    if not feasible_cuts_of_size_size_input_to_OR_gate:
        return {"None": {"is_there_an_implicit_OR": False, "number_of_nodes_available_for_removal": 0, "cut": [], "cone": []}}

    for cut_index, cut in enumerate(feasible_cuts_of_size_size_input_to_OR_gate):
        subgraph = generate_subgraph(G_copy, output_node, cut, draw=False)
        cone = set(subgraph.nodes()) - set(cut) - {output_node}
        truth_table = calculate_truth_table(subgraph)
        binary_str = ''.join(str(output[0]) for inputs, output in sorted(truth_table.items()))
        truth_table_int = int(binary_str, 2)

        key_name = f"implicit_OR_{cut_index}"
        if (size_input_to_OR_gate == 2 and truth_table_int == 7) or (size_input_to_OR_gate == 3 and truth_table_int == 127):
            # Found an implicit OR
            # number_of_nodes_available_for_removal = (nodes in the subgraph) - (nodes in the cut) - (output node, since this node is not a gate)
            number_of_nodes_available_for_removal = len(subgraph.nodes()) - len(cut) - 1
            results[key_name] = {
                "is_there_an_implicit_OR": True,
                "number_of_nodes_available_for_removal": number_of_nodes_available_for_removal,
                "cut": list(cut),
                "cone": list(cone)
            }
        else:
            results[key_name] = {
                "is_there_an_implicit_OR": False,
                "number_of_nodes_available_for_removal": 0,
                "cut": list(cut),
                "cone": list(cone)
            }

    # Filter out the entries where no implicit OR is found
    results = {k: v for k, v in results.items() if v["is_there_an_implicit_OR"]}

    return results if results else {"None": {"is_there_an_implicit_OR": False, "number_of_nodes_available_for_removal": 0, "cut": [], "cone": []}}

def check_implicit_OR_existence_v3(G, output_node, size_input_to_OR_gate):
    
    """
    This function uses a feasible cut rooted at output_node to check for any structure that
    computes an OR. 

    size_input_to_OR_gate is the input size to the implicit OR

    Can be used for multi-output circuits.  

    Returns in a list of tuple for each possible OR
    
    is_there_an_implicit_OR: Boolean representing whether it exisits
    number_of_nodes_available_for_removal: nodes that can be removed if changes is made
    cut: nodes in the cut
    code: nodes in the cone (subgraph - cut - target node)
    """        

    if size_input_to_OR_gate > 3:
        raise ValueError("Input size larger than 3 currently not supported")
    if size_input_to_OR_gate < 2:
        raise ValueError("Input size smaller than 2 is not allowed")

    G_copy = copy.deepcopy(G)
    results = {}
    
    # Previous code: 
    #feasible_cuts = find_feasible_cuts(G_copy, output_node, size_input_to_OR_gate, filter_redundant=True)
    #feasible_cuts_of_size_size_input_to_OR_gate = [cut for cut in feasible_cuts if len(cut) == size_input_to_OR_gate]
    
    # New code:
    feasible_cuts = exhaustive_cut_enumeration_dag(G_copy, size_input_to_OR_gate, output_node, filter_redundant=True)
    feasible_cuts_of_size_n = [cut for cut in feasible_cuts if len(cut) == size_input_to_OR_gate]
    feasible_cuts_of_size_n_fof = [cut for cut in feasible_cuts_of_size_n if is_fanout_free_standalone(G_copy, output_node, cut) == True]    
    feasible_cuts_of_size_input_to_OR_gate = feasible_cuts_of_size_n_fof
    
    if not feasible_cuts_of_size_input_to_OR_gate:
        return {"None": {"is_there_an_implicit_OR": False, "number_of_nodes_available_for_removal": 0, "cut": [], "cone": []}}

    for cut_index, cut in enumerate(feasible_cuts_of_size_input_to_OR_gate):
        subgraph = generate_subgraph(G_copy, output_node, cut, draw=False)
        
        #ensure the cut is clean input boundary
        if len([n for n in subgraph if subgraph.in_degree(n) == 0]) != len(cut):
            continue
        
        cone = set(subgraph.nodes()) - set(cut) - {output_node}
        truth_table = calculate_truth_table(subgraph)
        binary_str = ''.join(str(output[0]) for inputs, output in sorted(truth_table.items()))
        truth_table_int = int(binary_str, 2)

        key_name = f"implicit_OR_{cut_index}"
        if (size_input_to_OR_gate == 2 and truth_table_int == 7) or (size_input_to_OR_gate == 3 and truth_table_int == 127):
            # Found an implicit OR
            # number_of_nodes_available_for_removal = (nodes in the subgraph) - (nodes in the cut) - (output node, since this node is not a gate)
            number_of_nodes_available_for_removal = len(subgraph.nodes()) - len(cut) - 1
            results[key_name] = {
                "is_there_an_implicit_OR": True,
                "number_of_nodes_available_for_removal": number_of_nodes_available_for_removal,
                "cut": list(cut),
                "cone": list(cone)
            }
        else:
            results[key_name] = {
                "is_there_an_implicit_OR": False,
                "number_of_nodes_available_for_removal": 0,
                "cut": list(cut),
                "cone": list(cone)
            }

    # Filter out the entries where no implicit OR is found
    results = {k: v for k, v in results.items() if v["is_there_an_implicit_OR"]}

    return results if results else {"None": {"is_there_an_implicit_OR": False, "number_of_nodes_available_for_removal": 0, "cut": [], "cone": []}}


def delete_files_in_folder(folder_path):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"The folder {folder_path} does not exist.")
        return

    # Iterate over the files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            # Check if the path is a file and delete it
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            # Check if the path is a directory and delete it recursively
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

'''
def process_permutation_batch(permutations, expected_logic, adj_matrix, cello_v1_hill_function_parameters, input_signals_list_small_molecules, gate_toxicity_df):
    
    """
    Thsi function search a batch of permutations exhaustively. It is meant to be used for multi-process exhaustive search. 
    """    
        
    results = []

    for current_solution in permutations:
        Gp = assign_representations_with_io_nodes_3(adj_matrix, cello_v1_hill_function_parameters, current_solution)
        toxicity_score, detailed_results = calculate_toxicity_score(input_signals_list_small_molecules, Gp, gate_toxicity_df)
        actual_logic = simulate_signal_propagation(Gp, input_signals_list_small_molecules)
        current_score = calculate_circuit_score(expected_logic, actual_logic)
        roadblocking_flag = is_roadblocking(Gp)
        results.append((current_solution, toxicity_score, current_score, roadblocking_flag))
    return results
'''


def assign_random_repressors(adj_matrix, df, gate_max_incoming_signals_df):
    """
    Assigns random repressors/RBS from the library to nodes in a DAG, ensuring no repressor is repeated and
    that nodes do not exceed the maximum allowed incoming edges for their assigned gate.

    Parameters:
    adj_matrix (numpy.ndarray): Adjacency matrix representing the graph.
    df (pandas.DataFrame): DataFrame containing the library of repressors/RBS and their properties.
    gate_max_incoming_signals_df (pandas.DataFrame): DataFrame containing maximum allowed incoming signals for each gate.

    Returns:
    networkx.DiGraph: The graph with assigned repressors/RBS and node attributes.
    list: A list of repressor indices corresponding to the assigned repressors, in the order of non-input/output nodes.
    """
    # Create the DAG from the adjacency matrix
    G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
    
    # Remove disconnected nodes (nodes with neither incoming nor outgoing edges)
    disconnected_nodes = [node for node in G.nodes() if G.in_degree(node) == 0 and G.out_degree(node) == 0]
    G.remove_nodes_from(disconnected_nodes)
    
    # Identify input and output nodes
    input_nodes = [node for node in G.nodes() if G.in_degree(node) == 0]
    output_nodes = [node for node in G.nodes() if G.out_degree(node) == 0]
    
    # List of non-input/output nodes
    non_io_nodes = [node for node in G.nodes() if node not in input_nodes and node not in output_nodes]
    # Sort non_io_nodes to maintain a consistent order
    non_io_nodes.sort()
    
    # Initialize the list to keep track of assigned repressor indices
    node_representation_indices = []
    
    # Copy the dataframe to keep track of available repressors
    available_repressors = df.copy()
    # Add the 'gate_max_incoming_signals' to the available_repressors DataFrame
    available_repressors = available_repressors.merge(gate_max_incoming_signals_df, left_index=True, right_index=True)
    
    # Assign repressors to nodes
    for node in non_io_nodes:
        incoming_edges = G.in_degree(node)
        
        # Filter repressors that can handle at least this many incoming edges
        valid_repressors = available_repressors[
            available_repressors['gate_max_incoming_signals'] >= incoming_edges
        ]
        
        # Remove repressors that have already been assigned
        assigned_repressors = [G.nodes[n]['Repressor'] for n in G.nodes() if 'Repressor' in G.nodes[n]]
        valid_repressors = valid_repressors[~valid_repressors['Repressor'].isin(assigned_repressors)]
        
        if valid_repressors.empty:
            raise ValueError(f"No valid repressors available for node {node} with {incoming_edges} incoming edges.")
        
        # Randomly select one repressor
        selected_repressor = valid_repressors.sample(n=1)
        rep_index = selected_repressor.index[0]
        
        # Assign all columns as attributes to the node
        for col in df.columns:
            G.nodes[node][col] = df.loc[rep_index, col]
        
        # Keep track of the assigned repressor index
        node_representation_indices.append(rep_index)
        
        # Remove the assigned repressor from available_repressors
        available_repressors = available_repressors[available_repressors['Repressor'] != df.loc[rep_index, 'Repressor']]
    
    # Set 'type' attribute for input and output nodes
    for node in input_nodes:
        G.nodes[node]['type'] = 'input'
    for node in output_nodes:
        G.nodes[node]['type'] = 'output'
    
    return G, node_representation_indices



def validate_graph_incoming_signals(G, cello_v1_hill_function_parameters, gate_max_incoming_signals_df):
    """
    Validates whether the graph G complies with the maximum allowed incoming signals for each gate.
    For each non-input/output node, it checks if the number of incoming edges does not exceed the 
    specified limit in gate_max_incoming_signals_df based on the gate's Repressor and RBS.
    
    Parameters:
    G (networkx.DiGraph): The directed graph with node attributes 'Repressor' and 'RBS'.
    cello_v1_hill_function_parameters (pd.DataFrame): DataFrame containing 'Repressor' and 'RBS' columns, indexed from 0 to 19.
    gate_max_incoming_signals_df (pd.DataFrame): DataFrame indexed by gate indices with 'gate_max_incoming_signals' column.
    
    Returns:
    bool: True if the graph is valid (no violations), False otherwise.
    """
    for node, attributes in G.nodes(data=True):
        # Skip input and output nodes
        if attributes.get("type") in {"input", "output"}:
            continue

        # Get the number of incoming signals for the node
        incoming_signals = G.in_degree(node)

        # Retrieve the Repressor and RBS from the node attributes
        repressor = attributes.get('Repressor')
        rbs = attributes.get('RBS')

        if repressor is None or rbs is None:
            raise ValueError(f"Node {node} is missing 'Repressor' or 'RBS' attribute.")

        # Find the index in cello_v1_hill_function_parameters where Repressor and RBS match
        matching_rows = cello_v1_hill_function_parameters[
            (cello_v1_hill_function_parameters['Repressor'] == repressor) &
            (cello_v1_hill_function_parameters['RBS'] == rbs)
        ]

        if matching_rows.empty:
            raise ValueError(f"No entry found in cello_v1_hill_function_parameters for Repressor '{repressor}' and RBS '{rbs}'.")

        gate_index = matching_rows.index[0]

        # Get the maximum allowed incoming signals for this gate from gate_max_incoming_signals_df
        if gate_index not in gate_max_incoming_signals_df.index:
            raise ValueError(f"Gate index '{gate_index}' not found in gate_max_incoming_signals_df.")

        max_incoming_signals = gate_max_incoming_signals_df.loc[gate_index, 'gate_max_incoming_signals']

        # Check if the node violates the maximum allowed incoming signals
        if incoming_signals > max_incoming_signals:
            #print(f"Node {node} with Repressor '{repressor}' and RBS '{rbs}' has {incoming_signals} incoming signals, "
            #      f"which exceeds the maximum allowed {max_incoming_signals}.")
            return False  # Invalid graph

    return True  # Valid graph

#Updated version that removes the target node from the solution 
def exhaustive_cut_enumeration_dag(G, k, target_node=None, filter_redundant=False):
    """
    Enumerate all 'cuts' (as sets of node IDs) in a DAG by a bottom-up approach,
    where a 'source' (no in-edges) is treated as a leaf with only one cut: {source}.
    For any other node u, we take the cross-product of the cuts of its
    predecessors, union them, and also include {u}. Any cut exceeding size k
    is discarded.

    Parameters
    ----------
    G : nx.DiGraph
        Directed acyclic graph (DAG).
    k : int
        The maximum allowed size of any cut.
    target_node : int or None, optional
        If provided, the function returns only the cuts of the specified node
        (as sorted tuples). Otherwise, it returns a dict for all nodes.
    filter_redundant : bool, optional
        If True, remove any cut that is a superset of another cut (i.e., keep
        only minimal cuts). Default: False.

    Returns
    -------
    dict[int, list[tuple[int]]] or list[tuple[int]]
        - If target_node is None: returns {node: [cuts_as_tuples]}.
        - Otherwise: returns [cuts_as_tuples] for just the target node.
        In both cases, if filter_redundant=True, supersets are removed.
    """
    import networkx as nx
    from itertools import product

    # Ensure G is acyclic
    nx.algorithms.dag.topological_sort(G)

    # Topological order (sources first, sinks last)
    topo_order = list(nx.topological_sort(G))

    # Memo for storing the valid cuts (as frozensets) for each node
    memo = {}

    # Build up cuts for each node in topological order
    for node in topo_order:
        # Predecessors = "children" in a bottom-up sense
        preds = list(G.predecessors(node))

        # If no predecessors, it's a source => only one cut: {node}
        if not preds:
            memo[node] = [frozenset([node])]
            continue

        # Cross-product of predecessor cuts
        pred_cut_lists = [memo[p] for p in preds]
        combined_cuts = set()
        for combo in product(*pred_cut_lists):
            union_set = set()
            for cset in combo:
                union_set.update(cset)
            if len(union_set) <= k:
                combined_cuts.add(frozenset(union_set))

        # Include the singleton {node} itself
        combined_cuts.add(frozenset([node]))

        memo[node] = list(combined_cuts)

    # Convert each node's cuts from frozenset to sorted tuples
    memo_tuples = {}
    for node, cuts_list in memo.items():
        tuple_list = [tuple(sorted(cut_set)) for cut_set in cuts_list]
        # Sort so the output is in a deterministic order
        tuple_list.sort()
        memo_tuples[node] = tuple_list

    # Helper function to remove supersets
    def remove_supersets(list_of_tuples):
        """
        Given a list of sorted tuples, removes any tuple that is a superset
        of another tuple in the list. Returns a new list of sorted tuples.
        """
        # Convert tuples to sets for easier subset checks
        sets_list = [set(t) for t in list_of_tuples]
        # Sort by ascending size (so we keep smaller sets first)
        sets_sorted = sorted(sets_list, key=len)

        kept = []
        for s in sets_sorted:
            # If there's already a kept set that is a subset of s,
            # then s is a superset => skip it.
            if any(ks.issubset(s) for ks in kept):
                continue
            else:
                kept.append(s)

        # Convert back to sorted tuples and sort again for final output
        final_cuts = [tuple(sorted(s)) for s in kept]
        final_cuts.sort()
        return final_cuts

    # If target_node was specified, return only that node's cuts
    if target_node is not None:
        result = memo_tuples.get(target_node, [])
        if filter_redundant:
            result = remove_supersets(result)
        # **Exclude the singleton {target_node}**
        result = [cut for cut in result if cut != (target_node,)]
        return result

    # Otherwise, return a dict of cuts for ALL nodes
    if filter_redundant:
        filtered_dict = {}
        for node, cuts_for_node in memo_tuples.items():
            filtered_dict[node] = remove_supersets(cuts_for_node)
        return filtered_dict
    else:
        return memo_tuples

def is_fanout_free_standalone(G, target_node, cut):
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


def energy_score(G: nx.DiGraph, implicit_or_fn,*,fanin_size: int = 2) -> Tuple[int, Dict[str, Any]]:
    """Return an *energy* cost for a circuit graph.

    Parameters
    ----------
    G : nx.DiGraph
        Circuit graph (acyclic).  Nodes can be any hashable type.
    implicit_or_fn : callable
        Function that detects implicit‑OR patterns.  It must follow the
        signature ``implicit_or_fn(G, output_node, fanin_size) -> dict`` where
        the dict maps keys to a sub‑dict containing at least:
        - ``is_there_an_implicit_OR`` : bool
        - ``number_of_nodes_available_for_removal`` : int
    fanin_size : int, default 2
        Fan‑in size to probe when searching for implicit‑ORs.

    Returns
    -------
    energy : int
        ``|G.nodes| - #inputs - #outputs - max_removal``
    details : dict
        Diagnostic counts: inputs, outputs, max_removal, best_key.
    """

    if G.number_of_nodes() == 0:
        raise ValueError("Graph must contain at least one node")

    # 1) Identify primary outputs (sink nodes)
    output_nodes = [n for n in G.nodes() if G.out_degree(n) == 0]
    if not output_nodes:
        raise ValueError("Graph has no sink/output nodes")

    # 2) Identify primary inputs (source nodes)
    input_nodes = [n for n in G.nodes() if G.in_degree(n) == 0]

    # 3) Detect implicit‑OR opportunities anchored at the first output
    max_removal = 0
    best_key = None
    implicit_or_results = implicit_or_fn(G, output_nodes[0], fanin_size)

    for key, val in implicit_or_results.items():
        if val.get("is_there_an_implicit_OR", False) and val.get("number_of_nodes_available_for_removal", 0) > max_removal:
            max_removal = val["number_of_nodes_available_for_removal"]
            best_key = key

    # 4) Compute energy
    energy = G.number_of_nodes() - len(input_nodes) - len(output_nodes) - max_removal

    details = dict(
        num_nodes=G.number_of_nodes(),
        num_inputs=len(input_nodes),
        num_outputs=len(output_nodes),
        max_removal=max_removal,
        best_pattern_key=best_key,
    )
    return energy, details


def plot_circuit_layered(
    G,
    outdir="circuit_plot",
    name="circuit",
    fmt="svg",               # svg|png|pdf|dot
    rankdir="LR",            # LR or TB
    show_in_notebook=False,  # inline display in Jupyter
    notebook_width=None,     # width in px for inline display (Graphviz path)
    use_graphviz=True,       # False forces Matplotlib fallback
    compact=False,           # tighter Graphviz spacing & fonts
    gv_size=None,            # e.g. "6,4" or "6,4!" (ignored if show_in_notebook=True)
    gv_dpi=None,             # e.g. 96 or 120
    gv_margin="0.15",        # extra page margin (inches) to prevent clipping
    gv_pad="0.15",           # extra drawing pad (inches) to prevent clipping
    mpl_scale=1.0,           # scale Matplotlib fallback
    save = True
):


    if not isinstance(G, nx.DiGraph):
        raise ValueError("G must be a networkx.DiGraph")
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("G must be a DAG")

    def _infer_io_local(G):
        ins  = [n for n in G.nodes if G.in_degree(n)  == 0]
        outs = [n for n in G.nodes if G.out_degree(n) == 0]
        try:
            ins  = sorted(ins);  outs = sorted(outs)
        except TypeError:
            ins  = sorted(ins, key=str);  outs = sorted(outs, key=str)
        if not ins:  raise ValueError("No inputs (in_degree==0) found.")
        if not outs: raise ValueError("No outputs (out_degree==0) found.")
        return ins, outs

    def _label_for(n):
        indeg, outdeg = G.in_degree(n), G.out_degree(n)
        t = str(G.nodes[n].get("type", "")).lower()
        if indeg == 0 or t == "input":
            return "INPUT"
        if outdeg == 0 or t == "output":
            if indeg == 1: return "OUTPUT"
            if indeg == 2: return "OR/OUTPUT"
            return "OUT"
        if indeg == 1: return "NOT"
        if indeg == 2: return "NOR"
        return ""

    def _det_sorted(iterable):
        try:    return sorted(iterable)
        except: return sorted(iterable, key=str)

    ins, outs = _infer_io_local(G)
    for i in ins:
        if G.nodes[i].get("type") is None: G.nodes[i]["type"] = "input"
    for o in outs:
        if G.nodes[o].get("type") is None: G.nodes[o]["type"] = "output"

    # Graphviz path 
    if use_graphviz:
        try:
            import graphviz
            have_dot = shutil.which("dot") is not None
        except Exception:
            have_dot = False

        if have_dot:
            # topo distance from inputs -> layering
            dist = {}
            for n in nx.topological_sort(G):
                preds = list(G.predecessors(n))
                dist[n] = 0 if not preds else 1 + max(dist[p] for p in preds)

            g = graphviz.Digraph(name, format=fmt)
            nodesep = "0.35" if compact else "0.5"
            ranksep = "0.55" if compact else "0.8"
            fontsize = "9" if compact else "10"

            # Key tweaks: larger margin/pad; and only apply size if NOT inline
            g.attr(
                rankdir=rankdir, splines="polyline", concentrate="True",
                nodesep=nodesep, ranksep=ranksep, fontname="Helvetica",
                fontsize=fontsize, margin=str(gv_margin), pad=str(gv_pad),
                labelloc="t"
            )
            if gv_dpi:  g.attr(dpi=str(gv_dpi))
            if gv_size and not show_in_notebook:
                g.attr(size=str(gv_size))  # avoid when inline to prevent clipping

            g.node_attr.update(
                shape="box", style="rounded,filled", color="#585858",
                fontname="Helvetica", fontsize=fontsize, penwidth="1",
                margin="0.03,0.02" if compact else "0.06,0.04"
            )
            
            g.edge_attr.update(arrowsize="0.75", penwidth="2", color="#585858")

            # nodes
            for n in _det_sorted(G.nodes()):
                t = str(G.nodes[n].get("type", "")).lower()
                label_role = _label_for(n)
                main = str(n)
                sub  = ("\\n" + label_role) if label_role else ""
                fill = "#f2fef2" if t == "input" else ("#fcffe5" if t == "output" else "#c1e6fb")
                g.node(str(n), label=f"{main}{sub}", fillcolor=fill)

            # rank by level
            by_level = {}
            for n, d in dist.items():
                by_level.setdefault(d, []).append(n)
            for level in sorted(by_level.keys()):
                with g.subgraph() as s:
                    s.attr(rank="same")
                    for n in _det_sorted(by_level[level]):
                        s.node(str(n))

            # edges
            for u, v in G.edges():
                g.edge(str(u), str(v), tailport="e", headport="w")

            outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
            outfile = str((outdir / name).resolve())
            
            # Optional inline display (with width control)
            if show_in_notebook:
                try:
                    from IPython.display import display, SVG, Image, HTML
                    if fmt.lower() == "svg":
                        svg_bytes = g.pipe(format="svg")
                        if notebook_width:
                            svg_text = svg_bytes.decode("utf-8", "ignore")
                            svg_text = re.sub(
                                r"<svg",
                                f'<svg style="max-width:{int(notebook_width)}px;width:{int(notebook_width)}px;height:auto;"',
                                svg_text, count=1
                            )
                            display(HTML(svg_text))
                        else:
                            display(SVG(svg_bytes))
                    elif fmt.lower() == "png":
                        png_bytes = g.pipe(format="png")
                        display(Image(png_bytes, width=int(notebook_width) if notebook_width else None))
                except Exception:
                    pass

            if save:
                path = g.render(outfile, cleanup=True)
                return Path(path)
            else:
                path = None
                return None   
    
def plot_circuit_with_parts_layered(
    G,
    outdir="circuit_plot",
    name="circuit",
    fmt="svg",               # svg|png|pdf|dot
    rankdir="LR",            # LR or TB
    show_in_notebook=False,  # inline display in Jupyter
    notebook_width=None,     # width in px for inline display (Graphviz path)
    use_graphviz=True,       # False forces Matplotlib fallback
    compact=False,           # tighter Graphviz spacing & fonts
    gv_size=None,            # e.g. "6,4" or "6,4!" (ignored if show_in_notebook=True)
    gv_dpi=None,             # e.g. 96 or 120
    gv_margin="0.15",        # extra page margin (inches) to prevent clipping
    gv_pad="0.15",           # extra drawing pad (inches) to prevent clipping
    mpl_scale=1.0,           # scale Matplotlib fallback
    save=True,
    seed=42,
    color_mapping=None       # optional override for repressor->border color
):
    """
    Layered circuit plot with node labels & colors. INPUT/OUTPUT nodes match
    plot_circuit_layered (node\\nROLE, green/yellow fills). Internal nodes show
    'Repressor\\nRBS' and use palette colors for borders with a light tint fill.
    """
    if not isinstance(G, nx.DiGraph):
        raise ValueError("G must be a networkx.DiGraph")
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("G must be a DAG")

    # ---- palette ----
    if color_mapping is None:
        color_mapping  = {
            "AmeR":  "#76c6c7",
            "AmtR":  "#80c582",
            "BetI":  "#cee3f4",
            "Bm3R1": "#f59390", "BM3R1": "#f59390",
            "HlyIIR":"#f8e3ae",
            "IcaRA": "#cce1b9",
            "LitR":  "#f1bd69",
            "LmrA":  "#7ebde4", "LmRA": "#7ebde4",
            "PhlF":  "#e5a06c",
            "PsrA":  "#c6a5cc",
            "QacR":  "#f06069",
            "SrpR":  "#3582c0",
            "YFP":   "#f3d04f",
        }

    # ---- identify inputs/outputs ----
    input_nodes  = [n for n in G.nodes() if G.in_degree(n)  == 0]
    output_nodes = [n for n in G.nodes() if G.out_degree(n) == 0]

    for i in input_nodes:
        if G.nodes[i].get("type") is None:
            G.nodes[i]["type"] = "input"
    for o in output_nodes:
        if G.nodes[o].get("type") is None:
            G.nodes[o]["type"] = "output"

    # ---- labels + border/fill colors ----
    labels, border_colors, fill_colors = {}, {}, {}
    for node, data in G.nodes(data=True):
        indeg, outdeg = G.in_degree(node), G.out_degree(node)
        t = str(G.nodes[node].get("type", "")).lower()

        # INPUT
        if indeg == 0 or t == "input":
            labels[node] = f"{node}\nINPUT"
            border_colors[node] = "#585858"
            fill_colors[node]   = "#f2fef2"   

        # OUTPUT family
        elif outdeg == 0 or t == "output":
            role = "OUTPUT" if indeg == 1 else ("OR/OUTPUT" if indeg == 2 else "OUT")
            labels[node] = f"{node}\n{role}"
            border_colors[node] = "#585858"
            fill_colors[node]   = "#fcffe5"   

        # Internal nodes with parts info
        elif 'Repressor' in data and 'RBS' in data:
            rep = data['Repressor']; rbs = data['RBS']
            labels[node] = f"{rep}\n{rbs}"
            c = color_mapping.get(rep, "#666666")
            border_colors[node] = "#666666"
            # inline lighten toward white (no helper function created)
            try:
                r = int(c[1:3],16); g = int(c[3:5],16); b = int(c[5:7],16)
                factor = 0.82  # higher = lighter
                rf = int(r + (255 - r) * factor)
                gf = int(g + (255 - g) * factor)
                bf = int(b + (255 - b) * factor)
                fill_colors[node] = f"#{rf:02x}{gf:02x}{bf:02x}"
            except Exception:
                fill_colors[node] = "#ffffff"

        # Fallback
        else:
            labels[node] = "N/A"
            border_colors[node] = "#585858"
            fill_colors[node]   = "#c1e6fb"  # same default internal fill as plot_circuit_layered

    # ---- Graphviz layered drawing path ----
    have_dot = False
    if use_graphviz:
        try:
            import graphviz  # noqa: F401
            have_dot = shutil.which("dot") is not None
        except Exception:
            have_dot = False

    if use_graphviz and have_dot:
        import graphviz

        # Topological layering
        dist = {}
        for n in nx.topological_sort(G):
            preds = list(G.predecessors(n))
            dist[n] = 0 if not preds else 1 + max(dist[p] for p in preds)

        g = graphviz.Digraph(name, format=fmt)

        nodesep = "0.35" if compact else "0.5"
        ranksep = "0.55" if compact else "0.8"
        fontsize = "9" if compact else "10"

        g.attr(
            rankdir=rankdir, splines="polyline", concentrate="True",
            nodesep=nodesep, ranksep=ranksep, fontname="Helvetica",
            fontsize=fontsize, margin=str(gv_margin), pad=str(gv_pad),
            labelloc="t"
        )
        if gv_dpi:
            g.attr(dpi=str(gv_dpi))
        if gv_size and not show_in_notebook:
            g.attr(size=str(gv_size))

        # Match plot_circuit_layered borders/edges
        g.node_attr.update(
            shape="box", style="rounded,filled",
            fontname="Helvetica", fontsize=fontsize,
            penwidth="1", color="#585858", fillcolor="white",
            margin="0.03,0.02" if compact else "0.06,0.04",
        )
        g.edge_attr.update(arrowsize="0.75", penwidth="2", color="#585858")

        def _det_sorted(iterable):
            try:    return sorted(iterable)
            except: return sorted(iterable, key=str)

        # Nodes (apply per-node label, border, and fill)
        for n in _det_sorted(G.nodes()):
            gv_label = str(labels.get(n, str(n))).replace("\n", "\\n")
            g.node(
                str(n),
                label=gv_label,
                color=str(border_colors.get(n, "#585858")),
                fillcolor=str(fill_colors.get(n, "white")),
            )

        # Rank by level
        by_level = {}
        for n, d in dist.items():
            by_level.setdefault(d, []).append(n)
        for level in sorted(by_level.keys()):
            with g.subgraph() as s:
                s.attr(rank="same")
                for n in _det_sorted(by_level[level]):
                    s.node(str(n))

        # Edges
        for u, v in G.edges():
            g.edge(str(u), str(v), tailport="e", headport="w")

        # Output
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        outfile = str((outdir / name).resolve())

        if show_in_notebook:
            try:
                from IPython.display import display, SVG, Image, HTML
                if fmt.lower() == "svg":
                    svg_bytes = g.pipe(format="svg")
                    if notebook_width:
                        svg_text = svg_bytes.decode("utf-8", "ignore")
                        svg_text = re.sub(
                            r"<svg",
                            f'<svg style="max-width:{int(notebook_width)}px;width:{int(notebook_width)}px;height:auto;"',
                            svg_text, count=1
                        )
                        display(HTML(svg_text))
                    else:
                        display(SVG(svg_bytes))
                elif fmt.lower() == "png":
                    png_bytes = g.pipe(format="png")
                    display(Image(png_bytes, width=int(notebook_width) if notebook_width else None))
            except Exception:
                pass

        if save:
            path = g.render(outfile, cleanup=True)
            return Path(path)
        else:
            return None
        
def meets_criteria(perm):
    """
    Define the criteria that a permutation must meet.
    Return True if the permutation meets the criteria, False otherwise.
    """
    group1 = {3, 4, 5}
    group2 = {10, 11, 12}
    group3 = {14, 15}
    group4 = {16, 17, 18, 19}

    # Check that the permutation includes at most one element from each group
    group1_count = sum(1 for x in perm if x in group1)
    group2_count = sum(1 for x in perm if x in group2)
    group3_count = sum(1 for x in perm if x in group3)
    group4_count = sum(1 for x in perm if x in group4)

    # Ensure at most one element from each group
    return (group1_count <= 1 and
            group2_count <= 1 and
            group3_count <= 1 and
            group4_count <= 1)

def generate_and_filter_permutations(elements, r):
    """
    Generate permutations of r elements chosen from the given elements,
    and filter them based on the criteria.
    """
    total_count = math.perm(len(elements), r)  # Calculate the total number of permutations directly
    valid_permutations = []
    
    perm_iterator = itertools.permutations(elements, r)  # Create an iterator for permutations
    
    with tqdm(total=total_count, desc="Processing permutations") as pbar:
        for perm in perm_iterator:
            if meets_criteria(perm):
                valid_permutations.append(perm)
            pbar.update(1)
    
    return valid_permutations



def meets_criteria_v2(perm, groups, max_per_group: int = 1) -> bool:
    """
    Return True if `perm` contains at most `max_per_group` items drawn
    from each set in `groups`; otherwise return False.

    Parameters
    ----------
    perm : Sequence[int]
        The permutation being tested.
    groups : Iterable[set[int]]
        Any number of sets that define the rule.
    max_per_group : int, default = 1
        Upper bound allowed from each group.
    """
    # Early-exit as soon as one group is over the limit
    for group in groups:
        if sum(x in group for x in perm) > max_per_group:
            return False
    return True

def generate_and_filter_permutations_v2(
    elements, r, groups, max_per_group: int = 1
):
    """
    Generate all `r`-length permutations from `elements` and keep only
    those that satisfy `meets_criteria_v2`.

    Parameters
    ----------
    elements : Sequence[int]
        Pool to draw from (e.g. range(20)).
    r : int
        Length of each permutation.
    groups : Iterable[set[int]]
        The grouping rules supplied by the caller.
    max_per_group : int, default = 1
        Forwarded to `meets_criteria_v2`.

    Returns
    -------
    list[tuple[int]]
        All permutations that obey the group constraint.
    """
    total_count = math.perm(len(elements), r)
    valid_permutations = []

    with tqdm(total=total_count, desc="Processing permutations") as pbar:
        for perm in itertools.permutations(elements, r):
            if meets_criteria_v2(perm, groups, max_per_group):
                valid_permutations.append(perm)
            pbar.update(1)

    return valid_permutations

def save_valid_permutations(valid_permutations, file_path):
    """
    Save valid permutations to an HDF5 file.
    """
    df_valid_permutations = pd.DataFrame(valid_permutations)
    df_valid_permutations.to_hdf(file_path, key='df', mode='w', complevel=5, complib='blosc')
    print(f"Valid permutations saved to '{file_path}'")

def load_valid_permutations(file_path):
    """
    Load the valid permutations from an HDF5 file and convert it back to the original format.
    """
    df_valid_permutations = pd.read_hdf(file_path, key='df')
    valid_permutations = [tuple(row) for row in df_valid_permutations.to_numpy()]
    return valid_permutations

def load_input_data(path):
    """
    Load config from JSON and return:
      table: list of dicts {index:int -> value:float} in truth-table order
      names: dict {index:int -> name:str}
      order: list of indices used for ordering
      names_in_order: list of names following 'order'
    """
    with open(path, "r") as f:
        cfg = json.load(f)

    # parse ranges
    inputs = cfg.get("inputs", {})
    ranges = {int(k): (float(v["low"]), float(v["high"])) for k, v in inputs.items()}

    # parse names 
    names_cfg = cfg.get("names", {})
    names = {int(k): str(v) for k, v in names_cfg.items()}
    for i in ranges:
        names.setdefault(i, str(i))

    # normalize order
    order_cfg = cfg.get("order")
    if order_cfg is None:
        idxs = sorted(ranges)
    else:
        name_to_idx = {v: k for k, v in names.items()}
        idxs = []
        for x in order_cfg:
            if isinstance(x, int) or (isinstance(x, str) and x.isdigit()):
                idxs.append(int(x))
            else:
                idxs.append(name_to_idx[x])

    # build truth table 
    levels = [[ranges[i][0], ranges[i][1]] for i in idxs]
    table = [{i: v for i, v in zip(idxs, combo)} for combo in product(*levels)]

    return table, names, idxs, [names[i] for i in idxs]

def binary_truth_table(n):
    """
    Return a list of dicts representing the binary truth table
    for inputs 0..n-1, with the last index toggling fastest.
    """
    idxs = list(range(n))
    return [{i: b for i, b in zip(idxs, combo)} for combo in product([0, 1], repeat=n)]

def pt_to_in(pt):
    return pt / 72