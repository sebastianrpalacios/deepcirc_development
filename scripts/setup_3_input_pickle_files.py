#!/usr/bin/env python
"""
Convert the shared pickled 3-input-1-output NIG library into
individual *_NIG_unoptimized.pkl files.
"""

import pickle, os, pathlib, networkx as nx
from typing import Dict, Any

# ----------------------------------------------------------------------
SRC_PKL         = "/home/gridsan/spalacios/Designing complex biological circuits with deep neural networks/dgd/data/NIGs_3_inputs/NIGs_unoptimized_library_3_input_1_output.pkl"
DEST_DIR        = pathlib.Path(
    "/home/gridsan/spalacios/Designing complex biological circuits with deep neural networks/dgd/data/NIGs_3_inputs/"
)
DEST_DIR.mkdir(parents=True, exist_ok=True)

def _graph_to_triplet(G: nx.DiGraph) -> list[Any]:
    """Return [num_nodes, edges, node_attr_dict] – identical to 4-input files."""
    num_nodes   = G.number_of_nodes()
    edges       = list(G.edges())
    # the 4-input pickles keep *all* nodes in the attr-dict, even if attr is None
    attr_dict   = {n: G.nodes[n].get("type") for n in G.nodes()}
    return [num_nodes, edges, attr_dict]

# ----------------------------------------------------------------------
with open(SRC_PKL, "rb") as f:
    lib: Dict[str, Any] = pickle.load(f)          # key → graph-like object

print(f"Loaded {len(lib):,} circuits from {SRC_PKL}")

for hex_id, obj in lib.items():

    # ── turn the dict key into the desired string ───────────────────
    if isinstance(hex_id, int):
        # 3-input truth tables need only 2 hex digits (00–FF)
        hex_id_str = f"0x{hex_id:02X}"          # 26  →  0x1A
    else:
        hex_id_str = str(hex_id).upper()        # already a string key

    # ---- decide what to pickle -------------------------------------
    if isinstance(obj, nx.DiGraph):
        graph_list = _graph_to_triplet(obj)
    else:
        graph_list = obj                       # already in list form

    # ---- write to disk ---------------------------------------------
    out_name = f"{hex_id_str}_NIG_unoptimized.pkl"
    out_path = DEST_DIR / out_name
    with out_path.open("wb") as g:
        pickle.dump(graph_list, g)

    print(f"  → wrote {out_name}")

print(f"Finished.  Files are in: {DEST_DIR}")




