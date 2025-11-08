"""
yosys_extract.py
----------------
For every hex prefix listed in `circuits_hex_list`:

  •   Finds sub-folders whose names start with that hex
      (e.g. 0x00FB_V2, 0x00FB_revA …).

  •   Inside each matching folder, looks for the **first** “*.json”
      file produced by Yosys (`yosys -o … -json …`).

  •   Counts the number of gates (= number of cells in the JSON).

  •   Builds a directed NetworkX graph whose nodes are cells and whose
      edges follow the data-flow (producer → consumer on the same net),
      then saves it as `"<hex>_graph.graphml"` in the same folder.

  •   Writes a CSV summary (**hex, num_gates, graph_file**) called
      `gates_summary_yosys.csv`.

This script **ignores log.log** or any other file types; it is purely
for Yosys JSON processing.

Dependencies: `pip install networkx`
"""

import csv
import json
import re
import pickle
from pathlib import Path
import networkx as nx
from json import JSONDecodeError

# --------------------------------------------------------------------
# USER SETTINGS
# --------------------------------------------------------------------
circuits_hex_list = [
"0x000D",
"0x0239",
"0x0304",
"0x040B",
"0x0575",
"0x057A",
"0x0643",
"0x0760",
"0x09AF",
"0x0F42",
"0x1038",
"0x1048",
"0x10C9",
"0x1284",
"0x1323",
"0x13CE",
"0x1714",
"0x1858",
"0x1A60",
"0x1AC6",
"0x1CBF",
"0x1D95",
"0x1FDE",
"0x226B",
"0x22C6",
"0x23A7",
"0x240F",
"0x2A38",
"0x2A56",
"0x2FC7",
"0x3060",
"0x30CE",
"0x32AA",
"0x35C3",
"0x36DC",
"0x3812",
"0x3A17",
"0x3B31",
"0x3B60",
"0x3B68",
"0x409B",
"0x41A2",
"0x41B2",
"0x429B",
"0x4724",
"0x47FD",
"0x48C1",
"0x4A32",
"0x4BF8",
"0x5215",
"0x53AF",
"0x53D7",
"0x599A",
"0x5AAD",
"0x5B30",
"0x5DA9",
"0x5F01",
"0x5FE2",
"0x6060",
"0x616A",
"0x648B",
"0x6572",
"0x680A",
"0x6847",
"0x699D",
"0x6F2A",
"0x7096",
"0x70EC",
"0x7176",
"0x822B",
"0x850E",
"0x8F63",
"0x914C",
"0x918A",
"0x93AC",
"0x9591",
"0x96F7",
"0x9917",
"0x9BF5",
"0x9F8A",
"0xA2DA",
"0xA7B2",
"0xA960",
"0xB744",
"0xB8AD",
"0xBC16",
"0xBCA3",
"0xBDF1",
"0xBEE9",
"0xBF36",
"0xC248",
"0xC4B2",
"0xC766",
"0xCB82",
"0xCBD6",
"0xCE97",
"0xD319",
"0xD326",
"0xD477",
"0xD4E4",
"0xD550",
"0xDA80",
"0xDBFA",
"0xE605",
"0xE677",
"0xE93A",
"0xECF1",
"0xEFEB",
"0xF43F",
"0xF4E7",
"0xF5A4",
"0xFC79",
"0xD477",
"0xD4E4",
"0xD550",
"0xDA80",
"0xDBFA",
"0xE605",
"0xE677",
"0xE93A",
"0xECF1",
"0xEFEB",
"0xF43F",
"0xF4E7",
"0xF5A4",
"0xFC79"]

ROOT_DIR = Path(
    "/home/gridsan/spalacios/Designing complex biological circuits "
    "with deep neural networks/dgd/data/Cello_2_designs"
)
OUT_DIR = Path(                      # <-- where both outputs go
    "/home/gridsan/spalacios/Designing complex biological circuits "
    "with deep neural networks/dgd/data/yosys_designs"
)

CSV_NAME = "Yosys_circuit_sizes.csv"       # stays inside OUT_DIR
PKL_NAME = "hex_graphs.pkl"                # stays inside OUT_DIR
# --------------------------------------------------------------------

def get_primary_inputs(mod):
    """
    Return dict  port_name -> set(net_ids)  for all *input* ports
    in the Yosys module object.
    """
    inputs = {}
    for pname, pdata in mod.get("ports", {}).items():
        if pdata.get("direction") == "input":
            inputs[pname] = set(pdata["bits"])
    return inputs

def build_graph_from_yosys(json_path):
    """
    Return (gate_count, graph) for one Yosys JSON netlist.

    EXTRA NODES ADDED
    -----------------
    • primary-input ports  (type="PRIMARY_INPUT")
    • primary-output ports (type="PRIMARY_OUTPUT")

    Edges:
      INPUT → consuming cell
      driving cell → OUTPUT
    """
    import json, networkx as nx
    from pathlib import Path
    from json import JSONDecodeError

    # ---------------- load JSON ----------------
    with Path(json_path).open() as fh:
        data = json.load(fh)

    modules = data["modules"]
    top_mod = next(
        (n for n, m in modules.items() if m.get("attributes", {}).get("top")),
        next(iter(modules)),
    )
    mod   = modules[top_mod]
    cells = mod.get("cells", {})

    # ---------------- collect port nets ----------------
    pi_nets, po_nets = {}, {}             # port_name → set(net_ids)
    for pname, pinfo in mod.get("ports", {}).items():
        if pinfo.get("direction") == "input":
            pi_nets[pname] = set(pinfo["bits"])
        elif pinfo.get("direction") == "output":
            po_nets[pname] = set(pinfo["bits"])

    # ---------------- create graph & cell nodes ---------
    g = nx.DiGraph()
    for cname, cinfo in cells.items():
        g.add_node(cname, type=cinfo.get("type", ""))

    # ---------------- build net → producers / consumers -
    net_map = {}
    for cname, cinfo in cells.items():
        dirs  = cinfo.get("port_directions", {})
        conns = cinfo.get("connections", {})
        for port, nets in conns.items():
            direction = dirs.get(port, "")
            for net in nets:
                entry = net_map.setdefault(net, {"out": [], "in": []})
                entry["out" if direction == "output" else "in"].append(cname)

    # ---------------- add PRIMARY-INPUT nodes & edges ---
    for port_name, nets in pi_nets.items():
        g.add_node(port_name, type="PRIMARY_INPUT")
        for net in nets:
            for dst in net_map.get(net, {}).get("in", []):
                g.add_edge(port_name, dst, net=net)

    # ---------------- add PRIMARY-OUTPUT nodes & edges --
    for port_name, nets in po_nets.items():
        g.add_node(port_name, type="PRIMARY_OUTPUT")
        for net in nets:
            for src in net_map.get(net, {}).get("out", []):
                g.add_edge(src, port_name, net=net)

    # ---------------- connect cell → cell as before ----
    for net, ends in net_map.items():
        for src in ends["out"]:
            for dst in ends["in"]:
                g.add_edge(src, dst, net=net)

    return len(cells), g


def process_folder(hex_code, folder):
    """
    Return (gate_count, graph) for the *Yosys* JSON in `folder`.

    • The Yosys file is the one whose stem (name without extension)
      matches the folder name exactly.
    • Any other *.json files are ignored.
    • If the Yosys JSON is malformed or missing, returns (None, None).
    """
    yosys_json = folder / f"{folder.name}.json"        # exact-match file
    if not yosys_json.exists():
        return None, None

    try:
        return build_graph_from_yosys(yosys_json)
    except JSONDecodeError as e:
        print(f"Skipping invalid JSON: {yosys_json} — {e}")
        return None, None


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_rows = []           # (hex, num_gates)
    graphs = {}             # hex → networkx.DiGraph

    for hex_code in circuits_hex_list:
        dir_pat = re.compile(rf"^{re.escape(hex_code)}", re.IGNORECASE)
        found = False

        for entry in ROOT_DIR.iterdir():
            if entry.is_dir() and dir_pat.match(entry.name):
                found = True
                gates, graph = process_folder(hex_code, entry)
                csv_rows.append((hex_code, gates))
                if graph is not None:
                    graphs[hex_code] = graph

        if not found:
            csv_rows.append((hex_code, None))

    # ---------- write CSV ----------
    csv_path = OUT_DIR / CSV_NAME
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["hex", "num_gates"])
        writer.writerows(csv_rows)

    # ---------- pickle graph dictionary ----------
    pkl_path = OUT_DIR / PKL_NAME
    with pkl_path.open("wb") as fh:
        pickle.dump(graphs, fh)

    print(f"CSV  saved to: {csv_path.resolve()}")
    print(f"Dict saved to: {pkl_path.resolve()}")
    print(f"Graphs stored for {len(graphs)} hex codes.")


if __name__ == "__main__":
    main()