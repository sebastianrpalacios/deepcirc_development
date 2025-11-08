import networkx as nx
from pathlib import Path
import itertools
import networkx as nx
import itertools
import re
from pathlib import Path
import json, itertools, shutil, subprocess, re
from pathlib import Path
import re, subprocess, shutil, itertools
from pathlib import Path
import networkx as nx  
from pathlib import Path
import csv

from dgd.utils.utils5 import (
    calculate_truth_table_v2,
)

def validate_dag(
    G,
    n_outputs,
    raise_on_error=True,
    plot_on_error=True,
    plot_title="DAG (validation failed)",
):
    """
    Validates a NetworkX directed graph `G`:

      1) Graph is a DAG
      2) Graph is weakly connected (ignores edge direction)
      3) Inputs = nodes with in-degree 0
      4) Outputs = nodes with out-degree 0
      5) Exactly `n_outputs` outputs
      6) All node labels are ints (bools rejected)
      7) Input labels are 0..k-1 (k = #inputs)  -> WARNING if not consecutive
      8) Inputs have node attr 'type' == 'input'
      9) Outputs have node attr 'type' == 'output'
     10) **No node has more than 2 inputs** (in-degree <= 2)

    On error, prints a report, node attributes (including any 'note'),
    and plots the graph (if matplotlib is available and plot_on_error=True).

    Returns a dict with: {"ok", "inputs", "outputs", "errors", "warnings"}.
    """
    errors = []
    warnings = []

    # Basic type/emptiness checks
    if not isinstance(G, nx.DiGraph):
        errors.append("Graph must be a networkx.DiGraph (not MultiDiGraph/Graph/etc.).")

    if G.number_of_nodes() == 0:
        errors.append("Graph has no nodes (empty).")

    # 1) DAG check
    if not nx.is_directed_acyclic_graph(G):
        errors.append("Graph is not a DAG (contains a directed cycle).")

    # 2) Weak connectivity
    if G.number_of_nodes() > 0:
        try:
            if nx.number_weakly_connected_components(G) != 1:
                comps = [sorted(list(c)) for c in nx.weakly_connected_components(G)]
                errors.append(
                    f"Graph is not weakly connected; found {len(comps)} components: {comps}"
                )
        except nx.NetworkXError as e:
            errors.append(f"Connectivity check failed: {e!s}")

    # 3) & 4) Identify inputs/outputs
    inputs = [n for n, deg in G.in_degree() if deg == 0]
    outputs = [n for n, deg in G.out_degree() if deg == 0]

    # 5) Exactly n outputs
    if len(outputs) != n_outputs:
        errors.append(
            f"Expected exactly {n_outputs} outputs, found {len(outputs)}: {sorted(outputs)}"
        )

    # 6) All nodes are integers (exclude bool which is a subclass of int)
    non_int_nodes = [n for n in G.nodes if not (isinstance(n, int) and not isinstance(n, bool))]
    if non_int_nodes:
        errors.append(
            "All nodes must have integer labels; non-integer (or bool) nodes: "
            + ", ".join(map(str, non_int_nodes))
        )

    # 7) Inputs are first k integers starting at 0  -> WARNING (not error)
    if not non_int_nodes:
        sorted_inputs = sorted(inputs)
        k = len(sorted_inputs)
        expected_inputs = list(range(k))
        if sorted_inputs != expected_inputs:
            warnings.append(
                f"Inputs are not labeled consecutively as 0..{k-1}; found {sorted_inputs}"
            )

    # 8) Inputs have type='input'
    bad_input_type = [n for n in inputs if G.nodes[n].get("type") != "input"]
    if bad_input_type:
        errors.append(
            "All inputs must have node attribute type='input'; offending nodes: "
            + ", ".join(map(str, sorted(bad_input_type)))
        )

    # 9) Outputs have type='output'
    bad_output_type = [n for n in outputs if G.nodes[n].get("type") != "output"]
    if bad_output_type:
        errors.append(
            "All outputs must have node attribute type='output'; offending nodes: "
            + ", ".join(map(str, sorted(bad_output_type)))
        )

    # 10) No node has more than 2 inputs (fan-in <= 2)
    fanin_violations = [(n, d) for n, d in G.in_degree() if d > 2]
    if fanin_violations:
        fanin_txt = ", ".join(f"{n}({d})" for n, d in sorted(fanin_violations, key=lambda x: x[0]))
        errors.append(f"Nodes with >2 inputs (in-degree): {fanin_txt}")

    result = {
        "ok": len(errors) == 0,
        "inputs": sorted(inputs) if not non_int_nodes else inputs,
        "outputs": sorted(outputs) if not non_int_nodes else outputs,
        "errors": errors,
        "warnings": warnings,
    }

    # If errors: print report, attributes (incl. 'note'), and plot
    if errors:
        report = "DAG validation failed:\n- " + "\n- ".join(errors)
        print(report)

        if warnings:
            print("\nWarnings:")
            for w in warnings:
                print(f"- {w}")

        # Print node attributes in a stable order
        print("\nNode attributes:")
        try:
            def _key(x):
                return (0, x) if isinstance(x, int) else (1, str(x))
            for n, attrs in sorted(G.nodes(data=True), key=lambda t: _key(t[0])):
                print(f"  {n}: {dict(attrs)}")
        except Exception as e:
            print(f"(Could not enumerate node attributes cleanly: {e!s})")

        # Print any 'note' attributes explicitly
        print("\nNode 'note' attributes (if present):")
        try:
            for n, attrs in sorted(G.nodes(data=True), key=lambda t: _key(t[0])):
                if "note" in attrs:
                    print(f"  {n}: {attrs['note']}")
        except Exception as e:
            print(f"(Could not enumerate 'note' attributes cleanly: {e!s})")

        if plot_on_error:
            _plot_graph_with_labels(G, title=plot_title)

        if raise_on_error:
            raise ValueError(report)

    # If no errors but warnings exist, return them silently (caller can inspect)
    return result


def _plot_graph_with_labels(G, title="DAG"):
    """Plot a DiGraph with node labels; inputs green, outputs red, others gray."""
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(Plotting skipped: matplotlib not available: {e!s})")
        return

    # Layout: topological layering if DAG; otherwise spring layout
    try:
        if nx.is_directed_acyclic_graph(G):
            topo = list(nx.topological_sort(G))
            dist = {}
            for n in topo:
                preds = list(G.predecessors(n))
                dist[n] = 0 if not preds else 1 + max(dist[p] for p in preds)
            pos = {n: (i, -dist[n]) for i, n in enumerate(topo)}
        else:
            pos = nx.spring_layout(G, seed=42)
    except Exception:
        pos = nx.spring_layout(G, seed=42)

    node_colors = []
    for n in G.nodes:
        t = G.nodes[n].get("type")
        if t == "input":
            node_colors.append("#2ca02c")  # green
        elif t == "output":
            node_colors.append("#d62728")  # red
        else:
            node_colors.append("#7f7f7f")  # gray

    plt.figure(figsize=(2, 2))
    nx.draw_networkx(
        G,
        pos=pos,
        with_labels=True,
        labels={n: str(n) for n in G.nodes()},
        node_color=node_colors,
        node_size=100,
        font_size=12,
        arrows=True,
        width=1,
        arrowsize=5,
    )
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def idify(s):
    """Sanitize a node name to a legal Verilog identifier."""
    t = re.sub(r'[^a-zA-Z0-9_]', '_', str(s))
    return t if t and not t[0].isdigit() else f"_{t}"

def infer_io(G: nx.DiGraph):
    """Infer PIs and POs. Keep numeric order if possible; else fall back to string order."""
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Graph must be a DAG")
    ins  = [n for n in G.nodes if G.in_degree(n)  == 0]
    outs = [n for n in G.nodes if G.out_degree(n) == 0]
    try:
        ins  = sorted(ins)
        outs = sorted(outs)
    except TypeError:
        ins  = sorted(ins,  key=str)
        outs = sorted(outs, key=str)
    if not ins:  raise ValueError("No inputs (in_degree==0) found.")
    if not outs: raise ValueError("No outputs (out_degree==0) found.")
    return ins, outs

def find_yosys():
    """Prefer YoWASP on clusters; fall back to native yosys."""
    for exe in ("yowasp-yosys", "yosys"):
        p = shutil.which(exe)
        if p:
            return p
    raise FileNotFoundError("Neither 'yowasp-yosys' nor 'yosys' found on PATH.")


def nx_to_verilog(G: nx.DiGraph, input_names=None, output_names=None, module="gate"):
    if input_names is None or output_names is None:
        input_names, output_names = infer_io(G)

    node_id = {n: idify(n) for n in G.nodes}
    ins  = [idify(x) for x in input_names]
    outs = [idify(x) for x in output_names]

    lines = []
    ports = ", ".join(ins + outs)
    lines.append(f"module {module}({ports});")
    lines.append("input " + ", ".join(ins) + ";")
    lines.append("output " + ", ".join(outs) + ";")

    internals = [node_id[n] for n in G.nodes if (G.in_degree(n) > 0 and G.out_degree(n) > 0)]
    wire_names = sorted(set(internals) - set(ins) - set(outs))
    if wire_names:
        lines.append("wire " + ", ".join(wire_names) + ";")

    for n in nx.topological_sort(G):
        indeg, outdeg = G.in_degree(n), G.out_degree(n)
        if indeg == 0:
            continue
        preds = list(G.predecessors(n))
        ps = [node_id[p] for p in preds]
        tgt = node_id[n]
        if outdeg == 0:
            if len(ps) == 0:
                raise ValueError(f"Output node {n} has no drivers.")
            expr = " | ".join(ps)                      # OUTPUT = OR of preds
        else:
            if indeg == 1:
                expr = f"~{ps[0]}"                    # NOT
            elif indeg == 2:
                expr = f"~({ps[0]} | {ps[1]})"        # NOR
            else:
                raise ValueError(f"Node {n}: in_degree={indeg} unsupported (expect 1 or 2).")
        lines.append(f"assign {tgt} = {expr};")

    lines.append("endmodule\n")
    return "\n".join(lines)

def write_gate_verilog(G, input_names=None, output_names=None, module="gate", folder = None):
    v = nx_to_verilog(G, input_names, output_names, module)
    path = folder / f"{module}.v"
    path.write_text(v)
    return path

# Cell 2: baseline evaluator matching your gate rules
def eval_once(G: nx.DiGraph, assignment: dict):
    """Evaluate one input assignment. Outputs are sink nodes (out_degree==0)."""
    value = {}
    # set inputs
    for n in G.nodes:
        if G.in_degree(n) == 0:
            value[n] = int(assignment[str(n)])
    # topo simulate
    for n in nx.topological_sort(G):
        indeg, outdeg = G.in_degree(n), G.out_degree(n)
        if indeg == 0:
            continue  # input already set
        preds = list(G.predecessors(n))
        pv = [value[p] for p in preds]
        if outdeg == 0:
            # OUTPUT node: OR of all predecessors
            if len(pv) == 0:
                raise ValueError(f"Output node {n} has no drivers.")
            value[n] = int(any(pv))
        else:
            # INTERNAL node
            if indeg == 1:
                value[n] = 1 - pv[0]                # NOT
            elif indeg == 2:
                value[n] = 1 - int(any(pv))         # NOR
            else:
                raise ValueError(
                    f"Internal node {n} has in_degree={indeg}; only 1 (NOT) or 2 (NOR) supported."
                )
    # collect outputs (sinks)
    outs = [n for n in G.nodes if G.out_degree(n) == 0]
    return {str(o): int(value[o]) for o in outs}

def truth_table_baseline(G: nx.DiGraph, input_names=None, output_names=None, lsb_fast=True):
    """Return (inputs_list, outputs_list) where each list is a list of tuples."""
    if input_names is None or output_names is None:
        input_names, output_names = infer_io(G)
    n = len(input_names)
    it = itertools.product([0,1], repeat=n)
    if not lsb_fast:
        it = (bits[::-1] for bits in it)
    inputs_rows, output_rows = [], []
    for bits in it:
        a = {str(k): int(v) for k,v in zip(input_names, bits)}
        y = eval_once(G, a)
        inputs_rows.append(tuple(bits))
        output_rows.append(tuple(y[str(o)] for o in output_names))
    return inputs_rows, output_rows

def normalize_rows(rows):
    """Accept list[tuple|str] -> list[tuple[int,...]]."""
    norm = []
    for r in rows:
        if isinstance(r, str):
            norm.append(tuple(int(c) for c in r.strip()))
        else:
            norm.append(tuple(int(x) for x in r))
    return norm

def dict_tt_to_rows(tt_dict, input_names, output_names, lsb_fast=True):
    """
    Convert {(in_bits)->(out_bits)} into a list of output rows ordered by binary count
    over input_names, with the LAST input toggling fastest.
    """
    n_in, n_out = len(input_names), len(output_names)
    # validate & normalize values
    fixed = {}
    for k, v in tt_dict.items():
        if not (isinstance(k, tuple) and len(k) == n_in and all(b in (0,1) for b in k)):
            raise ValueError(f"Bad key {k!r}: must be a {n_in}-tuple of 0/1.")
        if isinstance(v, str):
            v = tuple(int(c) for c in v.strip())
        else:
            v = tuple(int(x) for x in v)
        if len(v) != n_out or any(x not in (0,1) for x in v):
            raise ValueError(f"Bad value {v!r} for key {k!r}: need {n_out} bits.")
        fixed[k] = v

    it = itertools.product([0,1], repeat=n_in)
    if not lsb_fast:
        it = (bits[::-1] for bits in it)

    rows, missing = [], []
    for bits in it:
        if bits not in fixed:
            missing.append(bits)
            rows.append(None)
        else:
            rows.append(fixed[bits])
    if missing:
        head = ", ".join(map(str, missing[:8]))
        more = " ..." if len(missing) > 8 else ""
        raise ValueError(f"Truth table missing {len(missing)} combo(s): {head}{more}")
    return rows

def table_to_verilog_from_impl(G, truth_table_impl, input_names=None, output_names=None, module="gold"):
    """User impl may return a dict {(in_bits)->(out_bits)} or a list of rows."""
    if input_names is None or output_names is None:
        input_names, output_names = infer_io(G)

    tt_obj = truth_table_impl(G, input_names, output_names)  # flexible signature
    if isinstance(tt_obj, dict):
        rows = dict_tt_to_rows(tt_obj, input_names, output_names, lsb_fast=True)
    else:
        rows = normalize_rows(tt_obj)

    n_in, n_out = len(input_names), len(output_names)
    if len(rows) != (1 << n_in):
        raise ValueError(f"Expected {1<<n_in} rows, got {len(rows)}.")
    for i, r in enumerate(rows):
        if len(r) != n_out:
            raise ValueError(f"Row {i} has {len(r)} outputs; expected {n_out}.")

    ins  = [idify(x) for x in input_names]
    outs = [idify(x) for x in output_names]
    header = []
    header.append(f"module {module}({', '.join(ins+outs)});")
    header.append("input "  + ", ".join(ins) + ";")
    header.append("output reg " + ", ".join(outs) + ";")
    header.append("always @* begin")
    header.append("  {" + ", ".join(outs) + "} = " + f"{n_out}'b" + "0"*n_out + ";")
    header.append("  case ({" + ", ".join(ins) + "})")

    body = []
    for i, out_row in enumerate(rows):
        key = format(i, f"0{n_in}b")                 # last input toggles fastest
        val = "".join(str(int(b)) for b in out_row)
        body.append(f"    {n_in}'b{key}: " +
                    "{" + ", ".join(outs) + "} = " + f"{n_out}'b{val};")

    footer = ["  endcase", "end", "endmodule\n"]
    return "\n".join(header + body + footer)

def write_gold_verilog(G, truth_table_impl, input_names=None, output_names=None, module="gold", folder=None):
    v = table_to_verilog_from_impl(G, truth_table_impl, input_names, output_names, module)
    path = folder / f"{module}.v"
    path.write_text(v)
    return path

def run_equiv_check(folder=None, top_gold="gold", top_gate="gate", yosys_exe=None):
    import subprocess, shutil
    yosys = yosys_exe or (shutil.which("yowasp-yosys") or shutil.which("yosys"))
    if not yosys:
        raise FileNotFoundError("Neither 'yowasp-yosys' nor 'yosys' found on PATH.")
    print(f"Using Yosys at: {yosys}")

    # --- equivalence flow (proc+mem lowered to pure logic) ---
    (folder / "check.ys").write_text(f"""
    read_verilog {top_gold}.v
    prep -top {top_gold}
    memory_map                # <--- turn any ROMs into mux/logic
    opt -fast
    design -stash gold

    read_verilog {top_gate}.v
    prep -top {top_gate}
    memory_map
    opt -fast
    design -stash gate

    design -copy-from gold -as {top_gold} {top_gold}
    design -copy-from gate -as {top_gate} {top_gate}

    equiv_make {top_gold} {top_gate} equiv
    prep -top equiv
    opt -fast
    equiv_simple -undef
    equiv_status -assert
    """)

    # --- SAT miter (also ensure no memories remain) ---
    (folder / "check_sat.ys").write_text(f"""
    read_verilog {top_gold}.v
    prep -top {top_gold}
    memory_map
    opt -fast
    design -stash gold

    read_verilog {top_gate}.v
    prep -top {top_gate}
    memory_map
    opt -fast
    design -stash gate

    design -copy-from gold -as {top_gold} {top_gold}
    design -copy-from gate -as {top_gate} {top_gate}

    miter -equiv -flatten {top_gold} {top_gate} miter
    prep -top miter
    memory_map
    opt -fast
    sat -verify -prove trigger 0 -show-ports -show-inputs -set-def-inputs -enable_undef -set-init-undef
    """)

    r = subprocess.run([yosys, "-q", "-s", "check.ys"], cwd=str(folder), text=True, capture_output=True)
    (folder / "check.out").write_text(r.stdout + r.stderr)
    print("== Yosys equivalence check ==")
    print(r.stdout, end=""); print(r.stderr, end="")
    if r.returncode == 0:
        print("Equivalent (equiv_status passed).")
        return True

    print("Equivalence failed. Trying SAT miter to find a counterexample...")
    r2 = subprocess.run([yosys, "-q", "-s", "check_sat.ys"], cwd=str(folder), text=True, capture_output=True)
    (folder / "check_sat.out").write_text(r2.stdout + r2.stderr)
    print("== SAT miter result ==")
    print(r2.stdout, end=""); print(r2.stderr, end="")
    return False

def validate_tt_using_yosys(
    G,
    input_names=None,
    output_names=None,
    folder=None,
    top_module="gate",
    verbose=False,
):
    """
    Compares the truth table computed by Yosys SAT enumeration vs. calculate_truth_table_v2.
    Returns True on match, False otherwise.

    Prints diagnostics only when `verbose=True`.
    """
    vprint = print if verbose else (lambda *args, **kwargs: None)

    # Infer IO (numeric order) if not given
    if input_names is None or output_names is None:
        input_names, output_names = infer_io(G)
    vprint(f"[test] inputs={input_names}  outputs={output_names}")

    # Ensure sinks act as OR in Python evaluator (mark on a copy)
    G_py = G.copy()
    for o in output_names:
        G_py.nodes[o]['type'] = 'output'

    # Write Verilog for the graph (so Yosys can read it)
    gate_v = write_gate_verilog(G, input_names, output_names, module=top_module, folder=folder)
    vprint(f"[test] wrote {gate_v}")

    # Truth table from Yosys (rows in binary-count order; last input toggles fastest)
    combos, yosys_rows = yosys_truth_table_with_yosys(
        folder=folder,
        top_module=top_module,
        input_names=input_names,
        output_names=output_names
    )

    # Truth table from your Python function (dict: {(bits)->(outbits)})
    py_tt = calculate_truth_table_v2(G_py)  # keys are tuples over sorted inputs
    # sanity: make sure the Python function’s input ordering matches input_names
    py_inputs_order = sorted([n for n in G_py.nodes() if G_py.in_degree(n) == 0])
    assert list(py_inputs_order) == list(input_names), \
        f"Input order mismatch: python uses {py_inputs_order}, test uses {input_names}"

    py_rows = [py_tt[bits] for bits in combos]  # align to same row order as Yosys

    # 5) Compare
    mismatches = []
    for i, (bits, y_row, p_row) in enumerate(zip(combos, yosys_rows, py_rows)):
        if tuple(y_row) != tuple(p_row):
            mismatches.append((i, bits, y_row, p_row))

    if mismatches:
        if verbose:
            vprint(f"Mismatch in {len(mismatches)}/{len(combos)} rows. Showing first 10:")
            for i, bits, y_row, p_row in mismatches[:10]:
                vprint(f"  row {i} inputs {dict(zip(input_names, bits))}: "
                       f"Yosys={y_row}  Python={p_row}")
        return False

    vprint(f"Truth tables match for all {len(combos)} input combinations.")
    return True



def _ys(name):  # escape for yosys CLI
    return "\\" + name

def _conv_bit(val):
    if isinstance(val, (int, bool)): return int(val)
    if isinstance(val, list) and val: return _conv_bit(val[0])
    s = str(val).strip()
    m = re.search(r"([01])$", s)
    if not m: raise ValueError(f"Cannot parse bit from {val!r}")
    return int(m.group(1))

def _norm_name(n: str) -> str:
    s = str(n)
    if s.startswith("\\"): s = s[1:]
    s = s.strip()
    if "." in s: s = s.split(".")[-1]
    return s

def _extract_from_json(data, want_plain):
    """
    Supports both WaveJSON (data['signal'] list with 'wave' strings) and
    the older dict/list 'model'/'signals' layouts.
    """
    target = _norm_name(want_plain)

    # --- WaveJSON (Wavedrom) ---
    sigs = data.get("signal")
    if isinstance(sigs, list):
        for e in sigs:
            name = e.get("name")
            if name and _norm_name(name) == target:
                w = (e.get("wave") or "").strip()
                # take the first meaningful symbol
                for ch in w:
                    if ch in "01xX":
                        if ch in "xX":
                            raise ValueError(f"{want_plain} is X/undef in WaveJSON.")
                        return int(ch)
                # If no 0/1/x symbol, try any explicit data value
                v = e.get("data")
                if v is not None:
                    return _conv_bit(v)
                raise KeyError(f"WaveJSON for {want_plain!r} has no value.")
    # --- Dict/list 'model' / 'signals' layouts ---
    for key in ("model", "outputs", "signals", None):  # None = top-level
        container = data if key is None else data.get(key)
        if isinstance(container, dict):
            for k, v in container.items():
                if _norm_name(k) == target:
                    return _conv_bit(v)
        if isinstance(container, list):
            for e in container:
                if not isinstance(e, dict):
                    continue
                en = e.get("name") or e.get("wire") or e.get("signal") or e.get("id")
                if en is not None and _norm_name(en) == target:
                    v = e.get("bits") or e.get("value") or e.get("val") or e.get("data")
                    return _conv_bit(v)
    raise KeyError

def _extract_from_txt(txt: str, want_plain: str) -> int:
    """
    Parse the SAT text log. Supports:
      1) The table printed after "model found:" (columns: Signal Name, Dec, Hex, Bin)
      2) Lines of the form "\_18 = 1"
    """
    target = _norm_name(want_plain)

    # --- (1) Table mode ---
    in_table = False
    for line in txt.splitlines():
        if not in_table and line.strip().startswith("Signal Name"):
            in_table = True
            continue
        if in_table:
            if not line.strip() or line.startswith("Dumping SAT model"):
                break
            # example: "  \_18                      0         0             0"
            m = re.match(r"\s*\\?([^\s]+)\s+(-?\d+)\s+", line)
            if m and _norm_name(m.group(1)) == target:
                dec = int(m.group(2))
                # outputs are 1-bit; any nonzero dec -> 1
                return 0 if dec == 0 else 1

    # --- (2) "name = value" mode (older prints) ---
    m = re.search(rf"(?:^|\s)\\?{re.escape(target)}\s*=\s*([01xX])(?=\s|$)", txt)
    if m:
        ch = m.group(1).lower()
        if ch == "x":
            raise ValueError(f"{want_plain} is X/undef in SAT log.")
        return int(ch)

    raise KeyError(f"'{want_plain}' not found in SAT text log.")

def yosys_truth_table_with_yosys(
    G: nx.DiGraph = None,
    folder: Path = None,
    top_module: str = "gate",
    input_names=None,
    output_names=None,
    yosys_exe=None,
    verbose: bool = False,
):
    """
    Compute a truth table using Yosys SAT by optionally taking a NetworkX DAG `G`.

    If `G` is provided:
      - Inputs/outputs are inferred (unless explicitly provided).
      - `{top_module}.v` is (re)generated from `G` into `folder`.

    If `G` is None:
      - Assumes `{top_module}.v` already exists in `folder` and that
        `input_names` and `output_names` are provided (old behavior).

    Returns:
      (combos, rows)
        combos: list of input tuples in binary-count order (last input toggles fastest)
        rows:   list of output tuples aligned with combos
    """
    import tempfile

    vprint = print if verbose else (lambda *args, **kwargs: None)

    # Resolve yosys
    yosys = yosys_exe or (shutil.which("yowasp-yosys") or shutil.which("yosys"))
    if not yosys:
        raise FileNotFoundError("Neither 'yowasp-yosys' nor 'yosys' found on PATH.")
    vprint(f"[Yosys] using: {yosys}")

    # Ensure a working folder
    if folder is None:
        folder = Path(tempfile.mkdtemp(prefix="yosys_tt_"))
    else:
        folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    # If a graph is provided, infer IO (unless overridden) and emit Verilog
    if G is not None:
        if input_names is None or output_names is None:
            input_names, output_names = infer_io(G)
        # Mark outputs as outputs on a copy (consistent with your eval)
        G_emit = G.copy()
        
        #Graph should have this already 
        #for o in output_names:
        #    G_emit.nodes[o]["type"] = "output"
        # Write the Verilog for Yosys to read
        write_gate_verilog(G_emit, input_names, output_names, module=top_module, folder=folder)
    else:
        # Back-compat mode requires names to drive SAT queries
        if input_names is None or output_names is None:
            raise ValueError("When G is None, you must provide input_names and output_names.")

    top  = top_module
    ins  = [idify(x) for x in input_names]
    outs = [idify(x) for x in output_names]
    n    = len(ins)

    # Generate all input combinations (last input toggles fastest)
    combos = list(itertools.product([0, 1], repeat=n))

    # Build Yosys script: prep once, then SAT per vector (dump json & text)
    lines = [
        f"read_verilog {top}.v",
        f"prep -top {top}",
        "memory_map",
        "opt -fast",
        "design -stash TT",
    ]
    for idx, bits in enumerate(combos):
        set_args  = " ".join(f"-set {_ys(ins[i])} {bits[i]}" for i in range(n))
        show_args = " ".join(_ys(o) for o in outs)
        lines += [
            f"design -copy-from TT -as {top} {top}",
            f"tee -o tt_{idx}.txt sat {set_args} -show-ports -show {show_args} -dump_json tt_{idx}.json"
        ]

    (folder / "truth.ys").write_text("\n".join(lines) + "\n")
    r = subprocess.run([yosys, "-q", "-s", "truth.ys"], cwd=str(folder),
                       text=True, capture_output=True)
    (folder / "truth.out").write_text(r.stdout + r.stderr)
    if r.returncode != 0:
        if verbose:
            vprint(r.stdout); vprint(r.stderr)
        raise RuntimeError(
            f"Yosys truth-table run failed (rc={r.returncode}). See {folder / 'truth.out'} for details."
        )

    # Parse outputs for each input vector
    rows = []
    for idx in range(len(combos)):
        json_path = folder / f"tt_{idx}.json"
        txt_path  = folder / f"tt_{idx}.txt"
        data = None
        if json_path.exists():
            try:
                data = json.loads(json_path.read_text())
            except Exception:
                data = None

        out_bits = []
        for o in outs:
            bit_val = None
            if data is not None:
                try:
                    bit_val = _extract_from_json(data, o)
                except Exception:
                    bit_val = None
            if bit_val is None:
                txt = txt_path.read_text() if txt_path.exists() else ""
                bit_val = _extract_from_txt(txt, o)
            out_bits.append(bit_val)
        rows.append(tuple(out_bits))

    # Pretty print (only if verbose)
    if verbose:
        header = " | ".join([*(str(x) for x in input_names), *(str(y) for y in output_names)])
        vprint(header)
        vprint("-" * len(header))
        for bits, outs_bits in zip(combos, rows):
            vprint(" ".join(map(str, bits)), "|", " ".join(map(str, outs_bits)))

    return combos, rows




def _bits_to_hex(bits):
    s = ''.join(str(int(b)) for b in bits)
    s = s.zfill((len(s) + 3) // 4 * 4)
    return '0x' + format(int(s, 2), f'0{len(s)//4}X')

def yosys_vs_python_truth_table(G, folder=None, module="gate", strict=True):
    """
    Build gate.v, compute truth tables with Yosys (SAT) and with calculate_truth_table_v2,
    verify equality, and return results.
    - folder: where to write gate.v and Yosys artifacts (created if missing)
    - strict: if True, raise on any mismatch; else just print and continue
    """
    folder = Path(folder) if folder is not None else Path("yosys_equiv")
    folder.mkdir(parents=True, exist_ok=True)

    # 1) inputs/outputs (numeric order preserved)
    ins, outs = infer_io(G)

    # 2) emit Verilog for the graph
    write_gate_verilog(G, ins, outs, module=module, folder=folder)

    # 3) truth table from Yosys (rows in binary count order; last input toggles fastest)
    combos, yosys_rows = yosys_truth_table_with_yosys(
        folder=folder, top_module=module, input_names=ins, output_names=outs
    )

    # 4) truth table from your Python function (mark sinks as outputs on a copy)
    G_py = G.copy()
    for o in outs:
        G_py.nodes[o]['type'] = 'output'
    py_tt = calculate_truth_table_v2(G_py)
    py_rows = [py_tt[b] for b in combos]  # align order to Yosys

    # 5) compare
    mism = [(i, b, y, p) for i,(b,y,p) in enumerate(zip(combos, yosys_rows, py_rows)) if tuple(y)!=tuple(p)]
    if mism:
        print(f"Yosys vs Python mismatch in {len(mism)}/{len(combos)} rows (showing up to 8):")
        for i,b,y,p in mism[:8]:
            print(f"  row {i} inputs {dict(zip(ins,b))}: Yosys={y} Python={p}")
        if strict:
            raise AssertionError("Yosys and Python truth tables differ.")
    else:
        print(f"Yosys and Python truth tables match for {len(combos)} rows.")

    # 6) optional hex (single-output convenience)
    yosys_hex = None
    py_hex    = None
    if len(outs) == 1:
        yosys_hex = _bits_to_hex([row[0] for row in yosys_rows])
        py_hex    = _bits_to_hex([row[0] for row in py_rows])

    return {
        "inputs": ins,
        "outputs": outs,
        "combos": combos,
        "yosys_rows": yosys_rows,
        "python_rows": py_rows,
        "yosys_hex": yosys_hex,
        "python_hex": py_hex,
        "equal": len(mism) == 0,
        "mismatches": mism,
    }

def _fmt_bit(b):
    if b in (0, 1): return str(int(b))
    if isinstance(b, bool): return "1" if b else "0"
    return "x" if b is None else str(b)

def write_tt_files_yosys(folder: Path, combos, rows, input_names, output_names):
    folder.mkdir(parents=True, exist_ok=True)
    header = [*(str(x) for x in input_names), *(str(y) for y in output_names)]

    # CSV
    with (folder / "truth_table.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for iv, ov in zip(combos, rows):
            w.writerow([*iv, *(_fmt_bit(b) for b in ov)])

    # Simple text (same columns/order as the verbose print)
    lines = []
    lines.append(" | ".join(header))
    lines.append("-" * len(lines[0]))
    for iv, ov in zip(combos, rows):
        lines.append(" ".join(map(str, iv)) + " | " + " ".join(_fmt_bit(b) for b in ov))
    (folder / "truth_table.txt").write_text("\n".join(lines) + "\n")