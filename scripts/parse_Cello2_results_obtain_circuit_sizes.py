
import csv
import re
from pathlib import Path

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
"0xFC79"]

ROOT_DIR = Path("/home/gridsan/spalacios/Designing complex biological circuits with deep neural networks/dgd/data/Cello_2_designs")                       
OUT_CSV  = Path("/home/gridsan/spalacios/Designing complex biological circuits with deep neural networks/dgd/data/Cello_2_designs/Cello_2_circuit_sizes.csv")       
# --------------------------------------------------------------------


def find_gate_count(log_path):
    """
    Return the gate count found in a log.log file.

    Priority:
      1)  'Number of gates :' → that integer.
      2)  If (1) is missing, add the integers after
          'ABC RESULTS:               NOR cells:' and
          'ABC RESULTS:               NOT cells:'.

    If none of the patterns are found, return None.
    """
    patt_gates = re.compile(r"Number\s+of\s+gates\s*:\s*(\d+)")
    patt_nor   = re.compile(r"ABC RESULTS:\s*NOR cells:\s*(\d+)")
    patt_not   = re.compile(r"ABC RESULTS:\s*NOT cells:\s*(\d+)")

    nor_count = not_count = None

    try:
        with Path(log_path).open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                # Highest-priority match
                m = patt_gates.search(line)
                if m:
                    return int(m.group(1)), 1

                # Collect fallback counts
                m = patt_nor.search(line)
                if m:
                    nor_count = int(m.group(1))
                    continue

                m = patt_not.search(line)
                if m:
                    not_count = int(m.group(1))
                    continue
    except FileNotFoundError:
        return None, None

    # Fallback when both ABC numbers are present
    if nor_count is not None and not_count is not None:
        return nor_count + not_count, 2

    return None, None


def process_hex(hex_code):
    """
    For a single hex code, find every directory whose name starts with it,
    look for log.log inside, and return a list of (hex, count_or_None, method_or_None).
    """
    results = []
    dir_pattern = re.compile(rf"^{re.escape(hex_code)}", re.IGNORECASE)

    for entry in ROOT_DIR.iterdir():
        if entry.is_dir() and dir_pattern.match(entry.name):
            count, method = find_gate_count(entry / "log.log")
            results.append((hex_code, count, method))

    if not results:  # no directory matched at all
        results.append((hex_code, None, None))

    return results


def main():
    rows = []
    for hex_code in circuits_hex_list:
        rows.extend(process_hex(hex_code))

    # write CSV
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["hex", "num_gates", "method_used (1: All,  2: Stopped after Yosys/ABC)"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} row(s) to {OUT_CSV.resolve()}")


if __name__ == "__main__":
    main()
    
    
    