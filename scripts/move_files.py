import shutil, pathlib

SRC = pathlib.Path(
    "/home/gridsan/spalacios/DRL1/supercloud-testing/ABC-and-PPO-testing1/"
    "Verilog_files_for_all_4_input_1_output_truth_tables_as_NIGs"
)
DST = pathlib.Path(
    "/home/gridsan/spalacios/Designing complex biological circuits with deep neural networks/"
    "dgd/data/NIGs_4_inputs"
)

DST.mkdir(parents=True, exist_ok=True)          

moved = 0
for fp in SRC.iterdir():                      
    if fp.is_file():                            
        shutil.move(fp, DST / fp.name)
        moved += 1
        print(f"{fp.name}")

print(f"\nMoved {moved} file(s).")
print(f"Destination now contains {len(list(DST.iterdir()))} file(s).")
