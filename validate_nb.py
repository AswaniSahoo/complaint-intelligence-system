import json
nb = json.load(open("notebooks/01_full_dataset_processing.ipynb"))
print(f"Valid notebook: {len(nb['cells'])} cells")
for i, c in enumerate(nb["cells"]):
    src = c["source"]
    first_line = src[0][:70].strip() if src else "(empty)"
    print(f"  Cell {i}: [{c['cell_type']:8s}] {first_line}")
