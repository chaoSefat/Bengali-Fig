import json

# Load data
with open("unified_dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

with open("similar_pairs_fuzzy.json", "r", encoding="utf-8") as f:
    pairs = json.load(f)

# Build mapping of duplicates
id_to_master = {}

def find_master(id_):
    while id_ in id_to_master:
        id_ = id_to_master[id_]
    return id_

# Union-find style grouping
for pair in pairs:
    master1 = find_master(pair["id1"])
    master2 = find_master(pair["id2"])
    if master1 != master2:
        # Always keep the smaller ID as master
        id_to_master[master2] = master1

# Filter unique entries
unique_ids = set()
deduplicated_data = []

for entry in dataset:
    entry_id = entry["id"]
    master_id = find_master(entry_id)

    if master_id not in unique_ids:
        deduplicated_data.append(entry)
        unique_ids.add(master_id)

with open("deduplicated_dataset.json", "w", encoding="utf-8") as f:
    json.dump(deduplicated_data, f, ensure_ascii=False, indent=2)

print(f"Deduplicated dataset saved with {len(deduplicated_data)} entries.")
