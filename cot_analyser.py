import json
import pathlib
from collections import defaultdict
import pandas as pd

ZERO_DIR = pathlib.Path("results/zero_shot")
COT_DIR  = pathlib.Path("results/chain_of_thought_hard_cases")

def load_jsons(folder: pathlib.Path):
    """Return dict keyed by (provider, model) -> list of full json dicts."""
    out = defaultdict(list)
    for f in folder.glob("*.json"):
        with open(f, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        key = (data["metadata"]["provider"], data["metadata"]["model"])
        out[key].append(data)
    return out

def build_index(results):
    """Map riddle_id -> is_correct for quick lookup."""
    return {r["riddle_id"]: r["is_correct"] for r in results}

def analyze():
    zero = load_jsons(ZERO_DIR)
    cot  = load_jsons(COT_DIR)

    rows = []
    for key, cot_runs in cot.items():
        provider, model = key
        if key not in zero:
            print(f"⚠️ No Zero-Shot match for {provider}/{model}, skipping.")
            continue

        # Use the most recent zero-shot file if multiple exist
        zfile = max(zero[key], key=lambda d: d["metadata"]["timestamp"])
        z_index = build_index(zfile["results"])

        # Combine all CoT runs for this model
        cot_results = []
        for cfile in cot_runs:
            cot_results.extend(cfile["results"])
        cot_index = build_index(cot_results)

        subset_ids = set(cot_index.keys())  # only CoT subset matters
        total_cases = len(subset_ids)
        if total_cases == 0:
            continue

        # Subset-specific metrics
        originally_wrong = {
            rid for rid in subset_ids
            if not z_index.get(rid, False)
        }

        improvements = sum(
            1 for rid in originally_wrong
            if cot_index[rid] is True
        )
        still_wrong = sum(
            1 for rid in originally_wrong
            if cot_index[rid] is False
        )
        regressions = sum(
            1 for rid in subset_ids
            if z_index.get(rid, False) and cot_index[rid] is False
        )

        # Accuracies
        zero_correct_subset = total_cases - len(originally_wrong)
        acc_zero_shot = zero_correct_subset / total_cases
        acc_cot = sum(cot_index.values()) / total_cases
        accuracy_gain = acc_cot - acc_zero_shot

        # Overall improvement
        overall_improvement = improvements - regressions
        overall_improvement_rate = overall_improvement / total_cases

        rows.append({
            "Provider": provider,
            "Model": model,
            "Total Cases": total_cases,
            "Originally Wrong": len(originally_wrong),
            "Improvements": improvements,
            "Regressions": regressions,
            "Still Wrong": still_wrong,
            "Overall Improvement": overall_improvement,
            "Overall Improvement Rate": round(overall_improvement_rate, 4),
            "Accuracy Zero-Shot (subset)": round(acc_zero_shot, 4),
            "Accuracy CoT": round(acc_cot, 4),
            "Accuracy Gain": round(accuracy_gain, 4)
        })

    df = pd.DataFrame(rows).sort_values(["Provider", "Model"])
    return df

if __name__ == "__main__":
    df = analyze()
    print(df.to_string(index=False))
    df.to_csv("cot_vs_zeroshot_final_summary.csv", index=False)
