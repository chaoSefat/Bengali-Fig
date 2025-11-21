# Bengali-Fig


---

## Running the Evaluation

- `zero_shot_riddle_eval.py` — Zero-shot evaluator: asks models to return a single letter (A/B/C/D) in JSON format for each riddle.
- `cot.py` — Chain-of-Thought (CoT) evaluator: prompts models to show step-by-step reasoning, expects a JSON response with reasoning + answer.
- `result_analyzer.py` — Aggregates result JSON files (usually from `results/zero_shot`) and prints/saves a detailed analysis and comparison CSV.
- `cot_analyser.py` — Compares CoT runs to corresponding zero-shot runs and reports accuracy deltas and improvement/regression statistics.

Common outputs are saved under `results/` and some CSV summaries are produced.

---

### Requirements

- Python 3.9+
- Packages: `openai` (or compatible client used in the repo), `python-dotenv`, `pandas`

Suggested install (create and activate a virtualenv first):

```bash
python -m venv .venv
source .venv/bin/activate
pip install pandas python-dotenv openai
```

or install from requirements.txt:

```bash
pip install -r requirements.txt
```
---

### Environment variables

Place secrets in a `.env` file or export them in your shell. The scripts check for the following keys (depending on provider):

- `OPENAI_API_KEY` — for OpenAI
- `NOVITA_API_KEY` — for Novita (base_url is set in code)
- `DEEPSEEK_API_KEY` — for Deepseek (base_url is set in code)
- `ANTHROPIC_API_KEY` — for Anthropic (base_url is set in code)

The repo uses `dotenv.load_dotenv()` so a `.env` file in the repo root with lines like `OPENAI_API_KEY=sk-...` will work.

---

### Data files

- `data/mcq_dataset.json` (or `data/v4_patched_mcq_dataset.json` depending on your dataset naming) — MCQ riddle dataset used by evaluators. Each entry is expected to contain fields like `id`, `question`, `options`, `correct_option`, `answer`, `reasoning_type`, `difficulty`, `trap_type`, `cultural_depth`, etc.
- `results/hardest_data_points.json` — used by `cot.py` main in the repository to focus CoT runs on a subset of hard cases. If not present, update the `data_file` variable inside the script or pass a custom path when running programmatically.

Make sure the data file has the structure expected by the evaluators (see `RiddleEvaluator` / `CoTRiddleEvaluator` constructors which call `json.load`).

---

### Files and how to use them

#### zero_shot_riddle_eval.py

Purpose: Run zero-shot evaluation on a dataset and save model responses.

What it does:

- Loads the dataset (`data_file` variable in `main()` defaults to `data/v4_patched_mcq_dataset.json`).
- Builds a simple prompt asking the model to return JSON like {"উত্তর":"A"}.
- Calls the provider/model defined in the `settings` list inside `main()`.
- Saves results to `results/zero_shot/` with a timestamped filename.

Inputs:

- Data file (JSON list of riddles).
- Provider & model definitions inside `main()` (edit the `settings` list).
- Environment API key(s).

Outputs:

- JSON file in `results/zero_shot/` with structure: `{ "metadata": {...}, "results": [{...}, ...] }`.

How to run:

- Option A (quick, using defaults):

  - Ensure environment variables are set and `data_file` path in `main()` is correct.
  - Run:
    ```bash
    python zero_shot_riddle_eval.py
    ```
  - The script runs the providers/models listed in the `settings` list sequentially.

- Option B (programmatic):
  - Import `RiddleEvaluator` in another script and instantiate with your provider/model/data_file. Call `run_evaluation()` with custom delays and start index.

Notes:

- The script uses internal `settings` in `main()` instead of CLI flags. Edit that list to add/remove models or change delays.
- There are built-in sleeps to avoid rate limits; tune `delay` and `batch_delay` in `settings`.

#### cot.py

Purpose: Run Chain-of-Thought (CoT) evaluation on (usually) hard cases and save the model's reasoning and final answer.

What it does:

- Loads dataset (defaults to `results/hardest_data_points.json` in `main()`).
- For each riddle, builds a CoT prompt in Bengali that asks the model to provide step-by-step reasoning and a JSON answer (with `যুক্তি` and `উত্তর`).
- Calls the model and parses the response (attempts JSON parsing and several regex fallbacks).
- Saves detailed results into `results/chain_of_thought_hard_cases/` with per-run JSON files. Also prints progress and periodic checkpoints.

Inputs:

- `data_file` path (set in `main()` for hard cases).
- Provider & model list in `main()`.
- Environment API key(s).

Outputs:

- JSON files in `results/chain_of_thought_hard_cases/` containing `metadata` and `results` (each result includes `raw_response`, parsed `reasoning`, predicted option and more).

How to run:

- Ensure API keys and `data_file` are correct.
- Run:
  ```bash
  python cot.py
  ```
- Or instantiate `CoTRiddleEvaluator` programmatically and call `run_evaluation(delay=..., batch_delay=..., start_index=...)`.

Notes:

- CoT prompts are verbose and in Bengali. The code attempts to parse structured JSON responses but also tolerates non-JSON outputs.
- You can change models in the `settings` list. Tweak delays to match the rate limits of your provider.

#### result_analyzer.py

Purpose: Aggregate many result JSON files (e.g., those in `results/zero_shot`) and generate a rich analysis including accuracy by model, by reasoning type, difficulty, trap susceptibility, and confusing distractors.

What it does:

- Reads all JSON files from `results/zero_shot` (default in `main()`) using glob.
- Computes primary metrics (overall accuracy, accuracy by reasoning type, difficulty, cultural depth).
- Computes secondary metrics (trap susceptibility, distractor confusion, confidence calibration).
- Prints an organized report to stdout and writes a comparison CSV to `result_analysis/zero_shot_model_comparison.csv`.

How to run:

- After you have result JSON files (from `zero_shot_riddle_eval.py`), run:
  ```bash
  python result_analyzer.py
  ```
- The script expects `.json` files in `results/zero_shot/`.

Notes:

- The analyzer currently uses a fixed `results_dir` variable in `main()`; modify it or call `ResultsAnalyzer` directly with a list of file paths.

#### cot_analyser.py

Purpose: Compare CoT runs (chain-of-thought) against zero-shot runs over the same IDs and report improvements/regressions.

What it does:

- Loads JSON files from `results/zero_shot` and `results/chain_of_thought_hard_cases` by provider/model metadata.
- Builds indices mapping `riddle_id` → correctness and computes metrics only on the CoT subset (so comparisons are aligned to cases CoT actually tried).
- Produces a pandas DataFrame summarizing improvements, regressions, and accuracy gains. Saves CSV `cot_vs_zeroshot_final_summary.csv`.

How to run:

- Ensure you have matching zero-shot and CoT output files (same provider/model pairs under `results/zero_shot` and `results/chain_of_thought_hard_cases`).
- Run:
  ```bash
  python cot_analyser.py
  ```
- The script writes `cot_vs_zeroshot_final_summary.csv` in the repo root.

Notes:

- The analyzer picks the most recent zero-shot file if multiple are present for a provider/model pair.

---

### Output directories and expected artifacts

- `results/zero_shot/` — JSON result files from `zero_shot_riddle_eval.py`.
- `results/chain_of_thought_hard_cases/` — JSON result files from `cot.py`.
- `result_analysis/` — CSV summaries produced by `result_analyzer.py` (script creates `result_analysis/zero_shot_model_comparison.csv`).
- `cot_vs_zeroshot_final_summary.csv` — summary CSV generated by `cot_analyser.py`.

---

### Tips, troubleshooting and suggestions

- If you get API errors, double-check the corresponding environment variable and the provider base_url used in the script.
- To add models/providers, edit the `settings` list in the `main()` function of the evaluator you want to run (zero-shot or CoT). Each entry should have at least `provider`, `model`, `delay`, and `batch_delay` keys.
- If your dataset file has a different name/path than the script expects, edit the `data_file` variable in `main()` or run the evaluator programmatically.

