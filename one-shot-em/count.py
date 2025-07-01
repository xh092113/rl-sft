import json
from pathlib import Path
from collections import defaultdict

# Root directory containing step folders
root = Path("checkpoints/Qwen2.5-Math-1.5B/one_shot_1.5b_t0.5")
pattern = "temp00/eval/amc23x8/test_qwen25-math-cot_-1_seed0_t0.0_s0_e-1.jsonl"

# Dictionary: idx -> list of steps where score == [True]
idx_to_true_steps = defaultdict(list)

# Sort step folders numerically by step number
step_dirs = sorted(root.glob("step_*"), key=lambda p: int(p.name.split("_")[1]))

for step_dir in step_dirs:
    step_name = step_dir.name  # e.g., "step_10"
    step_num = int(step_name.split("_")[1])
    jsonl_path = step_dir / pattern

    if not jsonl_path.exists():
        continue

    with open(jsonl_path) as f:
        for line in f:
            entry = json.loads(line)
            idx = entry["idx"]
            score = entry.get("score", [])
            if score == [True]:
                idx_to_true_steps[idx].append(step_num)

# Save as JSON
output_file = "all_correct_steps_per_problem00.json"
with open(output_file, "w") as f:
    json.dump(dict(sorted(idx_to_true_steps.items())), f, indent=2)

print(f"✅ Saved to {output_file}")