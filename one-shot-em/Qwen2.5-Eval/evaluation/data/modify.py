import os
import json
import re

input_path = "math500/test.jsonl"
output_folder = "math500-modified"
output_path = os.path.join(output_folder, "test.jsonl")

# Make sure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Pattern to match non-greedy inline math segments enclosed by single $
inline_math_pattern = re.compile(r"\$(.+?)\$", re.DOTALL)

def convert_math_delimiters(text):
    # Replace each $...$ with \(...\) safely
    return inline_math_pattern.sub(lambda m: f"\\({m.group(1)}\\)", text)

with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
    for line in infile:
        example = json.loads(line)
        for key in ["problem", "question"]:
            if key in example:
                example[key] = convert_math_delimiters(example[key])
        outfile.write(json.dumps(example, ensure_ascii=False) + "\n")

print(f"Modified file saved to {output_path}")