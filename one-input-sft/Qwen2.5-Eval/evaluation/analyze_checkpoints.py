import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def extract_scores_from_json(json_file):
    """Extract scores from a JSON evaluation result file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    if "acc" in data:
        return data["acc"]
    else:
        raise ValueError(f"No 'acc' key found in {json_file}")

def sortfunc(x):
    if x.name == "final":
        return 100000
    else:
        return int(x.name.split('_')[1])

def main():
    # Create results directory if it doesn't exist
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    # Find all checkpoint directories
    checkpoint_base = Path('../../checkpoints/Qwen2.5-Math-1.5B/one_shot')
    # get all directories in checkpoint_base
    checkpoint_dirs = [x for x in checkpoint_base.iterdir() if x.is_dir()]
    checkpoint_dirs = sorted(checkpoint_dirs, key=sortfunc)
    
    # Collect scores for each checkpoint
    all_scores = []
    for checkpoint_dir in checkpoint_dirs:
        checkpoint_name = checkpoint_dir.name
        print("checkpoint_name: ", checkpoint_name)
        scores = {}
        for dataset in ['amc23x8', 'math500', 'minerva_math', 'olympiadbench']:
            json_files = glob.glob(str(checkpoint_dir / "temp00" / "eval" / dataset / '*metrics.json'))
            if len(json_files) != 1:
                raise ValueError(f"Expected 1 metrics.json file, found {len(json_files)} for checkpoint {checkpoint_name}")
            json_file = json_files[0]
            scores[dataset] = extract_scores_from_json(json_file)
        scores['checkpoint'] = checkpoint_name
        all_scores.append(scores)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_scores)
    
    # Save raw data
    df.to_csv(results_dir / 'checkpoint_scores_1_5b.csv', index=False)
    
    # Create line plot, also plot the average score
    plt.figure(figsize=(12, 6))
    for dataset in ['amc23x8', 'math500', 'minerva_math', 'olympiadbench']:
        if dataset in df.columns:
            plt.plot(df['checkpoint'], df[dataset], marker='o', label=dataset)

    # plot the average score
    plt.plot(df['checkpoint'], df.drop(columns=['checkpoint']).mean(axis=1), marker='o', label='Average')

    plt.title('Evaluation Scores Across Checkpoints')
    plt.xlabel('Checkpoint')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    
    # Save plot
    plt.savefig(results_dir / 'checkpoint_scores_1_5b.png')
    plt.close()

    # give me the highest score for each dataset
    for dataset in ['amc23x8', 'math500', 'minerva_math', 'olympiadbench']:
        print(f"Highest score for {dataset}: {df[dataset].max()}")

    # give me the checkpoint name for the highest score for each dataset
    for dataset in ['amc23x8', 'math500', 'minerva_math', 'olympiadbench']:
        print(f"Checkpoint name for highest score for {dataset}: {df[df[dataset] == df[dataset].max()]['checkpoint'].values[0]}")
    
    # give me the checkpoint name for the highest average score (remember to exclude the 'checkpoint' column)
    print(f"Checkpoint name for highest average score: {df.drop(columns=['checkpoint']).mean(axis=1).idxmax()}")

if __name__ == '__main__':
    main() 