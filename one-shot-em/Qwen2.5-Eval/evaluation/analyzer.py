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
    
    scores = {}
    for dataset in ['amc23x8', 'math500', 'minerva', 'olympiad']:
        if dataset in data:
            scores[dataset] = data[dataset].get('score', 0)
    return scores

def sort_checkpoint(x):
    """Sort function for checkpoint names.
    Extracts the numeric part from checkpoint names like 'checkpoint-100' and returns it as an integer.
    """
    try:
        # Extract the number after 'checkpoint-'
        return int(x.split('-')[1])
    except (IndexError, ValueError):
        # If the format is unexpected, return 0 to put it at the start
        return 100

def main():
    # Create results directory if it doesn't exist
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    # Find all checkpoint directories
    checkpoint_base = Path('../../checkpoints/Qwen2.5-Math-1.5B/one_shot')
    checkpoint_dirs = sorted(glob.glob(str(checkpoint_base / '**/temp00'), recursive=True))
    
    # Collect scores for each checkpoint
    all_scores = []
    for temp_dir in checkpoint_dirs:
        checkpoint_path = Path(temp_dir)
        checkpoint_name = checkpoint_path.parent.name
        
        # Find all JSON result files
        json_files = glob.glob(str(checkpoint_path / '*.json'))
        
        ############### I believe there are some problems here. Fix later
        for json_file in json_files:
            # Check if the dataset name exists in the filename
            json_filename = Path(json_file).name
            if any(dataset in json_filename for dataset in ['amc23x8', 'math500', 'minerva', 'olympiad']):
                scores = extract_scores_from_json(json_file)
                if scores:
                    scores['checkpoint'] = checkpoint_name
                    all_scores.append(scores)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_scores)
    
    # Sort the DataFrame by checkpoint number
    df = df.sort_values(by='checkpoint', key=sort_checkpoint)
    
    # Save raw data
    df.to_csv(results_dir / 'checkpoint_scores.csv', index=False)
    
    # Create line plot
    plt.figure(figsize=(12, 6))
    for dataset in ['amc23x8', 'math500', 'minerva', 'olympiad']:
        if dataset in df.columns:
            plt.plot(df['checkpoint'], df[dataset], marker='o', label=dataset)
    
    plt.title('Evaluation Scores Across Checkpoints')
    plt.xlabel('Checkpoint')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    
    # Save plot
    plt.savefig(results_dir / 'checkpoint_scores.png')
    plt.close()

if __name__ == '__main__':
    main()