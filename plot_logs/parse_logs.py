import re
import pandas as pd
import sys
import os
import glob
import numpy as np

def parse_log_file(filepath):
    data = []
    current_step = None
    metrics = {}
    
    with open(filepath, 'r') as f:
        for line in f:
            step_match = re.search(r'INFO\s+\[step=(\d+)/', line)
            if step_match:
                if current_step is not None and metrics:
                    metrics['step'] = current_step
                    data.append(metrics.copy())
                current_step = int(step_match.group(1))
                metrics = {}
            
            if current_step is not None:
                metric_match = re.search(r'([\w/]+)=(?:[+-]?(?:(?:\d+\.\d*)|(?:\.\d+)|(?:\d+))(?:[eE][+-]?\d+)?|[\d,]+)', line)
                if metric_match:
                    key = metric_match.group(1)
                    val_str = metric_match.group(0).split('=')[1].replace(',', '')
                    try:
                        metrics[key] = float(val_str)
                    except ValueError:
                        pass
          
    if current_step is not None and metrics:
        metrics['step'] = current_step
        data.append(metrics)
        
    df = pd.DataFrame(data)
    if 'step' in df.columns:
        df = df.sort_values(by='step').reset_index(drop=True)
    
    # Formula: Tokens = Steps * BatchSize * SeqLen
    # BatchSize = 16, SeqLen = 4096 => Tokens = Step * 65536
    if 'step' in df.columns:
        df['Tokens (B)'] = df['step'] * 65536 / 1e9

    if 'eval/downstream/hellaswag_len_norm' in df.columns:
        df['down_hellaswag'] = df['eval/downstream/hellaswag_len_norm']
    
    mmlu_cols = [c for c in df.columns if 'mmlu' in c and 'var_len_norm' in c]
    if mmlu_cols:
        df['down_mmlu_var'] = df[mmlu_cols].mean(axis=1)

    mmlu_5shot_cols = [c for c in df.columns if 'mmlu' in c and 'mc_5shot' in c and 'test' not in c]
    if mmlu_5shot_cols:
        df['down_mmlu_5shot'] = df[mmlu_5shot_cols].mean(axis=1)
        
    return df

os.makedirs('run_csvs', exist_ok=True)
log_files = glob.glob('*.log') + ['../64exp8.out']
for log_file in log_files:
    run_name = os.path.basename(log_file).replace('.log', '').replace('.out', '')
    print(f"Parsing {log_file}...")
    df = parse_log_file(log_file)
    if 'step' in df.columns:
        df.to_csv(f'run_csvs/{run_name}.csv', index=False)
        print(f"Saved to run_csvs/{run_name}.csv")
