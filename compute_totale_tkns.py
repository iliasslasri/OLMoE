import os
import yaml

with open('configs/olmoe-small.yml', 'r') as f:
    config = yaml.safe_load(f)
data_paths = config['data']['paths']
files_size = 0
for path in data_paths:
    files_size += os.path.getsize(path)
    
# uint16 = 2 bytes per token
total_tokens = files_size // 2
print(f'File size: {files_size / 1e9:.2f} GB')
print(f'Total tokens (approx): {total_tokens:,}')

tokens_per_step = config['global_train_batch_size'] * config['model']['max_sequence_length']
total_steps = total_tokens // tokens_per_step
print(f'Tokens per step: {tokens_per_step:,}')
print(f'Steps for 1 epoch: {total_steps:,}')