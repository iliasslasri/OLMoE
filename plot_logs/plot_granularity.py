import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Option: Toggle between Steps (k) and Tokens (B)
USE_STEPS = True

# Load the CSVs parsed previously
df_64 = pd.read_csv('run_csvs/64exp8.csv')
df_32 = pd.read_csv('run_csvs/32exp4.csv')
df_8 = pd.read_csv('run_csvs/8exp1.csv')

def extract_data(df, shift=0, window=10):
    # Determine X column
    x_col = 'step' if USE_STEPS else 'Tokens (B)'
    x_div = 1000 if USE_STEPS else 1.0

    # Filter out NaNs and enforce perfectly sorted steps
    df = df.dropna(subset=['train/CrossEntropyLoss', x_col])
    df = df.sort_values(x_col).reset_index(drop=True)
    
    # Smooth training loss with a moving average
    smoothed_loss = df['train/CrossEntropyLoss'].rolling(window=window, min_periods=1).mean()
    
    # Structure: [(x, y), (x, y), ...]
    train_data = list(zip(df[x_col] / x_div, smoothed_loss))
    
    hellaswag_data = []
    if 'down_hellaswag' in df.columns:
        hella_df = df.dropna(subset=['down_hellaswag', x_col])
        if len(hella_df) > 0:
            hella_smoothed = hella_df['down_hellaswag'].rolling(window=max(1, window//2), min_periods=1).mean()
            hellaswag_data = list(zip(hella_df[x_col] / x_div, hella_smoothed))
            
    mmlu_data = []
    if 'down_mmlu_5shot' in df.columns:
        mmlu_df = df.dropna(subset=['down_mmlu_5shot', x_col])
        if len(mmlu_df) > 0:
            mmlu_smoothed = mmlu_df['down_mmlu_5shot'].rolling(window=max(1, window//2), min_periods=1).mean()
            mmlu_data = list(zip(mmlu_df[x_col] / x_div, mmlu_smoothed)) 

    mmlu_var_data = []
    if 'down_mmlu_var' in df.columns:
        mmlu_var_df = df.dropna(subset=['down_mmlu_var', x_col])
        if len(mmlu_var_df) > 0:
            mmlu_var_smoothed = mmlu_var_df['down_mmlu_var'].rolling(window=max(1, window//2), min_periods=1).mean()
            mmlu_var_data = list(zip(mmlu_var_df[x_col] / x_div, mmlu_var_smoothed)) 
    
    return {
        "train": train_data,
        "down_hellaswag": hellaswag_data,
        "down_mmlu_5shot": mmlu_data,
        "down_mmlu_var": mmlu_var_data
    }

DATA_FINEONE = extract_data(df_64, shift=0)   # 64
DATA_FINE = extract_data(df_32, shift=1)      # 32
DATA_COARSE = extract_data(df_8, shift=2)     # 8


FONTSIZE = 46
TOKENS_PER_STEP = 4096 * 1024

fig, axes = plt.subplots(figsize=(32, 8), ncols=4, nrows=1, sharex=True, layout='constrained')

titles = [
    "Training loss",
#    "Validation loss (The Pile)",
    "HellaSwag",# (Acc %)",
    "MMLU 5 shot",# (Acc %)",
    "MMLU Var",
]

colors = [
    "#F0539B",
    "#43C5E0",
    "#2E3168",
    "#FDBE15",
]

data = [
  (DATA_FINEONE["train"], DATA_FINE["train"], DATA_COARSE["train"]),
#  (DATA_FINEONE["val_pile"], DATA_FINE["val_pile"], DATA_COARSE["val_pile"]),
  (DATA_FINEONE["down_hellaswag"], DATA_FINE["down_hellaswag"], DATA_COARSE["down_hellaswag"]),
  (DATA_FINEONE["down_mmlu_5shot"], DATA_FINE["down_mmlu_5shot"], DATA_COARSE["down_mmlu_5shot"]),
  (DATA_FINEONE["down_mmlu_var"], DATA_FINE["down_mmlu_var"], DATA_COARSE["down_mmlu_var"]),
]

for i, ax in enumerate(axes.flatten()):
    mult = 100 if i >= 1 else 1
    # Check if data exists; simulate length might differ if runs crashed/stopped early
    if len(data[i][0]) > 0:
        ax.plot(
            [x[0] for x in data[i][0]],
            [x[1]*mult for x in data[i][0]],
            linewidth=6.0,
            color=colors[0],
            label="64",
        )
    if len(data[i][1]) > 0:
        ax.plot(
            [x[0] for x in data[i][1]],
            [x[1]*mult for x in data[i][1]],
            linewidth=6.0,
            color=colors[1],
            label="32",
        )
    if len(data[i][2]) > 0:
        ax.plot(
            [x[0] for x in data[i][2]],
            [x[1]*mult for x in data[i][2]],
            linewidth=6.0,
            color=colors[2],
            label="8",
        )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE)
    ax.set_title(titles[i], fontsize=FONTSIZE, fontweight='bold')

    if i < 1:
      ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
      ax.set_ylim(top=3.5)
    else:
      ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.0f'))

    ax.set_xlim(left=0)

    if i == 1:
      ax.legend(fontsize=FONTSIZE, frameon=False, title="   # experts", title_fontsize=FONTSIZE)

fig.supylabel('Performance', fontsize=FONTSIZE, fontweight='bold')
x_label = 'Steps (k)' if USE_STEPS else 'Tokens (B)'
fig.supxlabel(x_label, fontsize=FONTSIZE, fontweight='bold')

out_name = 'granularity_steps.pdf' if USE_STEPS else 'granularity_tokens.pdf'
plt.savefig(out_name, dpi=300, bbox_inches='tight')
print(f"Plot saved to {out_name}")
