import matplotlib.pyplot as plt
import pandas as pd

CSV_PATH = '/home/infres/lasri-22/OLMoE/plot_logs/wandb_export_2026-03-16T20_50_07.910+01_00.csv'
df_all = pd.read_csv(CSV_PATH)

def extract_metric(df, run_name, metric='train/CrossEntropyLoss', window=10):
    # Determine X column
    x_col = 'Step'
    x_div = 1000

    col_name = f"{run_name} - {metric}"
    if col_name not in df.columns:
        print(f"Warning: Column {col_name} not found in CSV")
        return []

    run_df = df.dropna(subset=[col_name, x_col])
    run_df = run_df.sort_values(x_col).reset_index(drop=True)
    
    if len(run_df) == 0:
        return []

    smoothed_loss = run_df[col_name].rolling(window=window, min_periods=1).mean()
    
    return list(zip(run_df[x_col] / x_div, smoothed_loss))

FONTSIZE = 46
plt.rcParams.update({'font.size': FONTSIZE})

colors = {
    "64": "#F0539B",
    "32": "#43C5E0",
    "8": "#2E3168",
    "dense": "#FDBE15",
    "nozloss": "#E63946",
    "shared": "#1D3557",
}

x_label = 'Steps (k)'

def create_plot(data_list, labels, color_keys, out_name):
    fig, ax = plt.subplots(figsize=(16, 12), layout='constrained')
    
    for data, label, color_key in zip(data_list, labels, color_keys):
        if data:
            ax.plot([x[0] for x in data], [x[1] for x in data], linewidth=6.0, color=colors[color_key], label=label)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    ax.set_xlim(left=0)
    ax.legend(fontsize=FONTSIZE, frameon=False)
    ax.set_ylabel('CrossEntropyLoss', fontsize=FONTSIZE, fontweight='bold')
    ax.set_xlabel(x_label, fontsize=FONTSIZE, fontweight='bold')
    ax.set_ylim(bottom=2.5, top=6)
    fig.savefig(out_name, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {out_name}")
    plt.close(fig)

# MoE Granularity: 8exp1, 32exp4, 64exp8
create_plot(
    [extract_metric(df_all, "64exp8"), extract_metric(df_all, "32exp4"), extract_metric(df_all, "8exp1")],
    ["64 experts", "32 experts", "8 experts"],
    ["64", "32", "8"],
    'plot_logs/plots/moe_granularity_comparison.pdf'
)
# MoE Granularity: 8exp1, 32exp4, 64exp8 with constant Compute (constant active params)
create_plot(
    [extract_metric(df_all, "64exp8"), extract_metric(df_all, "32exp4_ffn2")],
    ["64 experts", "32 experts", "8 experts"],
    ["64", "32", "8"],
    'plot_logs/plots/moe_granularity_constant_compute_comparison.pdf'
)
# MoE vs Dense: 32exp4, dense-32exp4
create_plot(
    [extract_metric(df_all, "32exp4"), extract_metric(df_all, "dense-32exp4")],
    ["MoE (32exp4)", "Dense (32exp4)"],
    ["32", "dense"],
    'plot_logs/plots/moe_vs_dense_comparison.pdf'
)

# Z-loss Effect: nozloss-8exp1, 8exp1
create_plot(
    [extract_metric(df_all, "nozloss-8exp1"), extract_metric(df_all, "8exp1")],
    ["No Z-loss", "With Z-loss"],
    ["nozloss", "8"],
    'plot_logs/plots/zloss_effect_comparison.pdf'
)

# Shared Expert Effect: shared-32exp4, 32exp4
create_plot(
    [extract_metric(df_all, "shared-32exp4"), extract_metric(df_all, "32exp4")],
    ["Shared Expert", "No Shared Expert"],
    ["shared", "32"],
    'plot_logs/plots/shared_expert_effect_comparison.pdf'
)
