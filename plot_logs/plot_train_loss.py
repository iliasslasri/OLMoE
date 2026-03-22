import matplotlib.pyplot as plt
import pandas as pd
import glob
import os

# Find the specific CSVs for each metric
CE_CSV_PATH = glob.glob(os.path.join('/home/infres/lasri-22/OLMoE/plot_logs/', 'CE_*.csv'))[0]
# LB_CSV_PATH = glob.glob(os.path.join('/home/infres/lasri-22/OLMoE/plot_logs/', 'LB_*.csv'))[0]
PERPxt_CSV_PATH = glob.glob(os.path.join('/home/infres/lasri-22/OLMoE/plot_logs/', 'P_*.csv'))[0]

df_ce = pd.read_csv(CE_CSV_PATH)
# df_lb = pd.read_csv(LB_CSV_PATH)
df_perp = pd.read_csv(PERPxt_CSV_PATH)

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
    "dense": "#2E3168",
    "nozloss": "#43C5E0",
    "shared": "#1D3557",
    "layernorm": "#1D3557",
}

x_label = 'Steps (k)'

def prep_ax(ax, data_list, labels, color_keys, ylabel, ylim=None):
    for data, label, color_key in zip(data_list, labels, color_keys):
        if data:
            ax.plot([x[0] for x in data], [x[1] for x in data], linewidth=6.0, color=colors[color_key], label=label)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE)
    if 'CrossEntropy' in ylabel:
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    ax.set_xlim(left=0)
    ax.legend(fontsize=FONTSIZE, frameon=False)
    ax.set_ylabel(ylabel, fontsize=FONTSIZE, fontweight='bold')
    ax.set_xlabel(x_label, fontsize=FONTSIZE, fontweight='bold')
    if ylim:
        ax.set_ylim(bottom=ylim[0], top=ylim[1])
    # ax.set_xlim(right=10)

def create_single_plot(data_list, labels, color_keys, out_name, ylabel, ylim=None):
    fig, ax = plt.subplots(figsize=(16, 12), layout='constrained')
    prep_ax(ax, data_list, labels, color_keys, ylabel, ylim)
    fig.savefig(out_name, dpi=300, bbox_inches='tight')
    
    img_out = out_name.replace('/plots/', '/plots/img/').replace('.pdf', '.png')
    os.makedirs(os.path.dirname(img_out), exist_ok=True)
    fig.savefig(img_out, dpi=300, bbox_inches='tight')
    
    print(f"Plot saved to {out_name} and {img_out}")
    plt.close(fig)

def create_combined_plot(data_lists, labels, color_keys, out_name, ylabels, ylims):
    fig, axes = plt.subplots(1, 3, figsize=(48, 12), layout='constrained')
    for ax, data_list, ylabel, ylim in zip(axes, data_lists, ylabels, ylims):
        prep_ax(ax, data_list, labels, color_keys, ylabel, ylim)
    fig.savefig(out_name, dpi=300, bbox_inches='tight')
    
    img_out = out_name.replace('/plots/', '/plots/img/').replace('.pdf', '.png')
    os.makedirs(os.path.dirname(img_out), exist_ok=True)
    fig.savefig(img_out, dpi=300, bbox_inches='tight')
    
    print(f"Plot saved to {out_name} and {img_out}")
    plt.close(fig)

def process_experiment(runs, labels, color_keys, base_out_name):
    # Extract data for all three metrics
    data_ce = [extract_metric(df_ce, run, 'train/CrossEntropyLoss') for run in runs]
    # data_lb = [extract_metric(df_lb, run, 'train/LoadBalancingLoss') for run in runs]
    data_perp = [extract_metric(df_perp, run, 'train/Perplexity') for run in runs]

    ylabels = ['CrossEntropyLoss', 'LoadBalancingLoss', 'Perplexity']
    ylims = [(2.8, 4.5), (0.010, 0.013), (15, 50)]

    # 1. Plot individual metrics
    create_single_plot(data_ce, labels, color_keys, f'/home/infres/lasri-22/OLMoE/plot_logs/plots/{base_out_name}_CE.pdf', ylabels[0], ylims[0])
    # create_single_plot(data_lb, labels, color_keys, f'/home/infres/lasri-22/OLMoE/plot_logs/plots/{base_out_name}_LB.pdf', ylabels[1], ylims[1])
    create_single_plot(data_perp, labels, color_keys, f'/home/infres/lasri-22/OLMoE/plot_logs/plots/{base_out_name}_PERP.pdf', ylabels[2], ylims[2])

    # Also keep the original name for backward compatibility
    create_single_plot(data_ce, labels, color_keys, f'/home/infres/lasri-22/OLMoE/plot_logs/plots/{base_out_name}.pdf', ylabels[0], ylims[0])

    # 2. Plot combined metrics (3 subplots)
    # create_combined_plot([data_ce, data_lb, data_perp], labels, color_keys, f'/home/infres/lasri-22/OLMoE/plot_logs/plots/{base_out_name}_combined.pdf', ylabels, ylims)

# plot gradient norms for 32exp4_layernorm vs 32exp4
def extract_gradient_norm(df, run_name, window=10):
    x_col = 'Step'
    x_div = 1000
    col_name = f"{run_name} - optim/total_grad_norm"

    if col_name not in df.columns:
        print(f"Warning: Column {col_name} not found in CSV")
        return []

    run_df = df.dropna(subset=[col_name, x_col])
    run_df = run_df.sort_values(x_col).reset_index(drop=True)
    
    if len(run_df) == 0:
        return []

    smoothed_norm = run_df[col_name].rolling(window=window, min_periods=1).mean()
    
    return list(zip(run_df[x_col] / x_div, smoothed_norm))

def create_gradient_norm_plot(data_list, labels, color_keys, out_name, ylabel, ylim):
    fig, ax = plt.subplots(figsize=(16, 12), layout='constrained')
    for data, label, color_key in zip(data_list, labels, color_keys):
        if data:
            ax.plot([x[0] for x in data], [x[1] for x in data], linewidth=6.0, color=colors[color_key], label=label)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE)
    ax.set_xlim(left=0)
    ax.legend(fontsize=FONTSIZE, frameon=False)
    ax.set_ylabel(ylabel, fontsize=FONTSIZE, fontweight='bold')
    ax.set_xlabel(x_label, fontsize=FONTSIZE, fontweight='bold')
    if ylim:
        ax.set_ylim(bottom=ylim[0], top=ylim[1])
    fig.savefig(out_name, dpi=300, bbox_inches='tight')
    
    img_out = out_name.replace('/plots/', '/plots/img/').replace('.pdf', '.png')
    os.makedirs(os.path.dirname(img_out), exist_ok=True)
    fig.savefig(img_out, dpi=300, bbox_inches='tight')
    
    print(f"Plot saved to {out_name} and {img_out}")
    plt.close(fig)

# LayerNorm vs RMSNorm: 32exp4_layernorm, 32exp4
process_experiment(
    ["32exp4_layernorm", "32exp4"],
    ["LayerNorm", "RMSNorm"],
    ["layernorm", "32"],
    'layernorm_rms_comparison'
)

# Extract gradient norm data for both runs
GRAD_NORM_PATH = glob.glob(os.path.join('/home/infres/lasri-22/OLMoE/plot_logs/', 'GRAD_*.csv'))[0]
df_grad_norm = pd.read_csv(GRAD_NORM_PATH)  # Assuming gradient norms are in the same CSV as CE
data_grad_norm_layernorm = extract_gradient_norm(df_grad_norm, "32exp4_layernorm")
data_grad_norm_rmsnorm = extract_gradient_norm(df_grad_norm, "32exp4")
# Create gradient norm plot

create_gradient_norm_plot(
    [data_grad_norm_layernorm, data_grad_norm_rmsnorm],
    ["LayerNorm", "RMSNorm"],
    ["layernorm", "32"],
    f'/home/infres/lasri-22/OLMoE/plot_logs/plots/layernorm_rms_gradient_norm_comparison.pdf',
    'Gradient Norm',
    (0.2, 1)
)

# MoE Granularity: 8exp1, 32exp4, 64exp8
process_experiment(
    ["64exp8", "32exp4", "8exp1"],
    ["64 experts", "32 experts", "8 experts"],
    ["64", "32", "8"],
    'moe_granularity_comparison'
)


# MoE Granularity: 8exp1, 32exp4, 64exp8 with constant Compute (constant active params)
process_experiment(
    ["64exp8", "32exp4_ffn2", "8exp1_ffn8"],
    ["64 experts", "32 experts", "8 experts"],
    ["64", "32", "8"],
    'moe_granularity_constant_compute_comparison'
)

# MoE vs Dense: 32exp4, dense-32exp4
process_experiment(
    ["32exp4", "dense-32exp4"],
    ["MoE (32exp4)", "Dense (32exp4)"],
    ["32", "dense"],
    'moe_vs_dense_comparison'
)

# Z-loss Effect: nozloss-8exp1, 8exp1
process_experiment(
    ["nozloss-8exp1", "8exp1"],
    ["No Z-loss", "With Z-loss"],
    ["nozloss", "8"],
    'zloss_effect_comparison'
)

# Shared Expert Effect: shared-32exp4, 32exp4
process_experiment(
    ["shared-32exp4", "32exp4"],
    ["Shared Expert", "No Shared Expert"],
    ["shared", "32"],
    'shared_expert_effect_comparison'
)
