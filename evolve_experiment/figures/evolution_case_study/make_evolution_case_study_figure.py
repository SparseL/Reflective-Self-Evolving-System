import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import json
import os
from pathlib import Path

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "requirements.txt").is_file())
OUT_DIR = ROOT / "evolve_experiment" / "figures" / "evolution_case_study"

# Nature Figure Style Settings
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})

def save_pub_py(fig, filename, dpi=600):
    fig.savefig(f"{filename}.svg", bbox_inches="tight")
    fig.savefig(f"{filename}.pdf", bbox_inches="tight")
    fig.savefig(f"{filename}.png", dpi=dpi, bbox_inches="tight")

def load_enron_data():
    csv_path = ROOT / "evolve_experiment" / "evolution" / "real" / "Enron_sampled_post_with_precompute_new" / "results" / "enron_sampled_pops_best_summary.csv"
    df = pd.read_csv(csv_path)
    return df

def load_ws_data():
    # Manual data from logs as summary CSV might not be easily accessible
    # Trajectory for WS_1000 from all_pops.json and summary
    gens = np.arange(1, 21)
    objs = np.zeros(20)
    objs[0:3] = 0.25825 # Gen 1-3
    objs[3:9] = 0.25765 # Gen 4-9
    objs[9:10] = 0.25506 # Gen 10
    objs[10:] = 0.22765 # Jump at Gen 11
    return gens, objs

def annotate_value(ax, x, y, text, xlim, ylim):
    x0, x1 = xlim
    y0, y1 = ylim

    dx, dy = 6, 5
    ha, va = "left", "bottom"

    if x >= x1 - 3:
        dx = -6
        ha = "right"

    if y >= y1 - (y1 - y0) * 0.08:
        dy = -5
        va = "top"

    ax.annotate(
        text,
        xy=(x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        ha=ha,
        va=va,
        fontsize=5,
        fontweight="bold",
        color="#2C3E50",
        zorder=6,
        annotation_clip=True,
    )

def plot_evolution_case_study():
    # Nature Figure Style Settings for individual plots
    mpl.rcParams.update({"font.size": 7})
    
    # --- Case 1: Enron_sample ---
    # Increased width to height ratio for rectangular shape
    fig_enron = plt.figure(figsize=(6.0, 3.1)) 
    ax_enron = fig_enron.add_subplot(111)
    df_enron = load_enron_data()
    df_enron = df_enron.sort_values("generation").reset_index(drop=True)
    
    # Detect all jump points
    df_enron['diff'] = df_enron['objective'].diff()
    jumps_enron = df_enron[df_enron['diff'] < -1e-7].copy()
    jumps_enron = pd.concat([df_enron.iloc[[0]], jumps_enron]).drop_duplicates(subset=['generation'])
    
    enron_last_gen = int(df_enron["generation"].max())
    enron_xmax = 50
    enron_x = df_enron["generation"].to_numpy()
    enron_y = df_enron["objective"].to_numpy()
    if enron_x[-1] < enron_xmax:
        enron_x = np.append(enron_x, enron_xmax)
        enron_y = np.append(enron_y, enron_y[-1])
    ax_enron.plot(enron_x, enron_y, color='#34495E', linewidth=1.0, zorder=1)
    
    enron_colors = plt.cm.turbo(np.linspace(0.08, 0.92, len(jumps_enron)))
    enron_ylim = (0.21, 0.242)
    enron_xlim = (0, enron_xmax + 0.8)
    for i, (idx, row) in enumerate(jumps_enron.iterrows()):
        gen = int(row['generation'])
        obj = row['objective']
        color = enron_colors[i]
        ax_enron.scatter(gen, obj, color=color, s=36, zorder=5, edgecolor='black', linewidth=0.6)
        if i != 0:
            annotate_value(ax_enron, gen, obj, f"{obj:.4f}", enron_xlim, enron_ylim)

    enron_final_obj = float(df_enron.loc[df_enron["generation"] == enron_last_gen, "objective"].iloc[0])
    ax_enron.scatter(
        enron_xmax,
        enron_final_obj,
        marker="D",
        s=60,
        facecolor="white",
        edgecolor="#2C3E50",
        linewidth=1.0,
        zorder=6,
    )

    ax_enron.set_xlabel('Generation')
    ax_enron.set_ylabel('Performance (ANC)')
    ax_enron.set_title('Evolution: Enron_sample', fontweight='bold', fontsize=8)
    ax_enron.set_ylim(*enron_ylim)
    ax_enron.set_xlim(*enron_xlim)
    ax_enron.grid(axis='y', linestyle=':', alpha=0.3)
    
    plt.tight_layout()
    save_pub_py(fig_enron, OUT_DIR / "evolution_enron_sample_rect")

    # --- Case 2: synthetic_ws_1000 ---
    # Increased width to height ratio for rectangular shape
    fig_ws = plt.figure(figsize=(6.0, 3.1))
    ax_ws = fig_ws.add_subplot(111)
    gens_ws, objs_ws = load_ws_data()
    ws_last_gen = int(np.max(gens_ws))
    ws_xmax = ws_last_gen + 8
    ws_x = gens_ws
    ws_y = objs_ws
    if ws_x[-1] < ws_xmax:
        ws_x = np.append(ws_x, ws_xmax)
        ws_y = np.append(ws_y, ws_y[-1])
    ax_ws.plot(ws_x, ws_y, color='#34495E', linewidth=1.0, zorder=1)
    
    ws_jumps = [(1, 0.25825), (4, 0.25765), (10, 0.25506), (11, 0.22765)]
    ws_colors = plt.cm.turbo(np.linspace(0.12, 0.88, len(ws_jumps)))
    ws_ylim = (0.22, 0.265)
    ws_xlim = (0, ws_xmax + 0.8)
    
    for i, (gen, obj) in enumerate(ws_jumps):
        color = ws_colors[i]
        ax_ws.scatter(gen, obj, color=color, s=36, zorder=5, edgecolor='black', linewidth=0.6)
        if i != 0:
            annotate_value(ax_ws, gen, obj, f"{obj:.4f}", ws_xlim, ws_ylim)

    ws_final_obj = float(objs_ws[-1])
    ax_ws.scatter(
        ws_xmax,
        ws_final_obj,
        marker="D",
        s=60,
        facecolor="white",
        edgecolor="#2C3E50",
        linewidth=1.0,
        zorder=6,
    )

    ax_ws.set_xlabel('Generation')
    ax_ws.set_ylabel('Performance (ANC)')
    ax_ws.set_title('Evolution: WS_1000', fontweight='bold', fontsize=8)
    ax_ws.set_ylim(*ws_ylim)
    ax_ws.set_xlim(*ws_xlim)
    ax_ws.grid(axis='y', linestyle=':', alpha=0.3)

    plt.tight_layout()
    save_pub_py(fig_ws, OUT_DIR / "evolution_ws_1000_rect")
    # plt.show()
    # plt.show()

if __name__ == "__main__":
    plot_evolution_case_study()
