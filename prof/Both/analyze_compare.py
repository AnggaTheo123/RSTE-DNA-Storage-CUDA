"""
RSTE Encoding: Sequential vs CUDA Comparison & Visualization
Paper: "Stable DNA Storage Encoding Scheme Based on Repeating Substring Tree"
       IEEE TCBB, Vol.22, No.5, Sept/Oct 2025

Generates:
  1. Merged comparison table (Excel)
  2. Speedup bar chart per dataset and per kernel
  3. Encoding rate comparison
  4. GC variance comparison
  5. Kernel time breakdown (stacked bar)
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, avoids Windows DLL crash
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

# ── paths ───────────────────────────────────────────────────────────────────
BASE   = r"c:\KomparUAS"
SEQ_CSV  = os.path.join(BASE, "hasil_sequential", "profil_sequential_all.csv")
CUDA_CSV = os.path.join(BASE, "hasil_cuda",       "profil_cuda_v2.csv")
OUT_DIR  = os.path.join(BASE, "hasil_analisis")
EXCEL_OUT = os.path.join(OUT_DIR, "result_summary.xlsx")
os.makedirs(OUT_DIR, exist_ok=True)

# ── load ─────────────────────────────────────────────────────────────────────
seq  = pd.read_csv(SEQ_CSV)
cuda = pd.read_csv(CUDA_CSV)

# Backward-compatible handling: older CUDA CSV may not have phase_sum_ms
if "phase_sum_ms" not in cuda.columns:
    cuda["phase_sum_ms"] = cuda["total_ms"]
if "gross_rate_bits_per_nt" not in seq.columns:
    seq["gross_rate_bits_per_nt"] = seq["encoding_rate_bits_per_nt"]
if "gross_rate_bits_per_nt" not in cuda.columns:
    cuda["gross_rate_bits_per_nt"] = cuda["encoding_rate_bits_per_nt"]
if "dna_hash_payload" not in seq.columns:
    seq["dna_hash_payload"] = ""
if "dna_hash_payload" not in cuda.columns:
    cuda["dna_hash_payload"] = ""

# Merge on dataset tag
merged = seq.merge(cuda, on="dataset", suffixes=("_seq", "_cuda"))
merged["dataset_label"] = merged["dataset"]

DATASETS = merged["dataset"].tolist()
FILE_SIZES_KB = merged["file_size_bytes_seq"] / 1024

# ── rename seq columns to avoid suffix confusion ──────────────────────────────
# sequential CSV cols: total_ms, lsbm_ms, huffman_ms, encode_ms, encoding_rate_bits_per_nt, gc_mean, gc_var ...
# after merge they become: total_ms_seq, lsbm_ms_seq, huffman_ms_seq, encode_ms_seq ...
# cuda CSV cols: total_ms, lsbm_kernel_ms, huffman_cpu_ms, encode_kernel_ms ...
# after merge they become: total_ms_cuda, lsbm_kernel_ms_cuda ...

# ── speedup metrics ───────────────────────────────────────────────────────────
merged["speedup_total"]  = merged["total_ms_seq"]    / merged["total_ms_cuda"]
merged["speedup_lsbm"]   = merged["lsbm_ms"]         / merged["lsbm_kernel_ms"]
merged["speedup_encode"] = merged["encode_ms"]       / merged["encode_kernel_ms"]

# Fairer stage-only compute comparison (excludes file I/O for sequential)
merged["seq_compute_ms"] = merged["lsbm_ms"] + merged["huffman_ms"] + merged["encode_ms"]
merged["cuda_compute_ms"] = (
    merged["bits_kernel_ms"]
    + merged["lsbm_kernel_ms"]
    + merged["huffman_cpu_ms"]
    + merged["encode_kernel_ms"]
    + merged["constraint_kernel_ms"]
)
merged["speedup_compute"] = merged["seq_compute_ms"] / merged["cuda_compute_ms"]
merged["payload_hash_match"] = merged["dna_hash_payload_seq"] == merged["dna_hash_payload_cuda"]

# ── Excel summary ─────────────────────────────────────────────────────────────
summary_cols = {
    "Dataset":            merged["dataset"],
    "File Size (KB)":     FILE_SIZES_KB.round(1),
    "Seq Total (ms)":     merged["total_ms_seq"].round(2),
    "CUDA Total (ms)":    merged["total_ms_cuda"].round(2),
    "CUDA Phase Sum (ms)": merged["phase_sum_ms"].round(2),
    "Speedup Total":      merged["speedup_total"].round(2),
    "Seq Compute (ms)":   merged["seq_compute_ms"].round(2),
    "CUDA Compute (ms)":  merged["cuda_compute_ms"].round(2),
    "Speedup Compute":    merged["speedup_compute"].round(2),
    "Seq LSBM (ms)":      merged["lsbm_ms"].round(2),
    "CUDA LSBM (ms)":     merged["lsbm_kernel_ms"].round(2),
    "Speedup LSBM":       merged["speedup_lsbm"].round(2),
    "Seq Encode (ms)":    merged["encode_ms"].round(2),
    "CUDA Encode (ms)":   merged["encode_kernel_ms"].round(2),
    "Speedup Encode":     merged["speedup_encode"].round(2),
    "Enc Rate Seq":       merged["encoding_rate_bits_per_nt_seq"].round(4),
    "Enc Rate CUDA":      merged["encoding_rate_bits_per_nt_cuda"].round(4),
    "Gross Rate Seq":     merged["gross_rate_bits_per_nt_seq"].round(4),
    "Gross Rate CUDA":    merged["gross_rate_bits_per_nt_cuda"].round(4),
    "GC Mean Seq":        merged["gc_mean_seq"].round(4),
    "GC Mean CUDA":       merged["gc_mean_cuda"].round(4),
    "GC Var Seq":         merged["gc_var_seq"].map(lambda x: f"{x:.6f}"),
    "GC Var CUDA":        merged["gc_var_cuda"].map(lambda x: f"{x:.6f}"),
    "RLL Max Seq":        merged["rll_max_seq"],
    "RLL Max CUDA":       merged["rll_max_cuda"],
    "End Viol Seq":       merged["end_violations_seq"],
    "End Viol CUDA":      merged["end_violations_cuda"],
    "Payload Hash Match": merged["payload_hash_match"],
}

df_summary = pd.DataFrame(summary_cols)
with pd.ExcelWriter(EXCEL_OUT, engine="openpyxl") as writer:
    df_summary.to_excel(writer, sheet_name="Comparison", index=False)
    merged.to_excel(writer, sheet_name="Raw Data", index=False)
print(f"[OK] Excel saved: {EXCEL_OUT}")

# ── plot style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})
COLORS = {"seq": "#2196F3", "cuda": "#FF5722", "speedup": "#4CAF50"}

def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Plot saved: {path}")

x   = np.arange(len(DATASETS))
w   = 0.35

# ── Figure 1: Total runtime comparison ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
b1 = ax.bar(x - w/2, merged["total_ms_seq"],  w, label="Sequential (CPU)", color=COLORS["seq"])
b2 = ax.bar(x + w/2, merged["total_ms_cuda"], w, label="CUDA (GPU)",       color=COLORS["cuda"])
ax.set_xlabel("Dataset")
ax.set_ylabel("Execution Time (ms)")
ax.set_title("Total Execution Time: Sequential vs CUDA")
ax.set_xticks(x); ax.set_xticklabels(DATASETS)
ax.legend()
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v,_: f"{v:,.0f}"))
for bar in [b1, b2]:
    for rect in bar:
        h = rect.get_height()
        ax.annotate(f"{h:,.0f}", xy=(rect.get_x()+rect.get_width()/2, h),
                    xytext=(0,3), textcoords="offset points", ha="center", fontsize=7)
save(fig, "fig1_total_runtime.png")

# ── Figure 2: Speedup per dataset ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(DATASETS, merged["speedup_total"], color=COLORS["speedup"], width=0.5)
ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, label="Baseline (1×)")
ax.set_xlabel("Dataset"); ax.set_ylabel("Speedup (×)")
ax.set_title("Overall Speedup: Sequential → CUDA")
ax.legend()
for rect in bars:
    h = rect.get_height()
    ax.text(rect.get_x()+rect.get_width()/2, h+0.05, f"{h:.2f}×",
            ha="center", fontsize=9, fontweight="bold")
save(fig, "fig2_speedup_total.png")

# ── Figure 3: LSBM speedup ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(DATASETS, merged["speedup_lsbm"], color="#9C27B0", width=0.5)
ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
ax.set_xlabel("Dataset"); ax.set_ylabel("Speedup (×)")
ax.set_title("LSBM Kernel Speedup: Sequential → CUDA")
for rect in bars:
    h = rect.get_height()
    ax.text(rect.get_x()+rect.get_width()/2, h+0.05, f"{h:.2f}×",
            ha="center", fontsize=9, fontweight="bold")
save(fig, "fig3_speedup_lsbm.png")

# ── Figure 4: Kernel time breakdown (stacked bar) ────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Sequential breakdown
seq_parts  = {
    "LSBM":     merged["lsbm_ms"],
    "Huffman":  merged["huffman_ms"],
    "Encode":   merged["encode_ms"],
}
bottoms = np.zeros(len(DATASETS))
part_colors = ["#EF5350","#42A5F5","#66BB6A","#FFA726","#AB47BC"]
for (label, vals), color in zip(seq_parts.items(), part_colors):
    axes[0].bar(DATASETS, vals, bottom=bottoms, label=label, color=color)
    bottoms += vals.values
axes[0].set_title("Sequential: Time Breakdown per Stage")
axes[0].set_ylabel("Time (ms)"); axes[0].legend(); axes[0].tick_params(axis="x", rotation=15)
axes[0].yaxis.set_major_formatter(ticker.FuncFormatter(lambda v,_: f"{v:,.0f}"))

# CUDA breakdown
cuda_parts = {
    "H2D":        merged["h2d_ms"],
    "Bits Kernel":merged["bits_kernel_ms"],
    "LSBM Kernel":merged["lsbm_kernel_ms"],
    "Huffman CPU":merged["huffman_cpu_ms"],
    "Encode Kern":merged["encode_kernel_ms"],
    "Constraint": merged["constraint_kernel_ms"],
    "D2H":        merged["d2h_ms"],
}
bottoms = np.zeros(len(DATASETS))
for (label, vals), color in zip(cuda_parts.items(), part_colors + ["#00BCD4","#FF7043"]):
    axes[1].bar(DATASETS, vals, bottom=bottoms, label=label, color=color)
    bottoms += vals.values
axes[1].set_title("CUDA: Time Breakdown per Kernel")
axes[1].set_ylabel("Time (ms)"); axes[1].legend(fontsize=7); axes[1].tick_params(axis="x", rotation=15)
axes[1].yaxis.set_major_formatter(ticker.FuncFormatter(lambda v,_: f"{v:,.0f}"))

plt.tight_layout()
save(fig, "fig4_kernel_breakdown.png")

# ── Figure 5: Encoding rate comparison ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - w/2, merged["encoding_rate_bits_per_nt_seq"],  w, label="Sequential", color=COLORS["seq"])
ax.bar(x + w/2, merged["encoding_rate_bits_per_nt_cuda"], w, label="CUDA",       color=COLORS["cuda"])
ax.axhline(2.0, color="red", linestyle="--", linewidth=1.0, label="Quaternary max (2.0 bits/nt)")
ax.axhline(2.26, color="green", linestyle=":", linewidth=1.0, label="Paper avg (2.26 bits/nt)")
ax.set_xlabel("Dataset"); ax.set_ylabel("Encoding Rate (bits/nt)")
ax.set_title("Encoding Rate: Sequential vs CUDA")
ax.set_xticks(x); ax.set_xticklabels(DATASETS); ax.legend()
save(fig, "fig5_encoding_rate.png")

# ── Figure 6: GC content variance ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - w/2, merged["gc_var_seq"].astype(float),  w, label="Sequential", color=COLORS["seq"])
ax.bar(x + w/2, merged["gc_var_cuda"].astype(float), w, label="CUDA",       color=COLORS["cuda"])
ax.set_xlabel("Dataset"); ax.set_ylabel("GC Variance")
ax.set_title("GC Content Variance per Dataset (lower = more stable)")
ax.set_xticks(x); ax.set_xticklabels(DATASETS); ax.legend()
save(fig, "fig6_gc_variance.png")

# ── Print summary table ───────────────────────────────────────────────────────
print("\n" + "="*70)
print("PROFILING SUMMARY")
print("="*70)
for _, row in merged.iterrows():
    print(f"\n[{row['dataset']}]  {row['file_name_seq']}  ({row['file_size_bytes_seq']/1024:.1f} KB)")
    print(f"  Sequential total : {row['total_ms_seq']:>10,.2f} ms")
    print(f"  CUDA total       : {row['total_ms_cuda']:>10,.2f} ms")
    print(f"  Speedup          : {row['speedup_total']:>10.2f}x")
    print(f"  LSBM speedup     : {row['speedup_lsbm']:>10.2f}x")
    print(f"  Enc rate (seq)   : {row['encoding_rate_bits_per_nt_seq']:.4f} bits/nt")
    print(f"  Enc rate (cuda)  : {row['encoding_rate_bits_per_nt_cuda']:.4f} bits/nt")
    print(f"  Gross rate (seq) : {row['gross_rate_bits_per_nt_seq']:.4f} bits/nt")
    print(f"  Gross rate (cuda): {row['gross_rate_bits_per_nt_cuda']:.4f} bits/nt")
    print(f"  Hash match       : {bool(row['payload_hash_match'])}")
    print(f"  GC mean (seq)    : {row['gc_mean_seq']:.4f}")
    print(f"  GC var  (seq)    : {float(row['gc_var_seq']):.6f}")
    print(f"  RLL max (seq)    : {row['rll_max_seq']}")
    print(f"  End viol (seq)   : {row['end_violations_seq']}")
print("="*70)
print(f"\nAverage total speedup: {merged['speedup_total'].mean():.2f}x")
print(f"Average LSBM speedup : {merged['speedup_lsbm'].mean():.2f}x")
print(f"Average compute speedup (fair stage-only): {merged['speedup_compute'].mean():.2f}x")
print(f"Payload hash matches : {merged['payload_hash_match'].sum()}/{len(merged)}")
