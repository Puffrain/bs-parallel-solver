#!/usr/bin/env python3
"""
plot_error.py — 绘制 L2 误差随网格加密的收敛曲线
用法:
    python3 tools/plot_error.py tools/error_table.csv
输出:
    results/error_convergence.png
    results/error_convergence.pdf
"""
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# --- 输入检查 ---
if len(sys.argv) != 2:
    print("Usage: plot_error.py  error_table.csv")
    sys.exit(1)

csv_in = Path(sys.argv[1])
if not csv_in.is_file():
    print(f"❌ 找不到文件: {csv_in}")
    sys.exit(1)

# --- 读取数据 ---
df = pd.read_csv(csv_in)
# 要求包含列: N_S, N_t, L2
if not {"N_S","L2"}.issubset(df.columns):
    print("❌ CSV 中缺少 N_S 或 L2 列")
    sys.exit(1)

# --- 准备输出目录 ---
proj_root = csv_in.parent.parent
out_dir   = proj_root / "results"
out_dir.mkdir(exist_ok=True)

# --- 绘制收敛图 ---
plt.figure(figsize=(6,4))
plt.loglog(df["N_S"], df["L2"], 'o-')
plt.xlabel("N_S (space points)")
plt.ylabel("L2 error")
plt.title("Error convergence (implicit Euler)")
plt.grid(True, which="both", ls="--")

png = out_dir / "error_convergence.png"
pdf = out_dir / "error_convergence.pdf"
plt.savefig(png, dpi=300, bbox_inches="tight")
plt.savefig(pdf,             bbox_inches="tight")
print(f"✅ 已保存收敛图到: {out_dir}")

plt.close()

