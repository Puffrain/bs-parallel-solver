#!/usr/bin/env python3
"""
refine_driver.py — 在不同网格/时间步下运行求解器并收集 L2 误差
用法：在项目根的 build/ 目录下执行
    python3 ../tools/refine_driver.py
结果：tools/error_table.csv
"""
import json, subprocess, os
from pathlib import Path

# 项目目录
proj_root = Path(__file__).parent.parent

# 原始参数文件
params_src = proj_root / "examples" / "params.json"
# 输出 CSV
csv_out    = proj_root / "tools" / "error_table.csv"

# 网格与时间步列表
Ns_list = [100, 200, 400, 800]
Nt_list = [250, 500, 1000, 2000]

# 读取基础参数
base_prm = json.load(open(params_src, 'r'))

with open(csv_out, 'w') as fout:
    fout.write("N_S,N_t,L2\n")
    for Ns, Nt in zip(Ns_list, Nt_list):
        # 写临时参数文件
        prm = base_prm.copy()
        prm["N_S"] = Ns
        prm["N_t"] = Nt
        tmp_json = proj_root / "tools" / "tmp_params.json"
        with open(tmp_json, 'w') as f:
            json.dump(prm, f, indent=2)

        # 调用并行求解器（隐式欧拉）
        subprocess.run([
            "mpirun", "-n", "4",
            str(proj_root / "build" / "bs_price"),
            str(tmp_json)
        ], check=True)

        # 调用误差脚本，计算 L2
        out = subprocess.check_output([
            "python3",
            str(proj_root / "tools" / "error_check.py"),
            str(proj_root / "build" / "bs_restart.h5"),
            str(tmp_json)
        ]).decode().strip()
        L2 = float(out)

        fout.write(f"{Ns},{Nt},{L2:.6e}\n")
        print(f"N_S={Ns}, N_t={Nt}, L2={L2:.3e}")

        # 删除临时文件
        os.remove(tmp_json)

