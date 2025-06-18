# 高性能计算期末项目 – Black–Scholes 方程 MPI 并行求解器

## 目录结构

```
final_project/
├── build/              # ⬅ 自动生成的编译目录
├── examples/           # 输入参数示例
│   └── params.json
├── include/
│   └── black_scholes.h # 全局数据结构与接口声明
├── src/                # C 源码
│   ├── main.c
│   ├── grid.c
│   ├── assemble.c
│   ├── timestep.c
│   └── io.c
├── tools/              # 误差 / 性能评测脚本
│   ├── error_check.py
│   ├── refine_driver.py
│   ├── plot_error.py
│   └── bench.sh
└── README.md           # 项目说明
```

## 快速开始

```bash
# 克隆并进入
git clone <repo-url> && cd final_project

# 生成构建目录
mkdir -p build && cd build
cmake ..
make -j               # 生成 ./bs_price

# 1. 显式欧拉
mpirun -n 4 ./bs_price ../examples/params.json --scheme ex

# 2. 隐式欧拉（默认）
mpirun -n 4 ./bs_price ../examples/params.json --scheme im

# 3. Crank–Nicolson
mpirun -n 4 ./bs_price ../examples/params.json --scheme cn
```

## 参数文件 (`examples/params.json`)

```json
{
  "S_min" : 0.0,
  "S_max" : 200.0,
  "N_S"   : 5000,
  "T"     : 1.0,
  "N_t"   : 1000,
  "sigma" : 0.2,
  "r"     : 0.05,
  "K"     : 100.0
}
```

## 重启功能

```bash
# 第一次：常规计算，每 10 步写入 bs_restart.h5
mpirun -n 4 ./bs_price ../examples/params.json --scheme im --write_every 10

# 第二次：从 bs_restart.h5 继续
mpirun -n 4 ./bs_price ../examples/params.json --restart bs_restart.h5
```

## 脚本说明

| 文件                     | 作用                       |
| ------------------------ | -------------------------- |
| `tools/error_check.py`   | 计算数值解与解析解误差     |
| `tools/refine_driver.py` | 网格加密实验自动执行       |
| `tools/plot_error.py`    | 绘制误差收敛曲线           |
| `tools/bench.sh`         | 不同核数性能测试并生成日志 |

## 依赖

- **MPI**: Open‑MPI ≥ 4.0  
- **HDF5**: `brew install hdf5`  
- **cJSON**: `brew install cjson`  
- **Python 3** + `numpy matplotlib h5py`（仅脚本需要）

```bash
# Homebrew 安装示例
brew install open-mpi hdf5 cjson
# 如提示找不到 cjson.pc，添加：
export PKG_CONFIG_PATH=/opt/homebrew/lib/pkgconfig:$PKG_CONFIG_PATH
```

## 代码模块简介

| 文件         | 功能                                                         |
| ------------ | ------------------------------------------------------------ |
| `grid.c`     | 生成全局/局部资产价格网格，处理 ghost cells                  |
| `assemble.c` | 组装离散方程矩阵（显式/隐式）                                |
| `timestep.c` | 显式欧拉、隐式欧拉、Crank–Nicolson 时间推进；包含 MPI halo 交换 |
| `io.c`       | 使用 HDF5 并行 I/O 输出/重启                                 |
| `main.c`     | 解析 JSON 参数、MPI 初始化、驱动整体流程                     |

## 提交内容

1. 全部源代码 (`src/`, `include/`)
2. `examples/params.json`
3. `tools/` 目录脚本
4. `README.md`（本文件）

> **注：** `build/` 目录为临时产物，提交前请先 `rm -rf build/`.

---

© 2025 HPC Course  
