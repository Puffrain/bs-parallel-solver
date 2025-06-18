#ifndef BLACK_SCHOLES_H
#define BLACK_SCHOLES_H

#include <hdf5.h>

/* ============ 参数结构 ============ */
typedef struct {
    double S_min, S_max;
    int    N_S;
    double T;
    int    N_t;
    double sigma, r, K;
} BSParams;

/* ============ 空间网格 (含左右 ghost) ============ */
typedef struct {
    double *S;          /* 长度 nloc+2: [ghostL | real(0..nloc-1) | ghostR] */
    int     nloc;       /* 本进程真实节点数 */
    double  dS;         /* 网格间距 */
} BSGrid;

/* ============ 时间推进向量 ============ */
typedef struct {
    double *V_old;      /* 上一步 (或当前) 解；长度 nloc+2 */
    double *V_new;      /* 指向新解（通常指向 V_old+1） */
} BSTime;

/* ============ HDF5 句柄 ============ */
typedef struct {
    hid_t file_id;
    hid_t dset_id;
    hid_t mspace;       /* 1×nloc 内存空间 */
} BSHDF5;

/* ============ 外部三对角系数 (assemble.c 定义) ============ */
extern double *a_loc, *b_loc, *c_loc;   /* 长度 nloc+2，与 ghost 对齐 */
extern int      n_loc;

/* ============ 接口函数原型 ============ */
void create_grid      (const BSParams *p, BSGrid *g);

void assemble_matrix  (const BSParams *p, const BSGrid *g);

/* 隐式欧拉时间推进 + HDF5 周期写
 * p, g      : 参数与网格
 * t         : 时间推进向量
 * h5        : 已初始化的 HDF5 句柄
 * write_every : 每隔多少步写一次
 * start_step  : 若重启则设为已完成的最后一步；新计算为 0
 */
void time_stepper(const BSParams *p, const BSGrid *g, BSTime *t,
                  const BSHDF5 *h5, int write_every, int start_step,
                  int scheme);          /* 0=ex,1=im,2=cn */

/* 并行 HDF5 初始化
 * global_N   : 全局节点总数
 * fname      : 文件名
 * restart    : 0=新建, 1=重启打开
 * h5         : 输出句柄
 * last_step  : 输出已有的最后一步索引 (仅重启时有效)
 */
void hdf5_io_init (const BSParams *p, const BSGrid *g,
                   int global_N, const char *fname, int restart,
                   BSHDF5 *h5, int *last_step);

void hdf5_io_write(const BSHDF5 *h5, const BSGrid *g,
                   const double *V_local, int step);

void hdf5_io_close(BSHDF5 *h5);

#endif /* BLACK_SCHOLES_H */

/* 旧内容保留，追加一行： */
void assemble_cn(const BSParams *p, const BSGrid *g,
                 double **a_cn, double **b_cn, double **c_cn,
                 double **rhs_coef);   /* CN 专用 */

