#include <stdio.h>
#include <stdlib.h>
#include <math.h>          /* 为 exp() 提供声明 */
#include <mpi.h>
#include "black_scholes.h"

/*---------------------------------------------*/
/*  串行三对角 Thomas 求解器                    */
/*---------------------------------------------*/
static void thomas_serial(int N, double *a, double *b,
                          double *c, double *d, double *x)
{
    /* 前向消元 */
    for (int i = 1; i < N; ++i) {
        double m = a[i] / b[i - 1];
        b[i]    -= m * c[i - 1];
        d[i]    -= m * d[i - 1];
    }
    /* 回代 */
    x[N - 1] = d[N - 1] / b[N - 1];
    for (int i = N - 2; i >= 0; --i)
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i];
}

/*---------------------------------------------*/
/*  “简单” 并行求解：Gather → Thomas → Scatter  */
/*---------------------------------------------*/
static void solve_tridiag_MPI(const BSGrid *g, double *rhs)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* 收集每个进程的行数 */
    int n    = g->nloc;
    int *cnt = NULL, *dis = NULL;
    if (rank == 0) {
        cnt = malloc(size * sizeof(int));
        dis = malloc(size * sizeof(int));
    }
    MPI_Gather(&n, 1, MPI_INT, cnt, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int Ntot = 0;
    if (rank == 0) {
        dis[0] = 0;
        for (int i = 0; i < size; ++i) {
            Ntot += cnt[i];
            if (i > 0) dis[i] = dis[i - 1] + cnt[i - 1];
        }
    }

    /* Gather 本地 a,b,c,rhs 到 rank 0 */
    double *aAll = NULL, *bAll = NULL, *cAll = NULL;
    double *dAll = NULL, *xAll = NULL;

    if (rank == 0) {
        aAll = malloc(Ntot * sizeof(double));
        bAll = malloc(Ntot * sizeof(double));
        cAll = malloc(Ntot * sizeof(double));
        dAll = malloc(Ntot * sizeof(double));
        xAll = malloc(Ntot * sizeof(double));
    }

    MPI_Gatherv(&a_loc[1], n, MPI_DOUBLE, aAll, cnt, dis,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(&b_loc[1], n, MPI_DOUBLE, bAll, cnt, dis,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(&c_loc[1], n, MPI_DOUBLE, cAll, cnt, dis,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(&rhs[1],   n, MPI_DOUBLE, dAll, cnt, dis,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* rank 0 串行求解 */
    if (rank == 0)
        thomas_serial(Ntot, aAll, bAll, cAll, dAll, xAll);

    /* Scatter 解回各进程 */
    MPI_Scatterv(xAll, cnt, dis, MPI_DOUBLE,
                 &rhs[1], n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* 释放临时缓冲 */
    if (rank == 0) {
        free(aAll); free(bAll); free(cAll);
        free(dAll); free(xAll);
        free(cnt);  free(dis);
    }
}

/* 欧式看涨 Payoff */
static inline double payoff(double S, double K)
{ return (S > K) ? (S - K) : 0.0; }

/*---------------------------------------------*/
/*           隐式欧拉时间推进主函数             */
/*---------------------------------------------*/
void time_stepper(const BSParams *p, const BSGrid *g, BSTime *t)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = g->nloc;          /* 本进程实节点数 */
    int N = n + 2;            /* 含 2 个 ghost */

    t->V_old = calloc(N, sizeof(double));
    if (!t->V_old) { perror("malloc"); MPI_Abort(MPI_COMM_WORLD, 1); }

    /* t=T Payoff */
    for (int i = 0; i < n; ++i)
        t->V_old[i + 1] = payoff(g->S[i + 1], p->K);

    double dt = p->T / p->N_t;

    for (int step = 1; step <= p->N_t; ++step)
    {
        double t_remain = p->T - step * dt;

        /* 左边界 */
        if (rank == 0) t->V_old[1] = 0.0;

        /* 右边界 */
        if (rank == size - 1) {
            double Smax = g->S[n];
            t->V_old[n] = Smax - p->K * exp(-p->r * t_remain);
        }

        /* 求解 A·V_new = V_old (结果覆盖到 V_old[1..n]) */
        solve_tridiag_MPI(g, t->V_old);

        if (rank == 0 && (step % 100 == 0 || step == p->N_t))
            printf("隐式步 %d / %d 完成\n", step, p->N_t);
    }

    /* 使 V_new 指向解向量首元素，便于主程序 free */
    t->V_new = &t->V_old[1];
    if (rank == 0) puts("隐式欧拉并行求解结束 ✅");
}

