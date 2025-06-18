#include <stdlib.h>
#include "black_scholes.h"

/* a_cn, b_cn, c_cn: CN 左矩阵 A
 * rhs_coef:         存 0.5*dt*(A±B)，供 RHS 计算 */
void assemble_cn(const BSParams *p, const BSGrid *g,
                 double **a_cn, double **b_cn, double **c_cn,
                 double **rhs_coef)
{
    int n = g->nloc;
    double dt = p->T / p->N_t, dS = g->dS;
    double sig2 = 0.5 * p->sigma * p->sigma, r = p->r;

    *a_cn     = malloc((n + 2) * sizeof(double));
    *b_cn     = malloc((n + 2) * sizeof(double));
    *c_cn     = malloc((n + 2) * sizeof(double));
    *rhs_coef = malloc((n + 2) * sizeof(double));      /* 存 (A±B) */

    for (int i = 0; i < n; ++i) {
        double S = g->S[i + 1];
        double A = sig2 * S * S / (dS * dS);
        double B =  r   * S / (2.0 * dS);

        (*a_cn)[i+1]      = -0.5 * dt * (A - B);
        (*b_cn)[i+1]      =  1.0 + dt * (A + r) ;      /* θ=0.5 */
        (*c_cn)[i+1]      = -0.5 * dt * (A + B);

        (*rhs_coef)[i+1]  =  0.5 * dt * (A + r);       /* 仅存对角系数方便 RHS */
    }

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    if (rank == 0)          (*a_cn)[1]      = 0.0;
    if (rank == size - 1)   (*c_cn)[n]      = 0.0;
}

