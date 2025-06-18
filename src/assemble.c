#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "black_scholes.h"

double *a_loc = NULL, *b_loc = NULL, *c_loc = NULL;
int      n_loc = 0;

void assemble_matrix(const BSParams *p, const BSGrid *g)
{
    int rank; MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    n_loc = g->nloc;

    a_loc = malloc((n_loc+2)*sizeof(double));
    b_loc = malloc((n_loc+2)*sizeof(double));
    c_loc = malloc((n_loc+2)*sizeof(double));

    double dt = p->T / p->N_t, dS = g->dS;
    double sig2 = 0.5 * p->sigma * p->sigma, r = p->r;

    for (int i = 0; i < n_loc; ++i) {
        double S = g->S[i+1];
        double A = sig2*S*S/(dS*dS);
        double B = r*S/(2.0*dS);

        a_loc[i+1] = -dt*(A-B);
        b_loc[i+1] =  1.0 + dt*(2*A + r);
        c_loc[i+1] = -dt*(A+B);
    }

    int size; MPI_Comm_size(MPI_COMM_WORLD,&size);
    if (rank==0)        a_loc[1]     = 0.0;
    if (rank==size-1)   c_loc[n_loc] = 0.0;

    if (rank==0) printf("隐式矩阵装配完毕 (本地行数 %d)\n", n_loc);
}

