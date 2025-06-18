#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "black_scholes.h"

void create_grid(const BSParams *p, BSGrid *g)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    int base = (p->N_S + 1) / size;       /* 基本分段 */
    int rem  = (p->N_S + 1) % size;       /* 余数补前 rem 个 */
    g->nloc  = base + (rank < rem);

    g->S  = malloc((g->nloc + 2) * sizeof(double));  /* +2 ghost */
    g->dS = (p->S_max - p->S_min) / p->N_S;

    long globalStart = rank*base + (rank < rem ? rank : rem);
    for (int i = 0; i < g->nloc; ++i)
        g->S[i + 1] = p->S_min + (globalStart + i) * g->dS;
}

