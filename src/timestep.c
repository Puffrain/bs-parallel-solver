/******************************************************************
 * time_stepper.c — 显式 / 隐式 / Crank-Nicolson 并行时间推进
 ******************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include "black_scholes.h"

/*==============================================================*/
/*   串行 Thomas（三对角）                                      */
/*==============================================================*/
static void thomas_serial(int N,double *a,double *b,double *c,
                          double *d,double *x)
{
    for(int i=1;i<N;++i){
        double m = a[i]/b[i-1];
        b[i]    -= m*c[i-1];
        d[i]    -= m*d[i-1];
    }
    x[N-1] = d[N-1]/b[N-1];
    for(int i=N-2;i>=0;--i)
        x[i] = (d[i]-c[i]*x[i+1])/b[i];
}

/*==============================================================*/
/*   简易并行三对角：Gather → Solve → Scatter                   */
/*==============================================================*/
static void solve_tridiag_MPI(const BSGrid *g,double *rhs,
                              double *a,double *b,double *c)
{
    int rank,size; MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    int n = g->nloc;
    int *cnt=NULL,*dis=NULL;
    if(rank==0){ cnt=malloc(size*sizeof(int)); dis=malloc(size*sizeof(int)); }
    MPI_Gather(&n,1,MPI_INT,cnt,1,MPI_INT,0,MPI_COMM_WORLD);

    int Ntot=0;
    if(rank==0){
        dis[0]=0;
        for(int i=0;i<size;++i){
            Ntot+=cnt[i];
            if(i) dis[i]=dis[i-1]+cnt[i-1];
        }
    }
    /* gather */
    double *aA=NULL,*bA=NULL,*cA=NULL,*dA=NULL,*xA=NULL;
    if(rank==0){
        aA=malloc(Ntot*sizeof(double));
        bA=malloc(Ntot*sizeof(double));
        cA=malloc(Ntot*sizeof(double));
        dA=malloc(Ntot*sizeof(double));
        xA=malloc(Ntot*sizeof(double));
    }
    MPI_Gatherv(a+1,n,MPI_DOUBLE,aA,cnt,dis,MPI_DOUBLE,0,MPI_COMM_WORLD);
    MPI_Gatherv(b+1,n,MPI_DOUBLE,bA,cnt,dis,MPI_DOUBLE,0,MPI_COMM_WORLD);
    MPI_Gatherv(c+1,n,MPI_DOUBLE,cA,cnt,dis,MPI_DOUBLE,0,MPI_COMM_WORLD);
    MPI_Gatherv(rhs+1,n,MPI_DOUBLE,dA,cnt,dis,MPI_DOUBLE,0,MPI_COMM_WORLD);

    if(rank==0) thomas_serial(Ntot,aA,bA,cA,dA,xA);

    MPI_Scatterv(xA,cnt,dis,MPI_DOUBLE,rhs+1,n,MPI_DOUBLE,0,MPI_COMM_WORLD);

    if(rank==0){ free(aA);free(bA);free(cA);free(dA);free(xA);free(cnt);free(dis); }
}

/*==============================================================*/
/*       欧式看涨 Payoff                                         */
/*==============================================================*/
static inline double payoff(double S,double K){ return (S>K)?S-K:0.0; }

/*==============================================================*/
/*          时间推进主函数                                       */
/*==============================================================*/
void time_stepper(const BSParams *p,const BSGrid *g,BSTime *t,
                  const BSHDF5 *h5,int write_every,int start_step,
                  int scheme)   /* 0=ex 1=im 2=cn */
{
    /* 进程信息 */
    int rank,size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    int n = g->nloc;
    int N = n+2;               /* +2 ghost */

    /* 分配解向量 */
    t->V_old = calloc(N,sizeof(double));
    if(!t->V_old){ perror("malloc"); MPI_Abort(MPI_COMM_WORLD,1); }

    if(start_step==0){
        for(int i=0;i<n;++i)
            t->V_old[i+1] = payoff(g->S[i+1],p->K);
    }

    /* 隐式欧拉系数（已由 assemble_matrix 填好到全局 a_loc/b_loc/c_loc） */
    double *a_imp=a_loc,*b_imp=b_loc,*c_imp=c_loc;

    /* 如 CN 需额外系数 */
    double *a_cn=NULL,*b_cn=NULL,*c_cn=NULL,*rhs_coef=NULL;
    if(scheme==2)
        assemble_cn(p,g,&a_cn,&b_cn,&c_cn,&rhs_coef);

    /* CN 使用单独 RHS 缓冲区 */
    double *rhs = (scheme==2) ? malloc(N*sizeof(double)) : t->V_old;

    double dt = p->T / p->N_t;

    /* 主循环 */
    for(int step=start_step+1; step<=p->N_t; ++step)
    {
        double t_remain = p->T - step*dt;

        /* 边界条件 */
        if(rank==0) t->V_old[1] = 0.0;
        if(rank==size-1){
            double Smax=g->S[n];
            t->V_old[n] = Smax - p->K*exp(-p->r*t_remain);
        }

        /* ---------- 构造 RHS ---------- */
        if(scheme==2){
            /* Crank–Nicolson: rhs = V^n + 0.5*dt*B*V^n （简化示例） */
            rhs[0]=rhs[n+1]=0.0;
            for(int i=1;i<=n;++i){
                double Vm=t->V_old[i-1], V=t->V_old[i], Vp=t->V_old[i+1];
                rhs[i] = V
                       + 0.5*dt*( a_cn[i]*Vm + b_cn[i]*V + c_cn[i]*Vp );
            }
        }

        /* ---------- 线性求解 / 显式更新 ---------- */
        if(scheme==0){                      /* 显式欧拉 */
            for(int i=1;i<=n;++i) t->V_old[i] = rhs[i];
        } else {                            /* IM 或 CN 解线性方程 */
            double *aa = (scheme==2) ? a_cn : a_imp;
            double *bb = (scheme==2) ? b_cn : b_imp;
            double *cc = (scheme==2) ? c_cn : c_imp;
            solve_tridiag_MPI(g,rhs,aa,bb,cc);
        }

        /* ---------- 写 HDF5 ---------- */
        if((step%write_every)==0)
            hdf5_io_write(h5,g,rhs,step);

        /* ---------- 准备下一步 ---------- */
        if(scheme!=0){             /* IM/CN: 把结果拷回 V_old */
            for(int i=1;i<=n;++i) t->V_old[i] = rhs[i];
        }

        if(rank==0 && (step%100==0 || step==p->N_t))
            printf("隐式步 %d / %d 完成\n",step,p->N_t);
    }

    t->V_new = &t->V_old[1];

    if(rank==0){
        const char *tag = scheme==0 ? "显式欧拉"
                          : scheme==1? "隐式欧拉"
                                     : "Crank-Nicolson";
        printf("%s 并行求解结束 ✅\n",tag);
    }

    if(scheme==2){ free(a_cn);free(b_cn);free(c_cn);free(rhs_coef);free(rhs);}
}

