/******************************************************************
 * main.c  ——  Black-Scholes 并行求解器入口
 *            显式 / 隐式 / Crank-Nicolson + HDF5 重启
 ******************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include "cjson/cJSON.h"
#include "black_scholes.h"

/* ---------- 把文件读成整块字符串 ---------- */
static char *read_file(const char *fname)
{
    FILE *fp = fopen(fname, "rb");
    if (!fp) { perror("打开文件失败"); return NULL; }
    fseek(fp, 0, SEEK_END);
    long n = ftell(fp);  rewind(fp);

    char *buf = (char*)malloc(n + 1);
    if (!buf) { perror("malloc"); fclose(fp); return NULL; }

    fread(buf, 1, n, fp);
    buf[n] = '\0';
    fclose(fp);
    return buf;
}

/* ---------- 解析 JSON 参数 ---------- */
static int parse_params(const char *fname, BSParams *p)
{
    char *txt = read_file(fname);
    if (!txt) return -1;

    cJSON *root = cJSON_Parse(txt);
    free(txt);
    if (!root) {
        fprintf(stderr, "JSON 解析失败: %s\n", cJSON_GetErrorPtr());
        return -1;
    }

    p->S_min = cJSON_GetObjectItem(root, "S_min")->valuedouble;
    p->S_max = cJSON_GetObjectItem(root, "S_max")->valuedouble;
    p->N_S   = cJSON_GetObjectItem(root, "N_S"  )->valueint;
    p->T     = cJSON_GetObjectItem(root, "T"    )->valuedouble;
    p->N_t   = cJSON_GetObjectItem(root, "N_t"  )->valueint;
    p->sigma = cJSON_GetObjectItem(root, "sigma")->valuedouble;
    p->r     = cJSON_GetObjectItem(root, "r"    )->valuedouble;
    p->K     = cJSON_GetObjectItem(root, "K"    )->valuedouble;

    cJSON_Delete(root);
    return 0;
}

/* ============================================================= */
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* -------- CLI 解析 -------- */
    const char *json_file = NULL;
    const char *rst_file  = NULL;
    int scheme = 1;          /* 0=ex, 1=im (default), 2=cn */

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--restart") && i + 1 < argc) {
            rst_file = argv[++i];
        }
        else if (!strcmp(argv[i], "--scheme") && i + 1 < argc) {
            const char *s = argv[++i];
            if      (!strcmp(s, "ex")) scheme = 0;
            else if (!strcmp(s, "im")) scheme = 1;
            else if (!strcmp(s, "cn")) scheme = 2;
            else {
                if (rank == 0) fprintf(stderr, "未知 scheme: %s\n", s);
                MPI_Finalize();  return EXIT_FAILURE;
            }
        }
        else {
            json_file = argv[i];
        }
    }

    if (!json_file) {
        if (rank == 0)
            fprintf(stderr,
                    "用法: %s params.json [--scheme ex|im|cn] "
                    "[--restart file.h5]\n", argv[0]);
        MPI_Finalize();  return EXIT_FAILURE;
    }

    /* -------- 读取模型参数 -------- */
    BSParams prm;
    if (parse_params(json_file, &prm) != 0) {
        if (rank == 0) fprintf(stderr, "参数解析失败\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (rank == 0)
        printf("参数已读入，N_S=%d, N_t=%d, scheme=%s\n",
               prm.N_S, prm.N_t,
               scheme==0?"ex":scheme==1?"im":"cn");

    /* -------- 创建网格 + 装配隐式矩阵（显式也可无碍） -------- */
    BSGrid grid = {0};
    create_grid(&prm, &grid);
    assemble_matrix(&prm, &grid);   /* CN 将在 time_stepper 内部重装 */

    /* -------- HDF5 初始化 / 重启 -------- */
    int global_N;
    MPI_Allreduce(&grid.nloc, &global_N, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    BSHDF5 h5;
    int last_step = 0;
    hdf5_io_init(&prm, &grid, global_N,
                 rst_file ? rst_file : "bs_restart.h5",
                 rst_file != NULL,
                 &h5, &last_step);

    if (rank == 0 && rst_file)
        printf("重启模式: 将从 step=%d 继续\n", last_step);

    /* -------- 时间推进 -------- */
    BSTime ts = {0};
    time_stepper(&prm, &grid, &ts,
                 &h5,
                 10,                /* 每 10 步写一帧 */
                 last_step,
                 scheme);           /* 0 ex, 1 im, 2 cn */

    /* -------- 结束清理 -------- */
    hdf5_io_close(&h5);
    if (ts.V_old) free(ts.V_old);
    if (grid.S)   free(grid.S);

    MPI_Finalize();
    return 0;
}

