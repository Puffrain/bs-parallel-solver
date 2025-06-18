#include <mpi.h>
#include "black_scholes.h"

/* ===========================================================
 * 并行 HDF5 初始化
 * -----------------------------------------------------------
 *  - 若 restart == 0  ⇒ 创建新文件并返回 last_step = 0
 *  - 若 restart == 1  ⇒ 打开文件、读取已有行数，返回最后步索引
 * =========================================================== */
void hdf5_io_init(const BSParams *p, const BSGrid *g,
                  int global_N,
                  const char *fname, int restart,
                  BSHDF5 *h5, int *last_step)
{
    /* 并行文件访问属性 */
    hid_t plist_fa = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plist_fa, MPI_COMM_WORLD, MPI_INFO_NULL);

    h5->file_id = restart
        ? H5Fopen(fname, H5F_ACC_RDWR, plist_fa)
        : H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, plist_fa);

    H5Pclose(plist_fa);

    /* 创建 (unlimited × global_N) dataspace */
    hsize_t dims[2]  = {0,            (hsize_t)global_N};
    hsize_t maxs[2]  = {H5S_UNLIMITED,(hsize_t)global_N};
    hid_t fspace0 = H5Screate_simple(2, dims, maxs);

    /* 设置 chunk，使每次写入一行 */
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    hsize_t chunk[2] = {1, (hsize_t)global_N};
    H5Pset_chunk(dcpl, 2, chunk);

    h5->dset_id = restart
        ? H5Dopen2(h5->file_id, "/V", H5P_DEFAULT)
        : H5Dcreate2(h5->file_id, "/V", H5T_IEEE_F64LE,
                     fspace0, H5P_DEFAULT, dcpl, H5P_DEFAULT);

    H5Pclose(dcpl);
    H5Sclose(fspace0);

    /* 本进程内存 dataspace：1 × nloc */
    hsize_t mdims[2] = {1, (hsize_t)g->nloc};
    h5->mspace = H5Screate_simple(2, mdims, NULL);

    /* 计算已有步数 → last_step */
    if (restart) {
        hsize_t cur_dims[2];
        hid_t cur_space = H5Dget_space(h5->dset_id);
        H5Sget_simple_extent_dims(cur_space, cur_dims, NULL);
        H5Sclose(cur_space);
        *last_step = (int)(cur_dims[0] - 1);      /* 0-based 索引 */
    } else {
        *last_step = 0;
    }
}

/* ===========================================================
 * 写入一步 (独立 I/O)
 * step 为 0-based 索引
 * =========================================================== */
void hdf5_io_write(const BSHDF5 *h5, const BSGrid *g,
                   const double *V_local, int step)
{
    /* 1. 扩展数据集行数到 step+1 */
    hsize_t newsize[2];
    hid_t fspace_file = H5Dget_space(h5->dset_id);
    H5Sget_simple_extent_dims(fspace_file, newsize, NULL);
    if (newsize[0] <= (hsize_t)step) {
        newsize[0] = step + 1;
        H5Dset_extent(h5->dset_id, newsize);
        H5Sclose(fspace_file);
        fspace_file = H5Dget_space(h5->dset_id);  /* 重新获取 */
    }

    /* 2. 计算本进程在列方向上的全局偏移 */
    hsize_t offset_col = 0;
    MPI_Exscan(&(g->nloc), &offset_col,
               1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    /* 3. 选定文件 hyperslab */
    hsize_t start[2] = {(hsize_t)step, offset_col};
    hsize_t count[2] = {1, (hsize_t)g->nloc};
    H5Sselect_hyperslab(fspace_file, H5S_SELECT_SET, start, NULL, count, NULL);

    /* 4. 写入 (独立模式) */
    hid_t dxpl = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(dxpl, H5FD_MPIO_INDEPENDENT);

    /* V_local+1 跳过左 ghost */
    H5Dwrite(h5->dset_id, H5T_NATIVE_DOUBLE,
             h5->mspace, fspace_file, dxpl, V_local + 1);

    H5Pclose(dxpl);
    H5Sclose(fspace_file);
}

/* =========================================================== */
void hdf5_io_close(BSHDF5 *h5)
{
    H5Sclose(h5->mspace);
    H5Dclose(h5->dset_id);
    H5Fclose(h5->file_id);
}

