cmake_minimum_required(VERSION 3.18)
project(bs_price LANGUAGES C)

# --------------------------------------------------
# 1. 查找依赖
# --------------------------------------------------
find_package(MPI   REQUIRED)                 # MPI::MPI_C
find_package(HDF5  REQUIRED COMPONENTS C)    # ${HDF5_LIBRARIES}

# ----- 手动指定 cJSON (如果 pkg-config 可用，也可以用 pkg_check_modules) -----
set(CJSON_INCLUDE_DIR /opt/homebrew/include/cjson)
set(CJSON_LIBRARY     /opt/homebrew/lib/libcjson.dylib)
if (NOT EXISTS ${CJSON_LIBRARY})
    message(FATAL_ERROR "未找到 libcjson.dylib，请先 `brew install cjson`")
endif()

# --------------------------------------------------
# 2. 源文件
# --------------------------------------------------
add_executable(bs_price
    src/main.c
    src/grid.c
    src/assemble.c
    src/assemble_cn.c        # <<< 新增 Crank–Nicolson 装配
    src/timestep.c
    src/io.c
)

# --------------------------------------------------
# 3. 头文件搜索路径
# --------------------------------------------------
target_include_directories(bs_price PUBLIC
    ${MPI_C_INCLUDE_DIRS}
    ${HDF5_INCLUDE_DIRS}
    ${CJSON_INCLUDE_DIR}
    include                  # 项目本地头文件
)

# --------------------------------------------------
# 4. 链接库
# --------------------------------------------------
target_link_libraries(bs_price PUBLIC
    MPI::MPI_C
    ${HDF5_LIBRARIES}
    ${CJSON_LIBRARY}
)

# --------------------------------------------------
# 5. 编译选项（可按需调整）
# --------------------------------------------------
target_compile_features(bs_price PRIVATE c_std_99)
target_compile_options(bs_price PRIVATE
    -Wall -Wextra -Wno-unused-parameter
)

