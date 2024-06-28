cmake_minimum_required(VERSION 2.8)
project(gemm)
include_directories(${PROJECT_SOURCE_DIR}/include)
add_compile_options(-Wall -std=c++11 -O2)
# test initialize value
set(nRepeats 10) # "repeat calculate,find the best time"
set(LDA -1) # "set -1,used the matrix leading dimension"
set(LDB -1) # "set -1,used the matrix leading dimension"
set(LDC -1) # "set -1,used the matrix leading dimension"
set(M -1)   # "set M dimension for -1,enjoy the loop"
set(N -1)   # "set N dimension for -1,enjoy the loop"
set(K -1)   # "set K dimension for -1,enjoy the loop"
set(PSTART 40) # "loop start"
set(PEND 1600)  # "loop end"
set(PINC 40)   # "loop increment"
option(USE_NEON "use neon or not" ON)
if(USE_NEON)
    message(">> USE NEON")
endif()
configure_file(${PROJECT_SOURCE_DIR}/config.h.in ${PROJECT_SOURCE_DIR}/include/config.h @ONLY)

aux_source_directory(${PROJECT_SOURCE_DIR}/utils UTILS_CPP)
set(SRC_CPP "./src/gemm_8x8_like_blas_neon.cpp")
set(SRC_CPP ${SRC_CPP} "./src/gemm_origin.cpp")
# aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC_CPP)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

# message(">> ${UTILS_CPP}")
# message(">> ${SRC_CPP}")
add_executable(test ${UTILS_CPP} ${SRC_CPP})

