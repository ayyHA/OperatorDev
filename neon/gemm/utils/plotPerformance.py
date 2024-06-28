import matplotlib.pyplot as plt
import numpy as np
import os

def solve(filename):
    with open(filename,'r') as f:
        sizes = []
        flops = []
        title = os.path.basename(filename)[7:]
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        for line in lines:
            slices = line.split(" ")
            size = int(slices[0])
            flop = float(slices[1])
            sizes.append(size)
            flops.append(flop)
    return title, sizes, flops

if __name__ == '__main__':
    # 这是一大堆的
    # file_list = ['../record_gemm_origin.txt','../record_gemm_4x1_origin.txt','../record_gemm_4x1_reg.txt',
    # '../record_gemm_4x1_reg_unroll.txt','../record_gemm_4x4_reg.txt','../record_gemm_4x4_neon.txt',
    # '../record_gemm_4x4_neon_block.txt',#'../record_gemm_4x4_neon_unroll4.txt','../record_gemm_4x4_neon_block_unroll4.txt',
    # '../record_gemm_4x4_neon_packed.txt',#'../record_gemm_4x4_neon_packed_mc128.txt']
    # '../record_gemm_4x4_like_blas_neon.txt']
    # 两两比较的
    file_list = ['../record_gemm_4x4_like_blas_asmV1.txt','../record_gemm_4x4_like_blas_asmV2.txt',
                '../record_gemm_4x4_like_blas_asmV3.txt','../record_gemm_4x4_like_blas_neonV2.txt']
    # 单独的
    # file_list = ['../record_gemm_4x4_like_blas_neon.txt']
    plt.xlabel('size')
    plt.ylabel('gflops')
    for file_name in file_list:
        t, x, y = solve(file_name)
        print(np.mean(y));
        plt.plot(x, y, label=t)
        plt.legend(bbox_to_anchor=(0.62, 0.65))
    # plt.savefig('../pics/gemm_to_like_blas_neon.png')
    
    plt.savefig('../pics/compare_gemm_like_blas_neon_arm')
    
    # plt.savefig('../pics/gemm_4x4_like_blas_neon.png')
    plt.show()