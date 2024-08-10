#include <cstdio>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cusparse.h>
#include <cuda_runtime.h>

using namespace std;

#define checkCudaError(func) {                                                      \
    cudaError_t e = (func);                                                         \
    if(e != cudaSuccess){                                                           \
        printf("CUDA ERROR: %s %d : %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
    }                                                                               \
}

#define checkCUSPARSEError(func){                                                                       \
    cusparseStatus_t status = (func);                                                                   \
    if(status != CUSPARSE_STATUS_SUCCESS){                                                              \
        printf("CUSPARSE ERROR: %s %d : %s\n",__FILE__,__LINE__,cusparseGetErrorString(status));        \
    }                                                                                                   \
}

#define DIV_UP(x,y) (((x)+(y-1))/(y))
#define NUM_THREADS_PER_BLOCK 256
#define MASK 0xffffffff


// 注意:这个构建出来的不是最终的CSR格式的阵,还需要进行调整
void buildBasicCSR(int i,int j,float v,int* rowStartNode,int* rowNode,int* colIdx,float* val,int& idx){
    colIdx[idx] = j;
    val[idx] = v;
    rowNode[idx] = rowStartNode[i];
    rowStartNode[i] = idx;
    idx++;
}

void readInitParams(int& row,int& col,int& nonzero_num,string& filename){
    ifstream f;
    f.open("matrix/" + filename + ".mtx");
    while(f.peek() == '%'){
        f.ignore(1024,'\n');
    }
    f >> row >> col >> nonzero_num;
    f.close();
}

void readMatrixParams(int row,int col,int nz,int* row_offsets,int* col_index,float* value,string& filename){
    ifstream f;
    f.open("matrix/" + filename + ".mtx");
    while(f.peek() == '%'){
        f.ignore(1024,'\n');
    }
    f >> row >> col >> nz;
    
    size_t roSize = sizeof(int)*(row);
    size_t nzSize_i = sizeof(int)*(nz);
    size_t nzSize_f = sizeof(float)*(nz);

    // 以下四个指针用来读取mtx文件中的数据
    // 和col_index作用类似,记录列数
    int* colIdx = (int*)malloc(nzSize_i);
    // 和value作用类似,记录列数对应的值
    float* val = (float*)malloc(nzSize_f);
    // 用来记录每行对应的元素索引的最末尾的值,这个元素索引是在rowNode中的,相当于head
    int* rowStartNode = (int*)malloc(roSize);
    // 用来记录同一行元素的索引,相当于指针
    int* rowNode = (int*)malloc(nzSize_i);

    // rowStartNode初值设为-1,相当于链表的nullptr的意思
    memset(rowStartNode,-1,roSize);

    int i,j;
    float v;
    int idx=0;
    while(f>>i>>j>>v){
        // 因为i,j是从1,1开始计数的,我们的下标是从0开始的,所以都要自减一波
        i--;
        j--;
        buildBasicCSR(i,j,v,rowStartNode,rowNode,colIdx,val,idx);
    }

    row_offsets[0] = 0;     // 第0行的起始非零值在value中的索引
    int cnt;
    for(i=0;i<row;i++){
        cnt = row_offsets[i];
        for(int j=rowStartNode[i];j!=-1;j=rowNode[j]){
            col_index[cnt] = colIdx[j];
            value[cnt] = val[j];
            cnt++;
        }
        // 需要注意,mtx的是列主序排序的,所以要局部reverse?不用!实际不用,我们有col,对着乘就好!而且这不是stl容器
        // reverse(col_index.begin()+row_offsets[i],col_index.begin()+cnt);
        // reverse(value.begin()+row_offsets[i],value.begin()+cnt);
        // 下一个row_offset的值
        row_offsets[i+1] = cnt;
    }
    row_offsets[row+1] =cnt;
    free(colIdx);
    free(val);
    free(rowStartNode);
    free(rowNode);
    f.close();
}

template<typename IdxType,typename ValType>
void spmv_cpu(vector<IdxType> row_offsets,vector<IdxType> col_index,vector<ValType> value,
            vector<ValType> x,vector<ValType>& y,int rowNum){
    IdxType rStart,rEnd;
    for(int i=0;i<rowNum;i++){
        rStart = row_offsets[i];
        rEnd = row_offsets[i+1];
        ValType res = 0.0f;
        for(IdxType j=rStart;j<rEnd;j++){
            res+= value[j] * x[col_index[j]];
        }
        y[i] = res;
    }
}

template<unsigned int WarpSize>
__device__ __forceinline__ float warpSum(float sum){
    if(WarpSize>=32) sum += __shfl_down_sync(MASK,sum,16);
    if(WarpSize>=16) sum += __shfl_down_sync(MASK,sum,8);
    if(WarpSize>=8) sum += __shfl_down_sync(MASK,sum,4);
    if(WarpSize>=4) sum += __shfl_down_sync(MASK,sum,2);
    if(WarpSize>=2) sum += __shfl_down_sync(MASK,sum,1);
    return sum;
}

template<typename IdxType,typename ValType,unsigned int VECTOR_PER_BLOCK,unsigned int THREAD_PER_VECTOR>
__global__ void spmv_gpu(int row_num,IdxType* __restrict__ row_offsets,IdxType* __restrict__ col_index,ValType* __restrict__ value,
                        ValType* __restrict__ x,ValType* __restrict__ y){
    const int THREADS_PER_BLOCK = VECTOR_PER_BLOCK * THREAD_PER_VECTOR;
    const int tid = blockIdx.x*THREADS_PER_BLOCK + threadIdx.x;
    const int trow = tid / THREAD_PER_VECTOR;   // thread在哪一个向量
    const int tlane = threadIdx.x & (THREAD_PER_VECTOR-1);  // thread在该向量负责哪一个元素/哪一个lane

    if(trow < row_num){
        ValType sum = 0.0f;
        IdxType rStart = row_offsets[trow];
        IdxType rEnd = row_offsets[trow+1];
        for(IdxType i=rStart+tlane;i<rEnd;i+=THREAD_PER_VECTOR){
            sum+=value[i]*x[col_index[i]];
        }
        sum = warpSum<THREAD_PER_VECTOR>(sum);
        if(tlane == 0)
            y[trow] = sum;
    }
}



int main(int argc,char** argv){
    // 首先要处理mtx文件,这是一种稀疏矩阵的文件格式,要读入并以CSR的形式进行表达
    if(argc < 3){
        fprintf(stderr,"You need input 3 parameters,like: ./spmv -f ./mtx_path");
        exit(-1);
    }

    string filename = argv[2];
    int row_num,col_num,nonzero_num;
    // 获取矩阵的行列以及有效元素值
    readInitParams(row_num,col_num,nonzero_num,filename);

    vector<int> row_offsets(row_num+1);
    vector<int> col_index(nonzero_num);
    vector<float> value(nonzero_num);
    vector<float> x(col_num,1.0f);
    vector<float> y(row_num,0.0f);
    vector<float> y_gpu(row_num,0.0f);
    vector<float> y_cusparse(row_num,0.0f);

    // 构建CSR的三个指针
    readMatrixParams(row_num,col_num,nonzero_num,row_offsets.data(),col_index.data(),value.data(),filename);

    // 打印一下,看下对不对
    // for(int i=0;i<3;i++){
    //     int rStart = row_offsets[i];
    //     int rEnd = row_offsets[i+1];
    //     for(int j=rStart;j<rEnd;j++){
    //         printf("[%d,%d]:[%.6f]\n",i,col_index[j],value[j]);
    //     }
    // }
    // printf("nonzero num:%d\n",row_offsets[row_num+1]);

    // 用CPU算出y,后面对比=====
    spmv_cpu<int,float>(row_offsets,col_index,value,x,y,row_num);
    // =======================

    size_t roSize = sizeof(int) * (row_num +1);
    size_t coSize = sizeof(int) * (nonzero_num);
    size_t valSize = sizeof(float) * (nonzero_num);
    size_t xSize = sizeof(float) * (col_num);
    size_t ySize = sizeof(float) * (row_num);

    int* dRowOffsets,*dColIndex;
    float* dValue,*dX,*dY,*dY_CUSPARSE;
    checkCudaError(cudaMalloc(&dRowOffsets,roSize));
    checkCudaError(cudaMalloc(&dColIndex,coSize));
    checkCudaError(cudaMalloc(&dValue,valSize));
    checkCudaError(cudaMalloc(&dX,xSize));
    checkCudaError(cudaMalloc(&dY,ySize));
    checkCudaError(cudaMalloc(&dY_CUSPARSE,ySize));

    checkCudaError(cudaMemcpy(dRowOffsets,row_offsets.data(),roSize,cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(dColIndex,col_index.data(),coSize,cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(dValue,value.data(),valSize,cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(dX,x.data(),xSize,cudaMemcpyHostToDevice));

    // 自测kernel
    const int MEAN_NUM_ROW = DIV_UP(nonzero_num,row_num);
    printf("MEAN NUM ROW IS: %d\n",MEAN_NUM_ROW);
    cudaEvent_t start,stop;
    checkCudaError(cudaEventCreate(&start));
    checkCudaError(cudaEventCreate(&stop));
    int nRepeats = 2000;
    float elapsed_time;
    checkCudaError(cudaEventRecord(start));
    // 用GPU算出dY=============
    for(int i=0;i<nRepeats;i++){
        if(MEAN_NUM_ROW<=2){
            const int THREAD_PER_VECTOR = 2;
            const int VECTOR_PER_BLOCK = NUM_THREADS_PER_BLOCK / THREAD_PER_VECTOR;
            dim3 block(NUM_THREADS_PER_BLOCK);
            dim3 grid(DIV_UP(row_num,VECTOR_PER_BLOCK));
            spmv_gpu<int,float,VECTOR_PER_BLOCK,THREAD_PER_VECTOR><<<grid,block>>>(row_num,dRowOffsets,dColIndex,dValue,dX,dY);
        }else if(MEAN_NUM_ROW>2 && MEAN_NUM_ROW<=4){
            const int THREAD_PER_VECTOR = 4;
            const int VECTOR_PER_BLOCK = NUM_THREADS_PER_BLOCK / THREAD_PER_VECTOR;
            dim3 block(NUM_THREADS_PER_BLOCK);
            dim3 grid(DIV_UP(row_num,VECTOR_PER_BLOCK));
            spmv_gpu<int,float,VECTOR_PER_BLOCK,THREAD_PER_VECTOR><<<grid,block>>>(row_num,dRowOffsets,dColIndex,dValue,dX,dY);
        }else if(MEAN_NUM_ROW>4 && MEAN_NUM_ROW<=8){
            const int THREAD_PER_VECTOR = 8;
            const int VECTOR_PER_BLOCK = NUM_THREADS_PER_BLOCK / THREAD_PER_VECTOR;
            dim3 block(NUM_THREADS_PER_BLOCK);
            dim3 grid(DIV_UP(row_num,VECTOR_PER_BLOCK));
            spmv_gpu<int,float,VECTOR_PER_BLOCK,THREAD_PER_VECTOR><<<grid,block>>>(row_num,dRowOffsets,dColIndex,dValue,dX,dY);
        }else if(MEAN_NUM_ROW>8 && MEAN_NUM_ROW<=16){
            const int THREAD_PER_VECTOR = 16;
            const int VECTOR_PER_BLOCK = NUM_THREADS_PER_BLOCK / THREAD_PER_VECTOR;
            dim3 block(NUM_THREADS_PER_BLOCK);
            dim3 grid(DIV_UP(row_num,VECTOR_PER_BLOCK));
            spmv_gpu<int,float,VECTOR_PER_BLOCK,THREAD_PER_VECTOR><<<grid,block>>>(row_num,dRowOffsets,dColIndex,dValue,dX,dY);

        }else if(MEAN_NUM_ROW>16){
            const int THREAD_PER_VECTOR = 32;
            const int VECTOR_PER_BLOCK = NUM_THREADS_PER_BLOCK / THREAD_PER_VECTOR;
            dim3 block(NUM_THREADS_PER_BLOCK);
            dim3 grid(DIV_UP(row_num,VECTOR_PER_BLOCK));
            spmv_gpu<int,float,VECTOR_PER_BLOCK,THREAD_PER_VECTOR><<<grid,block>>>(row_num,dRowOffsets,dColIndex,dValue,dX,dY);
        }
    }
    checkCudaError(cudaEventRecord(stop));
    checkCudaError(cudaEventSynchronize(stop));
    checkCudaError(cudaEventElapsedTime(&elapsed_time,start,stop));

    elapsed_time /= 1000;   // ms->s
    elapsed_time /= nRepeats; // avg time

    checkCudaError(cudaMemcpy(y_gpu.data(),dY,ySize,cudaMemcpyDeviceToHost));

    double maxError = 0.0f;
    double eps = 1e-2;
    for(int i=0;i<row_num;i++){
        maxError = fmax(maxError,fabs(y[i]-y_gpu[i]));
        // printf("[%d]:[%.6f] | [%.6f]\n",i,y[i],y_gpu[i]);
    }
    printf("MAXERROR: %.6f\n",maxError);
    printf("RESULT:%s\n",(maxError<eps)?"PASS":"FAILED");
    printf("MY SPMV COST TIME: %.9fs\n",elapsed_time);

    // =======================
    // 用cuSparse算出dY
    /*
    cusparseStatus_t
cusparseSpMV_bufferSize(cusparseHandle_t          handle,
                        cusparseOperation_t       opA,
                        const void*               alpha,
                        cusparseConstSpMatDescr_t matA,  // non-const descriptor supported
                        cusparseConstDnVecDescr_t vecX,  // non-const descriptor supported
                        const void*               beta,
                        cusparseDnVecDescr_t      vecY,
                        cudaDataType              computeType,
                        cusparseSpMVAlg_t         alg,
                        size_t*                   bufferSize)

    cusparseStatus_t
cusparseSpMV(cusparseHandle_t          handle,
             cusparseOperation_t       opA,
             const void*               alpha,
             cusparseConstSpMatDescr_t matA,  // non-const descriptor supported
             cusparseConstDnVecDescr_t vecX,  // non-const descriptor supported
             const void*               beta,
             cusparseDnVecDescr_t      vecY,
             cudaDataType              computeType,
             cusparseSpMVAlg_t         alg,
             void*                     externalBuffer)
     */
    float alpha=1.0f,beta=0.0f;
    // 构建cusparse用的上下文句柄
    cusparseHandle_t handle;
    checkCUSPARSEError(cusparseCreate(&handle));
    // 构建稀疏阵，并以CSR格式排列数据
    cusparseSpMatDescr_t spMatA;
    checkCUSPARSEError(cusparseCreateCsr(
        &spMatA,
        row_num,
        col_num,
        nonzero_num,
        dRowOffsets,
        dColIndex,
        dValue,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_32F
    ));             
    // 构建两个稠密向量X,Y
    cusparseDnVecDescr_t dnVecX;
    cusparseDnVecDescr_t dnVecY;
    cusparseCreateDnVec(&dnVecX,col_num,dX,CUDA_R_32F);
    cusparseCreateDnVec(&dnVecY,row_num,dY_CUSPARSE,CUDA_R_32F);
    // spmv使用的算法，用default是通用的: CUSPARSE_SPMV_ALG_DEFAULT
    // 获取buffersize
    size_t bufferSize;
    cusparseSpMV_bufferSize(handle,
                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha,
                            spMatA,
                            dnVecX,
                            &beta,
                            dnVecY,
                            CUDA_R_32F,
                            CUSPARSE_MV_ALG_DEFAULT,
                            &bufferSize);
    void* externalBuffer;
    checkCudaError(cudaMalloc(&externalBuffer,bufferSize));
    // 调用cusparse SpMV API,并计时
    checkCudaError(cudaEventRecord(start));
    for(int i=0;i<nRepeats;i++){
        cusparseSpMV(handle,
                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &alpha,
                    spMatA,
                    dnVecX,
                    &beta,
                    dnVecY,
                    CUDA_R_32F,
                    CUSPARSE_MV_ALG_DEFAULT,
                    externalBuffer);
    }
    checkCudaError(cudaEventRecord(stop));
    checkCudaError(cudaEventSynchronize(stop));
    checkCudaError(cudaEventElapsedTime(&elapsed_time,start,stop));
    elapsed_time /= 1000;       // ms->s;
    elapsed_time /= nRepeats;   // avg time;

    checkCudaError(cudaMemcpy(y_cusparse.data(),dY_CUSPARSE,ySize,cudaMemcpyDeviceToHost));
    // 校验正确性：
    maxError = 0.0f;
    for(int i=0;i<row_num;i++){
        maxError = fmax(maxError,fabs(y_cusparse[i]-y_gpu[i]));
        // printf("[%d]:[%.6f] | [%.6f]\n",i,y_cusparse[i],y_gpu[i]);
    }
    printf("MAXERROR: %.6f\n",maxError);
    printf("RESULT:%s\n",(maxError<eps)?"PASS":"FAILED");
    printf("CUSPARSE COST TIME: %.9fs\n",elapsed_time);

    // ======CUSPARSE清理======
    checkCUSPARSEError(cusparseDestroy(handle));
    checkCUSPARSEError(cusparseDestroyDnVec(dnVecX));
    checkCUSPARSEError(cusparseDestroyDnVec(dnVecY));
    checkCUSPARSEError(cusparseDestroySpMat(spMatA));
    checkCudaError(cudaFree(externalBuffer));
    // =======================
    checkCudaError(cudaEventDestroy(start));
    checkCudaError(cudaEventDestroy(stop));
    checkCudaError(cudaFree(dRowOffsets));
    checkCudaError(cudaFree(dColIndex));
    checkCudaError(cudaFree(dValue));
    checkCudaError(cudaFree(dX));
    checkCudaError(cudaFree(dY));
    return 0;
}