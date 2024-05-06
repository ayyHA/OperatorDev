void arrayWeightedSum(float* array1,float weighted1,
                      float* array2,float weighted2,
                      int len,float* dst){
    int i;
    for(i=0;i<len;i++)
        dst[i] = weighted1 * array1[i] + weighted2 * array2[i];
}