#ifndef _UTILS_H_
#define _UTILS_H_
void random_matrix(int,int,float*,int);
void copy_matrix(int,int,float*,int,float*,int);
double compare_matrix(int,int,float*,int,float*,int);
double get_time(struct timespec*,struct timespec*);

#endif //_UTILS_H_