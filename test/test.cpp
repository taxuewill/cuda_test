#include <stdio.h>
#include <iostream>
#include <string>
#include <string.h>
#include <sys/types.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

#include "../cuda/transform.cuh"

int64_t getCurrentTimestamp()
{

    struct timespec res;
    clock_gettime(CLOCK_REALTIME, &res);
    int64_t t0 = (int64_t)1000000 * res.tv_sec + (int64_t)res.tv_nsec / 1000;
    return t0;
}

using namespace std;
int main()
{
    std::cout<<"Hello Cuda"<<std::endl;
    // useCUDA();
    int dataSize = 1280*720;

    int * array1 = new int[dataSize];
    int * array2 = new int[dataSize];
    int * array3 = new int[dataSize];
    
    for(int i=0;i<dataSize;i++){
        array1[i]=1;
        array2[i]=2;
        array3[i]=0;
    }
    int64_t start = getCurrentTimestamp();
    for(int i = 0;i<dataSize;i++){
        array3[i]=array1[i]+array2[i];
    }
    // std::cout<< "size of int is "<<sizeof(float)<<endl;
     vectorAdd(array1,array2,array3,dataSize);
    int64_t end = getCurrentTimestamp();
    cout<<"pu cost " <<(end -start)<<endl;
    // for(int i = 0;i < dataSize;i++){
    //     //cout<<"array3["<<i<<"] is "<<array3[i]<<endl;
    //     if(array3[i]!= 3){
    //         cout<<"error ["<<i<<"] " <<array3[i]<<endl;
    //     }
    // }
    return 0;
}