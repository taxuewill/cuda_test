#include <stdio.h>
#include <iostream>

extern "C"
void useCUDA();

int main()
{
    std::cout<<"Hello C++"<<std::endl;
    useCUDA();
    return 0;
}