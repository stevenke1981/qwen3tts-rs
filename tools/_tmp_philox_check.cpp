#include <cstdio>
#include <cstdint>
#include "E:/qwentts.cpp-reference/src/philox.h"
int main(){
    uint32_t x;
    uint32_t y;
    uint32_t z;
    uint32_t w;
    uint32_t words[4] = {0,0,0,0};
    (void)words;
    float u;
    philox_uniform_fill(0,0,0,&u,1);
    Philox4 s = philox4x32_10({0,0,0,0},0,0);
    printf("%u %u %u %u\n", s.x, s.y, s.z, s.w);
    printf("%.16a\n", u);
    return 0;
}
