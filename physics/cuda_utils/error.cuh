#pragma once
#include <cstdio>
#include <cassert>

#ifdef DEBUG
#define cudaCheck(call)                                   \
{                                                     \
    const cudaError_t error_code = call;              \
    if (error_code != cudaSuccess)                    \
    {                                                 \
        fprintf(stderr, "CUDA Error:\n");                      \
        fprintf(stderr, "    File:       %s\n", __FILE__);     \
        fprintf(stderr, "    Line:       %d\n", __LINE__);     \
        fprintf(stderr, "    Error code: %d\n", error_code);   \
        fprintf(stderr, "    Error text: %s\n",                \
            cudaGetErrorString(error_code));          \
        assert(false);                                \
        exit(1);                                      \
    }                                                 \
}
#else
#define cudaCheck(call) call
#endif
