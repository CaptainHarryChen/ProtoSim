#pragma once

#define CUDA_BLOCK_SIZE 256
#define CUDA_GRID_SIZE(x) ((x + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE)
