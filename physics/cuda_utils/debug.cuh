#pragma once

#include <iostream>
#include <vector>

namespace cudaPhysics
{
    template <typename T>
    void print_device_array1d(T *array, size_t size, const std::string &name = "Array", size_t start = 0)
    {
        std::vector<T> host_array(size);
        cudaMemcpy(host_array.data(), array + start, size * sizeof(T), cudaMemcpyDeviceToHost);
        std::cout << name << " + "<< start << " : ";
        for (size_t i = 0; i < size; ++i)
        {
            std::cout << host_array[i] << " ";
        }
        std::cout << std::endl;
    }

    template <typename T>
    void print_device_array2d(T *array, size_t rows, size_t cols, const std::string &name = "Array", size_t start = 0)
    {
        std::vector<T> host_array(rows * cols);
        cudaMemcpy(host_array.data(), array + start * cols, rows * cols * sizeof(T), cudaMemcpyDeviceToHost);
        std::cout << name << ":" << std::endl;
        for (size_t i = 0; i < rows; ++i)
        {
            std::cout << i + start << ": ";
            for (size_t j = 0; j < cols; ++j)
            {
                std::cout << host_array[i * cols + j] << " ";
            }
            std::cout << std::endl;
        }
    }
}
