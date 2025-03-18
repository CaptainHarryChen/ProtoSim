#include "Solver.cuh"

template <typename Real>
void Solver<Real>::SynchronizeStep()
{
    cudaDeviceSynchronize();
}

template <typename Real>
Real *Solver<Real>::GetDevicePositions()
{
    return nullptr;
}

template <typename Real>
Real *Solver<Real>::GetDeviceColors()
{
    return nullptr;
}

template class Solver<float>;
template class Solver<double>;
