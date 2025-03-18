#pragma once

template <typename Real>
class Solver
{
public:
    Solver() = default;
    virtual ~Solver() = default;

    virtual void Step() = 0;
    /// @brief Synchronization, waiting the device to finish the computation. This is necessary for simulation FPS computation
    virtual void SynchronizeStep();
    /// @brief Get the position array on the device, for Connector
    virtual Real *GetDevicePositions();
    /// @brief Get the color array on the device, for Connector, especially for ParticleConnector
    virtual Real *GetDeviceColors();
};

extern template class Solver<float>;
extern template class Solver<double>;
