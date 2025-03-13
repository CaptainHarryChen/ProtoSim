#pragma once

template <typename Real>
class Solver
{
public:
    Solver() = default;
    virtual ~Solver() = default;

    virtual void Step() = 0;
    /// @brief Get the position array on the device, for Connector
    virtual inline Real *GetDevicePositions() { return nullptr; }
    /// @brief Get the color array on the device, for Connector, especially for ParticleConnector
    virtual inline Real *GetDeviceColors() { return nullptr; }
};
