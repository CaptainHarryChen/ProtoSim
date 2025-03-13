#pragma once

template <typename Real>
class Solver
{
public:
    Solver() = default;
    virtual ~Solver() = default;

    virtual void Step() = 0;
    /// @brief Get the position array on the device, for Connector
    virtual Real *GetDevicePositions() = 0;
};
