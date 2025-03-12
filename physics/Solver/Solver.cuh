#pragma once

template <typename Real>
class Solver
{
public:
    Solver() = default;
    virtual ~Solver() = default;

    virtual void Step() = 0;
};
