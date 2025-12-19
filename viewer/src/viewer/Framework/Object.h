#pragma once
#include <viewer/Event/Event.h>

namespace viewer {

class Object
{
public:
    Object() = default;
    virtual ~Object() = default;

    virtual void Update(double delta_time);
    virtual void ProcessEvent(const Event &event);
};

} // namespace viewer
