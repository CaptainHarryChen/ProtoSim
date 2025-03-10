#pragma once
#include <Event/Event.h>

class Object
{
public:
    Object() = default;
    virtual ~Object() = default;

    virtual void Update(double delta_time);
    virtual void ProcessEvent(const Event &event);
};
