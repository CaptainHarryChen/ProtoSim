#pragma once

class Connector
{
public:
    Connector() = default;
    virtual ~Connector() = default;

    virtual void TransferData() = 0;
};
