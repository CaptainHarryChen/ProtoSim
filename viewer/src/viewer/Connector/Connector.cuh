#pragma once

#define VIEWER_CONNECTOR_CUDA_BLOCK_SIZE 64

namespace viewer {
class Connector
{
public:
    Connector()          = default;
    virtual ~Connector() = default;

    virtual void TransferData() = 0;
};
}  // namespace viewer
