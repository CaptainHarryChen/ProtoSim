#pragma once
#include <any>
#include <string>
#include <unordered_map>
#include <vector>

template <typename Real>
struct PICVolumeCorrectorData
{
    unsigned int m_num_particle;
    Real *dev_particle_position;
    Real *dev_particle_volume;
    unsigned int *dev_particle_to_grid_id; // order: x y z
    Real *dev_particle_delta_position;

    unsigned int m_num_grid;
    Real m_grid_spacing;
    unsigned int m_boundary_thickness;
    Real m_inner_bbox[6];
    Real m_outer_bbox[6];
    unsigned int m_grid_size[3];
    Real *dev_grid_volume;
};

template <typename Real>
class PICVolumeCorrector
{
public:
    PICVolumeCorrector(
        unsigned int num_particle,
        Real *dev_particle_position,
        Real *dev_particle_volume,
        unsigned int *dev_particle_to_grid_id,
        std::vector<Real> bbox,
        Real grid_spacing,
        unsigned int boundary_thickness,
        unsigned int max_iteration
    );
    virtual ~PICVolumeCorrector();

    void Run();
    void CalculateParticleDeltaPosition();
    void ApplyBoxConstraint();

    bool m_verbose = false;
    unsigned int m_max_iteration;
    PICVolumeCorrectorData<Real> m_data;

    
};

extern template struct PICVolumeCorrectorData<float>;
extern template struct PICVolumeCorrectorData<double>;
extern template class PICVolumeCorrector<float>;
extern template class PICVolumeCorrector<double>;
