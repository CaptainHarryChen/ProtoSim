#pragma once
#include <any>
#include <string>
#include <unordered_map>
#include <vector>

template <typename Real>
struct SeparatingCorrectorData
{
    unsigned int m_num_particle;
    Real *dev_particle_position;
    Real *dev_particle_volume;
    unsigned int *dev_particle_to_grid_id; // order: x y z
    Real *dev_particle_radius;
    Real *dev_particle_delta_position;
    Real *dev_particle_id;

    unsigned int m_num_grid;
    Real m_grid_spacing;
    unsigned int m_boundary_thickness;
    Real m_inner_bbox[6];
    Real m_outer_bbox[6];
    unsigned int m_grid_size[3];

    Real *dev_grid_bin_start_idx;
    unsigned int *dev_grid_bin_size;
};

template <typename Real>
class SeparatingCorrector
{
public:
    SeparatingCorrector(
        unsigned int num_particle,
        Real *dev_particle_position,
        Real *dev_particle_volume,
        std::vector<Real> bbox,
        Real grid_spacing,
        unsigned int boundary_thickness,
        unsigned int max_iteration
    );
    virtual ~SeparatingCorrector();

    void Run();
    void CalculateParticleDeltaPosition();
    void ApplyBoxConstraint();

    bool m_verbose = false;
    unsigned int m_max_iteration;
    SeparatingCorrectorData<Real> m_data;
};

extern template struct SeparatingCorrectorData<float>;
extern template struct SeparatingCorrectorData<double>;
extern template class SeparatingCorrector<float>;
extern template class SeparatingCorrector<double>;
