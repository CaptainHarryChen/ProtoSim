#include "PDMPMHybridSolver.cuh"
#include <cuda_utils/error.cuh>
#include <cuda_utils/array.cuh>
#include <Math/algebra.cuh>
#include <Math/elastic_model.cuh>

namespace PDMPMHybridKernel
{
    template <typename Real>
    __global__ void update_F(PDMPMHybridSolverData<Real> *data)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= data->m_num_particle)
            return;
        if (data->dev_particle_type[i] == MPM_ELASTIC)
        {
            Real F[9], temp[9];
            cudaPhysics::matMul3(temp, data->m_time_step, &data->dev_particle_C[i * 9]);
            temp[0] += 1.;
            temp[4] += 1.;
            temp[8] += 1.;
            cudaPhysics::matMul3(F, temp, &data->dev_particle_F[i * 9]);
            cudaPhysics::vecCopy(&data->dev_particle_F[i * 9], F, 9);
        }
        else if (data->dev_particle_type[i] == MPM_FLUID)
        {
            // Only use the first element of F to store J
            Real J = data->dev_particle_F[i * 9] * (1 + data->m_time_step * (data->dev_particle_C[i * 9] + data->dev_particle_C[i * 9 + 4] + data->dev_particle_C[i * 9 + 8]));
            data->dev_particle_F[i * 9 + 0] = J;
        }
    }

    template <typename Real>
    __global__ void calc_particle_affine_momentum(PDMPMHybridSolverData<Real> *data)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= data->m_num_particle)
            return;
        Real *affine_momentum = &data->dev_particle_affine_momentum[i * 9];
        cudaPhysics::vecMul(affine_momentum, data->dev_particle_mass[i], &data->dev_particle_C[i * 9], 9);
        if (data->dev_particle_type[i] == MPM_ELASTIC)
        {
            Real stress[9];
            Real P[9];
            cudaPhysics::calc_neohookean_P(P, &data->dev_particle_F[i * 9], data->m_lame_mu, data->m_lame_lambda);
            cudaPhysics::matmatTMul3(stress, P, &data->dev_particle_F[i * 9]);
            cudaPhysics::vecMul(stress, -4 * data->m_time_step * data->dev_particle_volume[i] / data->m_grid_spacing / data->m_grid_spacing, stress, 9);
            cudaPhysics::vecAdd(affine_momentum, stress, affine_momentum, 9);
        }
        else if (data->dev_particle_type[i] == MPM_FLUID)
        {
            Real stress = -4 * data->m_time_step * data->dev_particle_volume[i] / data->m_grid_spacing / data->m_grid_spacing * data->m_lame_lambda * (data->dev_particle_F[i * 9] - 1);
            affine_momentum[0] += stress;
            affine_momentum[4] += stress;
            affine_momentum[8] += stress;
        }
    }

    template <typename Real>
    __global__ void calc_particle_to_leftbottom_grid_id(PDMPMHybridSolverData<Real> *data)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= data->m_num_particle)
            return;
        // find the nearest grid, so plus 0.5 and floor
        int x = floor((data->dev_particle_position[i * 3 + 0] - data->dev_outer_bbox[0]) / data->m_grid_spacing + 0.5);
        int y = floor((data->dev_particle_position[i * 3 + 1] - data->dev_outer_bbox[1]) / data->m_grid_spacing + 0.5);
        int z = floor((data->dev_particle_position[i * 3 + 2] - data->dev_outer_bbox[2]) / data->m_grid_spacing + 0.5);
        // get the left bottom grid id
        x--;
        y--;
        z--;
        if (x < 0 || y < 0 || z < 0 || x >= data->dev_grid_size[0] - 2 || y >= data->dev_grid_size[1] - 2 || z >= data->dev_grid_size[2] - 2)
        {
            x = max(0, min(x, (int)data->dev_grid_size[0] - 2));
            y = max(0, min(y, (int)data->dev_grid_size[1] - 2));
            z = max(0, min(z, (int)data->dev_grid_size[2] - 2));
            printf("Warning: particle %d is out of grid, set to %d %d %d\n [Particle position] %.10f %.10f %.10f", i, x, y, z, data->dev_particle_position[i * 3 + 0], data->dev_particle_position[i * 3 + 1], data->dev_particle_position[i * 3 + 2]);
        }

        data->dev_particle_to_grid_id[i] = x * data->dev_grid_size[1] * data->dev_grid_size[2] + y * data->dev_grid_size[2] + z;
    }

    template <typename Real>
    __device__ void get_grid_position(Real *grid_position, unsigned int id, PDMPMHybridSolverData<Real> *data)
    {
        unsigned int z = id % data->dev_grid_size[2];
        unsigned int y = (id / data->dev_grid_size[2]) % data->dev_grid_size[1];
        unsigned int x = id / data->dev_grid_size[2] / data->dev_grid_size[1];
        grid_position[0] = data->dev_outer_bbox[0] + x * data->m_grid_spacing;
        grid_position[1] = data->dev_outer_bbox[1] + y * data->m_grid_spacing;
        grid_position[2] = data->dev_outer_bbox[2] + z * data->m_grid_spacing;
    }

    template <typename Real>
    __device__ Real grid_particle_quadratic_weight(Real *grid_position, Real *particle_position, Real grid_spacing)
    {
        Real result = 1.;
        for (unsigned int i = 0; i < 3; ++i)
        {
            Real d = (particle_position[i] - grid_position[i]) / grid_spacing;
            Real w = 0;
            if (-0.5 < d && d < 0.5)
                w = 0.75 - d * d;
            else if (0.5 <= d && d < 1.5)
                w = 0.5 * (1.5 - d) * (1.5 - d);
            else if (-1.5 < d && d <= -0.5)
                w = 0.5 * (1.5 + d) * (1.5 + d);
            result *= w;
        }
        return result;
    }

    template <typename Real>
    __global__ void P2G_momentum_and_mass(PDMPMHybridSolverData<Real> *data)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= data->m_num_particle)
            return;
        Real *particle_position = &data->dev_particle_position[i * 3];
        Real *particle_velocity = &data->dev_particle_velocity[i * 3];
        Real *affine_momentum = &data->dev_particle_affine_momentum[i * 9];
        unsigned int leftbottom_grid_id = data->dev_particle_to_grid_id[i];
        unsigned int x = leftbottom_grid_id / data->dev_grid_size[1] / data->dev_grid_size[2];
        unsigned int y = (leftbottom_grid_id / data->dev_grid_size[2]) % data->dev_grid_size[1];
        unsigned int z = leftbottom_grid_id % data->dev_grid_size[2];
        for (unsigned int dx = 0; dx < 3; ++dx)
            for (unsigned int dy = 0; dy < 3; ++dy)
                for (unsigned int dz = 0; dz < 3; ++dz)
                {
                    unsigned int grid_id = (x + dx) * data->dev_grid_size[1] * data->dev_grid_size[2] + (y + dy) * data->dev_grid_size[2] + (z + dz);
                    Real grid_position[3];
                    get_grid_position(grid_position, grid_id, data);
                    Real weight = grid_particle_quadratic_weight(grid_position, particle_position, data->m_grid_spacing);
                    Real momentum[3];
                    Real delta_position[3];
                    cudaPhysics::vecSubs3(delta_position, grid_position, particle_position);
                    cudaPhysics::matVec3(momentum, affine_momentum, delta_position);
                    Real temp_momentum[3];
                    cudaPhysics::vecMul3(temp_momentum, data->dev_particle_mass[i], particle_velocity);
                    cudaPhysics::vecAdd3(momentum, temp_momentum, momentum);
                    cudaPhysics::vecMul3(momentum, weight, momentum);
                    atomicAdd(&data->dev_grid_momentum[grid_id * 3 + 0], momentum[0]);
                    atomicAdd(&data->dev_grid_momentum[grid_id * 3 + 1], momentum[1]);
                    atomicAdd(&data->dev_grid_momentum[grid_id * 3 + 2], momentum[2]);

                    atomicAdd(&data->dev_grid_mass[grid_id], weight * data->dev_particle_mass[i]);
                }
    }

    template <typename Real>
    __global__ void calc_grids_velocity(PDMPMHybridSolverData<Real> *data)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= data->m_num_grid)
            return;
        if (data->dev_grid_mass[i] > 0)
            cudaPhysics::vecMul3(&data->dev_grid_velocity[i * 3], (Real)(1. / data->dev_grid_mass[i]), &data->dev_grid_momentum[i * 3]);
        else
        {
            data->dev_grid_velocity[i * 3 + 0] = 0;
            data->dev_grid_velocity[i * 3 + 1] = 0;
            data->dev_grid_velocity[i * 3 + 2] = 0;
        }
    }

    template <typename Real>
    __global__ void grids_gravity(PDMPMHybridSolverData<Real> *data)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= data->m_num_grid)
            return;
        Real delta_velocity[3];
        cudaPhysics::vecMul3(delta_velocity, data->m_time_step, data->dev_gravity);
        cudaPhysics::vecAdd3(&data->dev_grid_velocity[i * 3], &data->dev_grid_velocity[i * 3], delta_velocity);
    }

    template <typename Real>
    __global__ void particles_gravity(PDMPMHybridSolverData<Real> *data)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= data->m_num_particle)
            return;
        Real delta_velocity[3];
        cudaPhysics::vecMul3(delta_velocity, data->m_time_step, data->dev_gravity);
        cudaPhysics::vecAdd3(&data->dev_particle_velocity[i * 3], &data->dev_particle_velocity[i * 3], delta_velocity);
    }

    template <typename Real>
    __global__ void grids_boundary_conditions(PDMPMHybridSolverData<Real> *data)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= data->m_num_grid)
            return;
        unsigned int x = i / data->dev_grid_size[1] / data->dev_grid_size[2];
        unsigned int y = (i / data->dev_grid_size[2]) % data->dev_grid_size[1];
        unsigned int z = i % data->dev_grid_size[2];
        if (x <= data->m_boundary_thickness)
            data->dev_grid_velocity[i * 3 + 0] = max(0., data->dev_grid_velocity[i * 3 + 0]);
        if (x >= data->dev_grid_size[0] - 1 - data->m_boundary_thickness)
            data->dev_grid_velocity[i * 3 + 0] = min(0., data->dev_grid_velocity[i * 3 + 0]);
        if (y <= data->m_boundary_thickness)
            data->dev_grid_velocity[i * 3 + 1] = max(0., data->dev_grid_velocity[i * 3 + 1]);
        if (y >= data->dev_grid_size[1] - 1 - data->m_boundary_thickness)
            data->dev_grid_velocity[i * 3 + 1] = min(0., data->dev_grid_velocity[i * 3 + 1]);
        if (z <= data->m_boundary_thickness)
            data->dev_grid_velocity[i * 3 + 2] = max(0., data->dev_grid_velocity[i * 3 + 2]);
        if (z >= data->dev_grid_size[2] - 1 - data->m_boundary_thickness)
            data->dev_grid_velocity[i * 3 + 2] = min(0., data->dev_grid_velocity[i * 3 + 2]);
    }

    template <typename Real>
    __global__ void G2P_velocity_and_C(PDMPMHybridSolverData<Real> *data)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= data->m_num_particle)
            return;
        if (data->dev_particle_type[i] == MPM_STATIC)
            return;
        Real *particle_position = &data->dev_particle_position[i * 3];
        unsigned int leftbottom_grid_id = data->dev_particle_to_grid_id[i];
        unsigned int x = leftbottom_grid_id / data->dev_grid_size[1] / data->dev_grid_size[2];
        unsigned int y = (leftbottom_grid_id / data->dev_grid_size[2]) % data->dev_grid_size[1];
        unsigned int z = leftbottom_grid_id % data->dev_grid_size[2];
        for (unsigned int dx = 0; dx < 3; ++dx)
            for (unsigned int dy = 0; dy < 3; ++dy)
                for (unsigned int dz = 0; dz < 3; ++dz)
                {
                    unsigned int grid_id = (x + dx) * data->dev_grid_size[1] * data->dev_grid_size[2] + (y + dy) * data->dev_grid_size[2] + (z + dz);
                    Real grid_position[3];
                    get_grid_position(grid_position, grid_id, data);
                    Real weight = grid_particle_quadratic_weight(grid_position, particle_position, data->m_grid_spacing);

                    Real velocity[3];
                    cudaPhysics::vecMul3(velocity, weight, &data->dev_grid_velocity[grid_id * 3]);
                    cudaPhysics::vecAdd3(&data->dev_particle_velocity[i * 3], &data->dev_particle_velocity[i * 3], velocity);

                    Real delta_position[3];
                    cudaPhysics::vecSubs3(delta_position, grid_position, particle_position);
                    Real temp_C[9];
                    cudaPhysics::vecvecT(temp_C, &data->dev_grid_velocity[grid_id * 3], delta_position, 3, 3);
                    cudaPhysics::matMul3(temp_C, weight * 4 / data->m_grid_spacing / data->m_grid_spacing, temp_C);
                    cudaPhysics::vecAdd(&data->dev_particle_C[i * 9], temp_C, &data->dev_particle_C[i * 9], 9);
                }
    }

    template <typename Real>
    __global__ void update_particle_positions(PDMPMHybridSolverData<Real> *data)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= data->m_num_particle)
            return;
        if (data->dev_particle_type[i] == 0)
            return;
        Real *position = &data->dev_particle_position[i * 3];
        Real delta_position[3];
        cudaPhysics::vecMul3(delta_position, data->m_time_step, &data->dev_particle_velocity[i * 3]);
        cudaPhysics::vecAdd3(position, position, delta_position);
    }

    template <typename Real>
    __global__ void particles_boundary_conditions(PDMPMHybridSolverData<Real> *data)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= data->m_num_particle)
            return;
        for (unsigned int j = 0; j < 3; ++j)
        {
            if (data->dev_particle_position[i * 3 + j] < data->dev_inner_bbox[j])
            {
                data->dev_particle_position[i * 3 + j] = data->dev_inner_bbox[j];
            }
            if (data->dev_particle_position[i * 3 + j] > data->dev_inner_bbox[j + 3])
            {
                data->dev_particle_position[i * 3 + j] = data->dev_inner_bbox[j + 3];
            }
        }
    }
}

template <typename Real>
PDMPMHybridSolver<Real>::PDMPMHybridSolver(
    const std::vector<Real> &node_position,
    const std::vector<unsigned int> &surface_triangle,
    const std::vector<unsigned int> &tetrahedron,
    const std::vector<Real> &tetrahedron_density,

    const std::vector<Real> &particle_position,
    const std::vector<unsigned int> &particle_type,
    const std::vector<Real> &particle_mass,
    const std::vector<Real> &particle_volume,
    std::vector<Real> bbox,
    Real grid_spacing,
    unsigned int boundary_thickness //
)
{
    m_data.m_num_vert = (unsigned int)node_position.size() / 3;
    m_data.m_num_tet = (unsigned int)tetrahedron.size() / 4;
    m_data.m_num_tri = (unsigned int)surface_triangle.size() / 3;

    cudaMalloc((void **)&m_data.dev_position_backup, sizeof(Real) * m_data.m_num_vert * 3);
    cudaMalloc((void **)&m_data.dev_position_guess, sizeof(Real) * m_data.m_num_vert * 3);
    cudaMalloc((void **)&m_data.dev_position_prev, sizeof(Real) * m_data.m_num_vert * 3);
    cudaMemcpy(m_data.dev_position_prev, node_position.data(), sizeof(Real) * m_data.m_num_vert * 3, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&m_data.dev_position, sizeof(Real) * m_data.m_num_vert * 3);
    cudaMemcpy(m_data.dev_position, node_position.data(), sizeof(Real) * m_data.m_num_vert * 3, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&m_data.dev_position_next, sizeof(Real) * m_data.m_num_vert * 3);
    cudaMalloc((void **)&m_data.dev_position_delta, sizeof(Real) * m_data.m_num_vert * 3);
    cudaMalloc((void **)&m_data.dev_velocity, sizeof(Real) * m_data.m_num_vert * 3);
    cudaMemset(m_data.dev_velocity, 0, sizeof(Real) * m_data.m_num_vert * 3);
    cudaMalloc((void **)&m_data.dev_mass, sizeof(Real) * m_data.m_num_vert);
    cudaMemset(m_data.dev_mass, 0, sizeof(Real) * m_data.m_num_vert);
    cudaMalloc((void **)&m_data.dev_vert_force, sizeof(Real) * m_data.m_num_vert * 3);
    cudaMalloc((void **)&m_data.dev_constraint_Hessian_diag, sizeof(Real) * m_data.m_num_vert);
    cudaMalloc((void **)&m_data.dev_stiffness_matrix_diag, sizeof(Real) * m_data.m_num_vert);
    cudaMemset(m_data.dev_stiffness_matrix_diag, 0, sizeof(Real) * m_data.m_num_vert);

    cudaMalloc((void **)&m_data.dev_triangle, sizeof(unsigned int) * surface_triangle.size());
    cudaMemcpy(m_data.dev_triangle, surface_triangle.data(), sizeof(unsigned int) * surface_triangle.size(), cudaMemcpyHostToDevice);

    cudaMalloc((void **)&m_data.dev_tetrahedron, sizeof(unsigned int) * tetrahedron.size());
    cudaMemcpy(m_data.dev_tetrahedron, tetrahedron.data(), sizeof(unsigned int) * tetrahedron.size(), cudaMemcpyHostToDevice);
    cudaMalloc((void **)&m_data.dev_tet_density, sizeof(Real) * m_data.m_num_tet);
    cudaMemcpy(m_data.dev_tet_density, tetrahedron_density.data(), sizeof(Real) * m_data.m_num_tet, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&m_data.dev_tet_volume, sizeof(Real) * m_data.m_num_tet);
    cudaMemset(m_data.dev_tet_volume, 0, sizeof(Real) * m_data.m_num_tet);
    cudaMalloc((void **)&m_data.dev_tet_force, sizeof(Real) * m_data.m_num_tet * 12);
    cudaMemset(m_data.dev_tet_force, 0, sizeof(Real) * m_data.m_num_tet * 12);
    cudaMalloc((void **)&m_data.dev_invDm, sizeof(Real) * m_data.m_num_tet * 9);

    assert(particle_position.size() % 3 == 0);
    m_data.m_num_particle = (unsigned int)particle_position.size() / 3;
    printf("num_particle = %u\n", m_data.m_num_particle);

    cudaMalloc(&m_data.dev_particle_position, sizeof(Real) * m_data.m_num_particle * 3);
    cudaMemcpy(m_data.dev_particle_position, particle_position.data(), sizeof(Real) * m_data.m_num_particle * 3, cudaMemcpyHostToDevice);
    cudaMalloc(&m_data.dev_particle_velocity, sizeof(Real) * m_data.m_num_particle * 3);
    cudaMemset(m_data.dev_particle_velocity, 0, sizeof(Real) * m_data.m_num_particle * 3);
    cudaMalloc(&m_data.dev_particle_mass, sizeof(Real) * particle_mass.size());
    cudaMemcpy(m_data.dev_particle_mass, particle_mass.data(), sizeof(Real) * m_data.m_num_particle, cudaMemcpyHostToDevice);
    cudaMalloc(&m_data.dev_particle_volume, sizeof(Real) * particle_volume.size());
    cudaMemcpy(m_data.dev_particle_volume, particle_volume.data(), sizeof(Real) * m_data.m_num_particle, cudaMemcpyHostToDevice);
    cudaMalloc(&m_data.dev_particle_type, sizeof(unsigned int) * particle_type.size());
    cudaMemcpy(m_data.dev_particle_type, particle_type.data(), sizeof(unsigned int) * m_data.m_num_particle, cudaMemcpyHostToDevice);
    cudaMalloc(&m_data.dev_particle_C, sizeof(Real) * m_data.m_num_particle * 9);
    cudaMemset(m_data.dev_particle_C, 0, sizeof(Real) * m_data.m_num_particle * 9);
    cudaMalloc(&m_data.dev_particle_affine_momentum, sizeof(Real) * m_data.m_num_particle * 9);
    cudaMemset(m_data.dev_particle_affine_momentum, 0, sizeof(Real) * m_data.m_num_particle * 9);
    cudaMalloc(&m_data.dev_particle_F, sizeof(Real) * m_data.m_num_particle * 9);
    cudaPhysics::fill_identity_matrix(m_data.dev_particle_F, m_data.m_num_particle, 3);
    cudaMalloc(&m_data.dev_particle_to_grid_id, sizeof(unsigned int) * m_data.m_num_particle);

    // need expand the bbox to ensure the particles can get 3x3x3 grids
    std::vector<Real> inner_bbox;
    std::vector<Real> outer_bbox;
    std::vector<unsigned int> grid_size; // order: x y z
    m_data.m_grid_spacing = grid_spacing;
    m_data.m_boundary_thickness = boundary_thickness;
    inner_bbox = bbox;
    outer_bbox = bbox;
    for (unsigned int i = 0; i < 3; ++i)
    {
        outer_bbox[i] -= m_data.m_boundary_thickness * m_data.m_grid_spacing;
        outer_bbox[i + 3] += m_data.m_boundary_thickness * m_data.m_grid_spacing;
    }
    grid_size.clear();
    for (unsigned int i = 0; i < 3; ++i)
        grid_size.push_back((unsigned int)(floor((outer_bbox[i + 3] - outer_bbox[i]) / m_data.m_grid_spacing + 0.5)) + 1);
    printf("grid_size = [%u %u %u]\n", grid_size[0], grid_size[1], grid_size[2]);
    for (unsigned int i = 0; i < 3; ++i)
        outer_bbox[i + 3] = outer_bbox[i] + m_data.m_grid_spacing * grid_size[i];
    m_data.m_num_grid = grid_size[0] * grid_size[1] * grid_size[2];
    printf("num_grid = %u\n", m_data.m_num_grid);

    cudaMalloc(&m_data.dev_inner_bbox, sizeof(Real) * 6);
    cudaMemcpy(m_data.dev_inner_bbox, inner_bbox.data(), sizeof(Real) * 6, cudaMemcpyHostToDevice);
    cudaMalloc(&m_data.dev_outer_bbox, sizeof(Real) * 6);
    cudaMemcpy(m_data.dev_outer_bbox, outer_bbox.data(), sizeof(Real) * 6, cudaMemcpyHostToDevice);
    cudaMalloc(&m_data.dev_grid_size, sizeof(unsigned int) * 3);
    cudaMemcpy(m_data.dev_grid_size, grid_size.data(), sizeof(unsigned int) * 3, cudaMemcpyHostToDevice);
    cudaMalloc(&m_data.dev_grid_momentum, sizeof(Real) * m_data.m_num_grid * 3);
    cudaMemset(m_data.dev_grid_momentum, 0, sizeof(Real) * m_data.m_num_grid * 3);
    cudaMalloc(&m_data.dev_grid_mass, sizeof(Real) * m_data.m_num_grid);
    cudaMemset(m_data.dev_grid_mass, 0, sizeof(Real) * m_data.m_num_grid);
    cudaMalloc(&m_data.dev_grid_velocity, sizeof(Real) * m_data.m_num_grid * 3);
    cudaMemset(m_data.dev_grid_velocity, 0, sizeof(Real) * m_data.m_num_grid * 3);

    m_data.m_time_step = TIME_STEP;
    m_data.m_time_step_inv = 1.0f / m_data.m_time_step;
    m_data.m_lame_mu = LAME_MU;
    m_data.m_lame_lambda = LAME_LAMBDA;
    m_data.m_ground_collision_stiffness = GROUND_COLLISION_STIFFNESS;
    m_data.m_under_relaxation = UNDER_RELAXATION;
    cudaMalloc(&m_data.dev_gravity, sizeof(Real) * 3);
    std::vector<Real> gravity = {0., -GRAVITY, 0.};
    cudaMemcpy(m_data.dev_gravity, gravity.data(), sizeof(Real) * 3, cudaMemcpyHostToDevice);

    cudaMalloc(&m_dev_data, sizeof(PDMPMHybridSolverData<Real>));
    cudaMemcpy(m_dev_data, &m_data, sizeof(PDMPMHybridSolverData<Real>), cudaMemcpyHostToDevice);

    // ProjectiveDynamicsSolverKernel::tetrahedron_initialize<Real><<<CUDA_GRID_SIZE(m_data.m_num_tet), CUDA_BLOCK_SIZE>>>(m_dev_data);
    // ProjectiveDynamicsSolverKernel::tet_stiffness_matrix_diag<Real><<<CUDA_GRID_SIZE(m_data.m_num_tet), CUDA_BLOCK_SIZE>>>(m_dev_data);

}

template <typename Real>
PDMPMHybridSolver<Real>::~PDMPMHybridSolver()
{
    cudaFree(m_data.dev_position_backup);
    cudaFree(m_data.dev_position_guess);
    cudaFree(m_data.dev_position_prev);
    cudaFree(m_data.dev_position);
    cudaFree(m_data.dev_position_next);
    cudaFree(m_data.dev_position_delta);
    cudaFree(m_data.dev_velocity);
    cudaFree(m_data.dev_mass);
    cudaFree(m_data.dev_vert_force);
    cudaFree(m_data.dev_constraint_Hessian_diag);
    cudaFree(m_data.dev_stiffness_matrix_diag);

    cudaFree(m_data.dev_tetrahedron);
    cudaFree(m_data.dev_tet_density);
    cudaFree(m_data.dev_tet_volume);
    cudaFree(m_data.dev_tet_force);
    cudaFree(m_data.dev_invDm);

    cudaFree(m_data.dev_particle_position);
    cudaFree(m_data.dev_particle_velocity);
    cudaFree(m_data.dev_particle_mass);
    cudaFree(m_data.dev_particle_volume);
    cudaFree(m_data.dev_particle_type);
    cudaFree(m_data.dev_particle_C);
    cudaFree(m_data.dev_particle_affine_momentum);
    cudaFree(m_data.dev_particle_F);
    cudaFree(m_data.dev_particle_to_grid_id);
    cudaFree(m_data.dev_inner_bbox);
    cudaFree(m_data.dev_outer_bbox);
    cudaFree(m_data.dev_grid_size);
    cudaFree(m_data.dev_grid_momentum);
    cudaFree(m_data.dev_grid_mass);
    cudaFree(m_data.dev_grid_velocity);
    cudaFree(m_data.dev_gravity);

    cudaFree(m_dev_data);
}

template <typename Real>
void PDMPMHybridSolver<Real>::Step()
{
    // P2G
    cudaMemset(m_data.dev_grid_momentum, 0, sizeof(Real) * m_data.m_num_grid * 3);
    cudaMemset(m_data.dev_grid_mass, 0, sizeof(Real) * m_data.m_num_grid);
    PDMPMHybridKernel::update_F<Real><<<CUDA_GRID_SIZE(m_data.m_num_particle), CUDA_BLOCK_SIZE>>>(m_dev_data);
    PDMPMHybridKernel::particles_gravity<Real><<<CUDA_GRID_SIZE(m_data.m_num_particle), CUDA_BLOCK_SIZE>>>(m_dev_data);
    PDMPMHybridKernel::calc_particle_affine_momentum<Real><<<CUDA_GRID_SIZE(m_data.m_num_particle), CUDA_BLOCK_SIZE>>>(m_dev_data);
    PDMPMHybridKernel::calc_particle_to_leftbottom_grid_id<Real><<<CUDA_GRID_SIZE(m_data.m_num_particle), CUDA_BLOCK_SIZE>>>(m_dev_data);
    PDMPMHybridKernel::P2G_momentum_and_mass<Real><<<CUDA_GRID_SIZE(m_data.m_num_particle), CUDA_BLOCK_SIZE>>>(m_dev_data);
    PDMPMHybridKernel::calc_grids_velocity<Real><<<CUDA_GRID_SIZE(m_data.m_num_grid), CUDA_BLOCK_SIZE>>>(m_dev_data);
    PDMPMHybridKernel::grids_gravity<Real><<<CUDA_GRID_SIZE(m_data.m_num_grid), CUDA_BLOCK_SIZE>>>(m_dev_data);
    PDMPMHybridKernel::grids_boundary_conditions<Real><<<CUDA_GRID_SIZE(m_data.m_num_grid), CUDA_BLOCK_SIZE>>>(m_dev_data);
    cudaMemset(m_data.dev_particle_velocity, 0, sizeof(Real) * m_data.m_num_particle * 3);
    cudaMemset(m_data.dev_particle_C, 0, sizeof(Real) * m_data.m_num_particle * 9);
    PDMPMHybridKernel::G2P_velocity_and_C<Real><<<CUDA_GRID_SIZE(m_data.m_num_particle), CUDA_BLOCK_SIZE>>>(m_dev_data);
    PDMPMHybridKernel::update_particle_positions<Real><<<CUDA_GRID_SIZE(m_data.m_num_particle), CUDA_BLOCK_SIZE>>>(m_dev_data);
    PDMPMHybridKernel::particles_boundary_conditions<Real><<<CUDA_GRID_SIZE(m_data.m_num_particle), CUDA_BLOCK_SIZE>>>(m_dev_data);
}

template <typename Real>
Real *PDMPMHybridSolver<Real>::GetDeviceNodePositions()
{
    return m_data.dev_position;
}

template <typename Real>
Real *PDMPMHybridSolver<Real>::GetDeviceParticlePositions()
{
    return m_data.dev_particle_position;
}

template class PDMPMHybridSolver<float>;
template class PDMPMHybridSolver<double>;
template struct PDMPMHybridSolverData<float>;
template struct PDMPMHybridSolverData<double>;
