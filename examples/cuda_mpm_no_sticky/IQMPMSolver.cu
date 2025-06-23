#include "IQMPMSolver.cuh"
#include <cuda_utils/cuda_utils.cuh>
#include <Math/algebra.cuh>
#include <Math/elastic_model.cuh>

namespace IQMPMSolverKernel
{
    template <typename Real>
    __global__ void update_F(IQMPMSolverData<Real> *data)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= data->m_num_particle)
            return;
        if (data->dev_object_type[data->dev_particle_object_id[i]] == MPM_ELASTIC)
        {
            Real F[9], temp[9];
            cudaPhysics::matMul3(temp, data->m_time_step, &data->dev_particle_C[i * 9]);
            temp[0] += 1.;
            temp[4] += 1.;
            temp[8] += 1.;
            cudaPhysics::matMul3(F, temp, &data->dev_particle_F[i * 9]);
            cudaPhysics::vecCopy(&data->dev_particle_F[i * 9], F, 9);
        }
        else if (data->dev_object_type[data->dev_particle_object_id[i]] == MPM_FLUID)
        {
            // Only use the first element of F to store J
            Real J = data->dev_particle_F[i * 9] * (1 + data->m_time_step * (data->dev_particle_C[i * 9] + data->dev_particle_C[i * 9 + 4] + data->dev_particle_C[i * 9 + 8]));
            data->dev_particle_F[i * 9 + 0] = J;
        }
    }

    template <typename Real>
    __global__ void calc_particle_affine_momentum(IQMPMSolverData<Real> *data)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= data->m_num_particle)
            return;
        Real *affine_momentum = &data->dev_particle_affine_momentum[i * 9];
        cudaPhysics::vecMul(affine_momentum, data->dev_particle_mass[i], &data->dev_particle_C[i * 9], 9);
        if (data->dev_object_type[data->dev_particle_object_id[i]] == MPM_ELASTIC)
        {
            Real stress[9];
            Real P[9];
            cudaPhysics::calc_neohookean_P(P, &data->dev_particle_F[i * 9], data->m_lame_mu, data->m_lame_lambda);
            cudaPhysics::matmatTMul3(stress, P, &data->dev_particle_F[i * 9]);
            cudaPhysics::vecMul(stress, -4 * data->m_time_step * data->dev_particle_volume[i] / data->m_grid_spacing / data->m_grid_spacing, stress, 9);
            cudaPhysics::vecAdd(affine_momentum, stress, affine_momentum, 9);
        }
        else if (data->dev_object_type[data->dev_particle_object_id[i]] == MPM_FLUID)
        {
            Real stress = -4 * data->m_time_step * data->dev_particle_volume[i] / data->m_grid_spacing / data->m_grid_spacing * 1000000.0 * (data->dev_particle_F[i * 9] - 1);
            affine_momentum[0] += stress;
            affine_momentum[4] += stress;
            affine_momentum[8] += stress;
        }
    }

    template <typename Real>
    __global__ void calc_particle_to_leftbottom_grid_id(IQMPMSolverData<Real> *data)
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

        data->dev_particle_to_grid_id[i] = (x * data->dev_grid_size[1] * data->dev_grid_size[2] + y * data->dev_grid_size[2] + z) + (data->dev_particle_object_id[i] * data->dev_grid_size[0] * data->dev_grid_size[1] * data->dev_grid_size[2]);
    }

    template <typename Real>
    __device__ void get_grid_xyz(unsigned int &x, unsigned int &y, unsigned int &z, unsigned int id, IQMPMSolverData<Real> *data)
    {
        id = id % (data->dev_grid_size[0] * data->dev_grid_size[1] * data->dev_grid_size[2]);
        z = id % data->dev_grid_size[2];
        y = (id / data->dev_grid_size[2]) % data->dev_grid_size[1];
        x = id / data->dev_grid_size[2] / data->dev_grid_size[1];
    }

    template <typename Real>
    __device__ unsigned int get_grid_id(unsigned int object_id, unsigned int x, unsigned int y, unsigned int z, IQMPMSolverData<Real> *data)
    {
        return (x * data->dev_grid_size[1] * data->dev_grid_size[2] + y * data->dev_grid_size[2] + z) + (object_id * data->dev_grid_size[0] * data->dev_grid_size[1] * data->dev_grid_size[2]);
    }

    template <typename Real>
    __device__ void get_grid_position(Real *grid_position, unsigned int id, IQMPMSolverData<Real> *data)
    {
        id = id % (data->dev_grid_size[0] * data->dev_grid_size[1] * data->dev_grid_size[2]);
        unsigned int x, y, z;
        get_grid_xyz(x, y, z, id, data);
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
    __global__ void P2G_momentum_and_mass(IQMPMSolverData<Real> *data)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= data->m_num_particle)
            return;
        Real *particle_position = &data->dev_particle_position[i * 3];
        Real *particle_velocity = &data->dev_particle_velocity[i * 3];
        Real *affine_momentum = &data->dev_particle_affine_momentum[i * 9];
        unsigned int leftbottom_grid_id = data->dev_particle_to_grid_id[i];
        unsigned int x, y, z;
        get_grid_xyz(x, y, z, leftbottom_grid_id, data);
        for (unsigned int dx = 0; dx < 3; ++dx)
            for (unsigned int dy = 0; dy < 3; ++dy)
                for (unsigned int dz = 0; dz < 3; ++dz)
                {
                    unsigned int grid_id = get_grid_id(data->dev_particle_object_id[i], x + dx, y + dy, z + dz, data);
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
    __global__ void P2G_normal_estimate(IQMPMSolverData<Real> *data)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= data->m_num_particle)
            return;
        if (data->dev_object_type[data->dev_particle_object_id[i]] == MPM_FLUID)
            return;
        Real *particle_position = &data->dev_particle_position[i * 3];
        unsigned int leftbottom_grid_id = data->dev_particle_to_grid_id[i];
        unsigned int x, y, z;
        get_grid_xyz(x, y, z, leftbottom_grid_id, data);
        for (unsigned int dx = 0; dx < 3; ++dx)
            for (unsigned int dy = 0; dy < 3; ++dy)
                for (unsigned int dz = 0; dz < 3; ++dz)
                {
                    unsigned int grid_id = get_grid_id(data->dev_particle_object_id[i], x + dx, y + dy, z + dz, data);
                    Real grid_position[3];
                    get_grid_position(grid_position, grid_id, data);
                    Real weight = grid_particle_quadratic_weight(grid_position, particle_position, data->m_grid_spacing);
                    Real delta_position[3];
                    cudaPhysics::vecSubs3(delta_position, grid_position, particle_position);
                    Real temp_normal[3];
                    cudaPhysics::vecMul3(temp_normal, weight * data->dev_particle_mass[i], delta_position);
                    atomicAdd(&data->dev_grid_normal[grid_id * 3 + 0], temp_normal[0]);
                    atomicAdd(&data->dev_grid_normal[grid_id * 3 + 1], temp_normal[1]);
                    atomicAdd(&data->dev_grid_normal[grid_id * 3 + 2], temp_normal[2]);
                }
    }

    template <typename Real>
    __global__ void grid_normal_normalize(IQMPMSolverData<Real> *data)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= data->m_num_grid)
            return;
        cudaPhysics::norm3InPlace(&data->dev_grid_normal[i * 3]);
    }

    template <typename Real>
    __global__ void calc_grids_velocity(IQMPMSolverData<Real> *data)
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
    __global__ void grids_gravity(IQMPMSolverData<Real> *data)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= data->m_num_grid)
            return;
        Real delta_velocity[3];
        cudaPhysics::vecMul3(delta_velocity, data->m_time_step, data->dev_gravity);
        cudaPhysics::vecAdd3(&data->dev_grid_velocity[i * 3], &data->dev_grid_velocity[i * 3], delta_velocity);
    }

    template <typename Real>
    __global__ void particles_gravity(IQMPMSolverData<Real> *data)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= data->m_num_particle)
            return;
        Real delta_velocity[3];
        cudaPhysics::vecMul3(delta_velocity, data->m_time_step, data->dev_gravity);
        cudaPhysics::vecAdd3(&data->dev_particle_velocity[i * 3], &data->dev_particle_velocity[i * 3], delta_velocity);
    }

    template <typename Real>
    __global__ void grids_couple(IQMPMSolverData<Real> *data)
    {
        unsigned int grid_id = blockIdx.x * blockDim.x + threadIdx.x;
        if (grid_id >= data->dev_grid_size[0] * data->dev_grid_size[1] * data->dev_grid_size[2])
            return;
        for (unsigned int i = 0; i < data->m_num_object; i++)
        {
            unsigned int grid_id_i = grid_id + i * data->dev_grid_size[0] * data->dev_grid_size[1] * data->dev_grid_size[2];
            if (data->dev_grid_mass[grid_id_i] <= 0)
                continue;
            for (unsigned int j = i + 1; j < data->m_num_object; j++)
            {
                unsigned int grid_id_j = grid_id + j * data->dev_grid_size[0] * data->dev_grid_size[1] * data->dev_grid_size[2];
                if (data->dev_grid_mass[grid_id_j] <= 0)
                    continue;
                Real normal[3];
                cudaPhysics::vecSubs3(normal, &data->dev_grid_normal[grid_id_j * 3], &data->dev_grid_normal[grid_id_i * 3]);
                cudaPhysics::norm3InPlace(normal);
                Real vel_i = cudaPhysics::dot3(&data->dev_grid_velocity[grid_id_i * 3], normal);
                Real vel_j = cudaPhysics::dot3(&data->dev_grid_velocity[grid_id_j * 3], normal);
                if (vel_j  - vel_i <= 0)
                    continue;
                // completely inelastic collision
                Real vel_res = (data->dev_grid_mass[grid_id_i] * vel_i + data->dev_grid_mass[grid_id_j] * vel_j) / (data->dev_grid_mass[grid_id_i] + data->dev_grid_mass[grid_id_j]);
                Real delta_vel_i[3], delta_vel_j[3];
                cudaPhysics::vecMul3(delta_vel_i, vel_res - vel_i, normal);
                cudaPhysics::vecMul3(delta_vel_j, vel_res - vel_j, normal);
                cudaPhysics::vecAdd3(&data->dev_grid_velocity[grid_id_i * 3], &data->dev_grid_velocity[grid_id_i * 3], delta_vel_i);
                cudaPhysics::vecAdd3(&data->dev_grid_velocity[grid_id_j * 3], &data->dev_grid_velocity[grid_id_j * 3], delta_vel_j);
            }
        }
    }

    template <typename Real>
    __global__ void grids_boundary_conditions(IQMPMSolverData<Real> *data)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= data->m_num_grid)
            return;
        unsigned int x, y, z;
        get_grid_xyz(x, y, z, i, data);
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
    __global__ void G2P_velocity_and_C(IQMPMSolverData<Real> *data)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= data->m_num_particle)
            return;
        if (data->dev_object_type[data->dev_particle_object_id[i]] == MPM_STATIC)
            return;
        Real *particle_position = &data->dev_particle_position[i * 3];
        unsigned int leftbottom_grid_id = data->dev_particle_to_grid_id[i];
        unsigned int x, y, z;
        get_grid_xyz(x, y, z, leftbottom_grid_id, data);
        for (unsigned int dx = 0; dx < 3; ++dx)
            for (unsigned int dy = 0; dy < 3; ++dy)
                for (unsigned int dz = 0; dz < 3; ++dz)
                {
                    unsigned int grid_id = get_grid_id(data->dev_particle_object_id[i], x + dx, y + dy, z + dz, data);
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
    __global__ void update_particle_positions(IQMPMSolverData<Real> *data)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= data->m_num_particle)
            return;
        if (data->dev_object_type[data->dev_particle_object_id[i]] == 0)
            return;
        Real *position = &data->dev_particle_position[i * 3];
        Real delta_position[3];
        cudaPhysics::vecMul3(delta_position, data->m_time_step, &data->dev_particle_velocity[i * 3]);
        cudaPhysics::vecAdd3(position, position, delta_position);
    }

    template <typename Real>
    __global__ void particles_boundary_conditions(IQMPMSolverData<Real> *data)
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

    template <typename Real>
    __global__ void get_max_particle_velocity(IQMPMSolverData<Real> *data)
    {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= data->m_num_particle)
            return;
        Real velocity[3];
        cudaPhysics::vecCopy(velocity, &data->dev_particle_velocity[i * 3], 3);
        Real v = cudaPhysics::len3(velocity);
        cudaPhysics::AtomicMax(data->dev_max_particle_velocity, v);
    }
}

template <typename Real>
IQMPMSolver<Real>::IQMPMSolver(
    const std::vector<unsigned int> &object_type, const std::vector<unsigned int> &particle_object_id,
    const std::vector<Real> &particle_position, const std::vector<Real> &particle_mass, const std::vector<Real> &particle_volume,
    std::vector<Real> bbox, Real grid_spacing, unsigned int boundary_thickness)
{
    m_data.m_num_object = (unsigned int)object_type.size();
    printf("num_object = %u\n", m_data.m_num_object);
    cudaMalloc(&m_data.dev_object_type, sizeof(unsigned int) * object_type.size());
    cudaMemcpy(m_data.dev_object_type, object_type.data(), sizeof(unsigned int) * object_type.size(), cudaMemcpyHostToDevice);

    assert(particle_position.size() % 3 == 0);
    m_data.m_num_particle = (unsigned int)particle_position.size() / 3;
    printf("num_particle = %u\n", m_data.m_num_particle);

    cudaMalloc(&m_data.dev_particle_object_id, sizeof(unsigned int) * m_data.m_num_particle);
    cudaMemcpy(m_data.dev_particle_object_id, particle_object_id.data(), sizeof(unsigned int) * m_data.m_num_particle, cudaMemcpyHostToDevice);
    cudaMalloc(&m_data.dev_particle_position, sizeof(Real) * m_data.m_num_particle * 3);
    cudaMemcpy(m_data.dev_particle_position, particle_position.data(), sizeof(Real) * m_data.m_num_particle * 3, cudaMemcpyHostToDevice);
    cudaMalloc(&m_data.dev_particle_velocity, sizeof(Real) * m_data.m_num_particle * 3);
    cudaMemset(m_data.dev_particle_velocity, 0, sizeof(Real) * m_data.m_num_particle * 3);
    cudaMalloc(&m_data.dev_particle_mass, sizeof(Real) * particle_mass.size());
    cudaMemcpy(m_data.dev_particle_mass, particle_mass.data(), sizeof(Real) * m_data.m_num_particle, cudaMemcpyHostToDevice);
    cudaMalloc(&m_data.dev_particle_volume, sizeof(Real) * particle_volume.size());
    cudaMemcpy(m_data.dev_particle_volume, particle_volume.data(), sizeof(Real) * m_data.m_num_particle, cudaMemcpyHostToDevice);
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
    printf("grid_size = %u x [%u %u %u]\n", m_data.m_num_object, grid_size[0], grid_size[1], grid_size[2]);
    for (unsigned int i = 0; i < 3; ++i)
        outer_bbox[i + 3] = outer_bbox[i] + m_data.m_grid_spacing * grid_size[i];
    m_data.m_num_grid = m_data.m_num_object * grid_size[0] * grid_size[1] * grid_size[2];
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
    cudaMalloc(&m_data.dev_grid_normal, sizeof(Real) * m_data.m_num_grid * 3);
    cudaMemset(m_data.dev_grid_normal, 0, sizeof(Real) * m_data.m_num_grid * 3);

    m_data.m_time_step = TIME_STEP;
    m_data.m_lame_mu = LAME_MU;
    m_data.m_lame_lambda = LAME_LAMBDA;
    cudaMalloc(&m_data.dev_gravity, sizeof(Real) * 3);
    std::vector<Real> gravity = {0., -GRAVITY, 0.};
    cudaMemcpy(m_data.dev_gravity, gravity.data(), sizeof(Real) * 3, cudaMemcpyHostToDevice);

    cudaMalloc(&m_data.dev_max_particle_velocity, sizeof(Real));

    cudaMalloc(&m_dev_data, sizeof(IQMPMSolverData<Real>));
    cudaMemcpy(m_dev_data, &m_data, sizeof(IQMPMSolverData<Real>), cudaMemcpyHostToDevice);
}

template <typename Real>
IQMPMSolver<Real>::~IQMPMSolver()
{
    cudaFree(m_data.dev_object_type);
    cudaFree(m_data.dev_particle_object_id);
    cudaFree(m_data.dev_particle_position);
    cudaFree(m_data.dev_particle_velocity);
    cudaFree(m_data.dev_particle_mass);
    cudaFree(m_data.dev_particle_volume);
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
    cudaFree(m_data.dev_max_particle_velocity);

    cudaFree(m_dev_data);
}

template <typename Real>
void IQMPMSolver<Real>::SetupInitVelocity(const std::vector<Real> &particle_velocity)
{
    cudaMemcpy(m_data.dev_particle_velocity, particle_velocity.data(), sizeof(Real) * particle_velocity.size(), cudaMemcpyHostToDevice);
}

template <typename Real>
void IQMPMSolver<Real>::Step()
{
    // P2G
    cudaMemset(m_data.dev_grid_momentum, 0, sizeof(Real) * m_data.m_num_grid * 3);
    cudaMemset(m_data.dev_grid_mass, 0, sizeof(Real) * m_data.m_num_grid);
    IQMPMSolverKernel::update_F<Real><<<CUDA_GRID_SIZE(m_data.m_num_particle), CUDA_BLOCK_SIZE>>>(m_dev_data);
    // IQMPMSolverKernel::particles_gravity<Real><<<CUDA_GRID_SIZE(m_data.m_num_particle), CUDA_BLOCK_SIZE>>>(m_dev_data);
    IQMPMSolverKernel::calc_particle_affine_momentum<Real><<<CUDA_GRID_SIZE(m_data.m_num_particle), CUDA_BLOCK_SIZE>>>(m_dev_data);
    IQMPMSolverKernel::calc_particle_to_leftbottom_grid_id<Real><<<CUDA_GRID_SIZE(m_data.m_num_particle), CUDA_BLOCK_SIZE>>>(m_dev_data);
    IQMPMSolverKernel::P2G_momentum_and_mass<Real><<<CUDA_GRID_SIZE(m_data.m_num_particle), CUDA_BLOCK_SIZE>>>(m_dev_data);

    cudaMemset(m_data.dev_grid_normal, 0, sizeof(Real) * m_data.m_num_grid * 3);
    IQMPMSolverKernel::P2G_normal_estimate<Real><<<CUDA_GRID_SIZE(m_data.m_num_particle), CUDA_BLOCK_SIZE>>>(m_dev_data);
    IQMPMSolverKernel::grid_normal_normalize<Real><<<CUDA_GRID_SIZE(m_data.m_num_grid), CUDA_BLOCK_SIZE>>>(m_dev_data);

    IQMPMSolverKernel::calc_grids_velocity<Real><<<CUDA_GRID_SIZE(m_data.m_num_grid), CUDA_BLOCK_SIZE>>>(m_dev_data);
    IQMPMSolverKernel::grids_gravity<Real><<<CUDA_GRID_SIZE(m_data.m_num_grid), CUDA_BLOCK_SIZE>>>(m_dev_data);
    for(unsigned int it = 0; it < 10; ++it)
        IQMPMSolverKernel::grids_couple<Real><<<CUDA_GRID_SIZE(m_data.m_num_grid / m_data.m_num_object), CUDA_BLOCK_SIZE>>>(m_dev_data);
    IQMPMSolverKernel::grids_boundary_conditions<Real><<<CUDA_GRID_SIZE(m_data.m_num_grid), CUDA_BLOCK_SIZE>>>(m_dev_data);
    cudaMemset(m_data.dev_particle_velocity, 0, sizeof(Real) * m_data.m_num_particle * 3);
    cudaMemset(m_data.dev_particle_C, 0, sizeof(Real) * m_data.m_num_particle * 9);
    IQMPMSolverKernel::G2P_velocity_and_C<Real><<<CUDA_GRID_SIZE(m_data.m_num_particle), CUDA_BLOCK_SIZE>>>(m_dev_data);
    
    // cudaMemset(m_data.dev_max_particle_velocity, 0, sizeof(Real));
    // IQMPMSolverKernel::get_max_particle_velocity<Real><<<CUDA_GRID_SIZE(m_data.m_num_particle), CUDA_BLOCK_SIZE>>>(m_dev_data);
    // Real max_velocity;
    // cudaMemcpy(&max_velocity, m_data.dev_max_particle_velocity, sizeof(Real), cudaMemcpyDeviceToHost);
    // printf("max particle velocity = %.10f\n", max_velocity);
    // printf("max particle movement = %.10f\n", max_velocity * m_data.m_time_step);
    // assert(max_velocity * m_data.m_time_step <= m_data.m_grid_spacing);

    IQMPMSolverKernel::update_particle_positions<Real><<<CUDA_GRID_SIZE(m_data.m_num_particle), CUDA_BLOCK_SIZE>>>(m_dev_data);
    // IQMPMSolverKernel::particles_boundary_conditions<Real><<<CUDA_GRID_SIZE(m_data.m_num_particle), CUDA_BLOCK_SIZE>>>(m_dev_data);
}

template <typename Real>
Real *IQMPMSolver<Real>::GetDevicePositions()
{
    return m_data.dev_particle_position;
}

template class IQMPMSolver<float>;
template class IQMPMSolver<double>;
template struct IQMPMSolverData<float>;
template struct IQMPMSolverData<double>;
