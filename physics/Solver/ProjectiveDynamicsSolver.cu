#include "ProjectiveDynamicsSolver.cuh"

template <typename Real>
ProjectiveDynamicsSolver<Real>::ProjectiveDynamicsSolver()
{
    // cudaMalloc((void**)&dev_masses, sizeof(Real) * this->m_nVerts);
    // cudaMemset(dev_masses, 0, sizeof(Real) * this->m_nVerts);
    // cudaMalloc((void**)&dev_masses_inv, sizeof(Real) * this->m_nVerts);

    // cudaMalloc((void**)&dev_volumes, sizeof(Real) * this->m_nTets);
    // cudaMemset(dev_volumes, 0, sizeof(Real) * this->m_nTets);
    // cudaMalloc((void**)&dev_positions_rest, sizeof(Real) * this->m_nVerts * 3);
    // cudaMemcpy(dev_positions_rest, this->dev_positions, sizeof(Real) * this->m_nVerts * 3, cudaMemcpyDeviceToDevice);

    // cudaMalloc((void**)&dev_invDm, sizeof(Real) * this->m_nTets * 9);

    // blocksPerGrid_verts = get_blocksPerGrid(this->m_nVerts);
    // blocksPerGrid_edges = get_blocksPerGrid(this->m_nEdges);
    // blocksPerGrid_tets = get_blocksPerGrid(this->m_nTets);

    // cudaMalloc((void**)&dev_positions_prev, sizeof(Real) * this->m_nVerts * 3);
    // cudaMalloc((void**)&dev_positions_last, sizeof(Real) * this->m_nVerts * 3);
    // cudaMalloc((void**)&dev_velocities_last, sizeof(Real) * this->m_nVerts * 3);
    // cudaMalloc((void**)&dev_positions_delta, sizeof(Real) * this->m_nVerts * 3);
    // cudaMalloc((void**)&dev_positions_delta_denominators, sizeof(Real) * this->m_nVerts);

    // cudaMalloc((void**)&dev_constrained, sizeof(unsigned int) * this->m_nVerts);
    // cudaMemcpy(dev_constrained, constrained.data(), sizeof(unsigned int) * this->m_nVerts, cudaMemcpyHostToDevice);

    // elasticity_initialization_kernel<Real> << < blocksPerGrid_tets, THREADS_PER_BLOCK >> > (dev_masses,
    //     dev_volumes, dev_invDm, dev_positions_rest, this->m_nTets, this->dev_tetrahedras, m_density, dev_constrained);
    // elasticity_mass_inverse_kernel <Real> << < blocksPerGrid_verts, THREADS_PER_BLOCK >> > (dev_masses_inv, dev_masses, this->m_nVerts, dev_constrained);

    // this->m_time_step = 1. / 200.;

    // cudaMalloc((void**)&dev_positions_next, sizeof(Real) * this->m_nVerts * 3);
    // cudaMemcpy(this->dev_positions_prev, this->dev_positions, sizeof(Real) * this->m_nVerts * 3, cudaMemcpyDeviceToDevice);
    // cudaMalloc((void**)&dev_positions_backup, sizeof(Real) * this->m_nVerts * 3);
    // cudaMalloc((void**)&dev_positions_delta, sizeof(Real) * this->m_nVerts * 3);
    // cudaMalloc((void**)&this->dev_inertia, sizeof(Real) * this->m_nVerts * 3);

    // cudaMalloc((void**)&dev_init_A, sizeof(Real) * this->m_nVerts);
    // cudaMalloc((void**)&dev_init_B, sizeof(Real) * this->m_nVerts * 3);

    // std::vector<unsigned int> verts_to_tets;
    // std::vector<unsigned int> verts_to_tets_offsets;
    // vertices_to_tetrahedras(tetrahedras, this->m_nTets, verts_to_tets, verts_to_tets_offsets);
    // cudaMalloc((void**)&dev_verts_to_tets, sizeof(unsigned int) * verts_to_tets.size());
    // cudaMalloc((void**)&dev_verts_to_tets_offsets, sizeof(unsigned int) * verts_to_tets_offsets.size());
    // cudaMemcpy(dev_verts_to_tets, verts_to_tets.data(), sizeof(unsigned int) * verts_to_tets.size(), cudaMemcpyHostToDevice);
    // cudaMemcpy(dev_verts_to_tets_offsets, verts_to_tets_offsets.data(), sizeof(unsigned int) * verts_to_tets_offsets.size(), cudaMemcpyHostToDevice);

    // cudaMalloc((void**)&dev_tet_force, sizeof(Real) * this->m_nTets * 12);
    // cudaMalloc((void**)&dev_vert_force, sizeof(Real) * this->m_nVerts * 3);
    // cudaMalloc((void**)&dev_vert_Hessian, sizeof(Real) * this->m_nVerts);

    // cudaMalloc((void**)&dev_energy, sizeof(Real));

    // // the size of diag_Hessian is m_nVerts
    // cudaMalloc((void**)&dev_diag_Hessian, sizeof(Real) * this->m_nVerts);
    // cudaMemset(dev_diag_Hessian, 0, sizeof(Real) * this->m_nVerts);
    // jacobi_preconditioning_kernel<Real> << < this->blocksPerGrid_tets, THREADS_PER_BLOCK >> > (this->m_nTets, this->dev_tetrahedras, m_stiffness,
    //     this->dev_volumes, this->dev_invDm, dev_diag_Hessian);

    // cudaMalloc((void**)&dev_constrained, sizeof(unsigned int) * this->m_nVerts);
    // cudaMemset(dev_constrained, 0, sizeof(unsigned int) * this->m_nVerts);

    // cudaMalloc((void**)&this->dev_ground_collision_count, sizeof(unsigned int));
    // cudaMalloc((void**)&this->dev_ground_collision_ids, sizeof(unsigned int) * this->m_nVerts);
}

template <typename Real>
ProjectiveDynamicsSolver<Real>::~ProjectiveDynamicsSolver()
{
}

template <typename Real>
void ProjectiveDynamicsSolver<Real>::Step()
{
    // cudaMemcpy(this->dev_position_backup, this->dev_position, sizeof(Real) * this->m_nVerts * 3, cudaMemcpyDeviceToDevice);

    // initial_guess_kernel<Real> << < this->blocksPerGrid_verts, THREADS_PER_BLOCK >> > (this->m_nVerts, this->m_time_step, this->dev_gravity, this->dev_position, this->dev_velocities, dev_constrained);
    // cudaMemcpy(this->dev_inertia, this->dev_position, sizeof(Real) * this->m_nVerts * 3, cudaMemcpyDeviceToDevice);
    // compute_mass_matrix_kernel<Real> << < this->blocksPerGrid_verts, THREADS_PER_BLOCK >> > (this->m_nVerts, this->dev_position, 1. / this->m_time_step, this->dev_masses, dev_init_A, dev_init_B);

    // Real omega;
    // Real rho = 0.9992;
    // unsigned int max_iter = 64;
    // for (unsigned int iter = 0; iter < max_iter; ++iter)
    // {
    //     projective_dynamics_constraint_kernel<Real> << <this->blocksPerGrid_tets, THREADS_PER_BLOCK >> > (this->m_nTets, this->dev_tetrahedras, m_stiffness, this->dev_position, this->dev_volumes, this->dev_invDm, dev_tet_force);

    //     cudaMemset(this->dev_ground_collision_count, 0, sizeof(unsigned int));
    //     ground_collision_detection<Real> << <this->blocksPerGrid_verts, THREADS_PER_BLOCK >> > (this->m_nVerts, this->dev_position, this->dev_ground_collision_count, this->dev_ground_collision_ids);
    //     this->m_ground_collision_count = visit_device(this->dev_ground_collision_count, 0);
    //     cudaMemset(this->dev_vert_force, 0, sizeof(Real) * this->m_nVerts * 3);
    //     cudaMemset(this->dev_vert_Hessian, 0, sizeof(Real) * this->m_nVerts);
    //     compute_ground_collision_force<Real> << <get_blocksPerGrid(this->m_ground_collision_count), THREADS_PER_BLOCK >> > (this->m_ground_collision_count, this->dev_position, this->m_collision_stiffness, this->dev_vert_force, this->dev_vert_Hessian, this->dev_ground_collision_ids);

    //     //compute_iteration_kernel<Real> << <this->blocksPerGrid_verts, THREADS_PER_BLOCK >> > (this->m_nVerts, this->dev_position, dev_position_next, dev_init_A, dev_init_B, dev_diag_Hessian, dev_verts_to_tets, dev_verts_to_tets_offsets,
    //     //    this->dev_tetrahedras, dev_tet_force, dev_constrained);

    //     compute_iteration_kernel<Real> << <this->blocksPerGrid_verts, THREADS_PER_BLOCK >> > (this->m_nVerts, this->dev_position, dev_position_next, dev_init_A, dev_init_B, dev_diag_Hessian, dev_verts_to_tets, dev_verts_to_tets_offsets,
    //         this->dev_tetrahedras, dev_tet_force, this->dev_vert_force, this->dev_vert_Hessian, this->dev_position_delta, dev_constrained);
    //     Real alpha = 1.;
    //     if (iter % 8 == 0)
    //     {
    //         alpha = this->line_searches();
    //     }
    //     //Real alpha = this->line_searches();
    //     update_position_kernel<Real> << <this->blocksPerGrid_verts, THREADS_PER_BLOCK >> > (this->m_nVerts, this->dev_position_next, this->dev_position, this->dev_position_delta, alpha);

    //     if (iter <= 10) omega = 1;
    //     else if (iter == 11) omega = 2 / (2 - rho * rho);
    //     else omega = 4 / (4 - rho * rho * omega);

    //     Chebyshev_kernel<Real> << <this->blocksPerGrid_verts, THREADS_PER_BLOCK >> > (this->m_nVerts, this->dev_position_prev, this->dev_position, this->dev_position_next, omega);

    //     swap<Real*>(this->dev_position, this->dev_position_prev);
    //     swap<Real*>(this->dev_position, this->dev_position_next);
    // }
    // update_velocity_kernel<Real> << <this->blocksPerGrid_verts, THREADS_PER_BLOCK >> > (this->m_nVerts, dev_position_backup, this->dev_position, 1. / this->m_time_step, this->dev_velocities);
}

template <typename Real>
Real *ProjectiveDynamicsSolver<Real>::GetDevicePositions()
{
    return this->m_data.dev_position;
}

template struct ProjectiveDynamicsSolverData<float>;
template struct ProjectiveDynamicsSolverData<double>;
template class ProjectiveDynamicsSolver<float>;
template class ProjectiveDynamicsSolver<double>;
