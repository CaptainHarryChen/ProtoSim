# ProtoSim

A framework for rapidly validating physical simulation algorithms.

## Algorithm

1. Explicit MPM. Includes elastic material and fluid.
2. Multi-grid explicit MPM. No sticking issue between objects. Includes elastic material and fluid.
3. Projective Dynamics. Use Chebyshev speed up and Jacobi iteration to solve implicit FEM.
4. PCG FEM. Use PCG iteration to solve implicit FEM.
5. PCG MPM. Use PCG iteration to solve implicit MPM. Includes elastic material and fluid.
6. Coupled PCG MPM. Use CPIC to couple the FEM and MPM.

