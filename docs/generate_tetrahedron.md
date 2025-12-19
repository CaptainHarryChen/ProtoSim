# Generate Tetrahedron

1. Get an `.obj` file of the model
2. Save the model as `.ply` with *MeshLab*
3. Get the *tetgen* from https://github.com/TetGen/TetGen
4. Compile the *tetgen* project
5. Run `tetgen -p -q xxx.obj` and get `.ele`, `.face`, `.node` files.
