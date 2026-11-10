# matlab_io.py
# The bridge between Python designs and the provided MATLAB analysis file.
#
# To include:
#   - Build the six input matrices from a design: connection matrix, x reaction matrix,
#     y reaction matrix, joint x and y locations, external loads matrix
#   - Coordinate rounding for export (e.g. 0.01 in)
#   - Run MATLAB, either by writing .mat files and calling the script or via a persistent matlab.engine session
#   - Parse the returned member force and reaction force matrices
#   - Keep sign conventions and matrix shapes in one place so they match truss_analysis.m
#   - Save each call's inputs and outputs to results/matlab_io/ for reproducibility
