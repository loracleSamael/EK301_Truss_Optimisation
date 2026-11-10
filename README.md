# Truss Optimiser

A pipeline that generates, analyses and optimises 2D pin-jointed bridge trusses.

Candidate trusses are generated as determinate, fully triangulated structures (M = 2J - 3)
spanning 30-33 in between a pin and a roller, subject to member length, joint height and
material (10 ft) constraints. Each design is analysed for its maximum load before a member
buckles, then scored and used to guide the generation of improved designs.

Two final trusses are selected:

1. **Maximum load**: highest load capacity, ignoring cost.
2. **Maximum load / cost**: best load capacity per unit cost (joint = 10, 1 per unit length of member).

Fast analysis runs in Python during optimisation; the provided MATLAB file
(`matlab/truss_analysis.m`) confirms the final designs. Both finalists are then drawn to scale
with member lengths, forces and joint coordinates.

## Structure

- `config.py`: problem constants and optimiser settings
- `matlab/`: provided MATLAB analysis file
- `src/`: geometry, topology generation, analysis, scoring, optimiser, MATLAB bridge, visualisation
- `scripts/`: pipeline entry point and MATLAB validation check
- `tests/`: unit tests
- `results/`: generated designs, MATLAB input/output and figures (git-ignored)

## Usage

*To be completed once the pipeline is implemented.*
