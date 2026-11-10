# config.py
# Single source of truth for all problem constants and settings. Nothing else should hard-code these.
#
# To include:
#   - Geometry limits: span range (30-33 in), member length range (8.5-14 in),
#     load joint height range above supports (0.5-1.5 in), vertical-strip tolerance under the load joint
#   - Material budget: total member length available (120 in)
#   - Cost model: cost per joint (10), cost per unit member length (1)
#   - Buckling fit: F = 3391.349 * L^-2.012 (F in oz, L in in)
#   - Joint index convention: 0 = pin, 1 = roller, 2 = load joint, 3+ = free joints
#   - Joint count range for random designs (derived from the material budget)
#   - Optimiser settings: number of starts, iterations, random seed, number of finalists sent to MATLAB
#   - Paths: MATLAB file location, results folders
