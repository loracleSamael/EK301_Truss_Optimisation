# analysis.py
# Fast Python equilibrium solver used inside the optimisation loop.
#
# To include:
#   - Build the joint equilibrium matrix for a design (members plus pin Rx, pin Ry and roller Ry)
#   - Solve for member forces under a 1 oz downward load at the load joint (+ tension, - compression)
#   - Singularity / mechanism detection (condition number or rank check)
#   - Buckling capacity per member from its length
#   - Maximum load P_max: the smallest capacity/force ratio across compression members
#   - Per-member utilisation at P_max (used to guide mutations)
#   - Total cost: joints x 10 + total member length
