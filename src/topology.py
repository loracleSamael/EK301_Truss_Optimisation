# topology.py
# Builds and modifies the member layout (which joints are connected).
#
# To include:
#   - Sweep-line Henneberg construction: sort joints by x, attach each new joint to the two ends of a
#     legal boundary edge, so M = 2J - 3 holds and the truss is rigid and planar by construction
#   - Legality checks for an attachment (member lengths in range, no crossings, no near-collinear triangles)
#   - Random joint sampling and random valid design generator (start points for the optimiser)
#   - Seed designs: Warren, Pratt, Howe and arch-like layouts snapped to the constraints
#   - Topology mutations: re-roll attachments, add a joint, remove a low-force joint, flip a diagonal
