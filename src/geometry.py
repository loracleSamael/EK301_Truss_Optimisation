# geometry.py
# Pure geometry helpers with no knowledge of loads or scoring.
#
# To include:
#   - Conversion between the optimiser parameter vector and joint coordinates
#   - Member length and angle calculations
#   - Segment crossing test and a count of crossing member pairs (planarity check)
#   - Check for any joint directly below the load joint (strip rule, with tolerance)
#   - Check for members hanging vertically below the load joint
#   - Mirror/canonical form of a design (x -> L - x) for caching and de-duplication
