# visualise.py
# Draws the final trusses.
#
# To include:
#   - To-scale drawing (equal axes) of joints and members
#   - Member labels: length and force, optionally angle; red = compression, blue = tension
#   - Joint labels with index and (x, y) coordinates
#   - Pin, roller, load arrow with failure load, and span dimension line
#   - Header with P_max, cost, P/cost, joint/member counts and material used
#   - Member table: ID, joints, length, angle, force, tension/compression, buckling utilisation
#   - Accept MATLAB-verified forces so figures show the confirmed values
#   - Save figures to results/figures/
