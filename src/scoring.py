# scoring.py
# Turns a design into numbers the optimiser can minimise.
#
# To include:
#   - Constraint violation: graded penalty for span, nothing below supports, load joint height,
#     strip under the load joint, member length limits, 120 in material budget and member crossings
#   - Objective 1: maximise P_max (cost ignored)
#   - Objective 2: maximise P_max / cost
#   - Penalty handling for invalid or singular designs (large value that still guides the search)
#   - Logging of P_max and cost separately so the metric can be changed without re-running
