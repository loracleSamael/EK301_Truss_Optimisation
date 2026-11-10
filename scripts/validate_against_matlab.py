# validate_against_matlab.py
# One-off check to run before trusting the optimiser.
#
# To include:
#   - Generate about 50 random valid designs
#   - Analyse each with the Python solver and with matlab/truss_analysis.m
#   - Report member force and reaction differences and any P_max mismatch
#   - Flag sign convention or matrix shape problems
#   - Exit with a clear pass/fail result
