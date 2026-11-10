# optimiser.py
# The search loop that improves designs based on analysis feedback.
#
# To include:
#   - Multi-start local search: random starts, adaptive Gaussian geometry steps, accept improvements
#   - Occasional topology re-rolls on the candidate geometry
#   - Utilisation-guided moves: shorten or re-angle the governing compression member, trim low-force members
#   - Optional CMA-ES for the geometry inner loop
#   - A final polish step on the best few designs
#   - Evaluation cache keyed by canonical (mirror-reduced) design
#   - Run once per objective and return a ranked list of finalists
