# run_pipeline.py
# Entry point that runs the whole project.
#
# To include:
#   1. Load settings from config.py
#   2. Generate valid starting designs
#   3. Optimise for maximum load (cost ignored)
#   4. Optimise for maximum load / cost
#   5. Send the top few designs of each objective through MATLAB and compare with the Python results
#   6. Re-rank on the MATLAB results and select the final two trusses
#   7. Save the designs (JSON) and draw both trusses
#   8. Print a short summary
