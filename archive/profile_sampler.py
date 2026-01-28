import time
import pandas as pd
import os
import numpy as np

# Import the refactored script.
# NOTE: This will trigger the one-time setup (data loading,
# likelihood init), which may take a few minutes. This is expected.
print("Importing refactored sampler and running one-time setup...")
import simon_goode_profiling as sampler
print("One-time setup complete. Starting profiling sweep.")

# =============================================================================
# PROFILING PARAMETERS
# =============================================================================

# Define the parameter space for the sweep
# Start small to find the "fast mode"
n_live_points = [10, 25, 50, 75, 100]

# It's often better to vary N_DELETE as a fraction of N_LIVE
delete_fractions = [0.5, 0.7, 0.9]

# Termination criterion
# You could also vary this, but we'll keep it fixed for now
termination_dlogz = 0.1

# Directory to store the (many) profiling outputs
output_dir = "results/profiling"
os.makedirs(output_dir, exist_ok=True)

# =============================================================================
# PROFILING LOOP
# =============================================================================

results = []
total_start_time = time.perf_counter()

try:
    for n_live in n_live_points:
        for frac in delete_fractions:
            n_delete = int(n_live * frac)
            
            # Ensure at least one point is deleted
            if n_delete == 0:
                n_delete = 1
            
            print("\n" + "="*50)
            print(f"Testing: N_LIVE={n_live}, N_DELETE={n_delete} (Frac={frac:.2f})")
            print("="*50)

            # Define a unique prefix for this run's output files
            prefix = f"profile_nlive{n_live}_ndelete{n_delete}"
            output_prefix = os.path.join(output_dir, prefix)

            # Start the timer
            start_time = time.perf_counter()

            # Run the analysis
            # We set verbose=False to keep the console clean
            # We set output_prefix=None to disable file saving and speed up the loop.
            # (Change to output_prefix=output_prefix if you want all files)
            logZ, _ = sampler.run_analysis(
                n_live=n_live,
                n_delete=n_delete,
                termination_dlogz=termination_dlogz,
                output_prefix=None, # Set to None to avoid saving files
                verbose=False       # Set to False for clean output
            )

            # Stop the timer
            end_time = time.perf_counter()
            exec_time_s = end_time - start_time

            print(f" -> Execution Time: {exec_time_s:.2f} s")
            print(f" -> Log Evidence (LogZ): {logZ:.2f}")

            # Store the results
            results.append({
                "N_LIVE": n_live,
                "N_DELETE": n_delete,
                "DeleteFrac": frac,
                "ExecTime_s": exec_time_s,
                "LogZ": logZ
            })

except KeyboardInterrupt:
    print("\nProfiling loop interrupted by user.")

total_end_time = time.perf_counter()
print(f"\nTotal profiling time: {(total_end_time - total_start_time) / 60:.2f} minutes")

# =============================================================================
# RESULTS ANALYSIS
# =============================================================================

if not results:
    print("No results collected.")
    exit()

# Convert results to a pandas DataFrame for easy analysis
df = pd.DataFrame(results)

# Sort by execution time
df = df.sort_values(by="ExecTime_s")

print("\n\n" + "="*70)
print("PROFILING SUMMARY")
print("="*70)
print(df.to_string(index=False, float_format="%.2f"))

# --- Fast Mode Analysis ---
print("\n\n" + "="*70)
print(f"FAST MODE ANALYSIS (Target < 120 seconds)")
print("="*70)

fast_modes = df[df['ExecTime_s'] < 120].copy()

if fast_modes.empty:
    print("No configurations finished under 120 seconds.")
    print("Consider testing with smaller N_LIVE values.")
else:
    # Find the run with the "best" LogZ among the fast ones
    # We use np.isclose to find runs near the max LogZ, as it can vary.
    max_logZ = fast_modes['LogZ'].max()
    fast_modes['IsBestLogZ'] = np.isclose(fast_modes['LogZ'], max_logZ, atol=0.5)
    
    # Sort by LogZ (best first)
    fast_modes = fast_modes.sort_values(by='LogZ', ascending=False)
    
    print("Configurations that finished under 2 minutes, sorted by LogZ (best first):")
    print(fast_modes.to_string(index=False, float_format="%.2f"))
    
    best_fast_mode = fast_modes.iloc[0]
    print("\n--- Recommendation ---")
    print("The best 'fast mode' configuration found is:")
    print(f"  N_LIVE   = {int(best_fast_mode['N_LIVE'])}")
    print(f"  N_DELETE = {int(best_fast_mode['N_DELETE'])}")
    print(f"  Time     = {best_fast_mode['ExecTime_s']:.2f} s")
    print(f"  LogZ     = {best_fast_mode['LogZ']:.2f}")

print("="*70)