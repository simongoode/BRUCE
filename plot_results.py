from anesthetic import read_chains
import matplotlib.pyplot as plt

# Load the nested sample results
samples = read_chains('results/simon_goode_results')

# Plot posterior distributions using anesthetic's built-in corner plot function
fig = samples.plot_2d()
fig.savefig("results/posterior_cornerplot_anesthetic.png")
