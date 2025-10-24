import pandas as pd
import matplotlib.pyplot as plt
import corner  # Use the 'corner' library instead of bilby

# --- Injection Parameters ---
# These are the "true" values from simon_goode.py, which will be
# overlaid on the corner plot.
injection_params = {
    "M_c": 35.0,
    "q": 0.9,
    "s1_z": 0.4,
    "s2_z": -0.3,
    "d_L": 1000.0,
    "iota": 0.4,
    "t_c": 0.0,
    "phase_c": 1.3,
    "ra": 1.375,
    "dec": -1.2108,
    "psi": 2.659,
}

# Calculate the derived 'chi_eff' for the injection parameters
injection_params["chi_eff"] = (
    injection_params["s1_z"] +
    injection_params["s2_z"] * injection_params["q"]
) / (1.0 + injection_params["q"])

# --- Parameters to Plot ---
# We'll select the main physical parameters, including the derived chi_eff
parameters_to_plot = [
    'M_c',
    'q',
    'chi_eff',
    'd_L',
    'iota',
    'ra',
    'dec',
    'psi',
    't_c'
]

# --- LaTeX Labels for Plot ---
# Create a dictionary for our labels
labels = {
    'M_c': r'$M_c$',
    'q': r'$q$',
    'chi_eff': r'$\chi_{\rm eff}$',
    'd_L': r'$d_L$',
    'iota': r'$\iota$',
    'ra': r'$\alpha$',
    'dec': r'$\delta$',
    'psi': r'$\psi$',
    't_c': r'$t_c$',
}

# --- Load the Data ---
# The CSV file from anesthetic has two extra header rows (labels, weights)
# We use pandas.read_csv to load it, specifying:
#   header=0: Use the first row as the column names
#   skiprows=[1, 2]: Skip the 'labels' and 'weights' rows
try:
    samples_df = pd.read_csv('results/simon_goode_results.csv', header=0, skiprows=[1, 2])
    num_samples = len(samples_df)
    print(f"Successfully loaded 'simon_goode_results.csv'. Found {num_samples} posterior samples.")
    
    # Drop any potential unnamed index columns if they exist
    unnamed_cols = [col for col in samples_df.columns if 'Unnamed:' in col]
    if unnamed_cols:
        print(f"Dropping unused columns: {unnamed_cols}")
        samples_df = samples_df.drop(columns=unnamed_cols)

    # --- Prepare Data for Corner Plot ---
    
    # 1. Select only the data columns we want to plot
    data_to_plot = samples_df[parameters_to_plot]
    
    # 2. Create an ordered list of labels for the plot
    ordered_labels = [labels[param] for param in parameters_to_plot]
    
    # 3. Create an ordered list of the "true" injection values
    ordered_truths = [injection_params[param] for param in parameters_to_plot]

    # --- Generate the Corner Plot ---
    output_filename = f'results/simon_goode_corner_NPOST{num_samples}.png'
    title_str = f'Posterior Corner Plot ({num_samples} posterior samples)'
    print(f"Generating corner plot and saving to '{output_filename}'...")

    # Set the font size for the plot
    plt.rcParams['font.size'] = 12

    # Call the corner library
    fig = corner.corner(
        data_to_plot,
        labels=ordered_labels,
        truths=ordered_truths,
        quantiles=[0.05, 0.5, 0.95],
        show_titles=True,
        title_quantiles=[0.05, 0.5, 0.95]
    )
    # Add number of posterior samples to the figure title
    plt.suptitle(title_str, fontsize=16)

    # Save the figure
    fig.savefig(output_filename, dpi=300, bbox_inches='tight')
    
    print("Corner plot generated successfully!")
    print(f"\nTo view the plot, check your directory for '{output_filename}'.")

except FileNotFoundError:
    print("Error: 'simon_goode_results.csv' not found.")
    print("Please make sure the CSV file is in the same directory as this script.")
except Exception as e:
    print(f"An error occurred: {e}")
    print("Please check the CSV file format.")

