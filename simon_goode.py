"""
Gravitational Wave Parameter Estimation using BlackJAX Nested Sampling.

This script, simon_goode.py, is an updated version of blackjax_4s.py,
designed to show how jimgw can more directly be intergrated. It performs parameter estimation for a
gravitational wave signal using the BlackJAX nested sampling library
and the jimgw toolkit.

Key Features:
-------------
- jimgw Version: 0.2.0
- Likelihood: Uses jimgw's HeterodynedTransientLikelihoodFD for efficient
  likelihood evaluation via relative binning. It is initialized with the
  known injection parameters as the reference parameters. Phase marginalization
  is enabled.
- Priors: Employs unit cube sampling, with prior transforms from the
  unit cube [0,1] to physical space. These transforms are based on the
  definitions within jimgw's prior classes.
- Sampler: Uses a custom acceptance-walk nested sampler built on BlackJAX.

How to Use:
-----------
1. Ensure you have jimgw==0.2.0, blackjax, and other dependencies installed.
2. Place the required data files ('4s_frequency_array.npy',
   '4s_H1_strain.npy', etc.) in the same directory.
3. Update the ASD (noise curve) paths if necessary.
4. Run the script from the command line: `python simon_goode.py`

How to Modify:
--------------
- Prior Configuration: Adjust the `param_config` dictionary to change
  prior bounds (min, max) and types ('uniform', 'sine', 'cosine',
  'powerlaw'). The script automatically constructs the necessary priors.
- Sampler Settings: Change `N_LIVE` and `N_DELETE` constants to control
  the nested sampler's behavior.
- Detector Setup: Modify the `detectors` list and `asd_paths` dictionary
  to change the detector network or noise curves.
- Heterodyned Likelihood: Adjust `N_BINS` to control the number of frequency
  bins used in the relative binning scheme (higher = more accurate but slower).

Technical Notes:
----------------
The heterodyned (relative binning) likelihood significantly speeds up
likelihood evaluations by pre-computing coefficients based on a reference
waveform. This is especially beneficial for nested sampling which requires
many likelihood evaluations.

Phase marginalization is handled automatically by the likelihood, so phase_c
is not included in the sampled parameters.

The unit cube sampling approach ([0,1]^n) with periodic boundary conditions
for angular parameters provides efficient exploration of the parameter space.

Author: Based on blackjax_4s.py, adapted for demonstration purposes
Date: 2025
"""

# Memory configuration
import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.6"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
import numpy as np
import blackjax
from astropy.time import Time
import tqdm
import pickle
import bilby

jax.config.update("jax_enable_x64", True)

# Import gravitational wave functions
from jimgw.single_event.detector import H1, L1, V1
from jimgw.single_event.likelihood import HeterodynedTransientLikelihoodFD
from jimgw.single_event.waveform import RippleIMRPhenomD

# Import custom BlackJAX nested sampling kernels
from blackjax_ns_gw.src.custom_kernels import (
    acceptance_walk_sampler,
    create_unit_cube_functions,
    init_unit_cube_particles,
    transform_to_physical
)

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# Detector and signal parameters
TRIGGER_TIME = 1126259642.413
DURATION = 4.0
POST_TRIGGER_DURATION = 2.0
F_REF = 50.0  # Reference frequency for waveform

# Nested sampling parameters
N_LIVE = 140  # Number of live points (calibrated to match Bilby compression)
N_DELETE = int(N_LIVE * 0.5)  # Number of points to delete per iteration
TERMINATION_DLOGZ = 0.1  # Terminate when remaining evidence < this fraction

# Heterodyned likelihood parameters
N_BINS = 500  # Number of frequency bins for relative binning

# =============================================================================
# WAVEFORM AND DETECTOR SETUP
# =============================================================================

# Initialize waveform model
waveform = RippleIMRPhenomD(f_ref=F_REF)

# Noise curve paths - UPDATE THESE PATHS FOR YOUR SYSTEM
asd_paths = {
    "H1": f"{os.path.dirname(bilby.__file__)}/gw/detector/noise_curves/aLIGO_O4_high_asd.txt",
    "L1": f"{os.path.dirname(bilby.__file__)}/gw/detector/noise_curves/aLIGO_O4_high_asd.txt",
    "V1": f"{os.path.dirname(bilby.__file__)}/gw/detector/noise_curves/AdV_asd.txt",
}

# Injection parameters - these define the "true" signal
injection_params = {
    "M_c": 35.0,      # Chirp mass (solar masses)
    "q": 0.9,         # Mass ratio (m2/m1, m1 >= m2)
    "s1_z": 0.4,      # Primary spin (aligned component)
    "s2_z": -0.3,     # Secondary spin (aligned component)
    "d_L": 1000.0,    # Luminosity distance (Mpc)
    "iota": 0.4,      # Inclination angle (rad)
    "t_c": 0.0,       # Coalescence time relative to trigger (s)
    "phase_c": 1.3,   # Coalescence phase (rad)
    "ra": 1.375,      # Right ascension (rad)
    "dec": -1.2108,   # Declination (rad)
    "psi": 2.659,     # Polarization angle (rad)
}

# Add derived parameter eta (needed by heterodyned likelihood)
injection_params["eta"] = injection_params["q"] / (1 + injection_params["q"]) ** 2

# =============================================================================
# DATA LOADING AND DETECTOR CONFIGURATION
# =============================================================================

print("Loading detector data...")

# Load detector data as JAX arrays
abs_path = 'blackjax_ns_gw/src/'
detector_data = {
    'frequencies': jnp.array(np.load(abs_path+'4s_frequency_array.npy')),
    'H1': jnp.array(np.load(abs_path+'4s_H1_strain.npy')),
    'L1': jnp.array(np.load(abs_path+'4s_L1_strain.npy')),
    'V1': jnp.array(np.load(abs_path+'4s_V1_strain.npy'))
}

# Configure detector frequency range
freq_range = {'min': 20.0, 'max': 1024.0}
freq_mask = (detector_data['frequencies'] >= freq_range['min']) & (detector_data['frequencies'] <= freq_range['max'])
filtered_frequencies = detector_data['frequencies'][freq_mask]

# Configure detectors
detectors = [H1, L1, V1]
detector_names = ['H1', 'L1', 'V1']

for det, name in zip(detectors, detector_names):
    det.frequencies = filtered_frequencies
    det.data = detector_data[name][freq_mask]

# Load PSD data for all detectors
def load_psd_data(asd_paths):
    """Load power spectral density data from ASD files."""
    psd_data = {}
    for name, path in asd_paths.items():
        f_np, asd_vals_np = np.loadtxt(path, unpack=True)
        psd_data[name] = {
            'frequencies': jnp.array(f_np),
            'psd': jnp.array(asd_vals_np**2)
        }
    return psd_data

psd_data = load_psd_data(asd_paths)

@jax.jit
def interpolate_psd(det_frequencies, psd_frequencies, psd_values):
    """Interpolate PSD to detector frequency grid."""
    return jnp.interp(det_frequencies, psd_frequencies, psd_values)

# Configure detector PSDs
for det in detectors:
    det.psd = interpolate_psd(
        det.frequencies,
        psd_data[det.name]['frequencies'],
        psd_data[det.name]['psd']
    )

print("Detector configuration complete.")

# =============================================================================
# LIKELIHOOD INITIALIZATION
# =============================================================================

print("Initializing Heterodyned Likelihood...")
print("This may take a few minutes as reference waveforms are computed...")

# Initialize the heterodyned likelihood from jimgw
# We use the injection parameters as the reference parameters, so no
# expensive maximization is needed. Phase marginalization is enabled.
likelihood = HeterodynedTransientLikelihoodFD(
    detectors=detectors,
    waveform=waveform,
    trigger_time=TRIGGER_TIME,
    duration=DURATION,
    post_trigger_duration=POST_TRIGGER_DURATION,
    ref_params=injection_params,  # Use injection params as reference
    n_bins=N_BINS,
    marginalization="phase",  # Enable phase marginalization
)

print("Heterodyned likelihood initialized successfully.")
print(f"Using {len(likelihood.freq_grid_center)} frequency bins for relative binning.")

# =============================================================================
# PRIOR CONFIGURATION
# =============================================================================

# Define sampled parameters
# NOTE: phase_c is excluded because it's marginalized in the likelihood
sample_keys = ["M_c", "q", "s1_z", "s2_z", "iota", "d_L", "t_c", "psi", "ra", "dec"]

# Parameter configuration
# Each parameter has: min, max, prior type, and wraparound flag
param_config = {
    "M_c": {
        "min": 25.0,
        "max": 50.0,
        "prior": "uniform",
        "wraparound": False,
        "description": "Chirp mass in solar masses"
    },
    "q": {
        "min": 0.25,
        "max": 1.0,
        "prior": "uniform",
        "wraparound": False,
        "description": "Mass ratio q = m2/m1 where m1 >= m2"
    },
    "s1_z": {
        "min": -1.0,
        "max": 1.0,
        "prior": "uniform",
        "wraparound": False,
        "description": "Aligned spin component of primary"
    },
    "s2_z": {
        "min": -1.0,
        "max": 1.0,
        "prior": "uniform",
        "wraparound": False,
        "description": "Aligned spin component of secondary"
    },
    "iota": {
        "min": 0.0,
        "max": jnp.pi,
        "prior": "sine",
        "wraparound": False,
        "description": "Inclination angle, sine prior for isotropic orientations"
    },
    "d_L": {
        "min": 100.0,
        "max": 5000.0,
        "prior": "powerlaw",
        "wraparound": False,
        "description": "Luminosity distance in Mpc, d_L^2 prior for uniform volume"
    },
    "t_c": {
        "min": -0.1,
        "max": 0.1,
        "prior": "uniform",
        "wraparound": False,
        "description": "Coalescence time relative to trigger in seconds"
    },
    "psi": {
        "min": 0.0,
        "max": jnp.pi,
        "prior": "uniform",
        "wraparound": True,
        "description": "Polarization angle, periodic at pi"
    },
    "ra": {
        "min": 0.0,
        "max": 2*jnp.pi,
        "prior": "uniform",
        "wraparound": True,
        "description": "Right ascension, periodic at 2pi"
    },
    "dec": {
        "min": -jnp.pi/2,
        "max": jnp.pi/2,
        "prior": "cosine",
        "wraparound": False,
        "description": "Declination, cosine prior for isotropic sky distribution"
    },
}

print("\nPrior Configuration:")
for key in sample_keys:
    cfg = param_config[key]
    print(f"  {key:8s}: [{cfg['min']:8.2f}, {cfg['max']:8.2f}] - {cfg['prior']:10s} - {cfg['description']}")

# =============================================================================
# PRIOR TRANSFORM FUNCTION
# =============================================================================

@jax.jit
def prior_transform_fn(u_params):
    """
    Transform unit cube samples [0,1] to physical parameter space.

    This function implements inverse CDF (percent-point function) transforms
    that map uniform [0,1] samples to the physical parameter space according
    to the prior distributions defined in param_config.

    The transforms are based on jimgw's prior definitions but adapted for
    unit cube sampling rather than normalizing flow sampling.

    Parameters:
    -----------
    u_params : dict
        Dictionary of parameters in unit cube [0,1]

    Returns:
    --------
    x_params : dict
        Dictionary of parameters in physical space

    Prior Types:
    ------------
    - uniform: p(x) = 1/(max-min)
      Transform: x = min + u*(max-min)

    - sine: p(x) = sin(x)/2 for x in [0, pi]
      Transform: x = arccos(1 - 2*u)
      Used for inclination to give isotropic orientations

    - cosine: p(x) = cos(x)/2 for x in [-pi/2, pi/2]
      Transform: x = arcsin(2*u - 1)
      Used for declination to give isotropic sky distribution

    - powerlaw: p(x) = (1+alpha)*x^alpha / (max^(1+alpha) - min^(1+alpha))
      Transform: x = (min^(1+alpha) + u*(max^(1+alpha) - min^(1+alpha)))^(1/(1+alpha))
      Used for luminosity distance with alpha=2 (d_L^2 prior for uniform volume)
    """
    x_params = {}

    for key, u_val in u_params.items():
        config = param_config[key]
        param_type = config["prior"]

        if param_type == "uniform":
            # Uniform prior: straightforward linear transform
            x_params[key] = config["min"] + u_val * (config["max"] - config["min"])

        elif param_type == "sine":
            # Sine prior: inverse CDF of sin(x)/2 for x in [0, pi]
            # CDF(x) = (1 - cos(x))/2, so inverse is arccos(1 - 2*u)
            x_params[key] = jnp.arccos(1.0 - 2.0 * u_val)

        elif param_type == "cosine":
            # Cosine prior: inverse CDF of cos(x)/2 for x in [-pi/2, pi/2]
            # CDF(x) = (sin(x) + 1)/2, so inverse is arcsin(2*u - 1)
            x_params[key] = jnp.arcsin(2.0 * u_val - 1.0)

        elif param_type == "powerlaw":
            # Power law prior with alpha=2 (d_L^2 prior)
            alpha = 2.0
            power = 1.0 + alpha
            min_val = config["min"]
            max_val = config["max"]
            term1 = min_val**power
            term2 = max_val**power - term1
            x_params[key] = (term1 + u_val * term2)**(1.0 / power)

        else:
            raise ValueError(f"Unknown prior type: {param_type}")

    return x_params

# =============================================================================
# LIKELIHOOD WRAPPER
# =============================================================================

def physical_loglikelihood_fn(params):
    """
    Wrapper for the jimgw likelihood.evaluate method.

    Takes a dictionary of physical parameters (sampled parameters only),
    adds derived parameters like eta, and calls the heterodyned likelihood.

    Parameters:
    -----------
    params : dict
        Dictionary of sampled physical parameters

    Returns:
    --------
    log_likelihood : float
        Log-likelihood value
    """
    # Create a full parameter dictionary by combining sampled parameters
    # with fixed injection parameters (for any non-sampled parameters)
    p = injection_params.copy()
    p.update(params)

    # Add derived parameter eta (symmetric mass ratio)
    # This is required by the waveform generator
    p['eta'] = p['q'] / (1.0 + p['q']) ** 2

    # Call the heterodyned likelihood
    # The 'data' argument is required by the base class but unused here,
    # as the data is already stored inside the likelihood object
    return likelihood.evaluate(p, data=None)

# =============================================================================
# NESTED SAMPLING SETUP
# =============================================================================

print("\nSetting up nested sampler...")

# Determine the order that JAX uses for flattening parameters
# This is needed for consistency with the unit cube wrapper
def get_ravel_order(particles_dict):
    """Determine the order that ravel_pytree uses for flattening."""
    example = jax.tree_util.tree_map(lambda x: x[0], particles_dict)
    flat, _ = jax.flatten_util.ravel_pytree(example)

    # Create a test dict with unique values to identify the order
    test_dict = {key: float(i) for i, key in enumerate(particles_dict.keys())}
    test_flat, _ = jax.flatten_util.ravel_pytree(test_dict)

    # The order is determined by the positions in the flattened array
    order = []
    for val in test_flat:
        for key, test_val in test_dict.items():
            if abs(val - test_val) < 1e-10:
                order.append(key)
                break
    return order

# Get sample_keys in correct ravel order
test_particles = {key: jax.random.uniform(jax.random.PRNGKey(42), (100,)) for key in sample_keys}
sample_keys = get_ravel_order(test_particles)

print(f"Parameter order for sampling: {sample_keys}")

# Create example parameter structure for initialization
example_params = {key: 0.0 for key in sample_keys}

# Initialize random key
rng_key = jax.random.PRNGKey(10)
rng_key, init_key = jax.random.split(rng_key, 2)

# Initialize unit cube particles [0,1]^n
unit_cube_particles = init_unit_cube_particles(init_key, example_params, N_LIVE)

# Create periodic mask for unit cube stepper
# This tells the sampler which parameters have periodic boundary conditions
periodic_mask = jax.tree_util.tree_map(lambda _: False, example_params)
for key in sample_keys:
    if param_config[key]["wraparound"]:
        periodic_mask[key] = True

print(f"Periodic parameters: {[k for k, v in periodic_mask.items() if v]}")

# Create the unit cube functions
# This wraps our physical likelihood and prior transform for unit cube sampling
unit_cube_fns = create_unit_cube_functions(
    physical_loglikelihood_fn=physical_loglikelihood_fn,
    prior_transform_fn=prior_transform_fn,
    mask_tree=periodic_mask
)

# Initialize the nested sampler
nested_sampler = acceptance_walk_sampler(
    logprior_fn=unit_cube_fns['logprior_fn'],
    loglikelihood_fn=unit_cube_fns['loglikelihood_fn'],
    nlive=N_LIVE,
    n_target=60,
    max_mcmc=5000,
    num_delete=N_DELETE,
    stepper_fn=unit_cube_fns['stepper_fn'],
    max_proposals=1000
)

# Initialize the sampler state
state = nested_sampler.init(unit_cube_particles)

print("Nested sampler initialized.")
print(f"  Live points: {N_LIVE}")
print(f"  Delete per iteration: {N_DELETE}")

# =============================================================================
# SAMPLING LOOP
# =============================================================================

@jax.jit
def one_step(carry, xs):
    """Single step of nested sampling."""
    state, k = carry
    k, subk = jax.random.split(k, 2)
    state, dead_point = nested_sampler.step(subk, state)
    return (state, k), dead_point

def terminate(state):
    """Termination criterion: stop when remaining evidence is small."""
    dlogz = jnp.logaddexp(0, state.logZ_live - state.logZ)
    return jnp.isfinite(dlogz) and dlogz < TERMINATION_DLOGZ

print("\nStarting nested sampling...")
print("Termination criterion: remaining evidence < 10% of total")

# Run nested sampling
dead = []
with tqdm.tqdm(desc="Dead points", unit=" dead points") as pbar:
    while not terminate(state):
        (state, rng_key), dead_info = one_step((state, rng_key), None)
        dead.append(dead_info)
        pbar.update(N_DELETE)

print("Nested sampling complete!")

# =============================================================================
# FINALIZATION AND OUTPUT
# =============================================================================

from blackjax.ns.utils import finalise
from anesthetic import NestedSamples

print("\nFinalizing results...")

# Finalize the nested sampling state
final_state = finalise(state, dead)

# Save the final state
with open('results/simon_goode_final_state.pkl', 'wb') as f:
    pickle.dump(final_state, f)

# Transform unit cube particles back to physical space
physical_particles = transform_to_physical(final_state.particles, prior_transform_fn)

# Add derived parameters for easier analysis
print("Calculating derived parameters...")
from jimgw.single_event.utils import Mc_q_to_m1_m2

# Add component masses
m1_samples, m2_samples = jax.vmap(Mc_q_to_m1_m2)(
    physical_particles["M_c"],
    physical_particles["q"]
)
physical_particles["m_1"] = m1_samples
physical_particles["m_2"] = m2_samples

# Add total mass
physical_particles["M_tot"] = m1_samples + m2_samples

# Add effective spin (chi_eff)
physical_particles["chi_eff"] = (
    physical_particles["s1_z"] +
    physical_particles["s2_z"] * physical_particles["q"]
) / (1.0 + physical_particles["q"])

# Add symmetric mass ratio (eta)
physical_particles["eta"] = physical_particles["q"] / (1.0 + physical_particles["q"]) ** 2

print("Derived parameters calculated.")

# Column labels for anesthetic output
column_to_label = {
    "M_c": r"$M_c$",
    "q": r"$q$",
    "m_1": r"$m_1$",
    "m_2": r"$m_2$",
    "M_tot": r"$M_{\rm tot}$",
    "eta": r"$\eta$",
    "chi_eff": r"$\chi_{\rm eff}$",
    "d_L": r"$d_L$",
    "iota": r"$\iota$",
    "ra": r"$\alpha$",
    "dec": r"$\delta$",
    "s1_z": r"$s_{1z}$",
    "s2_z": r"$s_{2z}$",
    "t_c": r"$t_c$",
    "psi": r"$\psi$",
}

# Clean up birth likelihoods (replace NaN with -inf)
logL_birth = final_state.loglikelihood_birth.copy()
logL_birth = jnp.where(jnp.isnan(logL_birth), -jnp.inf, logL_birth)

# Create NestedSamples object
samples = NestedSamples(
    physical_particles,
    logL=final_state.loglikelihood,
    logL_birth=logL_birth,
    labels=column_to_label,
    logzero=jnp.nan,
    dtype=jnp.float64,
)

# Save results
samples.to_csv("results/simon_goode_results.csv")
print("Results saved to results/simon_goode_results.csv")

# Save timing information
with open('results/simon_goode_timings.pkl', 'wb') as f:
    pickle.dump(pbar.format_dict, f)

# Print summary statistics
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)
print(f"jimgw version: 0.2.0")
print(f"Log Evidence (logZ): {state.logZ:.2f}")
print(f"Number of dead points: {len(dead) * N_DELETE}")
print(f"Number of live points: {N_LIVE}")
print("\nPosterior means and std deviations (sampled parameters):")
for key in sample_keys:
    mean = jnp.mean(physical_particles[key])
    std = jnp.std(physical_particles[key])
    true_val = injection_params.get(key, None)
    if true_val is not None:
        print(f"  {key:8s}: {mean:10.4f} ± {std:8.4f}  (true: {true_val:10.4f})")
    else:
        print(f"  {key:8s}: {mean:10.4f} ± {std:8.4f}")

print("\nDerived parameters:")
derived_keys = ["m_1", "m_2", "M_tot", "eta", "chi_eff"]
derived_true_vals = {
    "m_1": injection_params["M_c"] * (1 + injection_params["q"])**(1/5) / injection_params["q"]**(3/5),
    "m_2": injection_params["M_c"] * (1 + injection_params["q"])**(1/5) * injection_params["q"]**(2/5),
    "M_tot": injection_params["M_c"] * (1 + injection_params["q"])**(1/5) / injection_params["q"]**(3/5) * (1 + injection_params["q"]),
    "eta": injection_params["q"] / (1 + injection_params["q"])**2,
    "chi_eff": (injection_params["s1_z"] + injection_params["s2_z"] * injection_params["q"]) / (1 + injection_params["q"])
}
for key in derived_keys:
    mean = jnp.mean(physical_particles[key])
    std = jnp.std(physical_particles[key])
    true_val = derived_true_vals.get(key, None)
    if true_val is not None:
        print(f"  {key:8s}: {mean:10.4f} ± {std:8.4f}  (true: {true_val:10.4f})")
    else:
        print(f"  {key:8s}: {mean:10.4f} ± {std:8.4f}")

print("="*70)
print("\nAnalysis complete! Output files:")
print("  - simon_goode_results.csv: Posterior samples (including derived parameters)")
print("  - simon_goode_final_state.pkl: Final sampler state")
print("  - simon_goode_timings.pkl: Timing information")
print("\nTo analyze results, use anesthetic:")
print("  from anesthetic import NestedSamples")
print("  samples = NestedSamples(root='simon_goode_results')")
print("\nFor detailed corner plots:")
print("  samples.plot_2d(['M_c', 'q', 'd_L', 'iota'])")
print("="*70)