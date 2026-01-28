"""
Gravitational Wave Parameter Estimation using BlackJAX Nested Sampling.
(Refactored for profiling)

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
import time # Added for timing

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
from blackjax.ns.utils import finalise
from anesthetic import NestedSamples
from jimgw.single_event.utils import Mc_q_to_m1_m2


# =============================================================================
# CONFIGURATION CONSTANTS (Defaults for direct execution)
# =============================================================================

# Detector and signal parameters
TRIGGER_TIME = 1126259642.413
DURATION = 4.0
POST_TRIGGER_DURATION = 2.0
F_REF = 50.0  # Reference frequency for waveform

# Nested sampling parameters
N_LIVE = 1400  # Number of live points
N_DELETE = int(N_LIVE * 0.5)  # Number of points to delete per iteration
TERMINATION_DLOGZ = 0.1  # Terminate when remaining evidence < this fraction

# Heterodyned likelihood parameters
N_BINS = 500  # Number of frequency bins for relative binning

# =============================================================================
# WAVEFORM AND DETECTOR SETUP
# =============================================================================

# Initialize waveform model
waveform = RippleIMRPhenomD(f_ref=F_REF)

# Noise curve paths
asd_paths = {
    "H1": f"{os.path.dirname(bilby.__file__)}/gw/detector/noise_curves/aLIGO_O4_high_asd.txt",
    "L1": f"{os.path.dirname(bilby.__file__)}/gw/detector/noise_curves/aLIGO_O4_high_asd.txt",
    "V1": f"{os.path.dirname(bilby.__file__)}/gw/detector/noise_curves/AdV_asd.txt",
}

# Injection parameters
injection_params = {
    "M_c": 35.0, "q": 0.9, "s1_z": 0.4, "s2_z": -0.3, "d_L": 1000.0,
    "iota": 0.4, "t_c": 0.0, "phase_c": 1.3, "ra": 1.375, "dec": -1.2108,
    "psi": 2.659,
}
injection_params["eta"] = injection_params["q"] / (1 + injection_params["q"]) ** 2

# =============================================================================
# DATA LOADING AND DETECTOR CONFIGURATION
# (This section runs ONCE on import)
# =============================================================================

print("Loading detector data...")
abs_path = 'blackjax_ns_gw/src/'
detector_data = {
    'frequencies': jnp.array(np.load(abs_path+'4s_frequency_array.npy')),
    'H1': jnp.array(np.load(abs_path+'4s_H1_strain.npy')),
    'L1': jnp.array(np.load(abs_path+'4s_L1_strain.npy')),
    'V1': jnp.array(np.load(abs_path+'4s_V1_strain.npy'))
}

freq_range = {'min': 20.0, 'max': 1024.0}
freq_mask = (detector_data['frequencies'] >= freq_range['min']) & (detector_data['frequencies'] <= freq_range['max'])
filtered_frequencies = detector_data['frequencies'][freq_mask]

detectors = [H1, L1, V1]
detector_names = ['H1', 'L1', 'V1']
for det, name in zip(detectors, detector_names):
    det.frequencies = filtered_frequencies
    det.data = detector_data[name][freq_mask]

def load_psd_data(asd_paths):
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
    return jnp.interp(det_frequencies, psd_frequencies, psd_values)

for det in detectors:
    det.psd = interpolate_psd(
        det.frequencies,
        psd_data[det.name]['frequencies'],
        psd_data[det.name]['psd']
    )
print("Detector configuration complete.")

# =============================================================================
# LIKELIHOOD INITIALIZATION
# (This section runs ONCE on import - it's the slow part)
# =============================================================================

print("Initializing Heterodyned Likelihood...")
print("This may take a few minutes as reference waveforms are computed...")
likelihood_init_start = time.perf_counter()

likelihood = HeterodynedTransientLikelihoodFD(
    detectors=detectors,
    waveform=waveform,
    trigger_time=TRIGGER_TIME,
    duration=DURATION,
    post_trigger_duration=POST_TRIGGER_DURATION,
    ref_params=injection_params,
    n_bins=N_BINS,
    marginalization="phase",
)
likelihood_init_end = time.perf_counter()
print(f"Heterodyned likelihood initialized successfully in {likelihood_init_end - likelihood_init_start:.2f}s.")
print(f"Using {len(likelihood.freq_grid_center)} frequency bins for relative binning.")

# =============================================================================
# PRIOR CONFIGURATION
# (This also runs ONCE on import)
# =============================================================================

sample_keys = ["M_c", "q", "s1_z", "s2_z", "iota", "d_L", "t_c", "psi", "ra", "dec"]
param_config = {
    "M_c": {"min": 25.0, "max": 50.0, "prior": "uniform", "wraparound": False, "description": "Chirp mass"},
    "q": {"min": 0.25, "max": 1.0, "prior": "uniform", "wraparound": False, "description": "Mass ratio"},
    "s1_z": {"min": -1.0, "max": 1.0, "prior": "uniform", "wraparound": False, "description": "Spin 1z"},
    "s2_z": {"min": -1.0, "max": 1.0, "prior": "uniform", "wraparound": False, "description": "Spin 2z"},
    "iota": {"min": 0.0, "max": jnp.pi, "prior": "sine", "wraparound": False, "description": "Inclination"},
    "d_L": {"min": 100.0, "max": 5000.0, "prior": "powerlaw", "wraparound": False, "description": "Distance"},
    "t_c": {"min": -0.1, "max": 0.1, "prior": "uniform", "wraparound": False, "description": "Coalescence time"},
    "psi": {"min": 0.0, "max": jnp.pi, "prior": "uniform", "wraparound": True, "description": "Polarization"},
    "ra": {"min": 0.0, "max": 2*jnp.pi, "prior": "uniform", "wraparound": True, "description": "Right ascension"},
    "dec": {"min": -jnp.pi/2, "max": jnp.pi/2, "prior": "cosine", "wraparound": False, "description": "Declination"},
}

@jax.jit
def prior_transform_fn(u_params):
    x_params = {}
    for key, u_val in u_params.items():
        config = param_config[key]
        param_type = config["prior"]
        if param_type == "uniform":
            x_params[key] = config["min"] + u_val * (config["max"] - config["min"])
        elif param_type == "sine":
            x_params[key] = jnp.arccos(1.0 - 2.0 * u_val)
        elif param_type == "cosine":
            x_params[key] = jnp.arcsin(2.0 * u_val - 1.0)
        elif param_type == "powerlaw":
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
# (This also runs ONCE on import)
# =============================================================================

def physical_loglikelihood_fn(params):
    p = injection_params.copy()
    p.update(params)
    p['eta'] = p['q'] / (1.0 + p['q']) ** 2
    return likelihood.evaluate(p, data=None)

# =============================================================================
# REFACTORED ANALYSIS FUNCTION
# =============================================================================

def run_analysis(n_live, n_delete, termination_dlogz, 
                 output_prefix=None, verbose=True):
    """
    Runs the full nested sampling analysis for a given set of parameters.

    Parameters:
    ----------
    n_live : int
        Number of live points.
    n_delete : int
        Number of points to delete per iteration.
    termination_dlogz : float
        Termination criterion for dlogZ.
    output_prefix : str, optional
        If provided, saves results to files with this prefix
        (e.g., "results/MY_PREFIX_results.csv"). If None, skips saving.
    verbose : bool, optional
        If True, prints progress and summary statistics to console.
    
    Returns:
    -------
    logZ : float
        The calculated Log Evidence.
    final_state : object
        The final state object from blackjax.ns.utils.finalise.
    """
    
    # ===================================
    # NESTED SAMPLING SETUP
    # ===================================
    if verbose:
        print("\nSetting up nested sampler...")

    def get_ravel_order(particles_dict):
        example = jax.tree_util.tree_map(lambda x: x[0], particles_dict)
        flat, _ = jax.flatten_util.ravel_pytree(example)
        test_dict = {key: float(i) for i, key in enumerate(particles_dict.keys())}
        test_flat, _ = jax.flatten_util.ravel_pytree(test_dict)
        order = []
        for val in test_flat:
            for key, test_val in test_dict.items():
                if abs(val - test_val) < 1e-10:
                    order.append(key)
                    break
        return order

    # Get sample_keys in correct ravel order
    test_particles = {key: jax.random.uniform(jax.random.PRNGKey(42), (100,)) for key in sample_keys}
    ordered_sample_keys = get_ravel_order(test_particles)
    if verbose:
        print(f"Parameter order for sampling: {ordered_sample_keys}")

    example_params = {key: 0.0 for key in ordered_sample_keys}
    rng_key = jax.random.PRNGKey(10)
    rng_key, init_key = jax.random.split(rng_key, 2)
    unit_cube_particles = init_unit_cube_particles(init_key, example_params, n_live)

    periodic_mask = jax.tree_util.tree_map(lambda _: False, example_params)
    for key in ordered_sample_keys:
        if param_config[key]["wraparound"]:
            periodic_mask[key] = True
    if verbose:
        print(f"Periodic parameters: {[k for k, v in periodic_mask.items() if v]}")

    unit_cube_fns = create_unit_cube_functions(
        physical_loglikelihood_fn=physical_loglikelihood_fn,
        prior_transform_fn=prior_transform_fn,
        mask_tree=periodic_mask
    )

    nested_sampler = acceptance_walk_sampler(
        logprior_fn=unit_cube_fns['logprior_fn'],
        loglikelihood_fn=unit_cube_fns['loglikelihood_fn'],
        nlive=n_live,
        n_target=60,
        max_mcmc=5000,
        num_delete=n_delete,
        stepper_fn=unit_cube_fns['stepper_fn'],
        max_proposals=1000
    )
    state = nested_sampler.init(unit_cube_particles)
    
    if verbose:
        print("Nested sampler initialized.")
        print(f"   Live points: {n_live}")
        print(f"   Delete per iteration: {n_delete}")

    # ===================================
    # SAMPLING LOOP
    # ===================================

    @jax.jit
    def one_step(carry, xs):
        state, k = carry
        k, subk = jax.random.split(k, 2)
        state, dead_point = nested_sampler.step(subk, state)
        return (state, k), dead_point

    def terminate(state):
        dlogz = jnp.logaddexp(0, state.logZ_live - state.logZ)
        return jnp.isfinite(dlogz) and dlogz < termination_dlogz

    if verbose:
        print("\nStarting nested sampling...")
        print(f"Termination criterion: remaining evidence < {termination_dlogz * 100:.0f}% of total")

    dead = []
    with tqdm.tqdm(desc="Dead points", unit=" dead points", disable=not verbose) as pbar:
        while not terminate(state):
            (state, rng_key), dead_info = one_step((state, rng_key), None)
            dead.append(dead_info)
            pbar.update(n_delete)

    if verbose:
        print("Nested sampling complete!")

    # ===================================
    # FINALIZATION AND OUTPUT
    # ===================================
    if verbose:
        print("\nFinalizing results...")

    final_state = finalise(state, dead)

    if output_prefix:
        # Create results directory if it doesn't exist
        output_dir = os.path.dirname(output_prefix)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        with open(f'{output_prefix}_final_state.pkl', 'wb') as f:
            pickle.dump(final_state, f)

        physical_particles = transform_to_physical(final_state.particles, prior_transform_fn)
        
        m1_samples, m2_samples = jax.vmap(Mc_q_to_m1_m2)(
            physical_particles["M_c"], physical_particles["q"]
        )
        physical_particles["m_1"] = m1_samples
        physical_particles["m_2"] = m2_samples
        physical_particles["M_tot"] = m1_samples + m2_samples
        physical_particles["chi_eff"] = (
            physical_particles["s1_z"] +
            physical_particles["s2_z"] * physical_particles["q"]
        ) / (1.0 + physical_particles["q"])
        physical_particles["eta"] = physical_particles["q"] / (1.0 + physical_particles["q"]) ** 2

        column_to_label = {
            "M_c": r"$M_c$", "q": r"$q$", "m_1": r"$m_1$", "m_2": r"$m_2$",
            "M_tot": r"$M_{\rm tot}$", "eta": r"$\eta$", "chi_eff": r"$\chi_{\rm eff}$",
            "d_L": r"$d_L$", "iota": r"$\iota$", "ra": r"$\alpha$", "dec": r"$\delta$",
            "s1_z": r"$s_{1z}$", "s2_z": r"$s_{2z}$", "t_c": r"$t_c$", "psi": r"$\psi$",
        }
        
        logL_birth = final_state.loglikelihood_birth.copy()
        logL_birth = jnp.where(jnp.isnan(logL_birth), -jnp.inf, logL_birth)

        samples = NestedSamples(
            physical_particles, logL=final_state.loglikelihood,
            logL_birth=logL_birth, labels=column_to_label,
            logzero=jnp.nan, dtype=jnp.float64,
        )

        samples.to_csv(f"{output_prefix}_results.csv")
        if verbose:
            print(f"Results saved to {output_prefix}_results.csv")

    if verbose:
        print(f"\nLog Evidence (logZ): {state.logZ:.2f}")

    return state.logZ, final_state


# =============================================================================
# MAIN EXECUTION BLOCK
# =============================================================================

if __name__ == "__main__":
    
    # Run the analysis with the default parameters defined at the top
    logZ, final_state = run_analysis(
        n_live=N_LIVE,
        n_delete=N_DELETE,
        termination_dlogz=TERMINATION_DLOGZ,
        output_prefix="results/simon_goode", # Saves files to results/
        verbose=True
    )

    # --- Print full summary (as in original script) ---
    
    # Need to re-calculate physical particles if not saved, 
    # or just extract from saved file.
    # For simplicity, we just re-transform here for the summary.
    physical_particles = transform_to_physical(final_state.particles, prior_transform_fn)
    m1_samples, m2_samples = jax.vmap(Mc_q_to_m1_m2)(
        physical_particles["M_c"], physical_particles["q"]
    )
    physical_particles["m_1"] = m1_samples
    physical_particles["m_2"] = m2_samples
    physical_particles["M_tot"] = m1_samples + m2_samples
    physical_particles["chi_eff"] = (
        physical_particles["s1_z"] +
        physical_particles["s2_z"] * physical_particles["q"]
    ) / (1.0 + physical_particles["q"])
    physical_particles["eta"] = physical_particles["q"] / (1.0 + physical_particles["q"]) ** 2


    print("\n" + "="*70)
    print("SUMMARY STATISTICS (from __main__)")
    print("="*70)
    print(f"jimgw version: 0.2.0")
    print(f"Log Evidence (logZ): {logZ:.2f}")
    print(f"Number of dead points: {len(final_state.loglikelihood)}")
    print(f"Number of live points: {N_LIVE}")
    print("\nPosterior means and std deviations (sampled parameters):")
    
    # Get the ordered keys from the final state
    run_sample_keys = list(final_state.particles.keys())
    
    for key in run_sample_keys:
        mean = jnp.mean(physical_particles[key])
        std = jnp.std(physical_particles[key])
        true_val = injection_params.get(key, None)
        if true_val is not None:
            print(f"   {key:8s}: {mean:10.4f} ± {std:8.4f}   (true: {true_val:10.4f})")
        else:
            print(f"   {key:8s}: {mean:10.4f} ± {std:8.4f}")

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
            print(f"   {key:8s}: {mean:10.4f} ± {std:8.4f}   (true: {true_val:10.4f})")
        else:
            print(f"   {key:8s}: {mean:10.4f} ± {std:8.4f}")
    
    print("="*70)